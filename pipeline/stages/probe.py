"""Stage: probe. Measures the audio and writes the passport. Uses only ffprobe, ffmpeg and numpy.

Everything here is a measurement with a stated method; nothing is guessed. Fields that need a model
(language, speaker count, music) stay null with a status saying which phase measures them.
"""
from __future__ import annotations

import json
import math
import re
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from ..common import TrxError, env_version, load_config, sha256_file, utc_now

_URL = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]*://")


def _db(power: float) -> float:
    return 10.0 * math.log10(power + 1e-12)


def _ffprobe(path: Path) -> dict:
    cmd = ["ffprobe", "-v", "error", "-show_format", "-show_streams", "-of", "json", str(path)]
    try:
        res = subprocess.run(cmd, capture_output=True, timeout=120)
    except FileNotFoundError as e:
        raise TrxError("INTERNAL_ERROR", "ffprobe not found on PATH") from e
    if res.returncode != 0:
        raise TrxError("INPUT_UNREADABLE", res.stderr.decode("utf-8", "replace").strip()[:300])
    return json.loads(res.stdout.decode("utf-8", "replace"))


def _loudness(path: Path) -> tuple[float | None, float | None]:
    cmd = ["ffmpeg", "-nostdin", "-hide_banner", "-nostats", "-i", str(path), "-map", "0:a:0",
           "-af", "ebur128=peak=true", "-f", "null", "-"]
    res = subprocess.run(cmd, capture_output=True, timeout=3600)
    text = res.stderr.decode("utf-8", "replace")
    summary = text[text.rfind("Summary:"):] if "Summary:" in text else ""
    i_match = re.search(r"I:\s*(-?[\d.]+)\s*LUFS", summary)
    tp_part = summary[summary.find("True peak:"):] if "True peak:" in summary else ""
    p_match = re.search(r"Peak:\s*(-?[\d.]+|-inf)\s*dBFS", tp_part)
    lufs = float(i_match.group(1)) if i_match else None
    tp = None
    if p_match and p_match.group(1) != "-inf":
        tp = float(p_match.group(1))
    return lufs, tp


def _advanced(path: Path, duration: float, thr: dict) -> dict | None:
    """VAD + in-speech SNR + LID. Raises ImportError when the phase 1 runtime is not installed."""
    import faster_whisper  # noqa: F401  (ImportError -> phase 0 behaviour)

    from .. import gpu
    from ..audio import decode_mono
    from ..common import load_config
    from ..engines import cache
    from ..lid import acoustic_decision, detect, lid_engine_id, refine, windows_from_speech
    from .vad import speech_regions

    audio = decode_mono(path, 16000)  # mono mix, float32
    vad_params = {"threshold": thr.get("vad_threshold", 0.4), "min_speech_ms": 200, "min_silence_ms": 400,
                  "speech_pad_ms": 200, "max_segment_s": 28.0}
    speech = speech_regions(audio, vad_params)
    speech_sec = sum(e - s for s, e in speech)
    res = {"active_ratio": round(min(1.0, speech_sec / max(duration, 1e-6)), 4),
           "activity_method": f"silero_vad_v1 (threshold {vad_params['threshold']}; speech {speech_sec:.1f} s)",
           "warnings": []}
    # in-speech SNR: frames inside speech regions; codec-gated frames (near digital silence) excluded
    frame = int(16000 * thr["frame_ms"] / 1000)
    dbs = []
    for s, e in speech:
        seg = audio[int(s * 16000):int(e * 16000)]
        n = seg.size // frame
        if n:
            p = np.mean(np.square(seg[:n * frame].reshape(n, frame), dtype=np.float64), axis=1)
            dbs.append(10.0 * np.log10(p + 1e-12))
    if dbs:
        d = np.concatenate(dbs)
        speech_level = float(np.percentile(d, 95))
        d = d[d > max(-80.0, speech_level - thr.get("gated_frame_below_peak_db", 60.0))]
        if d.size >= 20:
            noise_floor = float(np.percentile(d, 10))
            res.update(noise_floor=noise_floor, speech_level=speech_level, snr=max(0.0, speech_level - noise_floor),
                       noise_method="vad_inspeech_percentile_v1 (10th vs 95th percentile of 30 ms frames inside VAD "
                                    "speech, codec-gated frames excluded)")
    # music / TV behind the voice: Demucs on the 10 s window with the most speech — feeds the music rule. It runs
    # BEFORE language ID: the language-ID model stays on the GPU for recognition and leaves ~1 GB free (E34)
    try:
        from .separate import AVAILABLE as SEP_OK
    except ImportError:
        SEP_OK = False
    if SEP_OK and speech_sec >= 2.0:
        from ..audio import decode_window, rms_dbfs
        from .separate import _separate
        win = float(thr.get("music_window_s", 10.0))
        starts = [s for s, _ in speech] or [0.0]
        best = max(starts, key=lambda a: sum(max(0.0, min(e, a + win) - max(s, a)) for s, e in speech))
        x = decode_window(path, best, win, 44100)
        if x.size > 44100:
            voc, bg = _separate(x, 44100, {"model": "htdemucs"})
            rel = rms_dbfs(bg) - rms_dbfs(voc)
            limit = float(thr.get("music_background_db", -12.0))
            res["music"] = {"present": rel > limit, "background_vs_speech_db": round(rel, 1),
                            "status": f"measured: Demucs htdemucs on {best:.0f}-{best + win:.0f} s; background "
                                      f"{rel:.1f} dB vs speech (present above {limit} dB)"}

    # language ID on windows spread over the speech
    eng_id = lid_engine_id()
    if eng_id and speech_sec >= 1.0:
        cfg = load_config("defaults")["languages"]
        cands = list(cfg["priority"])
        eng, _ = cache.get(eng_id, gpu.choose_device(), {})     # kept for recognition when it is the same model (E34)
        trusted = getattr(eng, "lid_trusted", False)
        try:
            wins = windows_from_speech(speech, duration, int(cfg.get("lid_windows", 6)), float(cfg.get("lid_window_sec", 30)))
            lid = detect(eng, audio, wins, cands)
            ref = None
            if lid["top3"]:
                ref = acoustic_decision(lid, eng_id) if trusted else refine(eng, audio, speech, lid)
        finally:
            cache.done(eng_id, eng)
        if lid["top3"] and ref:
            ev = ref["evidence"]
            tops = [w["top"] for w in lid["per_window"]]
            res["language"] = {
                "top3": lid["top3"],
                "decision": {"lang": ref["decision"], "basis": ref["basis"], "confidence": ref["confidence"]},
                "code_switch_risk": ref["code_switch_risk"],
                "status": (f"measured with {eng_id}: acoustic per-window tops {tops}; no forced decoding (the engine "
                           "tells Marathi from Hindi itself)") if trusted else
                          (f"measured with {eng_id}: acoustic per-window tops {tops}; "
                           f"forced-decode log-prob mr {ev['ll_mr']} / hi {ev['ll_hi']} on {ev['groups']} speech groups; "
                           f"distinct markers mr {ev['mr_markers']} / hi {ev['hi_markers']}"),
            }
            weak_indic = ref["decision"] in ("mr", "hi") and (
                ref["confidence"] is None or ref["confidence"] < float(cfg.get("text_lid_min_confidence", 0.65)))
            if weak_indic:
                res["warnings"].append(f"language_ambiguous: decided {ref['decision']} on weak evidence "
                                       f"({ref['basis']}); confirm or rerun with --lang")
    elif speech_sec < 1.0:
        res["warnings"].append("almost_no_speech: Silero VAD found under 1 s of speech")

    # speaker count (phase 4, sherpa-onnx) for files up to [probe].diarize_max_minutes — feeds rule R3
    try:
        from .diarize_sherpa import AVAILABLE, diarize
    except ImportError:
        AVAILABLE = False
    if AVAILABLE and speech_sec >= 2.0 and duration <= 60 * float(thr.get("diarize_max_minutes", 15)):
        turns = diarize(audio)
        speakers = {t["speaker"] for t in turns}
        res["speakers"] = {"estimate": len(speakers),
                           "status": f"measured: sherpa-onnx diarization, {len(turns)} turns"}
    return res


def probe(path: Path) -> dict:
    thr = load_config("thresholds")["probe"]
    sr = int(load_config("defaults")["runtime"]["analysis_sample_rate"])
    s = str(path)
    if _URL.match(s):
        raise TrxError("INPUT_UNREADABLE", "URLs are not allowed; give a local file path")
    path = Path(path)
    if not path.is_file():
        raise TrxError("INPUT_NOT_FOUND", s)

    meta = _ffprobe(path)
    audio = [st for st in meta.get("streams", []) if st.get("codec_type") == "audio"]
    if not audio:
        raise TrxError("INPUT_NO_AUDIO", path.name)
    st = audio[0]
    fmt = meta.get("format", {})
    channels = max(1, min(int(st.get("channels") or 1), 8))

    # ---- streaming decode: memory stays flat for long files
    frame = int(sr * thr["frame_ms"] / 1000)
    bytes_per_frame = frame * channels * 4
    cmd = ["ffmpeg", "-nostdin", "-v", "error", "-i", s, "-map", "0:a:0", "-ac", str(channels),
           "-ar", str(sr), "-f", "f32le", "-"]
    window = np.hanning(frame).astype(np.float32)
    nfft = 1 << (frame - 1).bit_length()
    freqs = np.fft.rfftfreq(nfft, 1.0 / sr)

    total = 0
    peak = 0.0
    clip = 0
    sumsq = 0.0
    sxy = sx2 = sy2 = 0.0
    frame_db: list[np.ndarray] = []
    roll_db: list[float] = []
    roll_hz: list[float] = []
    frame_index = 0
    leftover = b""
    with tempfile.TemporaryFile() as err:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=err)
        while True:
            buf = proc.stdout.read(bytes_per_frame * 1000)
            if not buf:
                break
            data = leftover + buf
            usable = len(data) // bytes_per_frame * bytes_per_frame
            chunk, leftover = data[:usable], data[usable:]
            if not chunk:
                continue
            x = np.frombuffer(chunk, dtype=np.float32).reshape(-1, channels)
            total += x.shape[0]
            ax = np.abs(x)
            peak = max(peak, float(ax.max()))
            clip += int((ax >= thr["clip_level"]).sum())
            sumsq += float(np.square(x, dtype=np.float64).sum())
            if channels >= 2:
                a, b = x[:, 0].astype(np.float64), x[:, 1].astype(np.float64)
                sxy += float((a * b).sum())
                sx2 += float((a * a).sum())
                sy2 += float((b * b).sum())
            mono = x.mean(axis=1).reshape(-1, frame)
            power = np.mean(np.square(mono, dtype=np.float64), axis=1)
            db = 10.0 * np.log10(power + 1e-12)
            frame_db.append(db)
            # spectral rolloff on every 4th frame (deterministic subsample)
            for k in range(mono.shape[0]):
                if (frame_index + k) % 4:
                    continue
                spec = np.abs(np.fft.rfft(mono[k] * window, n=nfft)) ** 2
                tot = spec.sum()
                if tot <= 0:
                    continue
                idx = int(np.searchsorted(np.cumsum(spec), 0.95 * tot))
                roll_hz.append(float(freqs[min(idx, len(freqs) - 1)]))
                roll_db.append(float(db[k]))
            frame_index += mono.shape[0]
        proc.wait()
        err.seek(0)
        err_text = err.read().decode("utf-8", "replace").strip()
    if total == 0:
        raise TrxError("INPUT_UNREADABLE", err_text[:300] or "decoder produced no samples")

    duration = total / sr
    if duration < thr["min_duration_sec"]:
        raise TrxError("INPUT_TOO_SHORT", f"{duration:.2f}s")

    fdb = np.clip(np.concatenate(frame_db), -100.0, None)
    non_digital_silence = fdb[fdb > -90.0]
    basis = non_digital_silence if non_digital_silence.size >= 0.2 * fdb.size else fdb
    noise_floor = float(np.percentile(basis, 10))
    speech_level = float(np.percentile(basis, 95))
    snr = max(0.0, speech_level - noise_floor)
    active_threshold = max(noise_floor + thr["activity_margin_db"], thr["activity_min_dbfs"])
    active_ratio = float(np.mean(fdb > active_threshold))

    rdb, rhz = np.array(roll_db), np.array(roll_hz)
    if rhz.size:
        loud = rhz[rdb >= np.percentile(rdb, 50)]
        rolloff = float(np.median(loud if loud.size else rhz))
    else:
        rolloff = 0.0
    if rolloff <= thr["narrowband_max_rolloff_hz"]:
        band = "narrowband"
    elif rolloff <= thr["wideband_max_rolloff_hz"]:
        band = "wideband"
    else:
        band = "fullband"

    corr = level_diff = None
    if channels == 1:
        layout = "mono"
    else:
        corr = sxy / math.sqrt(sx2 * sy2) if sx2 > 0 and sy2 > 0 else 0.0
        level_diff = _db(sx2) - _db(sy2)
        if corr >= thr["dup_mono_min_correlation"] and abs(level_diff) <= thr["dup_mono_max_level_diff_db"]:
            layout = "duplicated_mono"
        else:
            layout = "true_stereo" if channels == 2 else "multichannel"

    lufs, tp = _loudness(path)
    clipping_ratio = clip / (total * channels)
    warnings = []

    # Phase 1+: Silero VAD activity, in-speech SNR (E09) and Whisper LID — measured BEFORE the recipe is chosen.
    noise_method = "energy_percentile_v1 (10th vs 95th percentile of 30 ms frame energy)"
    activity_method = (f"energy_v1 (frame > max(noise_floor + {thr['activity_margin_db']} dB, "
                       f"{thr['activity_min_dbfs']} dBFS))")
    language = {"top3": None, "code_switch_risk": None, "status": "not_measured: no LID engine installed"}
    speakers = {"estimate": None, "status": "not_measured: diarization not installed or file too long"}
    music = {"present": None, "background_vs_speech_db": None,
             "status": "not_measured: Demucs not installed or too little speech"}
    try:
        adv = _advanced(path, duration, thr)
    except ImportError:
        adv = None
    if adv:
        active_ratio = adv["active_ratio"]
        activity_method = adv["activity_method"]
        if adv.get("snr") is not None:
            noise_floor, speech_level, snr = adv["noise_floor"], adv["speech_level"], adv["snr"]
            noise_method = adv["noise_method"]
        if adv.get("language"):
            language = adv["language"]
        if adv.get("speakers"):
            speakers = adv["speakers"]
        if adv.get("music"):
            music = adv["music"]
        warnings += adv.get("warnings", [])
    if layout == "duplicated_mono":
        warnings.append("duplicated_mono: both channels carry the same signal; treated as mono (never as 2 speakers)")
    if clipping_ratio > thr["warn_clipping_ratio"]:
        warnings.append(f"clipping: {clipping_ratio:.4%} of samples at full scale")
    if snr < thr["warn_low_snr_db"]:
        warnings.append(f"low_snr: estimated {snr:.1f} dB")
    if lufs is not None and lufs < thr["warn_quiet_lufs"]:
        warnings.append(f"very_quiet: {lufs:.1f} LUFS")
    if active_ratio < thr["warn_low_activity_ratio"]:
        warnings.append(f"low_activity: only {active_ratio:.0%} of frames above the noise floor")
    if layout == "multichannel":
        warnings.append("multichannel: only channels 1-2 were compared")
    if err_text:
        warnings.append(f"decoder_messages: {err_text[:200]}")

    def r(v, n=2):
        return None if v is None else round(float(v), n)

    return {
        "schema": "trx.passport/1",
        "env_version": env_version(),
        "created_utc": utc_now(),
        "source": {"file_name": path.name, "sha256": sha256_file(path), "size_bytes": path.stat().st_size},
        "container": {
            "format_name": str(fmt.get("format_name", "")),
            "duration_sec": round(duration, 3),
            "bit_rate": int(fmt["bit_rate"]) if str(fmt.get("bit_rate", "")).isdigit() else None,
        },
        "stream": {
            "codec": str(st.get("codec_name", "")),
            "sample_rate": int(st.get("sample_rate") or 0) or sr,
            "channels": int(st.get("channels") or 1),
            "channel_layout": st.get("channel_layout"),
        },
        "channels": {
            "count": channels,
            "layout_class": layout,
            "inter_channel_correlation": r(corr, 4),
            "level_difference_db": r(level_diff),
        },
        "levels": {
            "integrated_lufs": r(lufs, 1),
            "true_peak_dbtp": r(tp, 1),
            "rms_dbfs": r(_db(sumsq / (total * channels))),
            "peak_dbfs": r(20.0 * math.log10(peak + 1e-12)),
            "clipping_ratio": round(clipping_ratio, 6),
        },
        "noise": {
            "noise_floor_dbfs": r(noise_floor),
            "speech_level_dbfs": r(speech_level),
            "snr_db_estimate": r(snr),
            "method": noise_method,
        },
        "bandwidth": {"rolloff95_hz_median": r(rolloff, 0), "class": band, "analysis_sample_rate": sr},
        "activity": {"active_ratio": round(active_ratio, 4), "method": activity_method},
        "language": language,
        "speakers": speakers,
        "music": music,
        "warnings": warnings,
    }
