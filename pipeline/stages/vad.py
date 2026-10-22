"""Stage: vad. Silero VAD (the ONNX model bundled with faster-whisper, no torch) -> speech regions per track.

The same parameters are handed to the recognise stage, so segmentation is identical everywhere.
"""
from __future__ import annotations

from ..audio import read_wav
from ..common import TrxError, write_json
from ..context import Ctx


def speech_regions(audio, vad_params: dict) -> list[tuple[float, float]]:
    from faster_whisper.vad import get_speech_timestamps

    from ..engines.faster_whisper_engine import vad_options
    ts = get_speech_timestamps(audio, vad_options(vad_params), sampling_rate=16000)
    return [(round(t["start"] / 16000, 3), round(t["end"] / 16000, 3)) for t in ts]


def gap_audit(audio, regions, params: dict) -> list[tuple[float, float]]:
    """Gaps between speech regions that still hold speech-level sound are checked again with a lower VAD threshold
    (quiet speech that the main pass rejected, PLAN P4). Returns the extra regions found (absolute seconds)."""
    import numpy as np
    sr, frame = 16000, 480                                    # 30 ms frames
    n = audio.size // frame
    if n < 10:
        return []
    db = 10 * np.log10(np.mean(np.square(audio[:n * frame].reshape(n, frame), dtype=np.float64), axis=1) + 1e-12)
    live = db[db > -90.0]                                     # denoised audio has digital silence (-120 dB)
    if live.size < 10:
        return []
    floor = float(np.percentile(live, 10))
    min_gap = float(params.get("gap_min_s", 0.3))
    above = float(params.get("gap_energy_db", 15.0))
    low = dict(params, threshold=float(params.get("gap_threshold", 0.2)), speech_pad_ms=100)
    edges = [0.0] + [x for r in regions for x in r] + [audio.size / sr]
    found = []
    for a, b in zip(edges[0::2], edges[1::2]):
        if b - a < min_gap:
            continue
        f0, f1 = int(a * sr / frame), int(b * sr / frame)
        if f1 <= f0 or np.mean(db[f0:f1] > floor + above) < 0.3:   # mostly quiet: a real pause
            continue
        ctx_s = 0.5                                           # Silero needs a little context on each side
        c0, c1 = max(0.0, a - ctx_s), min(audio.size / sr, b + ctx_s)
        for s, e in speech_regions(audio[int(c0 * sr):int(c1 * sr)], low):
            s, e = max(a, c0 + s), min(b, c0 + e)               # keep only the part inside the gap
            if e - s >= 0.2:
                found.append((round(s, 3), round(e, 3)))
    return found


def run(ctx: Ctx, params: dict) -> None:
    if params.get("engine", "silero") != "silero":
        raise TrxError("CONFIG_INVALID", f"vad.engine '{params['engine']}' is not available (only 'silero')")
    total_speech = 0.0
    for tr in ctx.tracks:
        audio, _ = read_wav(tr.wav16)
        tr.speech = speech_regions(audio, params)
        quiet = gap_audit(audio, tr.speech, params) if params.get("gap_audit", True) else []
        if quiet:
            tr.speech = sorted(tr.speech + quiet)
            ctx.quiet_regions += [{"track": tr.label, "start": s, "end": e} for s, e in quiet]
            ctx.log.add("vad", f"track {tr.label}: {len(quiet)} quiet speech region(s) recovered from pauses",
                        "A pause that still held speech-level sound was checked again with a lower threshold "
                        "(quiet speech the first pass rejected).", regions=quiet)
        sp = sum(e - s for s, e in tr.speech)
        total_speech += sp
        path = ctx.work / f"vad_track{tr.index + 1}.json"
        write_json(path, {"track": tr.label, "params": params, "regions": tr.speech, "quiet_regions": quiet})
        ctx.custody.add(path, "derived", "vad")
        ctx.log.add("vad", f"track {tr.label}: {len(tr.speech)} speech regions, {sp:.1f} s of speech",
                    "Only speech is sent to recognition; silence is where Whisper invents text.",
                    threshold=params.get("threshold"), regions=len(tr.speech), speech_sec=round(sp, 2))
    ctx.metrics["speech_sec"] = round(total_speech, 2)
