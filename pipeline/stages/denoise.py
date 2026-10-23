"""Stage: denoise. DeepFilterNet3 (deep-filter binary, delay-compensated) -> ffmpeg afftdn -> copy-through.

Runs on the 48 kHz track, then re-derives the 16 kHz track. Never runs on near-silent tracks (artefacts).
"""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess

from ..audio import ffmpeg, read_wav, rms_dbfs, write_wav
from ..common import TRX_HOME, env_version, sha256_file, write_json
from ..context import Ctx

DEEP_FILTER = TRX_HOME / "toolchain" / "bin" / "deep-filter" / "deep-filter.exe"


def _deep_filter(src, out_dir, atten_db) -> tuple[bool, str]:
    if not DEEP_FILTER.is_file():
        return False, "deep-filter binary not installed"
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [str(DEEP_FILTER), "-D", "-a", str(float(atten_db)), "-o", str(out_dir), str(src)]
    res = subprocess.run(cmd, capture_output=True, timeout=7200)
    produced = out_dir / src.name
    if res.returncode != 0 or not produced.is_file():
        return False, (res.stderr or res.stdout).decode("utf-8", "replace")[-300:]
    return True, "ok"


def guard_spans(raw16, dn16, vad_params: dict, threshold: float, drop_db: float,
                win: float = 0.25) -> list[tuple[float, float]]:
    """Stretches that are speech in the ORIGINAL (Silero >= threshold, no padding) but that the denoiser turned down
    by >= drop_db: soft speech the denoiser treated as noise (on the Pappa call: short replies and a quiet sentence)."""
    from .vad import speech_regions
    sr = 16000
    spans: list[list[float]] = []
    for s, e in speech_regions(raw16, dict(vad_params, threshold=threshold, speech_pad_ms=0, min_silence_ms=200)):
        t = s
        while t < e:
            w = min(win, e - t)
            a, b = int(t * sr), int((t + w) * sr)
            if w >= 0.1 and rms_dbfs(raw16[a:b]) - rms_dbfs(dn16[a:b]) >= drop_db:
                if spans and abs(spans[-1][1] - t) < 1e-6:
                    spans[-1][1] = t + w
                else:
                    spans.append([t, t + w])
            t += win
    return [(round(a, 3), round(b, 3)) for a, b in spans]


def restore(raw48, dn48, spans, sr: int = 48000, fade: float = 0.02):
    """Put the original audio back into the denoised track on `spans`, with short crossfades."""
    import numpy as np
    out = dn48.copy()
    n = int(fade * sr)
    for s, e in spans:
        a, b = max(0, int(s * sr) - n), min(len(out), int(e * sr) + n)
        if b <= a:
            continue
        w = np.ones(b - a, dtype=np.float32)
        k = min(n, (b - a) // 2)
        if k:
            ramp = np.linspace(0.0, 1.0, k, dtype=np.float32)
            w[:k], w[-k:] = ramp, ramp[::-1]
        out[a:b] = w * raw48[a:b] + (1 - w) * dn48[a:b]
    return out


def run(ctx: Ctx, params: dict) -> None:
    order = [params.get("engine", "deep_filter"), *params.get("fallback", ["ffmpeg_afftdn", "copy_through"])]
    atten = params.get("attenuation_limit_db", 18.0)
    skip_below = float(params.get("skip_below_dbfs", -42.0))
    for tr in ctx.tracks:
        dn48 = ctx.work / f"track{tr.index + 1}_48k_dn.wav"
        wav16 = ctx.work / f"track{tr.index + 1}_16k_dn.wav"
        memo = ctx.work / f"denoise_track{tr.index + 1}.json"
        key = hashlib.sha256(json.dumps({"in": sha256_file(tr.wav48), "params": params, "env": env_version()},
                                        sort_keys=True).encode()).hexdigest()
        if memo.is_file() and dn48.is_file() and wav16.is_file():   # resumed run: same input, same settings
            m = json.loads(memo.read_text(encoding="utf-8"))
            if m.get("key") == key and m.get("out16") == sha256_file(wav16):
                tr.wav16_raw, tr.wav16 = tr.wav16, wav16
                ctx.log.add("denoise", f"track {tr.label}: reused saved result ({m['method']})",
                            "Same input and settings as before the interruption.")
                continue
        audio, _ = read_wav(tr.wav48)
        level = rms_dbfs(audio)
        used, note = "copy_through", ""
        if level < skip_below:
            note = f"track level {level:.1f} dBFS below {skip_below} dBFS: left untouched"
            shutil.copyfile(tr.wav48, dn48)
        else:
            for method in order:
                if method == "deep_filter":
                    ok, note = _deep_filter(tr.wav48, ctx.work / "deep_filter", atten)
                    if ok:
                        shutil.move(str(ctx.work / "deep_filter" / tr.wav48.name), dn48)
                        used = "deep_filter"
                        break
                elif method == "ffmpeg_afftdn":
                    try:
                        ffmpeg(["-i", str(tr.wav48), "-af", "afftdn=nr=12:nf=-50", "-c:a", "pcm_s16le", str(dn48)])
                        used, note = "ffmpeg_afftdn", "deep_filter unavailable or failed"
                        break
                    except Exception as e:  # noqa: BLE001
                        note = str(e)
                elif method == "copy_through":
                    shutil.copyfile(tr.wav48, dn48)
                    used = "copy_through"
                    break
        raw16 = tr.wav16                                        # the convert stage's (not denoised) 16 kHz track
        tr.wav16_raw = raw16
        tr.wav16 = ctx.work / f"track{tr.index + 1}_16k_dn.wav"
        ffmpeg(["-i", str(dn48), "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(tr.wav16)])
        restored = []
        if used != "copy_through" and params.get("speech_guard", True) and raw16 and raw16.is_file():
            r16, _ = read_wav(raw16)
            d16, _ = read_wav(tr.wav16)
            restored = guard_spans(r16, d16, ctx.stage_params("vad") or {}, float(params.get("speech_guard_threshold", 0.2)),
                                   float(params.get("speech_guard_db", 12.0)))
            if restored:
                r48, sr48 = read_wav(tr.wav48)
                d48, _ = read_wav(dn48)
                write_wav(dn48, restore(r48, d48[:len(r48)] if len(d48) >= len(r48) else
                                        __import__("numpy").pad(d48, (0, len(r48) - len(d48))), restored, sr48), sr48)
                ffmpeg(["-i", str(dn48), "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(tr.wav16)])
                ctx.log.add("denoise", f"track {tr.label}: kept the original audio on {len(restored)} stretch(es) "
                            f"({sum(b - a for a, b in restored):.1f} s) where the denoiser removed soft speech",
                            "Denoising must never erase speech: where the original is speech but the denoised track is "
                            f"{params.get('speech_guard_db', 12.0)} dB+ quieter, the original is kept.", spans=restored)
        ctx.custody.add(tr.wav16, "derived", "denoise")
        write_json(memo, {"key": key, "method": used, "out16": sha256_file(tr.wav16)})
        ctx.log.add("denoise", f"track {tr.label}: {used}",
                    "Noise is removed before recognition, limited so soft speech is not erased.",
                    method=used, attenuation_limit_db=atten, input_level_dbfs=round(level, 1), note=note)
