"""Stage: separate. Demucs v4 (htdemucs, MIT) splits each track into drums / bass / other / vocals; the VOCALS stem
(speech) replaces the track for the following stages, so music or TV behind the speaker is removed.

The stage first measures how much non-vocal energy there is. If the background is negligible
(< `apply_if_background_db_above` relative to the vocals), the original audio is kept: separation then only adds
artefacts. Results are reused when an interrupted run resumes.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np

from ..audio import ffmpeg, read_wav, rms_dbfs, write_wav
from ..common import TRX_HOME, env_version, sha256_file, write_json
from ..context import Ctx

MODEL_DIR = TRX_HOME / "toolchain" / "models" / "demucs-htdemucs"

try:
    import demucs  # noqa: F401
    import torch  # noqa: F401
    AVAILABLE = (MODEL_DIR / "SOURCE.json").is_file() and any(MODEL_DIR.glob("*.th"))
except ImportError:
    AVAILABLE = False


def _separate(x: np.ndarray, sr: int, params: dict) -> tuple[np.ndarray, np.ndarray]:
    """mono float32 at sr -> (vocals, background) mono at sr."""
    import torch
    from demucs.apply import apply_model
    from demucs.pretrained import get_model
    torch.manual_seed(0)
    model = get_model(params.get("model", "htdemucs"), repo=MODEL_DIR)
    model.eval()
    wav = torch.from_numpy(np.stack([x, x])).float()
    ref = wav.mean(0)
    m, s = ref.mean(), ref.std() + 1e-8
    with torch.no_grad():
        out = apply_model(model, ((wav - m) / s)[None], device="cpu", shifts=int(params.get("shifts", 1)),
                          split=True, overlap=float(params.get("overlap", 0.25)), progress=False)[0]
    out = out * s + m
    vi = model.sources.index("vocals")
    vocals = out[vi].mean(0).numpy()
    background = sum(out[i] for i in range(len(model.sources)) if i != vi).mean(0).numpy()
    return vocals.astype(np.float32), background.astype(np.float32)


def run(ctx: Ctx, params: dict) -> None:
    if params.get("engine", "demucs") != "demucs":
        from ..common import TrxError
        raise TrxError("CONFIG_INVALID", f"separate.engine '{params['engine']}' is not available (only 'demucs')")
    threshold = float(params.get("apply_if_background_db_above", -15.0))
    for tr in ctx.tracks:
        out48 = ctx.work / f"track{tr.index + 1}_48k_sep.wav"
        memo = ctx.work / f"separate_track{tr.index + 1}.json"
        key = hashlib.sha256(json.dumps({"in": sha256_file(tr.wav48), "params": params, "env": env_version()},
                                        sort_keys=True).encode()).hexdigest()
        if memo.is_file():
            mm = json.loads(memo.read_text(encoding="utf-8"))
            if mm.get("key") == key and (not mm["applied"] or out48.is_file()):
                if mm["applied"]:
                    tr.wav48 = out48
                    _derive16(ctx, tr)
                ctx.log.add("separate", f"track {tr.label}: reused saved result ({mm['decision']})",
                            "Same input and settings as before the interruption.")
                continue
        # Demucs works at 44.1 kHz: resample through ffmpeg (E12: decoding only via ffmpeg)
        tmp441 = ctx.work / f"track{tr.index + 1}_44k1.wav"
        ffmpeg(["-i", str(tr.wav48), "-ac", "1", "-ar", "44100", "-c:a", "pcm_s16le", str(tmp441)])
        x, sr = read_wav(tmp441)
        vocals, background = _separate(x.astype(np.float32), sr, params)
        tmp441.unlink(missing_ok=True)
        rel = rms_dbfs(background) - rms_dbfs(vocals)
        applied = rel > threshold
        decision = (f"applied: background {rel:.1f} dB vs speech (> {threshold} dB)" if applied else
                    f"kept the original: background only {rel:.1f} dB vs speech (<= {threshold} dB)")
        if applied:
            tmp = ctx.work / f"track{tr.index + 1}_vocals_44k1.wav"
            write_wav(tmp, vocals, sr)
            ffmpeg(["-i", str(tmp), "-ac", "1", "-ar", "48000", "-c:a", "pcm_s16le", str(out48)])
            tmp.unlink(missing_ok=True)
            tr.wav48 = out48
            ctx.custody.add(out48, "derived", "separate")
            _derive16(ctx, tr)
        write_json(memo, {"key": key, "applied": applied, "decision": decision, "background_vs_speech_db": round(rel, 1)})
        ctx.metrics["separation"] = decision
        ctx.log.add("separate", f"track {tr.label}: {decision}",
                    "Music/TV behind the voice is removed with Demucs (vocals stem) only when it is really there.",
                    model=params.get("model", "htdemucs"), background_vs_speech_db=round(rel, 1))


def _derive16(ctx: Ctx, tr) -> None:
    """Keep the 16 kHz track in step (used directly when the recipe has no denoise stage)."""
    tr.wav16 = ctx.work / f"track{tr.index + 1}_16k_sep.wav"
    if not tr.wav16.is_file():
        ffmpeg(["-i", str(tr.wav48), "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(tr.wav16)])
        ctx.custody.add(tr.wav16, "derived", "separate")
