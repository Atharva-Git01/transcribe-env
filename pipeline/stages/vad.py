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


def run(ctx: Ctx, params: dict) -> None:
    if params.get("engine", "silero") != "silero":
        raise TrxError("CONFIG_INVALID", f"vad.engine '{params['engine']}' is not available (only 'silero')")
    total_speech = 0.0
    for tr in ctx.tracks:
        audio, _ = read_wav(tr.wav16)
        tr.speech = speech_regions(audio, params)
        sp = sum(e - s for s, e in tr.speech)
        total_speech += sp
        path = ctx.work / f"vad_track{tr.index + 1}.json"
        write_json(path, {"track": tr.label, "params": params, "regions": tr.speech})
        ctx.custody.add(path, "derived", "vad")
        ctx.log.add("vad", f"track {tr.label}: {len(tr.speech)} speech regions, {sp:.1f} s of speech",
                    "Only speech is sent to recognition; silence is where Whisper invents text.",
                    threshold=params.get("threshold"), regions=len(tr.speech), speech_sec=round(sp, 2))
    ctx.metrics["speech_sec"] = round(total_speech, 2)
