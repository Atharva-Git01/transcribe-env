"""Stage: diarize. Runs BEFORE recognise (canonical order), so it labels tracks / prepares speaker turns.

mode = "channel": true-stereo tracks are already SPK1 / SPK2 (convert). Crosstalk check happens after recognition
                  in polish-time via track energy comparison (see recognise flags).
mode = "sherpa_onnx": phase 4 (speaker turns on a single track).
mode = "none": speaker stays SPK? (not determined).
"""
from __future__ import annotations

from ..common import TrxError
from ..context import Ctx


def run(ctx: Ctx, params: dict) -> None:
    mode = params.get("mode", "none")
    if mode == "channel":
        if len(ctx.tracks) < 2:
            ctx.log.add("diarize", "channel mode not applicable (one track)",
                        "Speakers cannot be separated by channel on mono or duplicated-mono audio (E01).")
            return
        ctx.log.add("diarize", "speakers = channels", "True stereo: each channel is one party.",
                    labels=[t.label for t in ctx.tracks],
                    crosstalk_suppression_db=params.get("crosstalk_suppression_db"))
        ctx.metrics["diarization"] = "channel"
        return
    if mode == "sherpa_onnx":
        from . import diarize_sherpa
        diarize_sherpa.run(ctx, params)
        return
    if mode == "none":
        ctx.log.add("diarize", "no diarization", "Recipe says none; speaker = SPK? (not determined).")
        return
    raise TrxError("CONFIG_INVALID", f"unknown diarize mode '{mode}'")
