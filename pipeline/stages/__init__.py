"""Stage registry: canonical order, the phase each stage arrives in, and which ones are implemented now.

A recipe may only list stages from CANONICAL_ORDER, in that order. When a phase lands, its stage function is
registered in IMPLEMENTED and the plan marks it "pending" instead of "not_installed".
"""
from __future__ import annotations

CANONICAL_ORDER = [
    "convert", "separate", "denoise", "vad", "lid", "diarize",
    "recognise", "verify", "merge", "polish", "gate", "export",
]

ARRIVES_IN_PHASE = {
    "convert": 1, "separate": 4, "denoise": 1, "vad": 1, "lid": 1, "diarize": 1,
    "recognise": 1, "verify": 2, "merge": 2, "polish": 1, "gate": 1, "export": 1,
}

# name -> callable(run_ctx, params). Empty in phase 0: only probe and select (which are not recipe stages) exist.
IMPLEMENTED: dict = {}


def phase_for(name: str, params: dict) -> int:
    if name == "diarize" and params.get("mode") == "sherpa_onnx":
        return 4
    if name == "polish" and params.get("llm_cleanup") is True:
        return 3
    return ARRIVES_IN_PHASE[name]
