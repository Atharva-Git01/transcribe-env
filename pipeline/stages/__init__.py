"""Stage registry: canonical order, the phase each stage arrives in, and which ones are implemented now.

A recipe may only list stages from CANONICAL_ORDER, in that order. Implemented stages are registered in IMPLEMENTED
(name -> run(ctx, params)); anything else is "not_installed" in the plan.
"""
from __future__ import annotations

import importlib

CANONICAL_ORDER = [
    "convert", "separate", "denoise", "vad", "lid", "diarize",
    "recognise", "verify", "merge", "polish", "gate", "export",
]

ARRIVES_IN_PHASE = {
    "convert": 1, "separate": 4, "denoise": 1, "vad": 1, "lid": 1, "diarize": 1,
    "recognise": 1, "verify": 2, "merge": 2, "polish": 1, "gate": 1, "export": 1,
}


def _available() -> dict:
    """Register a stage only if its runtime dependencies import (phase 0 env has no faster-whisper)."""
    found = {}
    try:
        importlib.import_module("faster_whisper")
        has_asr = True
    except ImportError:
        has_asr = False
    for name in ("convert", "denoise", "vad", "lid", "diarize", "recognise", "verify", "polish", "gate", "export",
                 "separate"):
        if name in ("vad", "lid", "recognise", "verify") and not has_asr:
            continue
        try:
            mod = importlib.import_module(f"{__name__}.{name}")
        except ImportError:
            continue
        if getattr(mod, "AVAILABLE", True):
            found[name] = mod.run
    return found


IMPLEMENTED: dict = _available()


def phase_for(name: str, params: dict) -> int:
    if name == "diarize" and params.get("mode") == "sherpa_onnx":
        return 4
    if name == "polish" and params.get("llm_cleanup") is True:
        return 3
    return ARRIVES_IN_PHASE[name]


def is_available(name: str, params: dict) -> bool:
    if name not in IMPLEMENTED:
        return False
    if name == "diarize" and params.get("mode") == "sherpa_onnx":
        try:
            from . import diarize_sherpa
            return diarize_sherpa.AVAILABLE
        except ImportError:
            return False
    return True
