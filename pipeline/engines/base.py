"""The ASR engine interface. Every engine adapter (faster-whisper, IndicConformer, whisper.cpp …) implements exactly
this, so recipes and the pipeline never depend on a specific model. Implemented from phase 1.

Rules for every engine:
- load from a local path in toolchain/models/<id>/ only (offline guard active; no downloads, no trust_remote_code)
- report the model sha256 (from SOURCE.json) so the run report can name the exact weights
- unload() must free GPU memory completely (one GPU model at a time)
- return Segment objects with confidences; never drop text silently
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Word:
    start: float
    end: float
    text: str
    probability: float | None = None


@dataclass
class Segment:
    start: float
    end: float
    text: str
    language: str
    avg_logprob: float | None = None      # Whisper-style; None if the engine has no equivalent
    no_speech_prob: float | None = None
    compression_ratio: float | None = None
    confidence: float | None = None       # engine-neutral 0..1 confidence when available
    words: list[Word] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)


class Engine:
    id: str = "base"
    languages: tuple[str, ...] = ()

    def load(self, device: str, params: dict) -> dict:
        """Load weights from toolchain/models. Return {'model_sha256': ..., 'device': ..., 'compute_type': ...}."""
        raise NotImplementedError

    def transcribe(self, wav_path: str, language: str | None, params: dict) -> list[Segment]:
        """Transcribe one 16 kHz mono wav (a VAD segment or a chunk)."""
        raise NotImplementedError

    def detect_language(self, wav_path: str, candidates: list[str]) -> list[tuple[str, float]]:
        """Top languages with probabilities, restricted to candidates. Engines without LID raise NotImplementedError."""
        raise NotImplementedError

    def unload(self) -> None:
        raise NotImplementedError
