"""Indic-Transcribe-core (Bodhan AI / AI4Bharat, Indic Open Model License v1.0): Canary FastConformer + Transformer
decoder, 1.2B parameters, 25+ languages incl. Indian English, code-mixed speech, native-script output.

Uses the model repo's reviewed inference code (revision e083ab7d) from the LOCAL model folder:
indic_transcribe.IndicTranscribe.from_pretrained(<folder>) (no download), transcribe(array, lang, mode="native").
"native" = the owner's convention (standard native spelling, numbers as words, English words in Devanagari).

torchaudio is only used by that code to resample non-16 kHz audio. The environment always hands it 16 kHz audio
(ffmpeg), so a small stand-in module is registered when torchaudio is absent; it raises if resampling is requested.

Licence (credit required): "Built with Indic-Transcribe-core from Bodhan AI / AI4Bharat". Hosting it as a service for
others needs Bodhan AI's written approval; no automated legal decisions without human oversight.
"""
from __future__ import annotations

import gc
import importlib
import importlib.machinery
import sys
import types

import numpy as np

from ..common import TrxError
from ..models import model_dir, require, source_info

LANGS = {"as", "bgc", "bhb", "bho", "bn", "brx", "doi", "en", "gu", "hi", "hne", "kn", "kok", "ks", "mai", "ml", "mni",
         "mr", "ne", "or", "pa", "sa", "sat", "sd", "ta", "te", "ur"}


def _torchaudio_stand_in() -> None:
    try:
        importlib.import_module("torchaudio")
        return
    except ImportError:
        pass

    def _no(*_a, **_k):
        raise TrxError("INTERNAL_ERROR", "Indic-Transcribe asked to resample: the environment always passes 16 kHz audio")

    import transformers  # noqa: F401  (import first: it must see torchaudio as NOT installed)
    ta = types.ModuleType("torchaudio")
    ta.__spec__ = importlib.machinery.ModuleSpec("torchaudio", None)   # no distribution metadata -> "not available"
    ta.transforms = types.SimpleNamespace(Resample=_no)
    ta.functional = types.SimpleNamespace(resample=_no)
    ta.__trx_stand_in__ = True
    sys.modules["torchaudio"] = ta


class IndicTranscribeEngine:
    family = "indic_transcribe"

    def __init__(self, engine_id: str):
        require(engine_id)
        self.id = engine_id
        self.path = model_dir(engine_id)
        self.asr = None
        self.max_s = 30.0

    def load(self, device: str, params: dict) -> dict:
        import torch
        torch.manual_seed(0)
        torch.set_num_threads(int(params.get("threads", 8)))
        _torchaudio_stand_in()
        sys.path.insert(0, str(self.path))           # the repo's flat imports (lid, modeling_indic_canary, …)
        try:
            mod = importlib.import_module("indic_transcribe")
        finally:
            sys.path.remove(str(self.path))
        want = params.get("device", "auto")
        dev = "cuda" if want in ("auto", "cuda") and torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if dev == "cuda" and params.get("dtype", "bfloat16") == "bfloat16" else torch.float32
        self.asr = mod.IndicTranscribe.from_pretrained(str(self.path), device=dev, dtype=dtype)
        self.max_s = float(params.get("piece_max_s", 30.0))
        return {"engine": self.id, "device": dev, "compute_type": str(dtype).replace("torch.", ""),
                "model_sha256": source_info(self.id).get("model_sha256")}

    def detect_language(self, audio: np.ndarray, candidates: list[str]) -> dict:
        top = self.asr.identify(np.asarray(audio, dtype=np.float32), topk=10)
        probs = {lang: p for lang, p in top if lang in candidates}
        total = sum(probs.values()) or 1.0
        return {c: probs.get(c, 0.0) / total for c in candidates}

    def transcribe(self, audio: np.ndarray, language: str | None, fw: dict, vad: dict | None,
                   initial_prompt: str | None, word_timestamps: bool, clip_timestamps: list[float] | None = None) -> list[dict]:
        lang = language or "mr"
        if lang not in LANGS:
            raise TrxError("CONFIG_INVALID", f"Indic-Transcribe does not support language '{lang}'")
        dur = len(audio) / 16000.0
        if clip_timestamps:
            pieces = list(zip(clip_timestamps[0::2], clip_timestamps[1::2]))
        elif vad is not None:
            from ..stages.recognise import speech_groups
            from ..stages.vad import speech_regions
            pieces = speech_groups(speech_regions(audio, vad), self.max_s, 0.8)
        else:
            pieces = [(a, min(dur, a + self.max_s)) for a in np.arange(0.0, dur, self.max_s)]
        out = []
        for a, b in pieces:
            while b - a > self.max_s + 1e-6:                # never past the 30 s the model was trained on
                out += self._piece(audio, a, a + self.max_s, lang)
                a += self.max_s
            out += self._piece(audio, a, b, lang)
        return out

    def _piece(self, audio, a: float, b: float, lang: str) -> list[dict]:
        x = np.ascontiguousarray(audio[int(a * 16000):int(b * 16000)], dtype=np.float32)
        if len(x) < 1600:
            return []
        text = self.asr.transcribe(x, lang, mode="native", max_new_tokens=int(min(512, 12 * (b - a) + 32))).strip()
        if not text:
            return []
        return [{"start": round(a, 3), "end": round(b, 3), "text": text, "avg_logprob": None, "no_speech_prob": None,
                 "compression_ratio": None, "temperature": None, "words": []}]

    def unload(self) -> None:
        self.asr = None
        gc.collect()
