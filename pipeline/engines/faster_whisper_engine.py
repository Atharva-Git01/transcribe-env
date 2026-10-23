"""faster-whisper engine (CTranslate2). Loads ONLY from a local folder; never downloads."""
from __future__ import annotations

import gc

import numpy as np

from .. import gpu
from ..common import TrxError, load_config
from ..models import model_dir, require, source_info


def vad_options(vad: dict):
    from faster_whisper.vad import VadOptions
    return VadOptions(
        threshold=float(vad.get("threshold", 0.5)),
        min_speech_duration_ms=int(vad.get("min_speech_ms", 250)),
        max_speech_duration_s=float(vad.get("max_segment_s", 28.0)),
        min_silence_duration_ms=int(vad.get("min_silence_ms", 500)),
        speech_pad_ms=int(vad.get("speech_pad_ms", 200)),
    )


class FasterWhisperEngine:
    family = "faster_whisper"

    def __init__(self, engine_id: str):
        require(engine_id)
        self.id = engine_id
        self.path = model_dir(engine_id)
        self.model = None
        self.device = None
        self.compute_type = None

    def load(self, device: str, params: dict) -> dict:
        gpu.prepare()
        from faster_whisper import WhisperModel
        ct = params.get("compute_type_gpu", "int8_float16") if device == "cuda" else params.get("compute_type_cpu", "int8")
        try:
            self.model = WhisperModel(str(self.path), device=device, compute_type=ct)
            self.device, self.compute_type = device, ct
        except Exception as e:  # noqa: BLE001
            if device != "cuda":
                raise TrxError("INTERNAL_ERROR", f"{self.id} failed to load on {device}: {e}") from e
            ct = params.get("compute_type_cpu", "int8")
            self.model = WhisperModel(str(self.path), device="cpu", compute_type=ct)
            self.device, self.compute_type = "cpu", ct
            return {"engine": self.id, "device": "cpu", "compute_type": ct, "fallback_reason": f"cuda load failed: {e}",
                    "model_sha256": source_info(self.id).get("model_sha256")}
        return {"engine": self.id, "device": self.device, "compute_type": self.compute_type,
                "model_sha256": source_info(self.id).get("model_sha256")}

    def detect_language(self, audio: np.ndarray, candidates: list[str]) -> dict:
        """Whisper LID on one window (<= 30 s). Returns {lang: prob} restricted to candidates, renormalised."""
        _, _, all_probs = self.model.detect_language(audio=audio)
        probs = {lang: p for lang, p in all_probs if lang in candidates}
        total = sum(probs.values()) or 1.0
        return {lang: probs.get(lang, 0.0) / total for lang in candidates}

    def transcribe(self, audio: np.ndarray, language: str | None, fw: dict, vad: dict | None,
                   initial_prompt: str | None, word_timestamps: bool,
                   clip_timestamps: list[float] | None = None) -> list[dict]:
        kwargs = dict(
            language=language,
            beam_size=int(fw.get("beam_size", 5)),
            best_of=int(fw.get("best_of", 5)),
            patience=float(fw.get("patience", 1.0)),
            temperature=list(fw.get("temperature", [0.0, 0.2, 0.4, 0.6])),
            compression_ratio_threshold=float(fw.get("compression_ratio_threshold", 2.4)),
            log_prob_threshold=float(fw.get("log_prob_threshold", -1.0)),
            no_speech_threshold=float(fw.get("no_speech_threshold", 0.6)),
            condition_on_previous_text=bool(fw.get("condition_on_previous_text", False)),
            repetition_penalty=float(fw.get("repetition_penalty", 1.0)),
            no_repeat_ngram_size=int(fw.get("no_repeat_ngram_size", 0)),
            initial_prompt=initial_prompt,
            hotwords=fw.get("hotwords") or None,
            word_timestamps=word_timestamps,
            hallucination_silence_threshold=(float(fw["hallucination_silence_threshold"])
                                             if word_timestamps and fw.get("hallucination_silence_threshold") else None),
        )
        if clip_timestamps:
            kwargs["clip_timestamps"] = clip_timestamps
            kwargs["vad_filter"] = False
        elif vad is not None:
            kwargs["vad_filter"] = True
            kwargs["vad_parameters"] = vad_options(vad)
        else:
            kwargs["vad_filter"] = False
        # temperature fallback samples at t > 0: fix the seed per call so the same input gives the same text
        import ctranslate2
        ctranslate2.set_random_seed(int(load_config("defaults")["runtime"]["seed"]))
        segments, _info = self.model.transcribe(audio, **kwargs)
        out = []
        for s in segments:
            out.append({
                "start": round(float(s.start), 3), "end": round(float(s.end), 3), "text": s.text.strip(),
                "avg_logprob": round(float(s.avg_logprob), 4), "no_speech_prob": round(float(s.no_speech_prob), 4),
                "compression_ratio": round(float(s.compression_ratio), 3),
                "temperature": float(s.temperature) if s.temperature is not None else None,
                "words": [{"start": round(float(w.start), 3), "end": round(float(w.end), 3), "word": w.word,
                           "probability": round(float(w.probability), 4)} for w in (s.words or [])],
            })
        return out

    def unload(self) -> None:
        self.model = None
        gc.collect()


def make(engine_id: str):
    if engine_id.startswith("faster_whisper"):
        return FasterWhisperEngine(engine_id)
    if engine_id.startswith("indic_conformer"):
        from .indic_conformer_engine import IndicConformerEngine
        return IndicConformerEngine(engine_id)
    if engine_id.startswith("indic_transcribe"):
        from .indic_transcribe_engine import IndicTranscribeEngine
        return IndicTranscribeEngine(engine_id)
    raise TrxError("STAGE_NOT_INSTALLED", f"no adapter for engine '{engine_id}'")
