"""wav2vec 2.0 CTC models through transformers (PLAN_v0.9 candidates), loaded from the local folder only:
- MMS-1b-all with its Marathi language adapter (Meta, CC-BY-NC-4.0: tests only, never paid work), and
- the multilingual phoneme recogniser wav2vec2-xlsr-53-espeak-cv-ft (Meta, Apache-2.0): it writes IPA phonemes, not
  words — the acoustic judge of spellings (pipeline/phonemes.py), never a transcript.
GPU float16 when CUDA is usable (owner's rule), else CPU float32; greedy CTC decoding, as the models were trained.
manifest.toml gives `target_lang` (MMS adapter) or `phonemes = true`.
"""
from __future__ import annotations

import gc
import sys

import numpy as np

from ..models import model, model_dir, require, source_info

PIECE_S = 30.0


class Wav2Vec2CtcEngine:
    family = "wav2vec2_ctc"

    def __init__(self, engine_id: str):
        require(engine_id)
        self.id = engine_id
        self.path = model_dir(engine_id)
        self.meta = model(engine_id)
        self.model = self.proc = None
        self.dev, self.dtype = "cpu", None

    def load(self, device: str, params: dict) -> dict:
        import torch
        from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForCTC
        torch.manual_seed(0)
        lang = params.get("target_lang") or self.meta.get("target_lang")
        if self.meta.get("phonemes"):
            from transformers import Wav2Vec2PhonemeCTCTokenizer
            tok = Wav2Vec2PhonemeCTCTokenizer.from_pretrained(self.path, do_phonemize=False)   # decoding needs no espeak
        else:
            from transformers import Wav2Vec2CTCTokenizer
            tok = Wav2Vec2CTCTokenizer.from_pretrained(self.path, **({"target_lang": lang} if lang else {}))
        self.proc = (Wav2Vec2FeatureExtractor.from_pretrained(self.path), tok)
        self.dev = "cuda" if device != "cpu" and torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.dev == "cuda" else torch.float32
        kw = {"target_lang": lang, "ignore_mismatched_sizes": True} if lang else {}
        self.model = Wav2Vec2ForCTC.from_pretrained(self.path, dtype=self.dtype, **kw).to(self.dev).eval()
        return {"engine": self.id, "device": self.dev, "compute_type": str(self.dtype).replace("torch.", ""),
                "model_sha256": source_info(self.id).get("model_sha256")}

    def _decode(self, x: np.ndarray) -> str:
        import torch
        fe, tok = self.proc
        inputs = fe(x, sampling_rate=16000, return_tensors="pt")
        with torch.inference_mode():
            logits = self.model(inputs.input_values.to(self.dev, self.dtype)).logits
        return tok.decode(logits.argmax(-1)[0].tolist()).strip()

    def transcribe(self, audio: np.ndarray, language: str | None, fw: dict, vad: dict | None,
                   initial_prompt: str | None, word_timestamps: bool, clip_timestamps: list[float] | None = None) -> list[dict]:
        dur = len(audio) / 16000.0
        spans = list(zip(clip_timestamps[0::2], clip_timestamps[1::2])) if clip_timestamps else [(0.0, dur)]
        out = []
        for a, b in spans:
            while a < b - 0.1:
                e = min(b, a + PIECE_S)
                x = np.ascontiguousarray(audio[int(a * 16000):int(e * 16000)], dtype=np.float32)
                text = self._decode(x) if len(x) >= 1600 else ""
                if text:
                    out.append({"start": round(a, 3), "end": round(e, 3), "text": text, "avg_logprob": None,
                                "no_speech_prob": None, "compression_ratio": None, "temperature": None, "words": []})
                a = e
        return out

    def unload(self) -> None:
        self.model = self.proc = None
        gc.collect()
        if "torch" in sys.modules and sys.modules["torch"].cuda.is_available():
            sys.modules["torch"].cuda.empty_cache()
