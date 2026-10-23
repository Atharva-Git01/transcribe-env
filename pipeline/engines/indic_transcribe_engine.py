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
import os
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


def _build(mod, path: str, dev: str, dtype):
    """IndicTranscribe as its own from_pretrained builds it. On the GPU the weights go straight from the file to the
    GPU (accelerate's device_map): 14 s instead of 77 s, and ~3 GB of RAM instead of ~7 GB, because Windows did not
    give back the CPU copy the plain load makes first (E33)."""
    if dev == "cuda":
        try:
            model = mod.IndicCanaryForConditionalGeneration.from_pretrained(path, dtype=dtype, device_map="cuda").eval()
        except ValueError:                           # accelerate not installed (CPU-only lock): the plain way
            return mod.IndicTranscribe.from_pretrained(path, device=dev, dtype=dtype)
        return mod.IndicTranscribe(model, mod.IndicCanaryFeatureExtractor.from_pretrained(path, device=dev),
                                   mod.IndicCanaryTokenizer.from_pretrained(path), dev)
    return mod.IndicTranscribe.from_pretrained(path, device=dev, dtype=dtype)


class IndicTranscribeEngine:
    family = "indic_transcribe"
    keep_loaded = True       # the probe's language ID and recognition share one load per run (E34)
    lid_trusted = True       # its own language ID tells Marathi from Hindi (5/5 fixtures, P = 1.0): no forced decoding

    def __init__(self, engine_id: str):
        require(engine_id)
        self.id = engine_id
        self.path = model_dir(engine_id)
        self.asr = None
        self.max_s = 30.0
        self.num_beams = 1

    def configure(self, params: dict) -> None:
        """Per-use settings of an already loaded model (the probe loads it with defaults, recognition reuses it)."""
        self.max_s = float(params.get("piece_max_s", self.max_s))
        self.num_beams = int(params.get("num_beams", 1))   # 1 = greedy, the model's own default

    def uses_gpu(self, device: str, params: dict) -> bool:
        """Where load() will put the model: the runtime device, this engine's setting, a CUDA build of PyTorch."""
        import torch
        return device != "cpu" and params.get("device", "auto") in ("auto", "cuda") and torch.cuda.is_available()

    def load(self, device: str, params: dict) -> dict:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")   # deterministic cuBLAS (E18); before CUDA starts
        import torch
        torch.manual_seed(0)
        torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
        torch.set_num_threads(int(params.get("threads", 8)))
        _torchaudio_stand_in()
        sys.path.insert(0, str(self.path))           # the repo's flat imports (lid, modeling_indic_canary, …)
        try:
            mod = importlib.import_module("indic_transcribe")
        finally:
            sys.path.remove(str(self.path))
        # GPU first (owner): the runtime device decides ([runtime] device = "cpu" forces the CPU), then this
        # engine's own setting; bf16 on the GPU (~2.4 GB of VRAM), float32 on the CPU
        dev = "cuda" if self.uses_gpu(device, params) else "cpu"
        dtype = torch.bfloat16 if dev == "cuda" and params.get("dtype", "bfloat16") == "bfloat16" else torch.float32
        fallback = None
        try:
            self.asr = _build(mod, str(self.path), dev, dtype)
        except RuntimeError as e:                    # torch.cuda.OutOfMemoryError is a RuntimeError
            if dev != "cuda":
                raise
            self.asr = None
            gc.collect()
            torch.cuda.empty_cache()
            dev, dtype, fallback = "cpu", torch.float32, f"cuda load failed: {e}"[:300]
            self.asr = _build(mod, str(self.path), dev, dtype)
        self.max_s = float(params.get("piece_max_s", 30.0))
        self.num_beams = int(params.get("num_beams", 1))
        info = {"engine": self.id, "device": dev, "compute_type": str(dtype).replace("torch.", ""),
                "model_sha256": source_info(self.id).get("model_sha256")}
        if fallback:
            info["fallback_reason"] = fallback
        return info

    def detect_language(self, audio: np.ndarray, candidates: list[str]) -> dict:
        # allowed_langs is a hard filter (model docs): speech in another language goes to the nearest candidate
        allowed = [c for c in candidates if c in LANGS]
        top = self.asr.identify(np.asarray(audio, dtype=np.float32), topk=len(allowed) or 10, allowed_langs=allowed or None)
        probs = {lang: p for lang, p in top if lang in candidates}
        total = sum(probs.values()) or 1.0
        return {c: probs.get(c, 0.0) / total for c in candidates}

    def transcribe(self, audio: np.ndarray, language: str | None, fw: dict, vad: dict | None,
                   initial_prompt: str | None, word_timestamps: bool, clip_timestamps: list[float] | None = None) -> list[dict]:
        lang = language or self._guess(audio)          # unknown language: the model's own LID on the speech
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

    def _guess(self, audio: np.ndarray) -> str:
        x = np.ascontiguousarray(audio[:int(self.max_s * 16000)], dtype=np.float32)
        return self.asr.identify(x, topk=1)[0][0]

    def _piece(self, audio, a: float, b: float, lang: str) -> list[dict]:
        x = np.ascontiguousarray(audio[int(a * 16000):int(b * 16000)], dtype=np.float32)
        if len(x) < 1600:
            return []
        beams = {"num_beams": self.num_beams} if self.num_beams > 1 else {}
        text, lp = self._decode(x, lang, int(min(512, 12 * (b - a) + 32)), beams)
        if not text:
            return []
        # the decoder's own certainty (E39): mean and lowest log-probability of the tokens it chose. Kept apart from
        # avg_logprob, whose flag threshold is Whisper's, until a threshold for this engine is measured.
        conf = {"token_logprob_mean": round(sum(lp) / len(lp), 4), "token_logprob_min": round(min(lp), 4)} if lp else {}
        return [{"start": round(a, 3), "end": round(b, 3), "text": text, "avg_logprob": None, "no_speech_prob": None,
                 "compression_ratio": None, "temperature": None, "words": [], **conf}]

    def _decode(self, x: np.ndarray, lang: str, max_new: int, beams: dict) -> tuple[str, list[float]]:
        """The wrapper's own transcribe(), with the model's generate() wrapped for this one call so the scores of the
        chosen tokens come back too; the text path and the greedy choice are unchanged. Beam search: no scores."""
        model = self.asr.model
        if beams:
            return self.asr.transcribe(x, lang, mode="native", max_new_tokens=max_new, **beams).strip(), []
        import torch
        own, orig, lp = "generate" in model.__dict__, model.generate, []

        def generate(*args, **kw):
            out = orig(*args, output_scores=True, return_dict_in_generate=True, **kw)
            steps = out.scores or ()
            chosen = out.sequences[0, out.sequences.shape[1] - len(steps):]
            eos = model.generation_config.eos_token_id
            eos = set(eos if isinstance(eos, (list, tuple)) else [eos])
            for step, tok in zip(steps, chosen.tolist()):
                if tok not in eos:
                    lp.append(float(torch.log_softmax(step[0].float(), dim=-1)[tok]))
            return out.sequences
        model.generate = generate
        try:
            text = self.asr.transcribe(x, lang, mode="native", max_new_tokens=max_new).strip()
        finally:
            if own:
                model.generate = orig
            else:
                del model.generate
        return text, lp

    def unload(self) -> None:
        self.asr = None
        gc.collect()
        if "torch" in sys.modules and sys.modules["torch"].cuda.is_available():
            sys.modules["torch"].cuda.empty_cache()     # one GPU model at a time (E06): hand the VRAM back
