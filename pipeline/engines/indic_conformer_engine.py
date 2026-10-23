"""IndicConformer-600M (AI4Bharat, MIT): hybrid CTC + RNNT Conformer for 22 Indic languages (no English).

Re-implements the reviewed inference of the model repo's `model_onnx.py` (revision 2a77d305) against LOCAL files only:
TorchScript NeMo mel preprocessor (assets/preprocessor.ts) -> ONNX encoder -> CTC decoder (greedy, per-language vocabulary
mask, word timestamps from frames) or RNNT (greedy, per-language joint post-net). `from_pretrained` / `snapshot_download`
from the repo are never called, so nothing can reach the network.

The engine decodes pieces: the given clip_timestamps, else the speech regions (VAD) grouped into pieces of <= 20 s,
else the whole audio in 30 s pieces. Each piece becomes one segment with word timings (CTC) — polish/split work as usual.
"""
from __future__ import annotations

import gc
import json
import math

import numpy as np

from ..common import TrxError
from ..models import model_dir, require, source_info

BLANK_ID = 256
SOS = 256
RNNT_MAX_SYMBOLS = 10
PRED_RNN_LAYERS = 2
PRED_RNN_HIDDEN_DIM = 640
FRAME_SEC = 0.08                     # encoder frame = 8 x 10 ms (config default in model_onnx.py)
LANGS = ["as", "bn", "brx", "doi", "gu", "hi", "kn", "kok", "ks", "mai", "ml", "mni", "mr", "ne", "or", "pa", "sa",
         "sat", "sd", "ta", "te", "ur"]


class IndicConformerEngine:
    family = "indic_conformer"

    def __init__(self, engine_id: str):
        require(engine_id)
        self.id = engine_id
        self.path = model_dir(engine_id)
        self.s = {}
        self.pre = None
        self.vocab = None
        self.masks = None
        self.decoder = "ctc"

    def load(self, device: str, params: dict) -> dict:
        import onnxruntime as ort
        import torch
        torch.manual_seed(0)
        a = self.path / "assets"
        self.pre = torch.jit.load(str(a / "preprocessor.ts"), map_location="cpu")
        self.pre.eval()
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = int(params.get("threads", 8))
        opts.log_severity_level = 3
        self.decoder = str(params.get("decoder", "ctc"))
        names = ["encoder", "ctc_decoder"] + (["rnnt_decoder", "joint_enc", "joint_pred", "joint_pre_net"]
                                               if self.decoder == "rnnt" else [])
        for n in names:
            self.s[n] = ort.InferenceSession(str(a / f"{n}.onnx"), sess_options=opts, providers=["CPUExecutionProvider"])
        self.vocab = json.loads((a / "vocab.json").read_text(encoding="utf-8"))
        self.masks = json.loads((a / "language_masks.json").read_text(encoding="utf-8"))
        return {"engine": self.id, "device": "cpu", "compute_type": f"onnx fp32 ({self.decoder})",
                "model_sha256": source_info(self.id).get("model_sha256")}

    def detect_language(self, audio: np.ndarray, candidates: list[str]) -> dict:
        raise TrxError("CONFIG_INVALID", "IndicConformer has no language identification; LID uses a Whisper engine")

    # ------------------------------------------------------------------ decoding of one piece
    def _encode(self, x: np.ndarray):
        import torch
        wav = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))[None]
        with torch.no_grad():
            feats, length = self.pre(input_signal=wav, length=torch.tensor([wav.shape[-1]]))
        out, lens = self.s["encoder"].run(["outputs", "encoded_lengths"],
                                          {"audio_signal": feats.numpy(), "length": length.numpy()})
        return out, lens

    def _ctc(self, enc, lens, lang):
        lp = self.s["ctc_decoder"].run(["logprobs"], {"encoder_output": enc})[0][0][:, self.masks[lang]]
        lp = lp - np.log(np.exp(lp - lp.max(-1, keepdims=True)).sum(-1, keepdims=True)) - lp.max(-1, keepdims=True)
        T = int(lens[0])
        lp = lp[:T]
        path = lp.argmax(-1)
        toks, cur, start, conf = [], None, 0, []
        for f, t in enumerate(path):
            t = int(t)
            if t == BLANK_ID:
                if cur is not None:
                    toks.append((cur, start, f, float(np.mean(conf))))
                    cur, conf = None, []
            elif t != cur:
                if cur is not None:
                    toks.append((cur, start, f, float(np.mean(conf))))
                cur, start, conf = t, f, [float(lp[f, t])]
            else:
                conf.append(float(lp[f, t]))
        if cur is not None:
            toks.append((cur, start, T, float(np.mean(conf))))
        words, w = [], None
        for tok, f0, f1, c in toks:
            piece = self.vocab[lang][tok]
            if "▁" in piece or w is None:
                if w and w["word"]:
                    words.append(w)
                w = {"word": piece.replace("▁", ""), "f0": f0, "f1": f1, "lp": [c]}
            else:
                w["word"] += piece
                w["f1"] = f1
                w["lp"].append(c)
        if w and w["word"]:
            words.append(w)
        nonblank = [float(lp[f, int(t)]) for f, t in enumerate(path) if int(t) != BLANK_ID]
        return words, (float(np.mean(nonblank)) if nonblank else None)

    def _rnnt(self, enc, lang):
        import onnxruntime as ort  # noqa: F401  (sessions already created)
        if f"joint_post_net_{lang}" not in self.s:
            import onnxruntime as ort2
            opts = ort2.SessionOptions()
            opts.log_severity_level = 3
            self.s[f"joint_post_net_{lang}"] = ort2.InferenceSession(
                str(self.path / "assets" / f"joint_post_net_{lang}.onnx"), sess_options=opts,
                providers=["CPUExecutionProvider"])
        je = self.s["joint_enc"].run(["output"], {"input": enc.transpose(0, 2, 1)})[0]
        hyp = [SOS]
        st = (np.zeros((PRED_RNN_LAYERS, 1, PRED_RNN_HIDDEN_DIM), dtype=np.float32),
              np.zeros((PRED_RNN_LAYERS, 1, PRED_RNN_HIDDEN_DIM), dtype=np.float32))
        lps = []
        for t in range(je.shape[1]):
            f = je[:, t, :][:, None, :]
            for _ in range(RNNT_MAX_SYMBOLS):
                g, _, s0, s1 = self.s["rnnt_decoder"].run(
                    ["outputs", "prednet_lengths", "states", "162"],
                    {"targets": np.array([[hyp[-1]]], dtype=np.int32), "target_length": np.array([1], dtype=np.int32),
                     "states.1": st[0], "onnx::Slice_3": st[1]})
                g = self.s["joint_pred"].run(["output"], {"input": g.transpose(0, 2, 1)})[0]
                j = self.s["joint_pre_net"].run(["output"], {"input": f + g})[0]
                logits = self.s[f"joint_post_net_{lang}"].run(["output"], {"input": j})[0].reshape(-1)
                m = logits.max()
                lsm = logits - m - math.log(np.exp(logits - m).sum())
                tok = int(lsm.argmax())
                if tok == BLANK_ID:
                    break
                hyp.append(tok)
                lps.append(float(lsm[tok]))
                st = (s0, s1)
        text = "".join(self.vocab[lang][x] for x in hyp[1:]).replace("▁", " ").strip()
        return text, (float(np.mean(lps)) if lps else None)

    # ------------------------------------------------------------------ public interface (same as faster-whisper)
    def transcribe(self, audio: np.ndarray, language: str | None, fw: dict, vad: dict | None,
                   initial_prompt: str | None, word_timestamps: bool, clip_timestamps: list[float] | None = None) -> list[dict]:
        lang = language or "mr"
        if lang not in self.masks:
            raise TrxError("CONFIG_INVALID", f"IndicConformer does not support language '{lang}' (no English)")
        dur = len(audio) / 16000.0
        if clip_timestamps:
            pieces = list(zip(clip_timestamps[0::2], clip_timestamps[1::2]))
        elif vad is not None:
            from ..stages.recognise import speech_groups
            from ..stages.vad import speech_regions
            pieces = speech_groups(speech_regions(audio, vad), float(fw.get("piece_max_s", 20.0)), 0.8)
        else:
            pieces = [(a, min(dur, a + 30.0)) for a in np.arange(0.0, dur, 30.0)]
        out = []
        for a, b in pieces:
            x = audio[int(a * 16000):int(b * 16000)]
            if len(x) < 1600:                       # < 0.1 s
                continue
            enc, lens = self._encode(x)
            if self.decoder == "rnnt":
                text, avg = self._rnnt(enc, lang)
                words = []
            else:
                ws, avg = self._ctc(enc, lens, lang)
                text = " ".join(w["word"] for w in ws).strip()
                words = [{"start": round(a + w["f0"] * FRAME_SEC, 3), "end": round(a + w["f1"] * FRAME_SEC, 3),
                          "word": " " + w["word"], "probability": round(float(np.exp(np.mean(w["lp"]))), 4)}
                         for w in ws]
            if not text:
                continue
            out.append({"start": round(words[0]["start"] if words else a, 3),
                        "end": round(words[-1]["end"] if words else b, 3), "text": text,
                        "avg_logprob": round(avg, 4) if avg is not None else None, "no_speech_prob": None,
                        "compression_ratio": None, "temperature": None, "words": words})
        return out

    def unload(self) -> None:
        self.s, self.pre = {}, None
        gc.collect()
