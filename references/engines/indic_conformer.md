# AI4Bharat IndicConformer 600M multilingual — first fallback for mr / hi (0.7.0)

- Built for the 22 official Indian languages (AI4Bharat); **no English**. ~600M parameters.
- Conformer with CTC and RNNT heads: non-autoregressive alignment, so no Whisper-style loops on silence.
  CTC gives word timings (from frames); RNNT is a little more accurate but has none, so it is decoded per speaker turn.
- Licence: **MIT** (model card, revision 2a77d305). Gated "auto" access on Hugging Face (the owner's login).
- Runtime here: `pipeline/engines/indic_conformer_engine.py` re-implements the reviewed `model_onnx.py` against the
  local files only (TorchScript mel preprocessor → ONNX encoder → CTC or RNNT with per-language vocabulary masks /
  joint post-nets). onnxruntime on the CPU, 8 threads. `from_pretrained` / `snapshot_download` are never called.
- Settings: `[recognise.indic_conformer]` in every recipe (`decoder`, `segmentation = "turns"`, `piece_max_s`).

## Measured here (seeded, same pipeline, decode per speaker turn)
| Date | Decoder | Fixture | CER | normalised | Notes |
|---|---|---|---|---|---|
| 2026-10-22 | CTC | Pappa call (pieces) | 0.390 | – | smoke test |
| 2026-10-22 | RNNT | Pappa call (pieces) | 0.383 | – | smoke test |
| 2026-10-22 | RNNT | Pappa call | **0.377** | 0.368 | speakers right 0.966, 5/26 turns missing, RTF 3.8 (whole pipeline) |
| 2026-10-22 | RNNT | FLEURS mr | 0.056 | 0.038 | RTF 2.3 |
| 2026-10-22 | RNNT | FLEURS hi | 0.034 | 0.016 | RTF 2.4 |

Indic-Transcribe-core was better on clean speech (0.048 / 0.021) and equal on the phone call, so it became primary
(knowledge/ERRORS.md E24). IndicConformer stays as the first fallback for mr/hi unless the owner removes it to free
2.4 GB (QUESTIONS.md 21).

Sources: huggingface.co/ai4bharat/indic-conformer-600m-multilingual · github.com/AI4Bharat/IndicConformerASR
