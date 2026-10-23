# Indic-Transcribe-core (Bodhan AI / AI4Bharat) — primary engine for every language (0.7.0)

- Canary architecture: FastConformer encoder (811M) + Transformer decoder (419M), 1.2B parameters, fine-tuned from
  nvidia/canary-1b-v2. 25+ languages: 22 Indic, English, Bhojpuri, Bhili; code-mixed speech handled natively.
- Output modes: `native` (used here) writes standard spelling in the language's own script, numbers as words and
  English words in Devanagari — the owner's convention. No word timings, so each speaker turn is decoded on its own
  (E28). Built-in language ID (`identify()`), right on 5/5 fixtures with P = 1.0: the probe's language-ID engine
  since 0.8.0, loaded once per run and reused by recognition (E34).
- **Licence: Indic Open Model License v1.0.** Free to self-host, commercial use included. Hosting it as an API or
  service for other people or companies needs Bodhan AI's written approval. Credit is required:
  "Built with Indic-Transcribe-core from Bodhan AI / AI4Bharat". Modified versions given to others keep the licence.
  Banned uses include mass surveillance and automated decisions about someone's legal rights without human oversight
  (the REVIEW rule stays). A separate licence is needed above 500M monthly users or $250M revenue.
- Runtime here: `pipeline/engines/indic_transcribe_engine.py` runs the repo's reviewed inference code (revision
  e083ab7d) from the local model folder; nothing is downloaded. torchaudio is replaced by a stand-in (E25): the
  environment always passes 16 kHz audio.
- **GPU (0.7.1, the default):** CUDA 12.6 PyTorch, bf16, weights loaded straight onto the GPU with accelerate
  (E33). 2.5 GB of VRAM on the RTX 3050 4 GB, ~2.7 GB of RAM, load 14–25 s, a 12 s piece decoded in ~3.5 s,
  deterministic (cuDNN/cuBLAS deterministic settings). Out of GPU memory → falls back to the CPU (logged).
- CPU (no CUDA build, or `[runtime] device = "cpu"`): float32, 8 threads, ~7 GB of RAM at peak, load 30–75 s.
- Pieces of at most 30 s (the length it was trained on); `max_new_tokens` scales with the piece length (loop guard).
- Settings: `[recognise.indic_transcribe]` in every recipe.

## Measured here (seeded, same pipeline, decode per speaker turn, Gemma cleanup off)
| Date | Device | Fixture | CER | normalised | Notes |
|---|---|---|---|---|---|
| 2026-10-22 | CPU | Pappa call (pieces ≤ 30 s) | 0.357 | 0.349 | speakers right only 0.696 (E28) |
| 2026-10-23 | CPU | Pappa call | 0.379 | 0.371 | speakers right 0.975, 4/26 turns missing, std 0.302, RTF 3.1 |
| 2026-10-23 | CPU | FLEURS mr / hi | 0.048 / 0.021 | 0.030 / 0.010 | RTF 2.3 |
| 2026-10-23 | CPU | FLEURS mr + music (Demucs, seeded) | 0.383 | 0.370 | 0.445 without Demucs (E27, E32) |
| 2026-10-23 | CPU | AMI meeting (en, 4 people) | 0.284 | 0.284 | speakers right 0.890, RTF 2.4 |
| 2026-10-23 | **GPU** | Pappa call | **0.374** | 0.366 | speakers right 0.975, 4/26 missing, RTF 2.5; repeat run identical |
| 2026-10-23 | **GPU** | FLEURS mr / hi | **0.047** / **0.021** | 0.029 / 0.010 | RTF 1.9 / 2.0 |
| 2026-10-23 | **GPU** | FLEURS mr + music (Demucs on GPU) | **0.388** | 0.376 | RTF 1.6 |
| 2026-10-23 | **GPU** | AMI meeting (en) | **0.284** | 0.284 | speakers right 0.916, RTF 2.0 |

Sources: huggingface.co/bodhan-ai/indic-transcribe-core (model card, licence files in the model folder).
