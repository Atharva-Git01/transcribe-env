# Indic-Transcribe-core (Bodhan AI / AI4Bharat) — primary engine for every language (0.7.0)

- Canary architecture: FastConformer encoder (811M) + Transformer decoder (419M), 1.2B parameters, fine-tuned from
  nvidia/canary-1b-v2. 25+ languages: 22 Indic, English, Bhojpuri, Bhili; code-mixed speech handled natively.
- Output modes: `native` (used here) writes standard spelling in the language's own script, numbers as words and
  English words in Devanagari — the owner's convention. No word timings, so each speaker turn is decoded on its own
  (E28). Built-in language ID (`identify()`), right on 5/5 fixtures with P = 1.0 (not used by the probe yet:
  CHANGELOG 0.7.0).
- **Licence: Indic Open Model License v1.0.** Free to self-host, commercial use included. Hosting it as an API or
  service for other people or companies needs Bodhan AI's written approval. Credit is required:
  "Built with Indic-Transcribe-core from Bodhan AI / AI4Bharat". Modified versions given to others keep the licence.
  Banned uses include mass surveillance and automated decisions about someone's legal rights without human oversight
  (the REVIEW rule stays). A separate licence is needed above 500M monthly users or $250M revenue.
- Runtime here: `pipeline/engines/indic_transcribe_engine.py` runs the repo's reviewed inference code (revision
  e083ab7d) from the local model folder; nothing is downloaded. torchaudio is replaced by a stand-in (E25): the
  environment always passes 16 kHz audio. CPU PyTorch, float32, 8 threads: ~5 GB RAM, model load 30–75 s.
  On a CUDA build of PyTorch it would use bf16 on the GPU (`device = "auto"`; not installed, QUESTIONS.md 20).
- Pieces of at most 30 s (the length it was trained on); `max_new_tokens` scales with the piece length (loop guard).
- Settings: `[recognise.indic_transcribe]` in every recipe.

## Measured here (seeded, same pipeline, decode per speaker turn, Gemma cleanup off)
| Date | Fixture | CER | normalised | Notes |
|---|---|---|---|---|
| 2026-10-22 | Pappa call (pieces ≤ 30 s) | 0.357 | 0.349 | speakers right only 0.696 (E28) |
| 2026-10-23 | Pappa call | **0.379** | 0.371 | speakers right 0.975, 4/26 turns missing, std 0.302, RTF 3.1 |
| 2026-10-23 | FLEURS mr | **0.048** | 0.030 | RTF 2.3 |
| 2026-10-23 | FLEURS hi | **0.021** | 0.010 | RTF 2.3 |
| 2026-10-23 | FLEURS mr + music (Demucs on, seeded) | **0.383** | 0.370 | 0.445 without Demucs (E27, E32) |
| 2026-10-23 | AMI meeting (en, 4 people) | **0.284** | 0.284 | speakers right 0.890, RTF 2.4 |

Sources: huggingface.co/bodhan-ai/indic-transcribe-core (model card, licence files in the model folder).
