# faster-whisper (CTranslate2 Whisper) — MIT

| Model | Disk | VRAM (int8_float16) | Speed on RTX 3050 4 GB | Role here |
|---|---|---|---|---|
| medium | 1.5 GB (cached) | ~1.5–2 GB | medium | phase 1 bootstrap |
| large-v3 | ~3.1 GB | ~3 GB (tight) | slow | cross-check; primary for en / mixed if eval agrees |
| large-v3-turbo | ~1.6 GB | ~2 GB | ~6–8× faster than large-v3 | speed mode / long files |

- Turbo has a 4-layer decoder (vs 32): faster, but published guides report lower accuracy on low-resource languages.
  No published Marathi head-to-head was found (2026-10-16) — `trx eval` decides.
- Strength: English, code-switched speech, word timestamps, punctuation. Weakness: autoregressive decoding can loop
  or hallucinate on silence/noise (references/failure_catalogue.md F1, F2).
- Parameters: references/decode_parameters.md.

## Measured here
| Date | Model | Fixture | CER | Notes |
|---|---|---|---|---|
| (FMS, 2026) | large-v3 | Pappa call | 0.7225 / 0.736 | FMS pipeline + metric; not directly comparable to trx's normalisation |
