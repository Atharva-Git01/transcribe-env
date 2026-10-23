# faster-whisper (CTranslate2 Whisper) — MIT

| Model | Disk | VRAM (int8_float16) | Speed on RTX 3050 4 GB | Role here |
|---|---|---|---|---|
| medium | 1.5 GB (cached) | ~1.5–2 GB | medium | last fallback for unknown languages |
| large-v3 | ~3.1 GB | ~3 GB (tight) | slow | fallback after the Indic engines (primary until 0.6.0) |
| large-v3-turbo | ~1.6 GB | ~2 GB | ~6–8× faster than large-v3 | language ID in the probe and per speech group (`--fast` until 0.7.1) |

Since 0.7.0 Indic-Transcribe-core is the primary engine for every language (knowledge/ERRORS.md E24). Gemma cleanup
still runs on Whisper output (E26). Demucs music separation made Whisper worse on the music fixture; it stays on
because the primary engine gains from it (E27).

- Turbo has a 4-layer decoder (vs 32): faster, but published guides report lower accuracy on low-resource languages.
  No published Marathi head-to-head was found (2026-10-16) — `trx eval` decides.
- Strength: English, code-switched speech, word timestamps, punctuation. Weakness: autoregressive decoding can loop
  or hallucinate on silence/noise (references/failure_catalogue.md F1, F2).
- Parameters: references/decode_parameters.md.

## Measured here
| Date | Model | Fixture | CER | Notes |
|---|---|---|---|---|
| (FMS, 2026) | large-v3 | Pappa call | 0.7225 / 0.736 | FMS pipeline + metric; not directly comparable to trx's normalisation |
| 2026-10-16 | large-v3 | Pappa call | 0.531 | 0.5.0 baseline |
| 2026-10-22 | large-v3 | Pappa call | 0.513 | 0.6.0: Gemma word gate + LM; speakers right 0.917, 4/26 turns missing |
| 2026-10-22 | large-v3 | FLEURS mr / hi | 0.273 / 0.072 | clean read speech |
| 2026-10-22 | large-v3 | FLEURS mr + music | 0.606 | 0.657 with Demucs (E27) |
| 2026-10-22 | large-v3 | AMI meeting (en) | 0.308 | speakers right 0.895 |
