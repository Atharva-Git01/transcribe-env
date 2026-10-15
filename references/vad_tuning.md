# VAD (Silero) tuning

The vad stage decides which parts of the audio are speech. Everything after it only sees those parts, so a VAD
mistake becomes a missing sentence. Prefer keeping a little noise over losing a soft reply.

| Parameter | Silero default | Meaning | Phone / soft speech | Noisy field audio | Clean audio |
|---|---|---|---|---|---|
| `threshold` | 0.5 | speech probability to start a segment | 0.35 | 0.30 | 0.50 |
| `min_speech_ms` | 250 | shorter bursts are dropped | 200 | 250 | 250 |
| `min_silence_ms` | 100 (faster-whisper: 2000) | silence needed to end a segment | 400 | 500 | 500 |
| `speech_pad_ms` | 30 (faster-whisper: 400) | padding added to both ends | 250 | 300 | 200 |
| `max_segment_s` | — | segments are split at the quietest point below this | 28 | 25 | 28 |

## Symptoms → fix
| Symptom | Likely cause | Fix |
|---|---|---|
| Short replies ("हो", "हं", "yes") missing | threshold too high or min_speech too long | lower threshold 0.05 at a time; min_speech 150–200 |
| First/last syllables clipped | padding too small | speech_pad 250–400 |
| Noise transcribed as words | threshold too low on noisy audio | raise threshold; rely on denoise first |
| Whole file "no speech" | very quiet input or VAD failure | fallback ladder: VAD off (OPERATOR.md, E04) |
| Words cut mid-sentence | max_segment too short | keep 25–28 s (Whisper window is 30 s) |

## Omission check (phase 2)
VAD speech ≥ `thresholds.toml [verify].omission_min_speech_sec` with no recognised text = an omission flag in
`review_list.md`. This catches ASR dropping speech that VAD heard.
