# faster-whisper decode parameters — what each does, safe ranges, what it breaks

Library defaults in brackets. Recipes set every value explicitly; this page explains them. Change only via eval.

| Parameter | Default | What it does | Safe range | Breaks when |
|---|---|---|---|---|
| `beam_size` | 5 | beams searched at temperature 0 | 1–5 on 4 GB | >5 costs VRAM/time for tiny gains |
| `best_of` | 5 | candidates sampled when temperature > 0 | 1–5 | — |
| `patience` | 1.0 | beam search patience | 1.0 | >1 slows decoding |
| `temperature` | [0.0…1.0] | fallback ladder: retried at the next value if a window fails the checks below | [0.0, 0.2, 0.4, 0.6] | high values (0.8–1.0) invent words on noisy audio |
| `compression_ratio_threshold` | 2.4 | text that compresses too well (= repetition) fails → retry hotter | 2.0–2.6 | too low rejects real repetitive speech |
| `log_prob_threshold` | -1.0 | average token log-prob below this fails → retry | -1.2 to -0.8 | too high forces needless retries |
| `no_speech_threshold` | 0.6 | a window is skipped as silence if no_speech_prob > this AND avg_logprob < log_prob_threshold | 0.4–0.7 | too low drops soft speech; too high transcribes noise |
| `condition_on_previous_text` | True | feeds the previous window's text as context | **False for forensic audio** | True lets one hallucination or loop spread to every later window |
| `prompt_reset_on_temperature` | 0.5 | drops the context prompt when a retry goes above this temperature | 0.5 | — |
| `initial_prompt` | None | text that primes script/register (prompts/initial_prompts) | short, neutral | long or specific prompts get copied into silent stretches |
| `hotwords` | None | biases toward listed words | case names only | generic words over-insert |
| `repetition_penalty` | 1.0 | penalises repeated tokens | 1.0–1.1 | >1.1 damages real repetition (doubled words) |
| `no_repeat_ngram_size` | 0 | forbids repeating n-grams | 0 | any value deletes real doubled words (हा हा) |
| `word_timestamps` | False | per-word timing | True | slightly slower |
| `hallucination_silence_threshold` | None | with word timestamps, skips silent gaps (seconds) where a hallucination is suspected | 1.5–2.0 | too low drops real short replies |
| `vad_filter` | False | built-in Silero VAD | False (the vad stage segments first) | double VAD = lost speech |
| `language` | auto | forced language code | from the lid stage | forcing the wrong language = fluent nonsense |
| `language_detection_segments` | 1 | windows used to detect language | 3–6 for long files | 1 window from a greeting misdetects |
| `compute_type` | — | precision | GPU `int8_float16`, CPU `int8` | `float16` large-v3 does not fit 4 GB |

## Fixed rules in this environment
- `condition_on_previous_text = false` everywhere (knowledge/ERRORS.md E03, failure_catalogue.md).
- `no_repeat_ngram_size = 0` everywhere (E02).
- The vad stage segments audio once; engines never run their own VAD on top.
- Every value actually used is copied into `plan.json`.
