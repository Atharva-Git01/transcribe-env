# Failure catalogue — symptom → cause → what the environment does

| # | Symptom in the transcript | Cause | Environment response |
|---|---|---|---|
| F1 | Same phrase repeated many times | decoder loop (hallucination) | compression-ratio check → temperature retry; residual loop flagged `[review]` (never deleted; doubled-word allowlist respected) |
| F2 | Fluent text during silence/music ("thank you for watching", "subscribe", धन्यवाद) | Whisper training artefact on non-speech | VAD segmentation; no_speech + hallucination_silence checks; flagged when VAD says no speech |
| F3 | Output switches to Hindi/Urdu/English mid-file | language drift | lid per file or per segment; language forced per segment after LID; drift flagged |
| F4 | Hindi spellings in a Marathi call (नहीं for नाही) | LID or prompt picked Hindi | check lid result + Marathi initial prompt (marathi_hindi_orthography.md) |
| F5 | Identical text under SPK1 and SPK2 | duplicated mono treated as stereo | probe layout detection (E01) |
| F6 | Empty transcript | VAD/no-speech thresholds, quiet input | fallback ladder; gate FAIL if still empty (E04) |
| F7 | Words lost at chunk/window boundaries | segment cut mid-word | speech_pad; overlap + timestamp de-duplication at merge |
| F8 | Worse result after denoising | denoiser artefacts on clean audio | clean_studio recipe has no denoise; attenuation limit in other recipes |
| F9 | Real repeated words removed | over-eager loop filter / no_repeat_ngram | doubled-word allowlist; no_repeat_ngram_size = 0 (E02) |
| F10 | Prompt words appear in the transcript | initial prompt copied into silence | short neutral prompts; flagged if prompt text appears verbatim |
| F11 | CUDA out of memory | two models on 4 GB | one GPU model at a time; int8; CPU fallback engine |
| F12 | Correct words "corrected" into other words | vocab map with valid words as variants | vocab disabled until reviewed (E07) |
| F13 | Run reached the internet | library first-use download / updater | offline guard → NETWORK_BLOCKED (E08) |
