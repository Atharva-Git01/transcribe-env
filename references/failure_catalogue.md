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
| F14 | A listener's short reply or "हं" between two lines of the other speaker is missing, or sits under the wrong speaker | diarization smoothing absorbs a one-window turn (one cost for a change mid-speech and at a pause) | lower switch cost across a pause in the phone recipe (E39); eval turn outcomes: `lost_backchannel`, `merged`, `wrong_speaker` |
| F15 | A short line (< 2 s) holds words nobody said: a personal name, an English word, an unrelated phrase | a short turn decoded alone has no context; Indic-Transcribe then invents (as Whisper did, E23) | soft flag `uncertain` from the decoder's own token confidence (E40: 16 of 18 implausible short lines on the 20-min call); eval reports content CER of lines < 2 s |
| F16 | A fluent line means the opposite or names another amount ("will" → "will not") | a misheard negation or number costs a few characters only | eval counts negations and numbers lost or added (`library/scoring/`); a reviewer must check every negation and amount in evidence |
| F17 | One person's name is written several ways in one transcript, or a surname appears that nobody said | names are outside the model's vocabulary; each turn is decoded alone | `--hotwords` for the owner's names; since 0.9.0 `review_list.md` lists rare words written in more than one way (`pipeline/names.py`, PLAN_v0.9 B4a), with the listed spelling when `--hotwords` has it |
| F18 | `trx eval` CER is near 1 on a transcript that reads well | the ground truth stops before the recording ends | `scored_until_sec` in truth.json / `make_fixture.py --scored-until` (E39) |
| F19 | The worst lines carry no review flag | Indic-Transcribe returns no confidence; flags cover quiet speech, loops and scripts only | soft flag `uncertain` (Indic-Transcribe's mean token log-prob < −0.21, E40): 57–83 % of the bad lines marked; eval reports "bad lines flagged" |
