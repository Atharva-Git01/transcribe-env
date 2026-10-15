# Questions for the owner (written overnight 2026-10-16 while you slept)

Each question shows the default I used so the work could continue. Nothing here stops the environment from working.
Reply with the number and your choice, e.g. "Q1 no, Q2 yes, Q4 b".

## Needs your decision

| # | Question | What I did (default) | Why it needs you |
|---|---|---|---|
| Q1 | **IndicConformer 600M** (AI4Bharat, probably the strongest free model for Marathi/Hindi) is MIT-licensed but **gated** on Hugging Face (log in + accept sharing your contact info), and it needs PyTorch (~1 GB more). Install it? | Not installed. Whisper medium, large-v3-turbo and large-v3 are installed. | Your HF account + disk. If yes: log in at huggingface.co, accept the terms on `ai4bharat/indic-conformer-600m-multilingual`, run `huggingface-cli login`, tell me. |
| Q2 | **Disk C:** has ~9 GB free (it dipped to 3.6 GB tonight). Your **pip cache is ~14 GB** (`%LOCALAPPDATA%\pip\cache`); `python -m pip cache purge` frees it safely (it only re-downloads when needed). Purge? | Not touched (outside this environment). | Your files. Also: the Windows pagefile grew to 10.6 GB tonight while large-v3 loaded and shrank again later (E15). |
| Q3 | **large-v3 (3.1 GB) and Gemma 3 (2.5 GB) live on D:** — a USB 2.0 stick (FAT32) — via junctions, because C: was full. Loading large-v3 from it takes ~2–3 min per run. Keep them there, or move them to C: after Q2 frees space? | Kept on D:. With D: unplugged, both show "not installed" and runs that need them stop with `MODEL_MISSING`. | Your drive; is the USB stick always connected? |
| Q4 | **Default engine.** a) large-v3 first, turbo when D: is unplugged (best CER 0.577; ~3–4 min per 1-min call from USB) b) turbo only (CER 0.720; ~1.5 min) | **a** — set in `config/defaults.toml [engines.primary]`. | Speed vs accuracy is your call. |
| Q5 | **Gemma 3 cleanup** works (~1 s per line after a ~1–2 min warm-up) but once changed a word's meaning while passing every gate (E17). Turn it on? | **Off.** It can be turned on with `[llm_cleanup] enabled = true` once an eval shows it lowers CER. | Forensic risk vs readability. |
| Q6 | **Demucs** (music/background-voice separation) needs PyTorch (~1 GB) + an 80 MB model. Install? | Not installed; the `separate` stage is skipped (`required = false`). | Disk + it is only useful for audio with music/TV behind the voice. |
| Q7 | **Long files (> 1 h):** chunked, resumable processing is not built; the long_file recipe runs in one pass (works, but a crash means starting over). Build it? | Not built; the fake "resumable/checkpoint" settings were removed from the recipe. | Do you expect recordings longer than an hour? |
| Q8 | **Language policy.** When the evidence for Marathi vs Hindi is weak, the environment picks the first language in `[languages] priority` (**Marathi**) and marks the run REVIEW. Is Marathi-first right for your cases? | mr → hi → en. | Your case mix. |
| Q9 | **Ollama** is installed on this PC and updates itself (E08). The environment no longer uses it (Gemma runs through llama.cpp); only its CUDA DLLs are hard-linked (hard links keep working even if Ollama updates or is uninstalled). Keep Ollama? | Left alone. | Your software. |

## Needs material from you (makes the environment better, not blocking)

| # | What | Why |
|---|---|---|
| Q10 | **More labelled audio:** 3–5 more recordings with a correct human transcript (ideally: one clean Marathi, one Hindi, one English/Hinglish, one noisy, one with 3+ speakers). Put them in `evals/fixtures/<name>/` as `audio.*` + a `truth.txt` and I'll convert them. | All tuning tonight used ONE call (Pappa). One fixture means every "improvement" may just fit that call. |
| Q11 | **Speaker separation on phone calls:** the Pappa truth has 2 speakers but diarization finds ~1 on this narrowband audio, so the phone recipe keeps speakers as `SPK?`. Do you have a call recorded in stereo (each side on its own channel)? | That is the reliable way to label speakers on calls (`true_stereo_call` recipe). |
| Q12 | **Vocabulary v2:** the FMS Marathi vocab rewrites real words into other words (E07), so it is disabled. Want me to build a reviewed version (you check a list of ~100 entries)? | Needs a native speaker's eye. |
| Q13 | **Git remote:** the environment has its own local git history (`git log` in transcribe-env). Push it to a private GitHub repo as a backup? | Needs your account. |
