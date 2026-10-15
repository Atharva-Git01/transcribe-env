# Gemma 3 4B instruct (Q4 GGUF) via llama.cpp — cleanup only (phase 3)

- Owner's choice for local LLM cleanup. Multilingual (Marathi and Hindi included).
- Runner: llama.cpp single binary — no background service, no auto-updater (knowledge/ERRORS.md E08).
- Memory: ~2.5 GB; runs on the GPU only after the ASR engine has unloaded, otherwise on CPU.
- Licence: Gemma Terms of Use; commercial use allowed under Google's prohibited-use policy. Google's official GGUF
  repo is gated (free Hugging Face account to accept the terms).

## What it is allowed to do
Suggest edits to one segment at a time: obvious ASR spelling errors, script errors (Hindi form in a Marathi
sentence), punctuation. Each suggestion must pass all 5 gates (config/thresholds.toml [llm_cleanup]):
1. banned prefix (e.g. "Here is the corrected text") → reject
2. empty output → reject
3. over-deletion (length ratio < 0.50) → reject
4. edit ratio > 0.40 → reject
5. entity guard: names, numbers, dates, places must be unchanged → reject otherwise

## What it must never do
Translate, summarise, add words not heard, remove fillers in verbatim style, or touch `[review]` / `[inaudible]` tags.
Every suggestion, accepted or rejected, is logged in decisions.json with the gate result.
