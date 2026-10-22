# Owner decisions (asked and answered 2026-10-16 / 2026-10-22)

| # | Topic | Decision | Status |
|---|---|---|---|
| 1 | Large models on the D: USB stick | Move to C: (USB copy kept as backup) | **done**: pip cache purged (15 GB), copied, sha256 re-verified, junctions replaced |
| 2 | pip cache (15 GB) | Purge | **done** |
| 3 | IndicConformer 600M (gated HF) | Install; owner logs in to Hugging Face | **waiting for the owner**: accept the terms on the model page, then `toolchain\python\.venv\Scripts\hf.exe auth login` |
| 4 | Gemma 3 cleanup | On only after `trx eval` proves it | **done — ON**: with the new word gate + language model, all 3 fixtures improved (CER Pappa 0.531→0.513, FLEURS mr 0.286→0.273, hi 0.083→0.074) |
| 5 | Demucs separation | Install, sharing PyTorch with IndicConformer | after 3 |
| 6 | Files > 1 h | Chunked, resumable processing | **done**: 10-min chunks cut at silences; running the same command again resumes an interrupted run |
| 7 | Default engine | large-v3 + a `--fast` option (turbo) per file | **done** |
| 8 | Language | Analyse, show the guess, **ask**; OK → continue, other language → use it | **done**: asked in a terminal; non-interactive operators get `LANGUAGE_CONFIRM_NEEDED` and rerun with `--lang` |
| 9 | Ollama | Leave it alone | done (nothing changed) |
| 10 | More fixtures | Both: owner's labelled recordings + free public test sets | public: FLEURS Marathi + Hindi added; **waiting for the owner's recordings** |
| 11 | Speakers | Diarization on every processed audio | **done**: every recipe diarizes (by channel or by voice, 94.5 % on the Pappa call, E19) |
| 12 | Vocabulary correction | Hybrid: language model + lexicon + hotwords (no RAG) | **done**: LM-checked word gate, `unknown_words` notes, `--hotwords`; N-best re-ranking tried and rejected (worse) |
| 13 | Backup | Private GitHub repo | **done**: `Atharva-Git01/transcribe-env` (no audio, transcripts or models) |
