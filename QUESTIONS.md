# Owner decisions (answered 2026-10-16)

| # | Topic | Decision | Status |
|---|---|---|---|
| 1 | Large models on the D: USB stick | Move to C: (USB copy kept as backup) | **done**: pip cache purged (15 GB), copied, sha256 re-verified, junctions replaced |
| 2 | pip cache (15 GB) | Purge | **done** |
| 3 | IndicConformer 600M (gated HF) | Install; owner logs in to Hugging Face | waiting for the owner's `hf auth login` |
| 4 | Gemma 3 cleanup | Off until `trx eval` proves it helps | to do: on/off eval |
| 5 | Demucs separation | Install, sharing PyTorch with IndicConformer | after 3 |
| 6 | Files > 1 h | Build chunked, resumable processing | to do |
| 7 | Default engine | large-v3 by default + a `--fast` option (turbo) per file | to do |
| 8 | Language | Analyse, show the guessed language, **ask the user**; OK → continue, other language → use it | to do |
| 9 | Ollama | Leave it alone | done (nothing changed) |
| 10 | More fixtures | Both: owner's labelled recordings + free public test sets (licences checked) | owner adds files; public sets need a download OK |
| 11 | Speakers | **Diarization on every processed audio** | to do (tune on mono calls) |
| 12 | Vocabulary correction | Hybrid chosen after comparison: KenLM n-gram language model (re-rank decoder candidates) + large Marathi/Hindi lexicon (non-word flags, confident fixes only) + per-case hotwords; no RAG | to do; corpora need a download OK |
| 13 | Backup | Private GitHub repo `Atharva-Git01/transcribe-env` (code/config/docs only; fixture audio + transcripts removed from history and git-ignored) | **done** |
