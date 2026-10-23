# Owner decisions (asked and answered 2026-10-16 / 2026-10-22 / 2026-10-23)

| # | Topic | Decision | Status |
|---|---|---|---|
| 1 | Large models on the D: USB stick | Move to C: (USB copy kept as backup) | **done**: pip cache purged (15 GB), copied, sha256 re-verified, junctions replaced |
| 2 | pip cache (15 GB) | Purge | **done** |
| 3 | IndicConformer 600M (gated HF) | Install; owner logs in to Hugging Face | **done** (2026-10-23): installed at a pinned revision and tested; first fallback for mr/hi. Indic-Transcribe-core came with the same login and is now the default engine (E24) |
| 4 | Gemma 3 cleanup | On only after `trx eval` proves it | **done**: ON for Whisper output (0.531→0.513); skipped for Indic-Transcribe, where it changed nothing and cost ~15 % of the time (E26) |
| 5 | Demucs separation | Install, sharing PyTorch with IndicConformer | **done**: htdemucs on the CPU PyTorch; used when the probe hears music/TV. Music fixture with Indic-Transcribe 0.445 → 0.383 (E27). A real recording with TV/music would confirm it |
| 6 | Files > 1 h | Chunked, resumable processing | **done**: 10-min chunks cut at silences; running the same command again resumes an interrupted run |
| 7 | Default engine | Best engine by eval + a `--fast` option per file | **done**: Indic-Transcribe-core since 0.7.0 (Pappa CER 0.513 → 0.379); `--fast` = Whisper turbo |
| 8 | Language | Analyse, show the guess, **ask**; OK → continue, other language → use it | **done**: asked in a terminal; non-interactive operators get `LANGUAGE_CONFIRM_NEEDED` and rerun with `--lang` |
| 9 | Ollama | Leave it alone | done (nothing changed) |
| 10 | More fixtures | Both: owner's labelled recordings + free public test sets | public: FLEURS mr/hi, FLEURS mr + music, AMI 4-person meeting; **waiting for the owner's recordings** (item 18) |
| 11 | Speakers | Diarization on every processed audio | **done**: every recipe diarizes; with per-turn decoding 97.5 % right on the Pappa call (E19, E28) |
| 12 | Vocabulary correction | Hybrid: language model + lexicon + hotwords (no RAG) | **done**: LM-checked word gate, `unknown_words` notes, `--hotwords`; N-best re-ranking tried and rejected (worse) |
| 13 | Backup | Private GitHub repo | **done**: `Atharva-Git01/transcribe-env` (no audio, transcripts or models). The call's words that 0.7 WIP pushed were removed by a history rewrite (E29, item 19) |
| 14 | Hugging Face login | You log in yourself | **done** (2026-10-23); both models downloaded |
| 19 | Pushed call text | Rewrite history + force-push | **done** (2026-10-23): every commit scanned clean |
| 22 | Disk space | Delete old eval runs, keep the newest per fixture | **done**: 57 runs, 1.4 GB |

## Open items (2026-10-23)

| # | Item | Status / what is needed from the owner |
|---|---|---|
| 14b | **Tokens pasted in chat** | They were never used. If you have not done it yet, revoke them at huggingface.co/settings/tokens. The login stored on this PC is only needed for future downloads (`hf auth logout` removes it). |
| 15 | **Marathi number table** (numbers are written as words) | Please check `library/numbers/mr.toml`, especially the spellings of 41, 71, 81, 91 and 99 (एक्केचाळीस, एक्काहत्तर, एक्क्याऐंशी, एक्क्याण्णव, नव्व्याण्णव). The file is plain text and can be edited directly. |
| 16 | **Standard-spelling version of the Pappa transcript** (your convention: standard spelling) | Claude drafted it: `evals/fixtures/pappa_260603_211622/truth_standard_DRAFT.txt` (this PC only). It is used as a second score (`std`: Whisper 0.487 → Indic-Transcribe 0.302). Unsure words were left exactly as you wrote them. Please correct the draft. |
| 17 | **English words in Devanagari** | **Solved by Indic-Transcribe**, which writes them in Devanagari itself. Only output of the Whisper fallback still lists Latin words for review (`latin_script`). |
| 18 | **Your recordings** | Later, as you said. Make each one a fixture with `python setup/make_fixture.py --id <id> --audio <file> --text <file> --lang mr` (one `speaker: text` line per turn). Wanted most: more phone calls, one Hindi and one Hinglish call, one with TV/music behind, one with 3+ people. |
| 19b | **Old commits still on GitHub** | History was cleaned twice on 2026-10-23 (the second pass also removed single words and speaker names). GitHub keeps the replaced commits reachable by their direct links: e07e138 (all 26 turns of the call) and 0a975f6 with its parents (a few single words). The repo is private, so only your account can open them. To remove them completely, ask GitHub Support to purge them ("remove sensitive data", repo Atharva-Git01/transcribe-env), or delete the repo on GitHub and let Claude push the clean history to a new one. |
| 20 | **Speed: CUDA PyTorch?** | Indic-Transcribe runs on the CPU: a 1-minute call takes about 3 minutes. The CUDA build of PyTorch (~2.5 GB download, replaces the CPU build) could run it on the RTX 3050 (bf16 needs ~2.4 GB of the 4 GB VRAM; untested). Needs your OK and disk space (item 22b). |
| 21 | **Keep IndicConformer?** | It is the first fallback for mr/hi and almost as good on the phone call (0.377 vs 0.379), worse on clean speech (0.056 vs 0.048). Deleting it frees 2.4 GB. |
| 22b | **Disk space and memory** | C: has 2–6 GB free. Windows' page file grew to 10 GB while the 1.2B model ran on the CPU (the PC has 15.6 GB of RAM); a restart gives most of it back. A run needs about 5.5 GB of free memory: with Chrome open (4.7 GB) three eval runs crashed, so runs now stop with `MEMORY_LOW` instead (E31). |
| 23 | **`--fast` no longer pays off** | On the phone call it saves only ~10 % of the time (RTF 2.7 vs 3.1) and doubles the errors (CER 0.771 vs 0.379, speakers right 0.717 vs 0.975). Keep it, remove it, or let it mean something else? |
