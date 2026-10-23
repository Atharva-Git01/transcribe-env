# Owner decisions (asked and answered 2026-10-16 / 2026-10-22 / 2026-10-23)

| # | Topic | Decision | Status |
|---|---|---|---|
| 1 | Large models on the D: USB stick | Move to C: (USB copy kept as backup) | **done**: pip cache purged (15 GB), copied, sha256 re-verified, junctions replaced |
| 2 | pip cache (15 GB) | Purge | **done** |
| 3 | IndicConformer 600M (gated HF) | Install; owner logs in to Hugging Face | **done** (2026-10-23): installed at a pinned revision and tested; first fallback for mr/hi. Indic-Transcribe-core came with the same login and is now the default engine (E24) |
| 4 | Gemma 3 cleanup | On only after `trx eval` proves it | **done**: ON for Whisper output (0.531→0.513); skipped for Indic-Transcribe, where it changed nothing and cost ~15 % of the time (E26) |
| 5 | Demucs separation | Install, sharing PyTorch with IndicConformer | **done**: htdemucs on the CPU PyTorch; used when the probe hears music/TV. Music fixture with Indic-Transcribe 0.445 → 0.383 (E27). A real recording with TV/music would confirm it |
| 6 | Files > 1 h | Chunked, resumable processing | **done**: 10-min chunks cut at silences; running the same command again resumes an interrupted run |
| 7 | Default engine | Best engine by eval + a `--fast` option per file | **done**: Indic-Transcribe-core since 0.7.0 (Pappa CER 0.513 → 0.379); `--fast` removed in 0.7.2 (item 23) |
| 8 | Language | Analyse, show the guess, **ask**; OK → continue, other language → use it | **done**: asked in a terminal; non-interactive operators get `LANGUAGE_CONFIRM_NEEDED` and rerun with `--lang` |
| 9 | Ollama | Leave it alone | done (nothing changed) |
| 10 | More fixtures | Both: owner's labelled recordings + free public test sets | public: FLEURS mr/hi, FLEURS mr + music, AMI 4-person meeting; **waiting for the owner's recordings** (item 18) |
| 11 | Speakers | Diarization on every processed audio | **done**: every recipe diarizes; with per-turn decoding 97.5 % right on the Pappa call (E19, E28) |
| 12 | Vocabulary correction | Hybrid: language model + lexicon + hotwords (no RAG) | **done**: LM-checked word gate, `unknown_words` notes, `--hotwords`; N-best re-ranking tried and rejected (worse) |
| 13 | Backup | Private GitHub repo | **done**: `Atharva-Git01/transcribe-env` (no audio, transcripts or models). The call's words that 0.7 WIP pushed were removed by a history rewrite (E29, item 19) |
| 14 | Hugging Face login | You log in yourself | **done** (2026-10-23); both models downloaded |
| 19 | Pushed call text | Rewrite history + force-push | **done** (2026-10-23): every commit scanned clean |
| 22 | Disk space | Delete old eval runs, keep the newest per fixture | **done**: 57 runs, 1.4 GB |
| 20 | CUDA PyTorch | Yes; and GPU first for every model from now on | **done** (0.7.1): Indic-Transcribe and Demucs on the RTX 3050 (bf16, 2.5 GB VRAM), loaded straight to the GPU (accelerate). Pappa RTF 3.06 → 2.54, music 3.57 → 1.58; RAM per run ~7 → ~2.7 GB; CER unchanged within noise |
| 23 | `--fast` (slower and worse since the GPU switch) | First: redefine as long pieces if really faster. It was not (Pappa 143 vs 140 s; AMI 6 % quicker), so: remove it, and speed up every run instead | **done**: `--fast` stops with a clear message (0.7.2); every run got quicker instead (0.8.0: Indic-Transcribe language ID + one model load per run; Pappa 140 → 111 s, same CER) |
| 21 | Keep IndicConformer? | Move it to the USB stick, plus everything not needed now; delete what is not needed | **done**: IndicConformer, Whisper large-v3 + medium, Gemma 3 and the setup downloads on D: (sha256-verified, junctions); pip/npm/conda/Composer caches and old crash dumps removed. C: 6.5 → 18.4 GB free |

## Open items (2026-10-23)

| # | Item | Status / what is needed from the owner |
|---|---|---|
| 14b | **Tokens pasted in chat** | They were never used. If you have not done it yet, revoke them at huggingface.co/settings/tokens. The login stored on this PC is only needed for future downloads (`hf auth logout` removes it). |
| 15 | **Marathi number table** (numbers are written as words) | Please check `library/numbers/mr.toml`, especially the spellings of 41, 71, 81, 91 and 99 (एक्केचाळीस, एक्काहत्तर, एक्क्याऐंशी, एक्क्याण्णव, नव्व्याण्णव). The file is plain text and can be edited directly. |
| 16 | **Standard-spelling version of the Pappa transcript** (your convention: standard spelling) | Claude drafted it: `evals/fixtures/pappa_260603_211622/truth_standard_DRAFT.txt` (this PC only). It is used as a second score (`std`: Whisper 0.487 → Indic-Transcribe 0.302). Unsure words were left exactly as you wrote them. Please correct the draft. |
| 17 | **English words in Devanagari** | **Solved by Indic-Transcribe**, which writes them in Devanagari itself. Only output of the Whisper fallback still lists Latin words for review (`latin_script`). |
| 18 | **Your recordings** | **Received** (2026-10-23): 15 voice notes, now in `evals/test_audio/` (git-ignored). Each was transcribed (again with 0.8.1, the better pipeline); the drafts are in `evals/test_audio/drafts/<id>.txt`. **Please correct them** (keep `speaker: text` lines, standard spelling, numbers as words, English in Devanagari); each corrected draft becomes a scored test file. |
| 19b | **Old commits still on GitHub** | History was cleaned twice on 2026-10-23 (the second pass also removed single words and speaker names). GitHub keeps the replaced commits reachable by their direct links: e07e138 (all 26 turns of the call) and 0a975f6 with its parents (a few single words). The repo is private, so only your account can open them. To remove them completely, ask GitHub Support to purge them ("remove sensitive data", repo Atharva-Git01/transcribe-env), or delete the repo on GitHub and let Claude push the clean history to a new one. |
| 23 | **The 20-minute call: transcript of 12:00–20:35** | Your transcript stops at 12:00 (turn 163); the call runs to 20:35. Only the first 12 minutes are scored now (`scored_until_sec`). If you transcribe the rest, the whole call becomes the test. |
| 24 | **Colloquial or standard spelling in the output?** | Your 20-minute transcript writes colloquial forms (contractions, colloquial negation, s for sh); the output follows your convention (standard spelling). The difference is 14 % of that call's CER. Keep standard spelling, or write as spoken? |
| 25 | **Names for the 20-minute call** | The same names come out in several forms. A list of the people and places in the call (`--hotwords`) lets the environment check them (planned name-consistency check). |
| 22b | **More space on C: (your call)** | C: has ~14 GB free (after the CUDA PyTorch install). Bigger wins need you: the hibernation file (6.3 GB; `powercfg /h off` in an admin terminal turns hibernation and Fast Startup off), Docker data (7.4 GB; `docker system prune`), Android emulator images (`.android`, 5.1 GB), OneDrive files kept on this PC (32 GB; "Free up space" in OneDrive), the uv cache (2.5 GB; not touched because running tools use it). A run needs ~5.5 GB of free memory: close Chrome for long runs (E31). |
