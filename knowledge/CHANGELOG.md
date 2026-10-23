# CHANGELOG

Every accepted change to the environment, newest first, with its `trx eval` result.
Format: `## <version> — <date>` · what changed · why · eval before → after (or "no scoring yet" before phase 1).

## 0.8.1 — 2026-10-23 (quality pass: denoising off, honest flags)
**Eval (seeded; new baseline saved).** Every fixture better or equal; mean CER 0.223 → **0.196**.

| Fixture | CER 0.8.0 → 0.8.1 | Real-time factor |
|---|---|---|
| Pappa call (mr, phone) | 0.374 → **0.361** | 2.10 → 1.63 |
| FLEURS mr | 0.047 → **0.044** | 1.75 → 1.21 |
| FLEURS hi | 0.021 → 0.021 | 1.92 → 1.31 |
| FLEURS mr + music | 0.388 → **0.305** | 1.55 → 1.24 |
| AMI meeting (en) | 0.284 → **0.249** | 1.81 → 1.26 |

- **Denoising off (E38):** Indic-Transcribe hears better without DeepFilterNet; the quiet-speech check now also runs
  on audio that still contains the speech (E22), so it recovers quiet stretches that were erased before. They are
  transcribed and marked `quiet_speech` (soft: REVIEW, never FAIL).
- Tried and not adopted: beam search (4 beams, ±0.007 per fixture both ways), recognising the original audio while
  VAD/diarization keep the denoised track (mean 0.207), wider turn margins (AMI worse).
- **Flags calibrated on human-written text (E37):** "repetitive" now means repeated phrases (≥ 35 % of a line's
  word triples), not a long line; whole-turn backchannels are not loops. On the owner's 15 recordings: no FAIL
  any more (it was a false alarm), REVIEW only for quiet speech found by the second check or a possible loop.
- **The owner's 15 recordings (15.3 min):** moved into `evals/test_audio/` (git-ignored), transcribed with 0.8.1:
  8 PASS, 7 REVIEW, 0 FAIL; drafts for correction in `evals/test_audio/drafts/` (`setup/make_drafts.py`). Accuracy
  will be measured once the owner's transcripts arrive.
- Fixed: the review list named every speaker "SPK?" (E35); the privacy check flagged common phone phrases (E36).
- Memory need on the GPU 3 → 2 GB (measured: the model adds ~1.5 GB once PyTorch is loaded).

## 0.8.0 — 2026-10-23 (one model load per run, Indic-Transcribe language ID)
**Eval (seeded; new baseline saved): every CER identical to 0.7.1, every run quicker.**

| Fixture | CER | Real-time factor 0.7.1 → 0.8.0 | Notes |
|---|---|---|---|
| Pappa call (mr, phone) | 0.374 | 2.54 → **2.10** (1.98 in a single clean run) | 140 s → 111 s; verdict REVIEW → PASS |
| FLEURS mr | 0.047 | 1.93 → **1.75** | |
| FLEURS hi | 0.021 | 1.99 → **1.92** | |
| FLEURS mr + music | 0.388 | 1.58 → **1.55** | the separation still reloads the model once (VRAM) |
| AMI meeting (en) | 0.284 | 1.95 → **1.81** | speakers right 0.916 |

- **Language ID by Indic-Transcribe** (E34): right on 5/5 fixtures with P = 1.0. The 28 s of forced Marathi/Hindi
  decoding that Whisper needed (E10) is gone, and so is the Whisper load (14 s). Whisper remains the fallback
  language-ID engine.
- **One load per run** (`pipeline/engines/cache.py`, E34): the probe's model is reused by the per-group language ID
  and by recognition (−27 s), then released. The probe's music check moved before language ID so that both fit on the
  GPU.
- The Pappa verdict became PASS: its REVIEW came only from Whisper's weak language evidence. The gate never measures
  accuracy: PASS means no flags.
- Where the time goes now (Pappa, 111 s): probe 52 s (model load ~29 s, speaker-count estimate 16 s, music check
  4 s, language ID 3 s), denoise 17 s, diarization 16 s, recognition 21 s.

## 0.7.2 — 2026-10-23 (`--fast` removed)
- **`--fast` removed (owner's decision).** As Whisper turbo it had become slower than the default (RTF 3.74 vs 2.54)
  and twice as wrong (CER 0.771 vs 0.374). Redefining it as "long speech pieces" was measured first and was not faster
  on the GPU: Pappa 143 s vs 140 s, AMI RTF 1.84 vs 1.95. Long pieces wrote better words (Pappa 0.361 vs 0.374, AMI
  0.248 vs 0.284) but put much more text on the wrong speaker (0.695 vs 0.975, 0.713 vs 0.916; E28).
  `trx transcribe --fast` now stops with USAGE and says why.
- **Where a 1-minute call's 140 s go (GPU):** probe 55 s (Whisper language ID, music check, speaker estimate),
  denoise 17 s, diarization 16 s, model load 27 s, recognition 22 s. Next step: Indic-Transcribe's own language ID
  and one model load per run.

## 0.7.1 — 2026-10-23 (GPU first; disk space)
**Eval (seeded; new baseline saved).** Same configuration as 0.7.0, with Indic-Transcribe and Demucs on the GPU:

| Fixture | CER 0.7.0 (CPU) → 0.7.1 (GPU) | Speakers right | Real-time factor |
|---|---|---|---|
| Pappa call (mr, phone) | 0.379 → **0.374** | 0.975 → 0.975 | 3.06 → **2.54** |
| FLEURS mr | 0.048 → **0.047** | – | 2.30 → **1.93** |
| FLEURS hi | 0.021 → 0.021 | – | 2.34 → **1.99** |
| FLEURS mr + music | 0.383 → 0.388 | – | 3.57 → **1.58** |
| AMI meeting (en) | 0.284 → 0.284 | 0.890 → **0.916** | 2.39 → **1.95** |

No fixture got worse beyond the noise margin. The repeated Pappa run gives the same text (deterministic on the GPU).
The baseline was saved with this code before `VERSION` was raised, so its entries still read 0.7.0.

- **GPU first (owner's rule):**
  - CUDA 12.6 build of PyTorch (hash-locked `setup/requirements-ml.lock`; the CPU build is kept as
    `requirements-ml-cpu.lock`).
  - Indic-Transcribe runs in bf16 (2.5 GB of VRAM), and so does Demucs. cuDNN and cuBLAS are set to deterministic
    (E18).
  - Out of memory on the GPU falls back to the CPU, logged.
- **Load straight to the GPU (E33):** `accelerate` + `psutil` (650 KB). A run holds ~2.7 GB of RAM instead of ~7 GB,
  and the model loads in 14–25 s instead of 77 s. The memory check asks for 3 GB on the GPU (5.5 GB on the CPU).
- **Disk (owner's decision):** IndicConformer, Whisper large-v3 and medium and Gemma 3 moved to the D: USB stick
  (sha256-verified, linked by junctions); setup downloads moved too; pip/npm/conda/Composer caches and old crash
  dumps cleaned. C: 6.5 → 18.4 GB free. `trx doctor` requires Gemma only when the default engines need it, and a
  missing Gemma skips the cleanup instead of failing a `--fast` run.
- **`--fast` is now slower than the default:** Pappa RTF 3.74 against 2.54 (Gemma cleanup runs on Whisper output),
  CER 0.771 against 0.374 (QUESTIONS.md 23).
- Still on the CPU, because no GPU build is installed or they are small: DeepFilterNet, Silero VAD, the speaker
  embeddings (sherpa-onnx), IndicConformer (onnxruntime).

## 0.7.0 — 2026-10-23 (Indic engines; PLAN_v0.7)
**Eval (seeded; same pipeline for every engine; new baseline saved).** CER (normalised CER in brackets).

| Fixture | Whisper large-v3 (0.6.0) | IndicConformer-600M | **Indic-Transcribe-core (default)** |
|---|---|---|---|
| Pappa call (mr, phone, 2 speakers) | 0.513 (0.498) · spk 0.917 · 4/26 turns missing | 0.377 (0.368) · spk 0.966 · 5/26 | **0.379 (0.371)** · spk **0.975** · 4/26 |
| FLEURS mr (10 read sentences) | 0.273 | 0.056 | **0.048 (0.030)** |
| FLEURS hi (10 read sentences) | 0.072 | 0.034 | **0.021 (0.010)** |
| FLEURS mr + music 5 dB below (new) | 0.606 (no Demucs; Demucs made it worse) | – | **0.383 (0.370)** with Demucs (0.445 without) |
| AMI ES2004a, 4-person English meeting (new) | 0.308 · spk 0.895 · 32/57 | no English | **0.284** · spk 0.890 · 35/57 |

- Pappa against the standard-spelling draft (`std`, QUESTIONS.md 16): 0.487 → **0.302**.
- Speed, whole pipeline, real-time factor: Pappa 3.1 (Whisper large-v3 + Gemma: 4.4), FLEURS 2.3, AMI 2.4, music 3.6.
  The Indic engines run
  on the CPU.

Engines (E24, E25)
- **Indic-Transcribe-core** (Bodhan AI / AI4Bharat, 1.2B, Indic Open Model License v1.0) is the primary engine for
  every language. It writes standard spelling, numbers as words and English words in Devanagari ("native" mode).
- **IndicConformer-600M** (AI4Bharat, MIT; ONNX, CTC + RNNT) is the first fallback for mr/hi. Whisper remains the
  last fallback, the `--fast` engine and the language-ID engine.
- Both adapters (`pipeline/engines/`) load local files only, at pinned revisions (`SOURCE.json`, 404 + 21 files).
  `from_pretrained` downloads are never called. Indic-Transcribe's code runs from its model folder, and a stand-in
  replaces torchaudio, which it would only need for resampling (E25).
- Settings per engine family, `[recognise.<family>]` in every recipe; `setup/` gained Hugging Face snapshots at a
  pinned revision and `requirements-ml.lock` (PyTorch 2.14 CPU, transformers, sentencepiece, Demucs; sha256).

Pipeline
- **Each speaker turn decoded on its own** (`segmentation = "turns"`) for engines without word timings (E28).
- **Demucs** music/TV separation: the probe measures music on a 10 s window. Rule R2b_music picks
  `noisy_field_audio`, and its `separate` stage keeps the voice when the background is louder than −15 dB (E27).
- **Gemma cleanup only for Whisper output** (`[llm_cleanup] engines`): with Indic-Transcribe it changed nothing (E26).
- **Output convention (owner):**
  - standard spelling (the engine's native mode)
  - numbers as words (`library/numbers/mr.toml`, `hi.toml`; times are left as written and reported)
  - English words in Devanagari (Indic-Transcribe; Whisper output still lists Latin words as `latin_script`)
- **Quiet speech:** a VAD gap audit (E22), and missed-turn recovery with a plausibility check (E23).
- **Rejected:** the speech guard (E21).

Eval and privacy (E29)
- `trx eval` now reports strict, normalised and loose CER; `std` CER against the standard-spelling draft; missing
  truth turns; Hindi forms in Marathi output; and the real-time factor.
- **`evals/results/` holds numbers only** (speakers anonymised as A, B, …). The truth-vs-output table per turn goes to
  `evals/compare/`, which never leaves this PC.
- The 0.7 WIP push had included the call's words. With the owner's OK the history was rewritten and force-pushed.
- `trx test` now fails if any file git would push holds words of a private recording or run (`pipeline/privacy.py`).
- **Fixed (E30):** an eval of several fixtures transcribed each fixture in the previous fixture's language.
- **Memory check (E31):** a run stops with exit 51 `MEMORY_LOW` (and a next step) instead of crashing when the
  1.2B model cannot fit in memory (`[runtime.memory_gb]`). `--save-baseline` never saves a run that stopped.
- **Fixed (E32):** Demucs separated the same audio differently on every run (an unseeded random time shift).
- **`--fast`** (Whisper turbo; Gemma cleanup still runs on its output) on the phone call: CER 0.771, speakers right
  0.717, 8/26 turns missing, RTF 2.7 against 3.1 for the default. It no longer pays off (QUESTIONS.md 23).
- **New fixtures:** FLEURS mr + music (`make_mix_fixture.py`) and AMI ES2004a, 3 minutes with 4 people
  (`make_ami_fixture.py`). The owner's recordings go in with `setup/make_fixture.py`.
- `trx test`: 32 tests (0.6.0: 26).

Tried and rejected
- **Whisper as cross-check engine** (verify) for Indic-Transcribe: it flagged 96 % of the phone call's lines and
  60 % of the clean read Marathi (none of those lines was bad), so the verdicts became FAIL. Verify stays off.
- **Indic-Transcribe's language ID in the probe:**
  - It was right on 5/5 fixtures with P = 1.0 on every window. Whisper's acoustic top was Hindi for all three Marathi fixtures.
  - The final decision is the same on every fixture, and it would add about 1 minute per file.
  - The probe keeps Whisper. The adapter supports it if a misguess appears (`[languages] lid_engines`).
- **Longer pieces (VAD groups ≤ 30 s) for Indic-Transcribe:** CER 0.357, but speakers right fell to 0.696 (E28).
- **Finer diarization on the phone call** (not adopted, to re-test on the owner's recordings):

  | Setting | CER | Speakers right | Missing turns |
  |---|---|---|---|
  | 1.0 s windows (kept) | 0.379 | 0.975 | 4/26 |
  | switch penalty 0.05 | 0.374 | 0.976 | 4/26 |
  | 0.75 s windows | 0.355 | 0.955 | 4/26 |
  | 0.75 s windows + switch penalty 0.05 | 0.359 | 0.948 | 3/26 |

  With 0.75 s windows CER improves by about 0.02 and speaker accuracy drops by about 0.02, measured on one 56-second
  call. That is too small to decide on one fixture (MAINTAINER.md).

Not solved yet
- **Short replies in fast exchanges:** 4 of 26 Pappa turns still have no text of their own.
  - Three of Whisper's four misses are now written: two short replies and the quiet sentence.
  - The new misses are short turns that diarization merged into a neighbour's turn.
- **Speed:** RTF ~3 on the CPU. The CUDA build of PyTorch could move Indic-Transcribe to the GPU (QUESTIONS.md 20).

## 0.6.0 — 2026-10-22 (owner decisions implemented)
**Eval (seeded; baseline saved):**

| Fixture | CER 0.5.0 → 0.6.0 | WER | Speakers right | Verdict |
|---|---|---|---|---|
| Pappa call (mr, phone, 2 speakers) | 0.5308 → **0.5132** | 0.8438 → 0.8047 | 0.917 (was: none) | REVIEW |
| FLEURS mr (10 clean sentences, new) | — → **0.2728** | 0.7198 | — | REVIEW |
| FLEURS hi (10 clean sentences, new) | — → **0.0717** | 0.1985 | — | PASS |

Owner decisions (QUESTIONS.md)
- **Models moved from the D: USB stick to C:.** The pip cache (15 GB) was purged first, and sha256 was re-verified after the copy. Runs got ~70 s faster.
- **Language is confirmed before transcribing.**
  - A person at a terminal is asked directly.
  - Other operators get exit 14 `LANGUAGE_CONFIRM_NEEDED` and rerun with `--lang`.
  - `--lang auto` skips the question.
- **New options:** `--fast` uses turbo; `--hotwords FILE` passes the case's names and places to the recogniser.
- **Diarization on every file (E19).**
  - The new mode `embed_cluster` uses ERes2NetV2 embeddings on 1 s windows, with k-means and Viterbi smoothing: 94.5 % of words on the right speaker in the experiment, 91.7 % in the pipeline.
  - Phone calls are fixed at 2 parties; other recordings estimate the speaker count.
  - Transcript lines are split where the speaker changes.
- **Long files:** 10-minute recognition chunks are cut at silences and saved one by one.
  - Running the same command again resumes an interrupted run. A file lock tells "interrupted" from "still running".
  - Diarization results are reused on resume.
- **Gemma cleanup is ON (E17).**
  - A word gate keeps only spelling, punctuation and Hindi→Marathi form fixes of the words that were heard.
  - A Marathi/Hindi language model must agree with each kept edit.
  - Cleanup off → on: Pappa 0.5308→0.5132, FLEURS mr 0.2856→0.2728, FLEURS hi 0.0827→0.0717.
- **Language models and lexicons for mr and hi.**
  - Built from IndicCorp v2 (CC-0; first 1 GiB each) into a numpy trigram model of 125 MB each. KenLM was not used because it has no Windows build.
  - Polish lists `unknown_words` for the reviewer. These are info only and never change the verdict. Unknown-word rate: ASR 12–34 % vs human transcripts 1–11 %.
- **Fixtures:** FLEURS Marathi and Hindi (CC-BY-4.0) were added. `trx eval` gained the `spk` column.

Fixed while measuring
- **E20:** per-group language ID could pick English for Marathi speech, and Whisper then translated it. English now needs P ≥ 0.80 against all other languages together.
- **Per-group language ID** chooses only between the main language and English, and still runs when the language is confirmed.
- **Unread recipe key removed:** `lid.candidates` (E05).
- **Tried and rejected:** re-decoding each line alone to get 5 alternatives for the language model to choose from. CER got worse (0.53 → 0.58–0.62).

## 0.5.0 — 2026-10-16 (overnight build: phases 1–5)
**Eval (fixture pappa_260603_211622, Marathi phone call, seeded):** large-v3 CER **0.531** / WER 0.844, verdict
REVIEW (FMS re-scored with the same metric: 0.745; it reported 0.7225 under its own normalisation). Turbo (fallback when D: is unplugged) CER 0.793. Baseline saved in `evals/baseline.json`.

Built
- Phase 1: hash-locked venv (`setup/requirements.lock`, 26 packages + cuDNN), CUDA 12 via hard links, GPU decoding on
  the RTX 3050; stages convert, denoise (DeepFilterNet3 → ffmpeg afftdn → copy), vad (Silero), lid, recognise,
  polish, gate, export; probe gets in-speech SNR (E09), Silero activity, LID and a speaker estimate.
- Phase 2: large-v3-turbo + large-v3 (D: USB via junction); 2-step LID for mr/hi (E10, E11); verify stage
  (cross-check engine, per-word-span agreement; off by default); `trx eval` with baseline comparison.
- Phase 3: Gemma 3 4B Q4_K_M through llama.cpp `llama-server` (Vulkan, localhost, no service); 5 rejection gates;
  **off by default** (E17).
- Phase 4: sherpa-onnx diarization (pyannote segmentation 3.0 + WeSpeaker). Demucs and resumable long files: not
  built (QUESTIONS.md Q6, Q7).
- Phase 5: `trx test` (16 unit tests), `trx doctor --deep`, dead-config check for recipes (`STAGE_KEYS`, E05),
  engine tables as ordered fallback lists, `trx explain` also finds eval runs.

Changed after measuring (all on the one fixture — see QUESTIONS.md Q10)
- Engines per language: large-v3 first, turbo fallback (medium looped in Bengali letters on Marathi, CER 1.52).
- Segmentation: `vad_concat` + split long segments at word boundaries (`split_max_s = 10`): large-v3 0.531 vs 0.577
  with per-group clips; turbo 0.793 vs 0.813 (E14).
- Per-segment compression ratio instead of faster-whisper's per-window value (one loop flagged a whole 30 s window).
- Seeded CTranslate2 sampling (E18): earlier unseeded numbers (turbo 0.720, large-v3 0.639) were not reproducible.
- Gate: soft flags (`hindi_forms_in_marathi`, `speaker_change_inside_segment`, `crosstalk`) can cause REVIEW but
  not FAIL; hard flags (loops, low confidence, wrong script …) over 40 % of the time still FAIL.
- Removed config keys that no code read (E05): long-file chunk/resume keys, numerals, tag templates, unused LID
  knobs, IndicConformer table; manifest rewritten to what is actually installed.

## 0.1.0-phase0 — 2026-10-16
- Built the skeleton: `trx` entry point (help, version, doctor, probe, transcribe, explain, eval, watch),
  offline guard, config (defaults, thresholds, selection rules, exit codes), JSON Schemas + validator, 8 recipes,
  run folders with read-only input copy, sha256 custody chain, decisions log, report, per-run git.
- Probe measures: channel layout (duplicated-mono detection), loudness (EBU R128), true peak, clipping, energy-based
  noise floor / SNR / activity, spectral-rolloff bandwidth class.
- Seeded knowledge/ERRORS.md (E01–E08), references, prompts, doubled-word lists; FMS vocab copied disabled (E07).
- Eval fixture: Pappa call (human ground truth). No scoring yet — the recognise stage arrives in phase 1.
- Config format is TOML (Python's built-in reader) instead of YAML: zero extra packages.
