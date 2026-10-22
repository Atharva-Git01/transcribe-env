# CHANGELOG

Every accepted change to the environment, newest first, with its `trx eval` result.
Format: `## <version> — <date>` · what changed · why · eval before → after (or "no scoring yet" before phase 1).

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
