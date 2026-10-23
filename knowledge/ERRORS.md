# ERRORS — what went wrong before, and the rule that now prevents it

Format per entry: **Seen** (what happened) · **Why** · **Fix** · **Check** (how the environment detects it) · **Rule**.
Read before changing anything (MAINTAINER.md). Append after every reviewed run that teaches something new.

## E01 — Mono audio transcribed twice as two speakers
- **Seen:** FMS turned a mono phone call into fake stereo (same signal on both channels) and transcribed it twice
  as SPEAKER_00 and SPEAKER_01 with identical text (Pappa call, 2026).
- **Why:** channel-as-speaker diarization was the default and ingest duplicated mono to stereo.
- **Fix:** probe classifies layout from measured inter-channel correlation and level difference.
- **Check:** `passport.channels.layout_class` = `duplicated_mono` when correlation ≥ 0.99 and level diff ≤ 1 dB.
- **Rule:** channel = speaker ONLY for `true_stereo`. Duplicated mono is downmixed.

## E02 — Real Marathi doubled words deleted as "hallucination"
- **Seen:** the FMS loop filter (last two words identical in a ≥4-word segment) could delete हा हा / हो हो, which
  are real speech in the ground truth.
- **Fix:** `library/doubled_words/<lang>.txt`; loop handling is `flag`, never delete.
- **Check:** eval fixtures contain doubled words; a regression drops them and raises CER.
- **Rule:** no stage silently deletes text. Uncertain text gets `[review]`.

## E03 — High error rate even with the best settings
- **Seen:** CER 0.7225 / 0.736 on the Pappa call with large-v3, CUDA, forced Marathi and DeepFilterNet.
- **Why:** hard colloquial code-switched phone speech; single engine; single VAD pass on the stereo path.
- **Fix:** dual-engine agreement + re-decode of weak windows (phase 2); Indic engine candidate; tuning via eval.
- **Check:** `trx eval` on fixtures; `evals/baseline.json`.
- **Rule:** no change is accepted without an eval result.

## E04 — A run returned zero segments
- **Seen:** FMS CPU + medium + no denoise + no prompt produced an empty transcript; VAD alone was not the cause.
- **Fix:** fixed fallback ladder (OPERATOR.md): VAD off → lower no-speech threshold → other engine → CPU engine → FAIL.
- **Check:** gate `fail_if_no_text`; a PASS can never have an empty transcript.
- **Rule:** never return empty output with a success status.

## E05 — The documented config was never loaded
- **Seen:** FMS `configs/default.yaml` looked authoritative but the app used hard-coded defaults.
- **Fix:** one parameter path: `config/defaults.toml` → recipe → `plan.json`; stages read ONLY plan.json.
- **Check:** plan.json is schema-validated and copies every parameter it uses (including the engine tables).
- **Rule:** a default lives in exactly one file.

## E06 — GPU memory not released; environment not reproducible
- **Seen:** FMS freed VRAM only when LLM cleanup ran; its Python environment had no lock file and later vanished.
- **Fix:** every stage loads → runs → unloads; own venv from `requirements.lock`; `trx doctor`.
- **Rule:** one GPU model at a time; pinned versions only.

## E07 — Vocabulary map rewrites real words into other words
- **Seen (2026-10-16, while building phase 0):** the FMS Marathi vocab lists real, different words as "misspellings"
  of a headword — e.g. आहेत under आहे, येते/येता under येतो. Applying it would change correct speech.
- **Fix:** copied as `library/vocab/mr_fms_v1.json` but **disabled** (`config/defaults.toml [vocabulary]`).
- **Check:** `trx doctor` lists variants that are themselves headwords.
- **Rule:** a variant may only map to a headword if the variant is not a valid word itself; vocab v2 needs review.

## E09 — Energy-based SNR is inflated on gated phone audio
- **Seen (2026-10-16, phase 0 probe of the Pappa call):** SNR estimate 42.6 dB, yet FMS found denoising essential
  for this call. Phone codecs gate the gaps between words to near-digital silence, so "quietest frames vs loudest
  frames" measures gap silence, not the noise under the speech.
- **Fix (phase 1, done):** `noise.method = vad_inspeech_percentile_v1` — noise floor measured inside Silero VAD speech
  regions, ignoring codec-gated frames (more than `gated_frame_below_peak_db` below the peak).
- **Check:** passport `noise.method` names the method; the energy value is kept next to it for comparison.
- **Rule:** rule R4 (noisy) reads only the in-speech estimate; the phone recipe denoises regardless.

## E08 — A tool reached the internet on its own
- **Seen (2026-10-16):** running `ollama list` started the Ollama app, which downloaded an update by itself.
- **Fix:** Gemma 3 runs through llama.cpp (no service, no updater); runtime offline guard; `OLLAMA_NO_UPDATE` set.
- **Check:** `trx doctor --offline`.
- **Rule:** nothing in the runtime may have a background updater or a first-use download.

## E10 — Whisper's language ID cannot see Marathi
- **Seen (2026-10-16, phase 2):** on the Pappa call (Marathi, human truth) every Whisper engine's acoustic LID gave
  mr ≈ 0.001 and hi ≈ 0.99 on every window. Marathi and Hindi share script and much vocabulary; Whisper had little
  Marathi training data. Trusting it would transcribe Marathi as Hindi.
- **Fix:** 2-step LID (`pipeline/lid.py`): (1) acoustic LID on 6 windows, (2) for mr/hi, forced decoding of speech
  groups in BOTH languages, compare average log-probs (margin `text_lid_ll_margin`), then count Marathi markers that
  survive in the Hindi-forced text vs Hindi markers in the Marathi-forced text (`text_lid_min_markers`).
  Confidence below `text_lid_min_confidence` → decision = priority language (mr) + warning `language_ambiguous`.
- **Check:** passport `language.decision` {lang, basis, confidence}; the gate turns `language_ambiguous` into REVIEW
  unless the operator forced `--lang`.
- **Rule:** never trust acoustic LID alone for mr/hi. A weak decision is a REVIEW, not a PASS.

## E11 — Marker counting inflated by decoder loops
- **Seen (2026-10-16):** a looping segment "आता हूँ. आता हूँ. …" counted as many Hindi markers and flipped the
  text-based language decision.
- **Fix:** count DISTINCT marker types per language; looping segments are dropped before counting; the cross-forced
  "survivor" test (markers that appear even when the OTHER language is forced) carries the decision.
- **Check:** `tests/test_core.py::TestLidEvidence` (a loop counts once).
- **Rule:** any count-based evidence must be robust to repetition.

## E12 — faster-whisper's audio decoder broke on PyAV 19
- **Seen (2026-10-16, phase 1):** `faster_whisper.decode_audio` passed a `metadata_errors` argument that av 19.0.1
  no longer accepts → crash before recognition.
- **Fix:** `pipeline/audio.decode_mono` decodes through ffmpeg (f32le pipe, 16 kHz mono); faster-whisper only ever
  receives numpy arrays.
- **Rule:** all decoding goes through ffmpeg; library decoders are not used.

## E13 — A half-downloaded model looked installed
- **Seen (2026-10-16, setup):** large-v3's model.bin download stopped at 75 MB; the file existed, so a "file exists"
  check would have called it installed. A later retry failed the sha256 check once more before succeeding.
- **Fix:** a model counts as installed only when `SOURCE.json` exists — written by setup only after every file's
  sha256 and size match `setup/sources.toml`. `trx doctor` checks sizes; `trx doctor --deep` re-hashes.
- **Rule:** never mark a component installed on file presence alone.

## E14 — Long recognition segments hide loops and make the gate FAIL everything
- **Seen (2026-10-16, phase 2 eval):** with `vad_filter` segmentation, faster-whisper merged the call's speech into
  two segments (0.4–33.6 s and 33.6–55.3 s). One "आता हूँ" loop flagged the first 33 s → 100 % flagged → FAIL,
  even though CER (0.72) equalled FMS large-v3.
- **Why (two causes):** long output segments, and faster-whisper giving every segment its 30 s window's
  compression ratio, so one loop marked the whole window "repetitive".
- **Fix:** keep `vad_concat` decoding (most context — per-group `clips` scored worse once seeded, E18) but cut output
  segments longer than `split_max_s` (10 s) at word boundaries (text unchanged), and compute the compression ratio
  per segment (window value kept as `window_compression_ratio`).
- **Check:** the gate's hard-flag ratio; `tests/test_core.py::TestSegmentation`.
- **Rule:** segment length is a recipe parameter, not an engine default.

## E15 — Disk ran out during installs
- **Seen (2026-10-16, setup):** C: dropped to 1.5 GB free while Gemma was downloading; later the Windows pagefile
  grew to 10.6 GB while large-v3 loaded and C: fell from ~9.8 GB to 3.6 GB free.
- **Fix:** large models went to D: via junctions for the night; on 2026-10-16 the owner cleared the 15 GB pip cache and
  they moved back to C: (sha256 re-verified). Downloads stop when C: is low; `[runtime] min_free_gb` in doctor.
- **Check:** `trx doctor` → `disk`.
- **Rule:** check free space before any install; the pagefile shrinks back only after a reboot.

## E16 — A character loop in the wrong script was not flagged
- **Seen (2026-10-16, phase 2):** one engine produced a long run of a Bengali letter ("ববববব…") inside Marathi text;
  the word-loop detector saw a single "word" and did not flag it.
- **Fix:** polish flags `CHAR_RUN` (same character ≥ 8 times) and `wrong_script` (letters outside Devanagari/Latin).
- **Check:** `tests/test_core.py::TestPolish`.
- **Rule:** loop detection works on characters as well as words.

## E17 — LLM cleanup changed meaning while passing every gate
- **Seen (2026-10-16 and 2026-10-22, Pappa call):** Gemma 3 4B fixed real errors (है→आहे, नहीं→नाही, spelling of an
  English loanword) but the same suggestions also changed meaning or added garbage that passed all 5 whole-sentence
  gates: a misheard verb "corrected" into a different verb that was never said, a word rewritten half in Latin
  letters, का→क्या (a new Hindi form), two words merged into one.
- **Fix (2026-10-22): a word gate** (`pipeline/stages/cleanup.py`). The suggestion is aligned word by word with what was
  heard; a word may change only by a small spelling fix (≤ `max_word_edit_ratio` = 25 % of its letters),
  punctuation, or a listed Hindi→Marathi form (`library/lang_forms/mr.toml`). No word added or removed, no new
  script, no new Hindi form. Everything else keeps the heard word. Pappa CER: no cleanup 0.5308 → old gates 0.5176
  (with the meaning changes) → word gate 0.5154 (none of them).
- **Check:** `tests/test_core.py::TestCleanupGates`; every word decision is in `work/llm_cleanup.json`.
- **Rule:** an LLM may only correct the form of a word that was heard, never choose a different word.

## E18 — The same file gave different transcripts on different runs
- **Seen (2026-10-16, phase 2 eval):** turbo on the Pappa call scored CER 0.720 in one run and 0.793 in the next
  with identical settings. Whisper's temperature fallback (t = 0.2 … 0.6 when a window looks like a loop) samples
  randomly, and CTranslate2's random generator was never seeded.
- **Fix:** `ctranslate2.set_random_seed(config/defaults.toml [runtime].seed)` before every decode
  (`pipeline/engines/faster_whisper_engine.py`). Two runs now give identical CER (0.7930, 0.7930).
- **Check:** run `trx eval` twice; the numbers must match exactly.
- **Rule:** every stochastic step is seeded from `[runtime].seed`; an eval result that cannot be reproduced is not
  evidence. Numbers measured before this fix (CHANGELOG) are marked "unseeded".

## E19 — Diarization could not separate the two people on a phone call
- **Seen (2026-10-22, Pappa call, mono narrowband):** sherpa-onnx (pyannote segmentation 3.0 + WeSpeaker ResNet34) put
  ~all speech under one speaker. Measured as the share of words with the right speaker: 54 % = "everyone is one
  speaker"; best pyannote-pipeline setting 68 %; forcing 2 speakers still gave 1. The segmentation model never saw a
  speaker change in telephone-band audio, and the VoxCeleb embeddings barely told the two voices apart (cos 0.90).
- **Fix:** mode `embed_cluster` (`pipeline/stages/diarize_embed.py`): 3D-Speaker ERes2NetV2 embeddings on 1 s windows
  (hop 0.25 s) inside VAD speech, cosine k-means, Viterbi smoothing, short turns merged → **94.5 %** (CAM++ 92–94 %,
  TitaNet-L 91–92 %). Transcript lines are split where the speaker changes (text unchanged).
- **Limit:** inside ONE speaker the halves can be as dissimilar (cos 0.62–0.75) as two speakers (0.67), so the number
  of speakers cannot be estimated reliably from 1 s windows. Phone calls are fixed at 2 parties; other recipes
  estimate and the report says "estimated".
- **Update 2026-10-23 (AMI 4-person meeting fixture):** with `split_cosine = 0.6` the meeting was estimated as ONE
  speaker (spk 0.628). Counting experiment (divisive split vs silhouette vs eigengap): only the divisive split with a
  threshold of 0.65–0.70 gets AMI → 3 (the 4th person has < 5 s of speech in the window), Pappa → 2 and a
  single speaker → 1. Silhouette cannot answer "1"; eigengap said 3 for one speaker. Now `split_cosine = 0.68`:
  AMI spk **0.895**, 3 speakers found.
- **Check:** `trx eval` column `spk` (speaker accuracy; needs a fixture whose truth has ≥ 2 speakers).
- **Rule:** a diarization change needs a `spk` result, like an ASR change needs CER.

## E20 — Marathi speech decoded as English came out TRANSLATED
- **Seen (2026-10-22, FLEURS Marathi fixture, meeting recipe):** per-group language ID was limited to "main language
  vs English". Whisper gives Marathi almost no probability (E10), so after renormalising English won 2 of 9 Marathi
  groups. Decoding a Marathi group as English made Whisper translate it ("Till 1970, there was a tendency…") or
  romanise it. CER 0.273 → 0.400.
- **Fix:** `segment_language()` in `pipeline/stages/lid.py`: candidates are ALL priority languages, and a group becomes
  English only if P(en) ≥ `[languages] segment_en_min` (0.80) against all of them together.
- **Check:** `tests/test_core.py::TestSegmentLanguage`; the FLEURS Marathi fixture in `trx eval`.
- **Rule:** never let a renormalised probability over a reduced candidate set decide a language; a wrong language
  forced on Whisper is worse than no decision.

## E21 — Splicing the original audio back into the denoised track broke recognition
- **Seen (2026-10-23, Pappa call):** DeepFilterNet turned soft speech down by its full 18 dB limit. This included
  short replies and a quiet sentence at 51.7–52.7 s (−30.5 → −48.5 dBFS). A "speech guard" put the original audio back
  wherever the original was speech (Silero ≥ 0.2) but the denoised track was 12 dB+ quieter (11 stretches, 6.2 s).
  Whisper then lost whole turns: CER 0.513 → **0.921**, missing turns 4 → 12.
- **Decision:** `speech_guard = false` in every recipe. The code stays for future engines, but needs an eval before
  re-enabling.
- **Also measured:** gentler denoising (10 dB limit) gave CER 0.515 and no denoising 0.518. On this call the lost
  short replies are **not** caused by denoising. Whisper does not write them down whatever the audio (see E23).
- **Rule:** mixing differently processed audio inside one track creates bursts that the recogniser treats as
  something else. Process whole tracks.

## E22 — The quiet-speech check could not see speech the denoiser had removed
- **Seen (2026-10-23):** the VAD gap audit runs on the denoised track, so the 51.7–52.7 s soft speech was already
  gone (only 31 % of frames above the floor; Silero found nothing). Its noise floor was also −120 dB on the denoised
  track (digital silence), so the energy test passed everything.
- **Fix:** the floor now ignores digital silence (frames below −90 dB). Silero gets 0.5 s of context on each side
  of a gap. The check still finds nothing on this call, so it stays as a safety net.
- **Rule:** any "is there speech here?" check must run on audio that still contains the speech.

## E23 — Whisper invents text when a short turn is decoded on its own
- **Seen (2026-10-23):** decoding each empty speaker turn alone (missed-turn recovery) returned repetitions that were
  never said (one word twice in a row, a filler word four times). CER 0.513 → 0.566, although missing turns fell from 4 to 1.
- **Fix:** recovered text must pass `_plausible()` (no loop or repeated words, no character runs, avg log-prob ≥ −0.8,
  no-speech ≤ 0.5) and is flagged `recovered_turn` (soft). With Whisper nothing passed on this call, so CER is back
  to 0.513. In `turns` mode (the Indic engines, E28) every turn is already decoded on its own, so recovery only runs
  in the other modes.
- **Rule:** text produced from a few hundred milliseconds of audio is guilty until proven plausible.

## E24 — A week of tuning around a weak recogniser gained 0.02; a better recogniser gained 0.13
- **Seen (2026-10-23):** every fix for the Pappa call since 0.5.0 (LID, denoise, diarization, Gemma word gate, language
  model) moved CER 0.531 → 0.513. Even clean read Marathi (FLEURS) stayed at 0.273 with Whisper large-v3.
- **Why:** Whisper's training is Hindi-heavy (E10). Wrong Marathi words and Hindi forms came from the model itself.
- **Fix:** same pipeline, different engine. Indic-Transcribe-core (Bodhan AI / AI4Bharat) is now primary for every
  language: Pappa 0.513 → **0.379**, FLEURS mr 0.273 → **0.048**, FLEURS hi 0.072 → **0.021**, AMI (en) 0.308 → 0.284.
  IndicConformer-600M (0.377 / 0.056 / 0.034, no English) is the first fallback for mr/hi. Whisper stays for `--fast`,
  for language ID and as the last fallback.
- **Check:** `trx eval` (all fixtures, normalised CER and `spk`).
- **Rule:** when every tuning attempt hits the same wall, measure a different model before tuning further.

## E25 — Indic-Transcribe needs torchaudio only to resample, which never happens here
- **Seen (2026-10-23):** the model's code imports torchaudio, which is not installed (a 300 MB+ package for one unused
  path). Two failures while working around it: `ValueError: torchaudio.__spec__ is None` (transformers checks the
  module spec), then `UnboundLocalError: importlib` (a local import shadowed the module).
- **Fix:** `_torchaudio_stand_in()` in `pipeline/engines/indic_transcribe_engine.py` imports transformers first (so it
  still sees torchaudio as absent), then registers a stand-in module with a real `ModuleSpec`. Any resampling call
  raises: the environment always passes 16 kHz audio (ffmpeg).
- **Rule:** never install a package for a code path the environment cannot reach; stub it so any use fails loudly.

## E26 — Gemma cleanup proven for Whisper changed nothing for Indic-Transcribe
- **Seen (2026-10-23):** cleanup off → on with Indic-Transcribe: Pappa 0.379 → 0.379, FLEURS mr 0.048 → 0.049,
  FLEURS hi 0.021 → 0.021, music 0.383 → 0.382 (seeded Demucs, E32), while the run took ~15 % longer (Pappa RTF 3.14 → 3.65). The engine already writes
  standard Marathi, so the word gate found almost nothing to keep.
- **Fix:** `[llm_cleanup] engines = ["faster_whisper"]` in `config/defaults.toml`: Gemma runs only on Whisper output
  (where it still gains 0.531 → 0.513). The decision log says when it is skipped.
- **Rule:** a post-processing step proven for one engine must be proven again for the next.

## E27 — Music separation hurt Whisper but helps Indic-Transcribe
- **Seen (2026-10-23, FLEURS Marathi + music 5 dB below the speech):** Whisper large-v3: no Demucs 0.606, voice only
  0.657, voice + background at −12 dB 0.651. Indic-Transcribe: no Demucs **0.445**, Demucs voice only **0.383**
  (seeded, E32; unseeded runs varied 0.358–0.383, Whisper's Demucs runs by as much).
- **Fix:** Demucs (htdemucs) stays on in `noisy_field_audio` (chosen by rule R2b_music when the probe measures music),
  applied only when the background is above −15 dB relative to the speech.
- **Rule:** audio pre-processing is judged per engine; re-measure it after every engine change.

## E28 — Text without word timings cannot be split at speaker changes
- **Seen (2026-10-23, Pappa call):** Indic-Transcribe on speech pieces of up to 30 s: CER 0.357 but speaker accuracy
  **0.696** and 19 of 26 turns without their own text. The engine returns one text per piece and no word timings,
  so a piece spanning two people cannot be split between them.
- **Fix:** `segmentation = "turns"` in `[recognise.indic_transcribe]` and `[recognise.indic_conformer]` of every recipe:
  each diarization turn (same-speaker turns merged up to 30 s) is decoded on its own. CER 0.379, speaker accuracy
  **0.975**, 4 of 26 turns missing. The model card also says shorter pieces cost a little accuracy; the owner's rule
  (diarization on every file) makes speakers the priority.
- **Re-measured on the GPU (0.7.2):** long pieces wrote better words on two fixtures (Pappa 0.361 vs 0.374, AMI
  0.248 vs 0.284) but speakers right fell to 0.695 / 0.713, and they were not faster. Per-turn decoding stays.
- **Rule:** an engine without word timings is always decoded per speaker turn.

## E29 — The words of the private call were pushed to GitHub inside eval results
- **Seen (2026-10-23):** the per-turn comparison added to `trx eval` (v0.7 WIP) wrote the truth AND the output text of
  every fixture into `evals/results/*.json/.md`, a tracked folder. Commit e07e138 pushed all 26 turns of the Pappa call
  to the private repo, against the owner's rule "keep the call off GitHub". Test strings had also been copied from
  the call (since 0.6.0), two words of it sat in the metric self-test (since phase 0), and single misheard words of
  it were quoted in E17 and QUESTIONS.md. The truth's speaker labels went into the results as well.
- **Fix:** `evals/results/` holds numbers only (`turns_without_text()` in `pipeline/evaluate.py`: no text, speaker
  names → A, B, …); truth vs output per turn goes to `evals/compare/`, which is git-ignored. Test strings and examples
  are now made up. With the owner's OK the history was rewritten (every commit scanned clean) and force-pushed.
  GitHub keeps the old commit reachable by its direct link until GitHub Support purges it (QUESTIONS.md 19b).
- **Check:** `trx test` → `TestEvalPrivacy`: `pipeline/privacy.py` looks for 3-word sequences of every private fixture
  (truth and all machine transcripts kept on this PC) and of every run in `runs/` in all files git would push, and
  for their single rare words (corpus count < 200) in docs, code, tests and results.
- **Rule:** nothing derived from a private recording goes into a tracked folder; test strings are invented, never
  copied from evidence; run `trx test` before every push.

## E30 — An eval of several fixtures transcribed each one in the previous fixture's language
- **Seen (2026-10-23):** the first full `trx eval` with Indic-Transcribe (all 5 fixtures in one process) wrote romanised
  loops for FLEURS Hindi ("John John John …"): it had been forced to English. `run_eval()` reused the name `lang`, the
  operator's `--lang` override, for the truth's language while scoring. From the second fixture on, every fixture
  was transcribed in the previous fixture's language. The 0.7 numbers before this came from one-fixture evals and
  are unaffected; 0.6.0 predates the bug.
- **Fix:** the truth's language has its own name (`t_lang`); `--lang` reaches every fixture unchanged.
- **Check:** `tests/test_core.py::TestEvalRun` (two fake fixtures, English then Marathi).
- **Rule:** never reuse a parameter's name for a per-item value inside a loop; a full eval must agree with
  one-fixture evals of the same configuration.

## E31 — The PC ran out of memory while loading Indic-Transcribe
- **Seen (2026-10-23):** three eval runs crashed while the 1.2B model loaded on the CPU: `OSError: The paging file is
  too small for this operation to complete (os error 1455)` and `MemoryError`. A whole Indic-Transcribe run peaks at
  ~7.0 GB of committed memory (6.3 GB in RAM). The PC has 15.6 GB of RAM; a web browser held 4.7 GB and the Claude app
  3.0 GB. The page file did not grow, even with 5.5 GB free on C:. A crashed run with `--save-baseline` also wrote
  CER 1.0 into `evals/baseline.json`.
- **Fix:**
  - Before loading an engine, recognise compares the free memory (RAM + page file, `pipeline/resources.py`) with
    `[runtime.memory_gb]` (Indic-Transcribe 5.5 GB, IndicConformer 3 GB). Below it, the run stops with exit 51
    `MEMORY_LOW`, and the next step is to close big programs or restart.
  - An out-of-memory crash anywhere in a run is reported as `MEMORY_LOW` too.
  - `--save-baseline` skips runs that stopped.
  - With the browser closed, the evals ran.
- **Check:** `tests/test_core.py::TestMemoryCheck`; exit code 51 in `config/exit_codes.toml`.
- **Rule:** measure a CPU model's memory need, configure it, and check it before loading. On a PC shared with daily
  work, the free memory decides whether a run can start.

## E32 — Demucs separated the same audio differently on every run
- **Seen (2026-10-23):** the music fixture scored 0.358 in one run and 0.376 in a later one. The configuration was
  the same, and the per-turn changes went both ways.
- **Why:** with `shifts = 1`, Demucs offsets the audio by a random 0–0.5 s drawn from Python's `random` module, which
  `torch.manual_seed` does not touch. Two offsets changed the separated voice by −14.9 dB relative to the voice:
  enough to change the words. The probe's music check (Demucs on 10 s) had the same randomness.
- **Fix:** `random.seed(0)` next to `torch.manual_seed(0)` in `pipeline/stages/separate.py`. The same 10 s now
  separate bit-identically, and two eval runs of the music fixture give the same CER.
- **Rule:** seed every random generator a library uses (Python, NumPy, torch), not only one, and prove it by running
  the same input twice (E18).

## E33 — On the GPU the model still held ~7 GB of RAM and took 77 s to load
- **Seen (2026-10-23):** with the CUDA build of PyTorch, Indic-Transcribe ran on the GPU (bf16, 2.45 GB of VRAM), but
  the process still held ~7.1 GB of committed memory, and loading took 77 s. Measured step by step: PyTorch and the
  CUDA context 1.4 GB; the model's own `from_pretrained` first builds all weights on the CPU (+3.2 GB); moving them to
  the GPU added another 2.5 GB of host memory. None of it was given back until the process ended.
- **Tried:** `disable_mmap=True` reads the whole 4.9 GB file into RAM first, which raised `MemoryError`.
- **Fix:** with `accelerate` (+ `psutil`; 650 KB, hash-locked), transformers loads the weights straight from the file
  onto the GPU (`device_map="cuda"`, `_build()` in `pipeline/engines/indic_transcribe_engine.py`). The process holds
  ~2.7 GB, loading takes 14–25 s, and the text is identical. `[runtime.memory_gb] indic_transcribe_gpu = 3.0`.
- **Rule:** on Windows, measure the free commit (RAM + page file) at every step, not just the VRAM; a load path can
  leave a full CPU copy behind.

## E34 — Every run spent ~45 s on Whisper language ID and loaded the speech model twice
- **Seen (2026-10-23, after the GPU switch):** the 56 s Pappa call took 140 s.
  - The probe's Whisper language ID took 47 s: load 14 s, 6 windows 4 s, and 28 s of forced Marathi/Hindi decoding.
    That decoding exists only because Whisper cannot tell the two languages apart (E10).
  - Recognition then loaded Indic-Transcribe again (27 s).
- **Fix:**
  - Indic-Transcribe identifies the language (`[languages] lid_engines`). Engines marked `lid_trusted` skip the forced
    decoding (`acoustic_decision()` in `pipeline/lid.py`).
  - `pipeline/engines/cache.py` keeps that model loaded into recognition. One kept model at a time; it is released
    by a request for any other engine, the end of recognition, the end of the run, or a Demucs separation short of
    VRAM.
  - The probe's music check now runs *before* language ID: the held model leaves only ~0.85–1 GB of the 4.29 GB free.
- **First attempt:** the music check ran after language ID, its VRAM guard released the model, and recognition
  loaded it again. No gain (131 s).
- **Result:** Pappa 140 → 111 s (RTF 2.54 → 1.98 in a clean run), same CER. Its verdict became PASS: the REVIEW
  had come only from Whisper's weak Marathi/Hindi evidence. Full eval: every CER identical; RTF Pappa 2.54 → 2.10,
  FLEURS mr 1.93 → 1.75, hi 1.99 → 1.92, music 1.58 → 1.55, AMI 1.95 → 1.81.
- **Also measured:** deterministic cuDNN costs nothing here (the same 12 pieces took 52–54 s either way, with
  identical output hashes across processes). An earlier "9× slower" reading came from a test that shared the GPU with
  a running eval.
- **Check:** `tests/test_core.py::TestEngineCache`; the decision log says "reused indic_transcribe_core".
- **Rule:** time every stage before optimising; load a model two stages need once; order the GPU users so that the
  held model and the next one fit together; never measure speed while another job uses the GPU.

## E35 — The review list named every speaker "SPK?"
- **Seen (2026-10-23, the owner's 15 test recordings):** `review_list.md` showed "SPK?" for every item, while the
  transcript had SPK1/SPK2. Polish created the review items in its flag loop, before it gave each line the speaker
  from diarization, so the items kept the track's placeholder label.
- **Fix:** after the speakers are assigned, polish copies each line's speaker into its review item. Only the review
  list was affected; transcripts and verdicts were right. Runs made before the fix keep their lists (runs are evidence).
- **Rule:** anything built from a line must be built, or refreshed, after the line is complete.

## E36 — The privacy test failed on common phone phrases (and a commit went out anyway)
- **Seen (2026-10-23):** with the owner's 15 calls on this PC, `trx test` reported three lines in `prompts/` as words of
  private recordings: a greeting and a filler phrase in the example prompts, written months before these calls, that
  also start many real calls. The maintainer's command chained `trx test | grep … && git commit`, so the commit and push
  went ahead on grep's success although the test had failed. Nothing private was pushed.
- **Fix:** a line counts only when it holds at least two private word triples (`MIN_HITS`); a copied private line
  holds several, and short fragments are still caught by the rare-word check.
- **Rule:** gate commits on the test's own exit status, never on a pipe's; a leak check must ignore what every
  conversation says.

## E37 — Long natural lines were flagged "repetitive", backchannels as "loops"
- **Seen (2026-10-23, the owner's 15 test recordings):**
  - 13 lines were flagged "repetitive" and made 6 runs REVIEW and one FAIL (t12, 65 % flagged).
  - None of those lines repeats a phrase: at most 5 % of their word triples repeat.
  - The check compared the zlib compression ratio with a fixed 2.4, a Whisper-era limit. Human-written Marathi
    crosses it by length alone: 8 of 8 pieces of ~370 characters, 58 of 231 overall.
  - Two "loops" were whole turns of acknowledgement ("हं" four times; the standard spelling "हां" was missing from
    the list of real repeats).
- **Fix:**
  - "repetitive" now means ≥ 35 % of a line's word triples repeat. In human-written text (373 pieces, conversation
    included) that share is at most 18 %; a phrase said twice gives 50 %, decoder loops 60–85 %.
  - The compression ratio stays only as a backstop, with a length-aware limit, max(2.4, 2.25 + 0.0016 × characters)
    + 0.35, that no human-written line reached and the real loops of E30 exceed.
  - A whole turn of one listed backchannel word, at most 5 times, is no longer a loop; "हां" joined the list.
- **Result (what-if on the 15 recordings):** 8 PASS / 6 REVIEW / 1 FAIL → 14 PASS / 1 REVIEW. The three remaining
  loops (a question word, a verb and "OK", each three times) stay flagged until the owner's transcripts confirm them.
- **Check:** `tests/test_core.py::TestRepetitionFlags`.
- **Rule:** calibrate every flag on human-written text of the same kind and length before it may change a verdict; a
  flag that fires on normal speech teaches reviewers to ignore flags.

## E38 — Denoising made Indic-Transcribe worse on every scored file
- **Seen (2026-10-23):** with denoising off, every scored file improved, and runs got 15–30 % quicker:

  | Fixture | CER, denoised → not denoised |
  |---|---|
  | Pappa call | 0.374 → 0.361 |
  | FLEURS mr | 0.047 → 0.044 |
  | FLEURS hi | 0.021 → 0.021 |
  | Music fixture | 0.388 → 0.305 |
  | AMI meeting | 0.284 → 0.249 |
  | Mean | 0.223 → 0.196 |

  Recognising the original audio while VAD and diarization kept the denoised track helped less (mean 0.207). On the
  music and meeting fixtures, VAD on denoised audio loses soft speech (E22).
- **Why:** Indic-Transcribe was trained on noisy Indian speech. DeepFilterNet (18 dB limit) removes soft speech and
  leaves artefacts the model never heard, also after Demucs separation. Whisper had preferred denoised audio slightly
  (0.513 vs 0.518 in 0.6.0); it is only the fallback now.
- **Fix:** `[denoise] engine = "copy_through"` in every recipe. DeepFilterNet stays installed and selectable.
  `[recognise] audio = "before_denoise"` exists for a future engine but is not used.
- **Speaker accuracy** moved too (Pappa 0.975 → 0.966, AMI 0.916 → 0.887). It moved the same way in the variant whose
  diarization was identical (original-audio recognition: 0.968, 0.891), so the change comes from aligning different
  text against the truth, not from worse speaker separation.
- **Also tried:**
  - Beam search (4 beams) instead of greedy decoding: mean 0.195 vs 0.196, ±0.007 per fixture in both directions.
    Greedy decoding (the model's own default) was kept.
  - Wider turn margins (0.3 s instead of 0.1 s): AMI got worse (0.299). Rejected.
- **Rule:** re-measure every pre-processing step after an engine change (E27 again); a step that helped one engine
  can hurt the next.

## E39 — The owner's 20-minute call: a half transcript, and failures the CER could not show
- **Seen (2026-10-23):** the owner's 20:35 Marathi phone call (fixture `call_240509_20min`, private) and the owner's
  transcript. First things that went wrong in the eval, then what the full failure check found.
  - **The transcript stops at 12:00.** The first eval scored the 8.5 untranscribed minutes as inserted errors: CER
    0.925, WER 1.18. Per minute, up to minute 11 about 80 % of the output characters matched the truth, from minute
    12 almost none. With only the lines that start before 720 s: CER 0.232 (Pappa: 0.361), spk 0.986.
  - **"Missing turns" (20 of 163) was misleading.** 8 of them were merged: two lines of one speaker joined because
    the other speaker's short reply between them was lost. Their words are in the output.
  - **Spelling conventions are 14 % of the CER.** The owner's transcript is colloquial (contractions, colloquial
    negation, s for sh, no word-final anusvara); the output writes standard spelling, the owner's convention.
    Folding the conventions on both sides: 0.232 → 0.198. The other 86 % is content: substituted 376, missing 312
    and extra 436 characters (part of the "extra" is real speech the transcript skipped).
  - **The listener's short turns are the weakest part.** 9 of ~30 acknowledgements were lost, 5 short
    interjections went under the other speaker. Lines under 2 s: content CER 0.28, longer lines 0.18. Decoded on its
    own, a short turn comes back with words nobody said: English words, personal names, unrelated phrases (the
    Indic-Transcribe form of E23).
  - **Meaning changes cost few characters.** Of 60 negations in the truth, about 10 were lost or added (a future
    "will happen" became "will not happen"; a "don't" appeared from nowhere); a phrase with an amount became an
    unrelated, damaging phrase. Numbers: 1 real error in 35.
  - **Names are not stable within one call:** one surname came out in four forms; a surname nobody said appeared.
  - **The review flags miss the bad lines.** Of the 17 lines with > 40 % content errors, 1 carried a review flag.
    Indic-Transcribe returns no confidence. Language-model fluency is a weak substitute: flagging 42 % of the lines
    would catch 56 % of the bad ones (Spearman −0.29), so it was not made a flag.
  - **Fine:** 20 minutes at real-time factor ~0.7–0.8, no memory problem, no detected speech left without text, and
    the unscored 8.5 minutes look like the scored part (fluency, unknown words; read through as Marathi).
- **Why:** the scores only said how much was wrong. Every number above came from a one-off analysis script; the
  next change would have been judged on CER alone.
- **Fix:**
  - `scored_until_sec` in truth.json (`setup/make_fixture.py --scored-until`): only output lines that start before
    it are scored.
  - **Failure monitor in `trx eval`** (`pipeline/metrics.py`, `evals/README.md`): content CER, turn outcomes
    (correct / misheard / merged / lost / lost backchannel / wrong speaker), negations and numbers lost or added
    (`library/scoring/<lang>.toml`), short lines against long lines, bad lines that carry a review flag. Numbers
    only in results; the turn-by-turn evidence stays in `evals/compare/`.
  - Alignment rows in numpy and a bit-parallel distance: identical results (checked in `self_test`), the 20-minute
    alignment 24.7 s → 1.1 s.
  - **Diarization:** a speaker change across a pause of ≥ 0.2 s now costs 0.02 instead of 0.1 (phone recipe,
    `switch_penalty_pause`). On the call: wrong speaker 6 → 4, merged 8 → 6, spk 0.986 → 0.987, CER 0.2316 →
    0.2318 (noise). Pappa identical. On the meeting recipe it made AMI worse (CER 0.249 → 0.254, spk 0.887 →
    0.876, wrong speaker 6 → 8): four people interrupt each other without pauses. Phone recipe only.
  - Privacy: renaming the fixture (its first name held a place and surname from the call) showed that eval runs
    counted as private only under their fixture's current name; now every eval run is private unless its fixture is
    open-licensed. A written filler in code matched the call; words that `library/` holds on purpose no longer count
    as leaks.
- **Open (next steps, measured with the monitor):** a confidence score from the decoder, to flag the bad and invented
  lines (F15, F19); a check for near-identical rare words (names) within one file (F17); the owner's choice between
  colloquial and standard spelling for the output; the transcript of 12:00–20:35.
- **Rule:** read the `how:` line, not only the CER. A change can keep the CER and still lose replies or flip
  meanings; a truth that stops early must say where.
