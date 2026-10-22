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
  to 0.513. Recovery stays on for engines trained on short utterances (Indic models).
- **Rule:** text produced from a few hundred milliseconds of audio is guilty until proven plausible.
