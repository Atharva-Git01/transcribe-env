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
- **Fix (phase 1):** add an in-speech estimate (WADA-SNR on VAD speech regions) and use it for rule R4.
- **Check:** passport `noise.method` names the method; compare with the phase 1 value on the same file.
- **Rule:** until then, rule R4 (noisy) can under-fire on phone audio; the phone recipe denoises regardless.

## E08 — A tool reached the internet on its own
- **Seen (2026-10-16):** running `ollama list` started the Ollama app, which downloaded an update by itself.
- **Fix:** Gemma 3 runs through llama.cpp (no service, no updater); runtime offline guard; `OLLAMA_NO_UPDATE` set.
- **Check:** `trx doctor --offline`.
- **Rule:** nothing in the runtime may have a background updater or a first-use download.
