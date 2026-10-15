# Morning report — transcribe-env 0.5.0 (overnight 2026-10-16)

## In one minute
- **All phases are built** (0–5) except three owner decisions: IndicConformer, Demucs and resumable long files (QUESTIONS.md).
- `trx doctor --offline` → **READY** (64 checks, 0 required failures). `trx test` → **16/16 pass**. Everything runs offline on the RTX 3050.
- **Pappa call** (the only labelled fixture): **CER 0.531** with large-v3. Re-scored with the same metric, FMS gets **0.745** (it reported 0.7225 under its own normalisation), so this is about **29 % fewer character errors**. Verdict **REVIEW**: no hard flags; 5 lines carry Hindi word forms, and the Marathi/Hindi decision was weak.
- Same input → same output: every random step is now seeded (two runs gave identical numbers).
- Your answers are needed in **QUESTIONS.md** (13 short questions, each with the default I used).

## How to use it
```
cd C:\Users\asr26\transcribe-env
bin\trx doctor --offline           ready + provably offline?
bin\trx transcribe "<file>"        one file -> runs\<case>-<time>\  (transcript in out\, verdict in report.md)
bin\trx explain <run-id>           what happened and why, in plain sentences
```
Maintenance: `trx eval` (score vs baseline), `trx test`, `trx doctor --deep` (re-hash all models).
A 1-minute call takes ~4 min with large-v3: about 2–3 min of that is loading the model from the D: USB stick. With turbo it takes ~1.5 min.

## What runs, in order (all local)
probe (measure) → select recipe (8 recipes, rule table) → plan.json → convert → denoise (DeepFilterNet3) → VAD
(Silero) → language ID (2-step for Marathi/Hindi) → [diarize] → recognise (faster-whisper large-v3, GPU) →
[verify: second engine, off] → polish (flags, never deletes) → [Gemma 3 cleanup, off] → gate (PASS/REVIEW/FAIL) →
export (txt/srt/vtt/json + review_list.md). Every step goes to decisions.json, and every file is hashed in custody.json.

## Eval results (fixture: Pappa call, 56.6 s, Marathi phone audio, human truth)
| Engine | Segmentation | Seeded | CER | WER | Verdict | Time |
|---|---|---|---|---|---|---|
| **large-v3 (default)** | vad_concat + split | yes | **0.531** | 0.844 | **REVIEW** | 250 s |
| large-v3 | clips | yes | 0.577 | 0.906 | REVIEW | 193 s |
| turbo (fallback) | vad_concat + split | yes | 0.793 | 0.961 | FAIL | 88 s |
| turbo | clips | yes | 0.813 | 1.008 | REVIEW | 91 s |
| medium | clips | no | 1.524 | 1.758 | FAIL (Bengali-letter loops) | 413 s |
| FMS (large-v3), re-scored with trx metrics, one copy | — | — | 0.745 | 0.914 | — | — |

Earlier unseeded numbers (turbo 0.720, large-v3 0.639) could not be reproduced (E18), so they were not used. The
"clips" and medium verdicts were measured before the per-segment repetition fix (E14), so only their CER is comparable.

## Defaults I chose (all reversible in config/defaults.toml)
- Engine: **large-v3 first, turbo when D: is unplugged** for mr/hi/mixed/unknown; turbo first for English (no English fixture yet).
- Segmentation `vad_concat` + splitting long segments at word boundaries (the text is never changed).
- Cross-check (second engine): **off**. Result of the smoke test: see "Verify smoke test" below.
- Gemma 3 cleanup: **off** (it once changed a word's meaning while passing all its gates, E17).
- Gate: soft flags (Hindi word forms in Marathi, speaker change, crosstalk) give REVIEW only. Hard flags (loops,
  low confidence, wrong script, prompt copy, known hallucinations) on more than 40 % of the time give FAIL.

## Honest limits (please read)
1. **One fixture only.** Every choice above was measured on a single 56-second call. It could be fitting that call, not your audio in general. Three to five more labelled clips (Q10) would turn these into real decisions.
2. **The language decision on this call is fragile.** Whisper's own language ID says "Hindi" for this Marathi call (E10). The text evidence also leaned Hindi (8 Hindi words vs 5 Marathi words). Marathi was chosen only because weak evidence falls back to your priority language, and the run is marked REVIEW. Possible fix: count Marathi words that survive Hindi decoding more heavily, since Whisper drifts toward Hindi. I did not apply it: it needs a Hindi fixture to prove it doesn't break Hindi audio.
3. **CER 0.53 is still a draft.** About half the characters differ from the human transcript. Some of that is spelling (Hindi forms such as है/मैं where Marathi आहे/मी was spoken). The output is a reviewed draft, not final evidence text, on audio this hard.
4. **Speakers on mono phone calls are not separated** (shown as `SPK?`). Diarization found ~1 of the 2 speakers on narrowband audio. Stereo recordings (one side per channel) are labelled reliably.
5. **D: USB drive dependency.** Without it, runs fall back to turbo (CER 0.79, often FAIL) and Gemma is unavailable.
6. **Not built:** IndicConformer (gated; needs your HF login and torch), Demucs (needs torch), chunked/resumable files over 1 h (a long file still runs, in one pass).

## Things that broke tonight and are now prevented (knowledge/ERRORS.md)
- E10 Whisper's language ID can't see Marathi.
- E11 Loops inflated marker counts.
- E12 PyAV 19 broke faster-whisper's decoder.
- E13 A half-downloaded model looked installed.
- E14 Long segments made one loop flag 30 s at a time.
- E15 Disk ran low (pagefile growth).
- E16 Character loops in Bengali letters were not flagged.
- E17 Gemma changed a meaning.
- E18 Runs were not reproducible.

## Verify smoke test (cross-check on, turbo checking large-v3)
- **The mechanism works.** It added only 20 s, and every segment received an agreement score plus the other engine's text as an alternative.
- **But turbo looped** ("हुआ हुआ …") on this call, so 22 of 25 segments came out as "disagree" and the verdict was FAIL. On audio this hard, turbo is too weak to check large-v3, so cross-check stays **off**. It becomes useful once a second strong engine exists (IndicConformer, Q1).

## Operator test
`bin\trx transcribe "evals\fixtures\pappa_260603_211622\audio.m4a" --case smoke`
- It ended with `"name": "OK", "verdict": "REVIEW"`.
- Output is in `runs\smoke-20261016-051619\`; open its `report.md` and `out\review_list.md` to see what a delivery looks like.

## Where things are
- Environment: `C:\Users\asr26\transcribe-env`, with its own git history (`git log`). Large models: `D:\transcribe-env-models` (junctions).
- Changes: `knowledge/CHANGELOG.md` (0.5.0). Questions: `QUESTIONS.md`. Plan: `forensic media suit/doc/TRANSCRIPTION_ENVIRONMENT_PLAN.md`.
- Eval history: `evals/results/*.json`; baseline: `evals/baseline.json`; logs: `setup/logs/`.
