# evals/

`trx eval` scores the pipeline on every fixture and compares with `baseline.json`. It is the gate for every change
(MAINTAINER.md).

## Fixture
`fixtures/<id>/truth.json` (schema `trx.truth/1`) + the audio file it names:
```
{ "schema": "trx.truth/1", "id": "...", "audio": "audio.m4a", "audio_sha256": "...", "language": "mr",
  "speakers": [...], "reference_text": "...", "segments": [{"speaker": "...", "text": "..."}],
  "source": "...", "notes": "...", "scored_until_sec": 720.0, "spelling": "as_spoken" }
```
Ground truth must be human-verified. Best source: transcripts the owner corrects after real runs.
`scored_until_sec` (optional): the truth covers only the start of the recording; only output lines that start before
it are scored, the run itself still processes the whole file (E39). `setup/make_fixture.py --scored-until <s>`.
`spelling` (optional, default `standard`): `as_spoken` = the transcript writes words as spoken (colloquial forms;
the owner's way since 2026-10-23). Such a fixture is judged by the **content CER** in the regression check, so a
change of spelling style in the output is neither a regression nor a gain. Both owner fixtures are `as_spoken`.

## Metrics (pipeline/metrics.py)
- **CER** (primary for Marathi/Hindi) and WER, with one fixed normalisation (NFC, lowercase, punctuation and tags
  removed).
- **Normalised CER** also ignores nukta, treats chandrabindu as anusvara, unifies the candra vowel signs and reads
  digits as number words. **Loose CER** also ignores i/u vowel length. **std** scores against the standard-spelling
  version of the truth, where one exists.
- **spk:** the share of characters on the right speaker, for truths with ≥ 2 speakers.
- **missing turns:** truth turns without text of their own. Also reported: Hindi forms in Marathi output, and the
  real-time factor.
- Diarization error rate (DER) is not scored: no fixture has timed speaker turns.
- Every decode is seeded (E18), so the same configuration must reproduce the same numbers exactly.

### Failure monitor: HOW it went wrong (E39; the `how:` line under each fixture, and in results/compare)
- **content CER:** CER after folding spelling conventions (colloquial vs standard Marathi forms, word-final
  anusvara, s/sh, n/retroflex n, "-ye" contractions), the spoken forms of `library/spoken_variants/<lang>.toml`
  (0.9.0) and hesitation fillers on BOTH sides. What is left is misheard, missing or extra words. On the owner's
  20-minute call the conventions were 14 % of the CER.
- **spoken forms** (truths written as spoken, 0.9.0, `pipeline/spelling.py`): of the truth words in a listed spoken
  form, how many the output wrote that way too (**spoken-form recall**), and of the truth words in a standard form
  that has a listed spoken variant, how many the output switched to a spoken form (**wrong switches**). Words the
  output does not have at all are left to the content CER. The gate: recall may not drop by more than 0.05 against
  the baseline, and at most 5 % of the standard words may be switched (PLAN_v0.9). Word by word, in
  `compare/` only.
- **turn outcomes:** each truth turn is `correct` (≥ 85 % of its characters), `misheard`, `lost`,
  `lost_backchannel` (an acknowledgement that vanished), `merged` (its words are in a line of a neighbouring turn of
  the same speaker: the reply between them was lost — the old "missing turns" counted this as missing),
  `merged_other` / `wrong_speaker` (its words are under the other speaker). The compare file has it per turn.
- **negations / numbers:** a negation or number word lost, changed or added — a fluent line whose meaning flipped.
  Word lists: `library/scoring/<lang>.toml`. A year written in digits in the truth is read as a plain number
  ("one thousand nine hundred …") while speakers say the year form: FLEURS mr counts 2 such words.
- **lines < 2 s vs longer:** content CER of short output lines (mostly the listener's replies) against the rest.
- **bad lines flagged:** of the output lines with > 40 % content errors, how many carry a review flag (anything but
  the info flags). This is the share of the worst lines a reviewer is pointed to.

## Where results go
- `results/eval_<time>_<engine>.json` + `.md`: numbers only (speakers as A, B, …). Pushed to GitHub.
- `compare/<same name>.md`: truth vs output per turn. It holds the words of the recordings, so it is git-ignored
  and stays on this PC (E29).
- `runs/`: the eval runs themselves (git-ignored; old ones may be deleted, `trx eval` rebuilds them).
- `test_audio/`: the owner's test recordings (15 WhatsApp voice notes, 15.3 min, received 2026-10-23;
  `manifest.json` = id, original name, sha256, length). Git-ignored. `drafts/` holds the environment's transcript
  of each (`python setup/make_drafts.py --manifest evals/test_audio/manifest.json`); once the owner has corrected a
  draft, `setup/make_fixture.py` turns it into a fixture that `trx eval` scores.

## Screening harness (PLAN_v0.9, `trx screen`, `pipeline/screen.py`)
Every candidate model is decoded on the same fixed lines, one model at a time on the GPU, in minutes:
- `trx screen build` cuts the lines once into `screening/v1/` (git-ignored: private audio and words): the scored
  part of every fixture written as spoken (the 20-minute call's 12 minutes and the Pappa call: 193 lines, 13.3 min),
  cut where the baseline transcript put its lines. The set is frozen.
- `trx screen run --candidate <id>` (candidates: `screen_candidates.toml`) scores four challenge sets: **S** spelling
  (50 spoken slots: words the owner wrote as spoken where the baseline wrote them standard; 241 standard slots:
  words both wrote the same way that have a known spoken variant) → recall and switches; **R** lines under 2 s →
  content CER and words not in the truth; **N** lines with negations or numbers → their errors; **G** the rest →
  content and strict CER. Also the load time, real-time factor and peak GPU memory.
- `trx screen report` writes `results/screen_<set>_summary.md` (numbers only); per line truth vs output:
  `compare/screen_<set>_<candidate>.md` (this PC only).

## Accepting a change
- The target fixture/recipe must improve.
- No other fixture may get worse by more than **0.01 CER absolute** (noise margin; revisit once there are ≥ 10 fixtures).

## Fixtures now (all on this PC only)
| id | language | length | notes | 0.8.1 baseline (Indic-Transcribe on the GPU, no denoising) |
|---|---|---|---|---|
| pappa_260603_211622 | mr | 56.6 s | hard real phone call (private); 2 speakers, 26 turns | CER 0.361 · spk 0.966 |
| fleurs_mr_test10 | mr | 127.7 s | FLEURS test (CC-BY-4.0), 10 read sentences | CER 0.044 |
| fleurs_hi_test10 | hi | 122.6 s | FLEURS test (CC-BY-4.0), 10 read sentences | CER 0.021 |
| fleurs_mr_test10_music5db | mr | 127.7 s | fleurs_mr_test10 + public-domain music 5 dB below the speech | CER 0.305 |
| ami_es2004a_450s | en | 180 s | AMI meeting ES2004a, 450–630 s (CC-BY-4.0); 4 speakers, 57 turns | CER 0.249 · spk 0.887 |
| call_240509_20min | mr | 1234.7 s | owner's real phone call (private), 2 speakers; the owner's transcript covers 0:00–12:00 (163 turns, colloquial spelling) → `scored_until_sec` 720 | CER 0.232 · content 0.192 · spk 0.986 |

Wanted: 5–10 short labelled real recordings from the owner (`setup/make_fixture.py`).
