# evals/

`trx eval` scores the pipeline on every fixture and compares with `baseline.json`. It is the gate for every change
(MAINTAINER.md).

## Fixture
`fixtures/<id>/truth.json` (schema `trx.truth/1`) + the audio file it names:
```
{ "schema": "trx.truth/1", "id": "...", "audio": "audio.m4a", "audio_sha256": "...", "language": "mr",
  "speakers": [...], "reference_text": "...", "segments": [{"speaker": "...", "text": "..."}],
  "source": "...", "notes": "..." }
```
Ground truth must be human-verified. Best source: transcripts the owner corrects after real runs.

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

## Where results go
- `results/eval_<time>_<engine>.json` + `.md`: numbers only (speakers as A, B, …). Pushed to GitHub.
- `compare/<same name>.md`: truth vs output per turn. It holds the words of the recordings, so it is git-ignored
  and stays on this PC (E29).
- `runs/`: the eval runs themselves (git-ignored; old ones may be deleted, `trx eval` rebuilds them).

## Accepting a change
- The target fixture/recipe must improve.
- No other fixture may get worse by more than **0.01 CER absolute** (noise margin; revisit once there are ≥ 10 fixtures).

## Fixtures now (all on this PC only)
| id | language | length | notes | 0.8.0 baseline (Indic-Transcribe on the GPU) |
|---|---|---|---|---|
| pappa_260603_211622 | mr | 56.6 s | hard real phone call (private); 2 speakers, 26 turns | CER 0.374 · spk 0.975 |
| fleurs_mr_test10 | mr | 127.7 s | FLEURS test (CC-BY-4.0), 10 read sentences | CER 0.047 |
| fleurs_hi_test10 | hi | 122.6 s | FLEURS test (CC-BY-4.0), 10 read sentences | CER 0.021 |
| fleurs_mr_test10_music5db | mr | 127.7 s | fleurs_mr_test10 + public-domain music 5 dB below the speech | CER 0.388 |
| ami_es2004a_450s | en | 180 s | AMI meeting ES2004a, 450–630 s (CC-BY-4.0); 4 speakers, 57 turns | CER 0.284 · spk 0.916 |

Wanted: 5–10 short labelled real recordings from the owner (`setup/make_fixture.py`).
