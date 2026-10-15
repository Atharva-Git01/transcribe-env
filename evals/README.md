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
CER (primary for Marathi/Hindi), WER; one fixed normalisation (NFC, lowercase, punctuation and tags removed).
Diarization error rate (DER) is not scored yet: no fixture has timed speaker turns.
Every decode is seeded (E18), so the same configuration must reproduce the same numbers exactly.

## Accepting a change
- The target fixture/recipe must improve.
- No other fixture may get worse by more than **0.01 CER absolute** (noise margin; revisit once there are ≥ 10 fixtures).

## Fixtures now
| id | language | length | notes |
|---|---|---|---|
| pappa_260603_211622 | mr | 56.6 s | hard real phone call; 2 speakers; baseline large-v3 CER 0.531 (0.5.0); FMS 0.745 re-scored with trx metrics (it reported 0.7225–0.736 with its own) |

Wanted: 5–10 more short labelled real clips (owner), plus small free public Marathi/Hindi sets later.
