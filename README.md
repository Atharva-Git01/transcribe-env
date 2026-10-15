# transcribe-env — local, offline transcription environment

A self-contained workshop that turns an audio (or video) file into a forensic-grade transcript using **local models
only**. It is not tied to any AI assistant: a person or any model can operate it by following `OPERATOR.md`, and the
environment itself makes every technical decision, check and record.

- **Zero cost** — no paid API, plan or service.
- **Offline at runtime** — an offline guard blocks every network call; `trx doctor --offline` proves it.
- **Autonomous runtime** — one command runs the whole pipeline; no AI steers a run.
- **Reviewed afterwards** — Claude (or any model) improves the environment's files only through the eval gate
  in `MAINTAINER.md`.
- **Forensic** — the original is never modified; read-only copy, sha256 custody chain, append-only decision log,
  nothing silently deleted.

## Three commands
```
trx doctor --offline            is it ready and provably offline?
trx transcribe "<file>"         one file -> runs/<case>-<time>/
trx explain <run-id>            what was done and why, in plain sentences
```
`trx` = `bin\trx.cmd` (cmd / PowerShell) or `bin/trx` (Git Bash). `trx help` lists everything.

## What a run produces (`runs/<run-id>/`)
| File | Meaning |
|---|---|
| `input/` | read-only copy of the original (hash-verified) |
| `passport.json` | measurements: layout (real stereo vs duplicated mono), loudness, clipping, noise, bandwidth |
| `plan.json` | recipe chosen (and the rule that chose it) + every parameter every stage will use |
| `decisions.json` | every action, in order, with the reason |
| `custody.json` | sha256 of the original and of every file the run produced |
| `out/` | transcript `.txt .srt .vtt .json` + `review_list.md` (from phase 1) |
| `report.md` / `report.json` | verdict PASS / REVIEW / FAIL with reasons |

## Layout
```
config/      defaults, thresholds, recipe-selection rules, exit codes (TOML)
recipes/     one file per audio situation, every parameter explicit
pipeline/    runtime code (deterministic, offline) + JSON Schemas
prompts/     Whisper initial prompts; Gemma 3 cleanup prompts (phase 3)
library/     vocab, doubled words, hotwords
references/  what the environment knows: parameters, VAD, orthography, forensic rules, failures, engines
knowledge/   ERRORS.md (read first), CHANGELOG.md
evals/       fixtures with human ground truth + baseline scores
inbox/       drop files here for `trx watch`
runs/        one folder (and git repo) per input — evidence, never edited
toolchain/   own venv, models (with SOURCE.json + sha256), binaries
setup/       the only internet-touching part, run once per tier after the owner's OK
```

## Phases
| Phase | Adds | Status |
|---|---|---|
| 0 | skeleton, probe, recipe selection, plan, custody, doctor, explain | **built** |
| 1 | venv; convert, denoise (DeepFilterNet3), VAD (Silero), language ID, faster-whisper medium, polish, gate, export | next (≈2 GB after OK) |
| 2 | large-v3 / turbo, IndicConformer, dual-engine verify + merge, eval picks defaults | ≈5–6 GB |
| 3 | Gemma 3 4B via llama.cpp cleanup (5 rejection gates) | ≈2.5 GB |
| 4 | Demucs separation, sherpa-onnx diarization, long files | ≈1 GB |
| 5 | weak-operator hardening | — |

Full plan: `forensic media suit/doc/TRANSCRIPTION_ENVIRONMENT_PLAN.md`.
