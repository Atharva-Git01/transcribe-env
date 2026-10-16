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
Maintenance: `trx eval`, `trx test`, `trx doctor --deep` (re-hashes every model), `trx probe <file>`, `trx watch`.
`trx` = `bin\trx.cmd` (cmd / PowerShell) or `bin/trx` (Git Bash). `trx help` lists everything.

## What a run produces (`runs/<run-id>/`)
| File | Meaning |
|---|---|
| `input/` | read-only copy of the original (hash-verified) |
| `passport.json` | measurements: layout (real stereo vs duplicated mono), loudness, clipping, noise, bandwidth |
| `plan.json` | recipe chosen (and the rule that chose it) + every parameter every stage will use |
| `decisions.json` | every action, in order, with the reason |
| `custody.json` | sha256 of the original and of every file the run produced |
| `out/` | transcript `.txt .srt .vtt .json` + `review_list.md` (+ `readable` variant with `--style readable`) |
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
| 1 | own venv (hash-locked); convert, denoise (DeepFilterNet3 → ffmpeg afftdn), VAD (Silero), 2-step language ID, faster-whisper medium on the GPU (CUDA 12), polish flags, gate, export | **built** |
| 2 | large-v3-turbo + large-v3; dual-engine verify stage; `trx eval` scoring + baseline | **built** — IndicConformer not installed (QUESTIONS.md Q1) |
| 3 | Gemma 3 4B (Q4_K_M) via llama.cpp `llama-server` (Vulkan), 5 rejection gates | **built, off by default** (E17) |
| 4 | sherpa-onnx speaker diarization | **built** — Demucs separation and resumable long files not built (QUESTIONS.md) |
| 5 | weak-operator hardening: one JSON status line + exit code per command, `trx test`, dead-config check | **built** |

Status, numbers and open questions: `MORNING_REPORT.md`, `QUESTIONS.md`, `knowledge/CHANGELOG.md`.
Full plan: `forensic media suit/doc/TRANSCRIPTION_ENVIRONMENT_PLAN.md`.

## Hardware notes (this machine)
- RTX 3050 Laptop 4 GB: one model on the GPU at a time (load → run → unload). `config/defaults.toml [runtime] device`.
- All models are on C: in `toolchain/models/` (a backup copy of large-v3 and Gemma 3 is on the D: USB stick).
