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
evals/       fixtures with human ground truth (this PC only) + baseline; results/ = numbers only (pushed),
             compare/ = truth vs output per turn (this PC only, git-ignored)
inbox/       drop files here for `trx watch`
runs/        one folder (and git repo) per input — evidence, never edited
toolchain/   own venv, models (with SOURCE.json + sha256), binaries
setup/       the only internet-touching part, run once per tier after the owner's OK
```

## Phases
| Phase | Adds | Status |
|---|---|---|
| 0 | skeleton, probe, recipe selection, plan, custody, doctor, explain | **built** |
| 1 | own venv (hash-locked); convert, denoise (DeepFilterNet3 → ffmpeg afftdn), VAD (Silero), 2-step language ID, faster-whisper on the GPU (CUDA 12), polish flags, gate, export | **built** |
| 2 | large-v3 + turbo (`--fast`); verify stage (cross-check; off: with Whisper as the second engine it flagged 96 % of the phone call and 60 % of clean read Marathi in 0.7.0); `trx eval` scoring + baseline | **built** |
| 3 | Gemma 3 4B via llama.cpp `llama-server` (Vulkan) — suggestions filtered by a word gate + language model | **built, ON for Whisper output only** (E17, E26) |
| 4 | diarization on every file: by channel (true stereo) or by voice (ERes2NetV2 embeddings, E19); lines split at speaker changes; Demucs music/TV separation (`noisy_field_audio`, E27) | **built** |
| 5 | weak-operator hardening: one JSON status line + exit code per command, `trx test`, dead-config check, language confirmation, resume of interrupted runs, 10-min chunks for long files | **built** |
| 6 | Marathi + Hindi word language models and lexicons (IndicCorp v2, CC-0); `--hotwords`; eval fixtures from FLEURS (CC-BY-4.0) | **built** |
| 7 | Indic engines (0.7.0): **Indic-Transcribe-core** primary for every language, IndicConformer-600M fallback for mr/hi; each speaker turn decoded on its own; numbers as words; normalised CER, missing turns and a per-turn report in `trx eval`; fixtures FLEURS + music and an AMI 4-person meeting | **built** (E24–E28) |

Status, numbers and open questions: `knowledge/CHANGELOG.md`, `QUESTIONS.md` (`MORNING_REPORT.md` is the 0.5.0 report).
Plans: `PLAN_v0.7.md`; overall: `forensic media suit/doc/TRANSCRIPTION_ENVIRONMENT_PLAN.md`.

## Hardware notes (this machine)
- RTX 3050 Laptop 4 GB: one model on the GPU at a time (load → run → unload). `config/defaults.toml [runtime] device`.
  The GPU runs Whisper (language ID, `--fast`) and Gemma; the Indic engines run on the CPU (CPU build of PyTorch).
  A 1-minute phone call takes about 3 minutes. Indic-Transcribe needs about 5.5 GB of free memory (15.6 GB of RAM
  here): with a big browser session open, a run may stop with `MEMORY_LOW` (E31).
- The models every run needs are on C: (`toolchain/models/`). The fallbacks (IndicConformer, Whisper large-v3 and
  medium) and Gemma 3 live on the D: USB stick, linked by directory junctions (`D:\transcribe-env-models\README.txt`).
  With the stick unplugged, normal runs are unaffected, `trx doctor` stays READY and `--fast` runs without the Gemma
  cleanup.

## Licences and credit
- Built with **Indic-Transcribe-core from Bodhan AI / AI4Bharat** (Indic Open Model License v1.0). Free to self-host,
  including commercial use. Offering it as a service that other people or companies call needs Bodhan AI's written
  approval first; automated decisions affecting someone's legal rights need human oversight (the REVIEW rule).
- IndicConformer-600M (AI4Bharat), Whisper, Demucs: MIT. Gemma 3: Gemma Terms of Use. The licence, source,
  revision and sha256 of every model is in `toolchain/models/<id>/SOURCE.json` and `manifest.toml`.
