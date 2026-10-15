# MAINTAINER.md — how the environment is improved (Claude or any model, never during a run)

The runtime is autonomous and offline. Improvements happen afterwards, only in environment files, and only when the
evaluation proves they help. Nothing here is ever done while a run is in progress.

## Inputs you review
- `runs/<run_id>/report.json`, `report.md`, `out/review_list.md`, `decisions.json`, `passport.json`
- a corrected transcript from the owner (best input: it becomes a new eval fixture)
- `knowledge/ERRORS.md` (read first, every time)

## What you may change
| File | Typical change |
|---|---|
| `config/selection_rules.toml` | add/reorder a rule, adjust a condition value |
| `config/thresholds.toml` | tune a threshold |
| `config/defaults.toml` | engine per language (only from eval winners), output defaults |
| `recipes/*.toml` | stage params; bump `[recipe].version` on every change |
| `prompts/**` | initial prompts, Gemma 3 cleanup prompts |
| `library/**` | vocab, doubled words, hotwords |
| `references/**`, `knowledge/**` | documentation, ERRORS entries |
| `pipeline/**` | code — only with the same eval gate plus `trx doctor` passing |

## The gate (no exceptions)
1. Make the change on a branch: `git switch -c tune/<short-name>`.
2. Run `trx doctor --offline` → must be OK.
3. Run `trx eval` → compare with `evals/baseline.json`.
   - Accept only if the target score improves AND no recipe × engine × fixture gets worse beyond the noise margin
     in `evals/README.md`.
4. Accepted: merge, update `evals/baseline.json`, add a `knowledge/CHANGELOG.md` entry (what, why, before → after),
   bump `VERSION` (patch for tuning, minor for new stages/engines).
5. Rejected: delete the branch. Record what was tried in `knowledge/ERRORS.md` if it taught something.

## Adding a new ASR engine
1. Add it to `manifest.toml` (licence must allow commercial use; size; source).
2. Owner approves the download; `setup/` installs it into `toolchain/models/<id>/` with sha256 in `SOURCE.json`.
3. Write `pipeline/engines/<id>_engine.py` implementing the interface in `pipeline/engines/base.py`.
4. `trx eval --engine <id>`: it becomes a default for a language only if it beats the current one there.

## Never
- Never change anything during a run or inside `runs/`.
- Never add a component that needs the internet at runtime, a background service, or an auto-updater.
- Never accept a change without an eval result (before phase 1 scoring exists: doctor + schema checks only).
