# OPERATOR.md — how to run this environment (for a person or any AI model)

You do not need to make technical decisions. The environment chooses the recipe, the parameters and the checks.
Your job: run the command, read the final JSON line, do what its `next` field says, report the result.

## Commands (use exactly these)

| Goal | Command |
|---|---|
| Is the environment ready? | `trx doctor --offline` |
| Transcribe one file | `trx transcribe "<path to file>"` |
| Transcribe everything in inbox/ | `trx watch --once` |
| Explain a finished run | `trx explain <run-id>` |

`trx` = `bin\trx.cmd` (Windows cmd/PowerShell) or `bin/trx` (Git Bash), run from this folder.
Quote paths that contain spaces.

## Checklist for every file

1. Run `trx doctor --offline`. The last line must contain `"name": "OK"`. If not, stop and report the FAIL lines.
2. Run `trx transcribe "<file>"`. Do not add options unless the owner asked for them.
3. Read the LAST line of the output (one JSON object). Find `name`.
4. Look up `name` in `config/exit_codes.toml` (the line also contains `meaning` and `next`). Do what `next` says.
5. If `name` is `OK`: open `runs/<run_id>/report.md` and `runs/<run_id>/out/review_list.md`.
   - Verdict `PASS`: deliver `runs/<run_id>/out/`.
   - Verdict `REVIEW`: deliver `out/` AND tell the owner which lines are in `review_list.md` (timestamps included).
   - Verdict `FAIL`: deliver nothing; report the reasons from `report.md`.
6. Run `trx explain <run_id>` and include its text in your report. Do not describe the run in your own words.

## Allowed options (only when the owner asks)

| Owner says | Add |
|---|---|
| "use recipe X" | `--recipe X` (names = files in recipes/ without .toml) |
| "readable transcript" | `--style readable` (default is forensic verbatim) |
| "it is Marathi / Hindi / English" | `--lang mr` / `--lang hi` / `--lang en` |
| "call the case Y" | `--case Y` |

## Forbidden — never do these

- Never edit anything in `config/`, `recipes/`, `prompts/`, `library/`, `pipeline/` or `references/` (that is the
  maintainer's job, see MAINTAINER.md).
- Never edit, move or delete anything inside `runs/` — it is evidence.
- Never change the original input file or the copy in `runs/<run_id>/input/`.
- Never install, download or update anything; never connect the environment to the internet.
- Never retry the same file more than once without a different instruction from the owner.
- Never present a transcript as final if the verdict is `REVIEW` or `FAIL`.

## Phase status

Phase 0 (current): `transcribe` creates the run, copies + hashes the input, measures it, picks a recipe and writes the
plan, then stops with `STAGE_NOT_INSTALLED` because the transcription stages arrive in phase 1. This is expected.
