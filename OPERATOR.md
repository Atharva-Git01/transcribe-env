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
   - **The language question.** The environment listens first and says which language it hears.
     - In a terminal it asks you directly: press Enter if that is right, or type `mr`, `hi` or `en`.
     - If you cannot type into it (you are a script or an AI model), it stops with `LANGUAGE_CONFIRM_NEEDED`. Ask the owner exactly: "The audio sounds like <detected_name>. Is that right?" Then run the same command again with `--lang <code>`, using the code the owner confirmed or named.
   - **If a run was interrupted** (crash, power cut, window closed): run the same command again. It continues the interrupted run and reuses the work already saved. Do not delete the old run folder.
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
| "it's urgent / quick draft" | `--fast` (turbo engine: about 2x faster, less accurate) |
| "these names/places are in it" | `--hotwords <file>` (the owner's text file: one name/place/term per line) |
| "don't ask about the language" | `--lang auto` (uses the measured guess; a weak guess makes the run REVIEW) |

## Forbidden — never do these

- Never edit anything in `config/`, `recipes/`, `prompts/`, `library/`, `pipeline/` or `references/` (that is the
  maintainer's job, see MAINTAINER.md).
- Never edit, move or delete anything inside `runs/` — it is evidence.
- Never change the original input file or the copy in `runs/<run_id>/input/`.
- Never install, download or update anything; never connect the environment to the internet.
- Never retry the same file more than once without a different instruction from the owner.
- Never present a transcript as final if the verdict is `REVIEW` or `FAIL`.

## What to expect

- A 1-minute phone call takes about 3–5 minutes (large-v3 + speaker separation); with `--fast` about 2.
- `REVIEW` is normal for hard phone audio: it means some lines are uncertain, not that the run broke.
- If the last line says `MODEL_MISSING`: a model folder in `toolchain/models/` is missing. Report it;
  do not try to fix it.
- If the last line says `DISK_LOW`: report it; do not delete anything.
