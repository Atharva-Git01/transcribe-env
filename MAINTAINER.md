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
| `recipes/*.toml` | stage params; bump `[recipe].version` on every change; a NEW key also needs code that reads it + an entry in `pipeline/recipes.py STAGE_KEYS` |
| `prompts/**` | initial prompts, Gemma 3 cleanup prompts |
| `library/**` | vocab, doubled words, hotwords |
| `references/**`, `knowledge/**` | documentation, ERRORS entries |
| `pipeline/**` | code — only with the same eval gate plus `trx doctor` passing |

## The gate (no exceptions)
1. Make the change on a branch: `git switch -c tune/<short-name>`.
2. Run `trx doctor --offline` and `trx test` → both must be OK (the test suite also rejects recipe keys no
   stage reads — see `pipeline/recipes.py STAGE_KEYS` — and any file git would push that holds words of a
   private recording, E29).
3. Run `trx eval` → compare with `evals/baseline.json`.
   - Accept only if the target score improves AND no recipe × engine × fixture gets worse beyond the noise margin
     in `evals/README.md`.
4. Accepted: merge, update `evals/baseline.json`, add a `knowledge/CHANGELOG.md` entry (what, why, before → after),
   bump `VERSION` (patch for tuning, minor for new stages/engines).
5. Rejected: delete the branch. Record what was tried in `knowledge/ERRORS.md` if it taught something.

## Fixtures and scores
- Every fixture is `evals/fixtures/<id>/audio.* + truth.json` (`trx.truth/1`). Audio and truth stay on this PC
  (git-ignored). Public test sets are rebuilt by `setup/setup.py install --phase 6` (see `setup/sources.toml`).
- `trx eval` reports strict / normalised / loose CER, WER, `spk` (speaker accuracy, truth with ≥ 2 speakers),
  missing truth turns, Hindi forms in Marathi output and the real-time factor.
- `evals/results/` is pushed to GitHub and holds numbers only (speaker names become A, B, …). The truth-vs-output
  table per turn is written to `evals/compare/`, which never leaves this PC. Never copy a private fixture's words
  into any tracked file, including test strings and ERRORS examples (E29).
- Current fixtures: the Pappa call (hard Marathi phone call, 2 speakers), FLEURS Marathi and Hindi (10 clean read
  sentences each), FLEURS Marathi + music 5 dB below the speech, and an AMI 4-person English meeting (3 min).
  A change must not make any of them worse by more than the noise margin.
- The owner's recordings become fixtures with `python setup/make_fixture.py --id <id> --audio <file> --text <file>
  --lang mr` (one `speaker: text` line per turn, written in the output convention).

## Language models and lexicons
- `toolchain/models/lm-<lang>-v1` is built by `setup/build_lm.py` from a corpus prefix (IndicCorp v2, CC-0).
- Used by the cleanup word gate (an edit must not make the line less natural) and by polish (`unknown_words`).
- KenLM is not used: it has no Windows wheels; the numpy trigram model in `pipeline/lm.py` does the same job here.
- Tried and rejected (2026-10-22): re-decoding each line on its own to get 5 alternatives and letting the LM choose.
  The line loses its context, and the alternatives were worse (CER 0.62 vs 0.53).

## Adding a new ASR engine
1. Add it to `manifest.toml` (licence must allow commercial use; size; source).
2. Owner approves the download; `setup/` installs it into `toolchain/models/<id>/` with sha256 in `SOURCE.json`.
3. Write `pipeline/engines/<id>_engine.py` implementing the interface in `pipeline/engines/base.py`; register it in
   `make()` and give its family a settings table `[recognise.<family>]` in every recipe (plus `STAGE_KEYS`).
   An engine without word timings must use `segmentation = "turns"` (E28).
4. `trx eval --engine <id>`: it becomes a default for a language only if it beats the current one there.
5. Re-measure the post-processing with it: Gemma cleanup (`[llm_cleanup] engines`, E26) and Demucs (E27) were
   decided per engine.
6. GPU first (owner's rule): use the GPU build when one exists, keep the CPU as the fallback, and measure the RAM a
   run holds (free commit, E33) as well as the VRAM; set `[runtime.memory_gb]` from that measurement.

## Never
- Never change anything during a run or inside `runs/`.
- Never add a component that needs the internet at runtime, a background service, or an auto-updater.
- Never accept a change without an eval result. With only one fixture, treat small gains as noise and ask the
  owner for more labelled audio (QUESTIONS.md).
