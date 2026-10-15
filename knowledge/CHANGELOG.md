# CHANGELOG

Every accepted change to the environment, newest first, with its `trx eval` result.
Format: `## <version> — <date>` · what changed · why · eval before → after (or "no scoring yet" before phase 1).

## 0.1.0-phase0 — 2026-10-16
- Built the skeleton: `trx` entry point (help, version, doctor, probe, transcribe, explain, eval, watch),
  offline guard, config (defaults, thresholds, selection rules, exit codes), JSON Schemas + validator, 8 recipes,
  run folders with read-only input copy, sha256 custody chain, decisions log, report, per-run git.
- Probe measures: channel layout (duplicated-mono detection), loudness (EBU R128), true peak, clipping, energy-based
  noise floor / SNR / activity, spectral-rolloff bandwidth class.
- Seeded knowledge/ERRORS.md (E01–E08), references, prompts, doubled-word lists; FMS vocab copied disabled (E07).
- Eval fixture: Pappa call (human ground truth). No scoring yet — the recognise stage arrives in phase 1.
- Config format is TOML (Python's built-in reader) instead of YAML: zero extra packages.
