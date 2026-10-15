# setup/ — the only part that may use the internet

Nothing here runs automatically. Each tier is installed once, after the owner approves it.

Planned (phase 1):
- `setup.ps1 -Tier <phase>` — prints the plan (components, sizes, licences, disk needed) and changes nothing.
- `setup.ps1 -Tier <phase> -Approve` — creates `toolchain/python/.venv`, installs `requirements.lock`
  (`pip --require-hashes`), fetches models/binaries into `toolchain/`, writes `SOURCE.json` with sha256.
- `fetch_models.py --check <phase>` — sizes via HEAD requests only (no download).
- `clean.py` — lists rebuildable files; deletes only with `--yes`.

After setup, the runtime is offline again: `trx doctor --offline` must pass.
