"""Shared helpers: paths, config loading, exit codes, the one-line JSON status, time, hashing."""
from __future__ import annotations

import hashlib
import json
import re
import sys
import tomllib
from datetime import datetime, timezone
from pathlib import Path

TRX_HOME = Path(__file__).resolve().parents[1]
CONFIG_DIR = TRX_HOME / "config"
RECIPES_DIR = TRX_HOME / "recipes"
RUNS_DIR = TRX_HOME / "runs"
SCHEMAS_DIR = TRX_HOME / "pipeline" / "schemas"


def env_version() -> str:
    return (TRX_HOME / "VERSION").read_text(encoding="utf-8").strip()


def load_toml(path: Path) -> dict:
    with open(path, "rb") as f:
        return tomllib.load(f)


def load_config(name: str) -> dict:
    return load_toml(CONFIG_DIR / f"{name}.toml")


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def local_stamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def sha256_file(path: Path, block: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(block):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def safe_case_name(stem: str, limit: int = 40) -> str:
    name = re.sub(r"[^A-Za-z0-9_-]+", "_", stem).strip("_")
    return (name[:limit].rstrip("_")) or "audio"


def out(msg: str = "") -> None:
    """Print human-readable text; survives consoles that cannot show Devanagari."""
    try:
        print(msg)
    except UnicodeEncodeError:
        print(msg.encode("ascii", "backslashreplace").decode("ascii"))


# --------------------------------------------------------------------------- exit codes
_EXIT = None


def exit_codes() -> dict:
    global _EXIT
    if _EXIT is None:
        _EXIT = load_config("exit_codes")
    return _EXIT


class TrxError(Exception):
    """Raise with a name from config/exit_codes.toml. trx.py turns it into the exit code + status line."""

    def __init__(self, name: str, detail: str = "", **extra):
        super().__init__(f"{name}: {detail}")
        self.name = name
        self.detail = detail
        self.extra = extra


def status_line(cmd: str, name: str, detail: str = "", **extra) -> int:
    """Print the single machine-readable JSON status line every command ends with. Returns the exit code."""
    codes = exit_codes()
    entry = codes.get(name) or codes["INTERNAL_ERROR"]
    payload = {
        "trx": env_version(),
        "cmd": cmd,
        "ok": entry["code"] == 0,
        "code": entry["code"],
        "name": name if name in codes else "INTERNAL_ERROR",
        "meaning": entry["meaning"],
        "next": entry["next"],
    }
    if detail:
        payload["detail"] = detail
    payload.update(extra)
    sys.stdout.flush()
    print(json.dumps(payload, ensure_ascii=True))
    return entry["code"]
