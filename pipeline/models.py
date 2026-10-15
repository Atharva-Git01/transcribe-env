"""Model registry from manifest.toml: engine id -> local folder, family, installed?, recorded hashes."""
from __future__ import annotations

import json

from .common import TRX_HOME, TrxError, load_toml


def components() -> list[dict]:
    return load_toml(TRX_HOME / "manifest.toml").get("component", [])


def model(engine_id: str) -> dict:
    for c in components():
        if c["id"] == engine_id and c.get("kind") == "model":
            return c
    raise TrxError("CONFIG_INVALID", f"engine/model '{engine_id}' is not in manifest.toml")


def model_dir(engine_id: str):
    return TRX_HOME / model(engine_id)["path"]


def installed(engine_id: str) -> bool:
    try:
        c = model(engine_id)
    except TrxError:
        return False
    d = TRX_HOME / c.get("path", "__none__")
    marker = c.get("marker", "model.bin")
    # SOURCE.json is written by setup/setup.py only after the sha256 check, so a half-downloaded file never counts
    return (d / marker).is_file() and (d / "SOURCE.json").is_file()


def source_info(engine_id: str) -> dict:
    p = model_dir(engine_id) / "SOURCE.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else {}


def first_installed(engine_ids: list[str]) -> str | None:
    for e in engine_ids:
        if e and installed(e):
            return e
    return None


def require(engine_id: str) -> None:
    if not installed(engine_id):
        raise TrxError("MODEL_MISSING", f"{engine_id} is not installed in {model(engine_id).get('path')}")
