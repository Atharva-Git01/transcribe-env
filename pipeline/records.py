"""Run records: decisions.json (append-only, why every action happened) and custody.json (hash chain)."""
from __future__ import annotations

import json
from pathlib import Path

from .common import sha256_file, utc_now, write_json


class Decisions:
    def __init__(self, run_dir: Path):
        self.path = run_dir / "decisions.json"
        self.items: list[dict] = []

    @classmethod
    def load(cls, run_dir: Path) -> "Decisions":
        """Reopen an interrupted run's log: new entries are appended after the existing ones (never rewritten)."""
        d = cls(run_dir)
        d.items = json.loads(d.path.read_text(encoding="utf-8"))
        return d

    def add(self, stage: str, action: str, reason: str, **data) -> None:
        self.items.append({"seq": len(self.items) + 1, "utc": utc_now(), "stage": stage,
                           "action": action, "reason": reason, "data": data})
        write_json(self.path, self.items)


class Custody:
    def __init__(self, run_dir: Path, run_id: str, original_path: Path, original_sha256: str, size: int):
        self.run_dir = run_dir
        self.path = run_dir / "custody.json"
        self.data = {
            "schema": "trx.custody/1",
            "run_id": run_id,
            "original": {"path": str(original_path), "sha256": original_sha256,
                         "size_bytes": size, "hashed_utc": utc_now()},
            "entries": [],
        }
        write_json(self.path, self.data)

    @classmethod
    def load(cls, run_dir: Path) -> "Custody":
        c = cls.__new__(cls)
        c.run_dir, c.path = run_dir, run_dir / "custody.json"
        c.data = json.loads(c.path.read_text(encoding="utf-8"))
        return c

    def add(self, file: Path, role: str, stage: str) -> str:
        digest = sha256_file(file)
        rel = file.relative_to(self.run_dir).as_posix()
        last = next((e for e in reversed(self.data["entries"]) if e["file"] == rel), None)
        if last and last["sha256"] == digest:  # regenerated identically on resume: already in the chain
            return digest
        self.data["entries"].append({
            "file": rel,
            "role": role, "stage": stage, "sha256": digest,
            "size_bytes": file.stat().st_size, "utc": utc_now(),
        })
        write_json(self.path, self.data)
        return digest
