"""Run records: decisions.json (append-only, why every action happened) and custody.json (hash chain)."""
from __future__ import annotations

from pathlib import Path

from .common import sha256_file, utc_now, write_json


class Decisions:
    def __init__(self, run_dir: Path):
        self.path = run_dir / "decisions.json"
        self.items: list[dict] = []

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

    def add(self, file: Path, role: str, stage: str) -> str:
        digest = sha256_file(file)
        self.data["entries"].append({
            "file": file.relative_to(self.run_dir).as_posix(),
            "role": role, "stage": stage, "sha256": digest,
            "size_bytes": file.stat().st_size, "utc": utc_now(),
        })
        write_json(self.path, self.data)
        return digest
