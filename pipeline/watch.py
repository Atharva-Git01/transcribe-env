"""trx watch: process every audio/video file dropped into inbox/. Files are never moved or changed; processed
files are remembered by sha256 in inbox/.processed.json so the same content is never run twice."""
from __future__ import annotations

import time

from .common import TRX_HOME, TrxError, out, read_json, sha256_file, utc_now, write_json
from .run import transcribe

INBOX = TRX_HOME / "inbox"
LEDGER = INBOX / ".processed.json"
MEDIA = {".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg", ".opus", ".wma", ".amr", ".3gp",
         ".mp4", ".mkv", ".mov", ".avi", ".webm"}


def _ledger() -> dict:
    return read_json(LEDGER) if LEDGER.is_file() else {}


def scan_once() -> list[dict]:
    ledger = _ledger()
    results = []
    for f in sorted(INBOX.iterdir()):
        if not f.is_file() or f.suffix.lower() not in MEDIA:
            continue
        digest = sha256_file(f)
        if digest in ledger:
            continue
        try:
            name, info = transcribe(str(f))
        except TrxError as e:
            name, info = e.name, {"run_id": None, "detail": e.detail}
        ledger[digest] = {"file": f.name, "run_id": info.get("run_id"), "exit": name, "utc": utc_now()}
        write_json(LEDGER, ledger)
        out(f"  {f.name} -> {info.get('run_id')} ({name})")
        results.append({"file": f.name, "run_id": info.get("run_id"), "exit": name})
    return results


def watch(interval: int, once: bool) -> list[dict]:
    if once:
        return scan_once()
    out(f"watching {INBOX} every {interval}s (Ctrl+C to stop)")
    done: list[dict] = []
    try:
        while True:
            done += scan_once()
            time.sleep(interval)
    except KeyboardInterrupt:
        return done
