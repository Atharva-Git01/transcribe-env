"""Per-case hotwords: names, places and terms that the recogniser should spell correctly (`trx transcribe --hotwords`).

File format: UTF-8 text, one term per line, `#` starts a comment. They are handed to faster-whisper's `hotwords`
hint (part of the decoder prompt). They bias spelling only; a term is never inserted unless it was spoken, and the
list is recorded in plan.json and decisions.json.
"""
from __future__ import annotations

from pathlib import Path

from .common import TrxError

MAX_TERMS = 40          # Whisper's prompt holds ~224 tokens shared with the initial prompt
MAX_CHARS = 300


def read_hotwords(path: str) -> list[str]:
    p = Path(path).expanduser()
    if not p.is_file():
        raise TrxError("INPUT_NOT_FOUND", f"hotwords file {p}")
    terms, seen = [], set()
    for line in p.read_text(encoding="utf-8").splitlines():
        t = line.split("#", 1)[0].strip()
        if t and t not in seen:
            terms.append(t)
            seen.add(t)
    if not terms:
        raise TrxError("USAGE", f"hotwords file {p} has no terms (one name/place/term per line)")
    if len(terms) > MAX_TERMS or len(" ".join(terms)) > MAX_CHARS:
        raise TrxError("USAGE", f"too many hotwords ({len(terms)} terms, {len(' '.join(terms))} characters); "
                                f"keep at most {MAX_TERMS} terms / {MAX_CHARS} characters, the most important first")
    return terms
