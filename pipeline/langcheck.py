"""Language confirmation (owner decision 2026-10-16): the environment measures the audio, guesses the language and
ASKS before transcribing. OK -> the guess is used as confirmed; another language -> that one is used.

- A person at a terminal is asked inline (`trx transcribe` waits for Enter or a language code).
- A non-interactive operator (a script or an AI model) gets exit LANGUAGE_CONFIRM_NEEDED with the guess; it asks the
  owner and reruns with `--lang <code>`. No run folder is created for the check.
- `--lang auto` accepts the guess without asking (eval, watch, or when the owner says so).

The check probes a verified temporary copy; the passport is cached by sha256 so the rerun does not measure twice.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

from .common import RUNS_DIR, TrxError, env_version, load_config, out, sha256_file, write_json

CHECK_DIR = RUNS_DIR / ".lang_check"
NAMES = {"mr": "Marathi", "hi": "Hindi", "en": "English"}


def candidates() -> list[str]:
    return list(load_config("defaults")["languages"]["priority"])


def cache_path(sha: str) -> Path:
    return CHECK_DIR / f"{sha}.passport.json"


def cached_passport(sha: str) -> dict | None:
    p = cache_path(sha)
    if not p.is_file():
        return None
    data = json.loads(p.read_text(encoding="utf-8"))
    return data["passport"] if data.get("env_version") == env_version() else None


def guess(passport: dict) -> dict:
    lang = passport.get("language", {})
    dec = lang.get("decision") or {}
    top = lang.get("top3") or []
    code = dec.get("lang") or (top[0]["lang"] if top else None)
    return {"lang": code, "name": NAMES.get(code, code), "confidence": dec.get("confidence"),
            "basis": dec.get("basis") or ("acoustic LID" if top else "not measured")}


def measure(src: Path) -> tuple[str, dict]:
    """Probe a verified temporary copy of the input (the original is only read) and cache the passport."""
    from .stages.probe import probe
    sha = sha256_file(src)
    cached = cached_passport(sha)
    if cached:
        return sha, cached
    tmp = CHECK_DIR / sha / src.name
    tmp.parent.mkdir(parents=True, exist_ok=True)
    try:
        shutil.copy2(src, tmp)
        if sha256_file(tmp) != sha:
            raise TrxError("INTERNAL_ERROR", "language-check copy hash differs from the original")
        passport = probe(tmp)
    finally:
        shutil.rmtree(tmp.parent, ignore_errors=True)
    write_json(cache_path(sha), {"env_version": env_version(), "sha256": sha, "passport": passport})
    return sha, passport


def ask(g: dict) -> str:
    """Interactive: Enter = accept the guess; a language code = use that language."""
    allowed = candidates()
    conf = f", confidence {g['confidence']:.2f}" if isinstance(g.get("confidence"), (int, float)) else ""
    out(f"Detected language: {g['name']} ({g['lang']}{conf}) — {g['basis']}")
    while True:
        try:
            ans = input(f"Is this right? Press Enter for yes, or type the language ({'/'.join(allowed)}): ").strip().lower()
        except EOFError:
            raise TrxError("LANGUAGE_CONFIRM_NEEDED", "no answer was given", detected=g["lang"],
                           confidence=g.get("confidence"))
        if ans in ("", "y", "yes", "ok", "ho", "haan", "हो", "हाँ"):
            if not g["lang"]:
                out("No language could be measured; please type one.")
                continue
            return g["lang"]
        if ans in allowed:
            return ans
        out(f"Please press Enter or type one of: {', '.join(allowed)}")


def confirm(file: str, lang_arg: str | None) -> tuple[str | None, str]:
    """Returns (language or None for auto, how it was decided). Raises LANGUAGE_CONFIRM_NEEDED when nobody can answer."""
    allowed = candidates()
    if lang_arg and lang_arg != "auto":
        if lang_arg not in allowed:
            raise TrxError("USAGE", f"--lang must be one of {allowed} or auto")
        return lang_arg, "given by the operator (--lang)"
    if lang_arg == "auto":
        return None, "auto: the measured guess is used without asking (--lang auto)"
    src = Path(file).expanduser()
    if not src.is_file():
        raise TrxError("INPUT_NOT_FOUND", str(src))
    _, passport = measure(src)
    g = guess(passport)
    if sys.stdin is not None and sys.stdin.isatty():
        chosen = ask(g)
        how = "confirmed by the operator" if chosen == g["lang"] else f"corrected by the operator (guess was {g['lang']})"
        return chosen, how
    raise TrxError("LANGUAGE_CONFIRM_NEEDED",
                   f"guessed {g['name']} ({g['lang']}); ask the owner, then rerun with --lang <code>",
                   detected=g["lang"], detected_name=g["name"], confidence=g.get("confidence"), basis=g["basis"])
