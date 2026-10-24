"""The words of private recordings never leave this PC (E29). A fixture is private unless its truth.json names an open
licence (CC-BY / CC-0); every real run in runs/ is private evidence. `leaks()` looks for 3-word sequences of those
texts (truth, and every machine transcript of them kept on this PC) in the files git would push. `trx test` runs it.
Only Devanagari words are compared: English word triples are too common to tell a leak from ordinary prose (the
structural protection stays: fixtures, runs and evals/compare are git-ignored, evals/results holds numbers only).
"""
from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

from .common import TRX_HOME

OPEN_LICENCE = re.compile(r"CC-BY|CC-0|CC0", re.I)
WORD = re.compile(r"[\u0900-\u0963\u0966-\u097F]+")   # Devanagari letters and digits, no danda


def _texts_of(path: Path) -> list[str]:
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    return [s.get("text", "") for k in ("segments", "segments_standard") for s in d.get(k, []) if isinstance(s, dict)]


def private_texts(home: Path = TRX_HOME) -> list[str]:
    texts, open_ids = [], set()
    for d in sorted((home / "evals" / "fixtures").glob("*/")):
        tj = d / "truth.json"
        if not tj.is_file():
            continue
        try:
            truth = json.loads(tj.read_text(encoding="utf-8"))
        except ValueError:
            continue
        if OPEN_LICENCE.search(str(truth.get("source", ""))):
            open_ids.add(d.name)
            continue
        texts += _texts_of(tj)
    # an eval run (evals/runs/eval_<fixture>-<date>-<time>) is private unless its fixture is open: the runs of a
    # renamed or removed fixture stay private (E39)
    for t in (home / "evals" / "runs").glob("eval_*/out/transcript.json"):
        if t.parents[1].name[len("eval_"):].rsplit("-", 2)[0] not in open_ids:
            texts += _texts_of(t)
    for t in (home / "runs").glob("*/out/transcript.json"):
        texts += _texts_of(t)
    # the screening lines and every candidate's output on them (PLAN_v0.9) are words of private fixtures
    for t in [*(home / "evals" / "screening").glob("*/lines.json"), *(home / "evals" / "screening").glob("*/out/*.json")]:
        texts += _texts_of(t)
    return texts


def phrases(texts: list[str], n: int = 3, min_chars: int = 12) -> set[str]:
    out = set()
    for x in texts:
        w = WORD.findall(x)
        out |= {g for g in (" ".join(w[i:i + n]) for i in range(len(w) - n + 1)) if len(g) >= min_chars}
    return out


def pushable_files(home: Path = TRX_HOME) -> list[str]:
    """Tracked files plus untracked files git would add (ignored ones excluded)."""
    def git(*args):
        res = subprocess.run(["git", "-C", str(home), *args], capture_output=True, text=True, encoding="utf-8")
        return [p for p in res.stdout.split("\n") if p] if res.returncode == 0 else []
    return sorted(set(git("ls-files")) | set(git("ls-files", "--others", "--exclude-standard")))


PROSE = ("*.md", "knowledge/*", "evals/results/*", "tests/*", "pipeline/*", "pipeline/*/*", "config/*", "recipes/*")
COMMON_MIN = 200                     # corpus count (lm-mr/hi unigrams) below which a word counts as rare
GENERIC = ("library/", "prompts/", "references/")


def generic_words(home: Path = TRX_HOME) -> set[str]:
    """Words already in library/, prompts/ and references/: generic speech (fillers such as a written "hmm",
    backchannels, common colloquial forms) kept there on purpose and pushed anyway, so never a sign of a leak (E39)."""
    words = set()
    for rel in pushable_files(home):
        if rel.startswith(GENERIC):
            try:
                words |= set(WORD.findall((home / rel).read_text(encoding="utf-8")))
            except (OSError, UnicodeDecodeError):
                continue
    return words


def rare_word_leaks(home: Path = TRX_HOME) -> list[tuple[str, int, list[str]]]:
    """Single rare words of private texts (colloquial or misheard forms; corpus count < COMMON_MIN) in prose and
    results. library/, prompts/ and references/ are left out: they hold generic colloquial Marathi on purpose."""
    from . import lm
    models = [m for m in (lm.load("mr"), lm.load("hi")) if m is not None]
    if not models:
        return []
    generic = generic_words(home)
    rare = {w for t in private_texts(home) for w in WORD.findall(t)
            if len(w) >= 2 and w not in generic and max(m.count(w) for m in models) < COMMON_MIN}
    if not rare:
        return []
    files = [r for r in pushable_files(home) if not r.startswith(GENERIC) and any(Path(r).match(g) for g in PROSE)]
    found = []
    for rel in files:
        try:
            body = (home / rel).read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        for i, line in enumerate(body.splitlines(), 1):
            hit = sorted({w for w in WORD.findall(line) if w in rare})
            if hit:
                found.append((rel, i, hit))
    return found


MIN_HITS = 2       # one common greeting phrase also starts many real calls; a copied line has several (E36)


def leaks(home: Path = TRX_HOME) -> list[tuple[str, int, int]]:
    """[(file, line, number of private 3-word sequences on it)] for every pushable text file; a line counts when it
    holds at least MIN_HITS of them. Short fragments are covered by rare_word_leaks()."""
    grams = phrases(private_texts(home))
    if not grams:
        return []
    found = []
    for rel in pushable_files(home):
        p = home / rel
        if not p.is_file() or p.stat().st_size > 5_000_000:
            continue
        try:
            body = p.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        for i, line in enumerate(body.splitlines(), 1):
            w = WORD.findall(line)
            n = sum(1 for j in range(len(w) - 2) if " ".join(w[j:j + 3]) in grams)
            if n >= MIN_HITS:
                found.append((rel, i, n))
    return found
