"""Scoring: character and word error rate with one fixed normalisation (pure Python, no packages).

Normalisation (applied identically to reference and hypothesis): Unicode NFC, lowercase, remove punctuation
(incl. Devanagari danda । and double danda ॥), collapse whitespace. Bracketed tags like [inaudible …] are removed.
Complexity is O(len(ref) * len(hyp)); fine for eval clips (minutes), not for hour-long files.
"""
from __future__ import annotations

import re
import unicodedata

_TAGS = re.compile(r"\[[^\]]*\]")
_PUNCT = re.compile(r"[।॥.,!?;:\"'()\-–—…“”‘’/\\]")
_SPACE = re.compile(r"\s+")


def normalise(text: str) -> str:
    t = unicodedata.normalize("NFC", text)
    t = _TAGS.sub(" ", t).lower()
    t = _PUNCT.sub(" ", t)
    return _SPACE.sub(" ", t).strip()


def _edit_distance(a, b) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def cer(reference: str, hypothesis: str) -> float:
    ref, hyp = normalise(reference).replace(" ", ""), normalise(hypothesis).replace(" ", "")
    if not ref:
        return 0.0 if not hyp else 1.0
    return _edit_distance(ref, hyp) / len(ref)


def wer(reference: str, hypothesis: str) -> float:
    ref, hyp = normalise(reference).split(), normalise(hypothesis).split()
    if not ref:
        return 0.0 if not hyp else 1.0
    return _edit_distance(ref, hyp) / len(ref)


def self_test() -> tuple[bool, str]:
    checks = [
        cer("नमस्कार मंडळी", "नमस्कार मंडळी") == 0.0,
        cer("abc", "abd") == 1 / 3,
        wer("a b c", "a x c") == 1 / 3,
        cer("हा हा।", "हा हा") == 0.0,
    ]
    return all(checks), f"{sum(checks)}/{len(checks)} metric checks passed"
