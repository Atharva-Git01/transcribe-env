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



def speaker_accuracy(ref_segments: list[dict], hyp_segments: list[dict]) -> float | None:
    """Share of aligned characters whose hypothesis speaker matches the reference speaker, under the best one-to-one
    mapping of hypothesis labels to reference speakers (labels are arbitrary: SPK1 may be "B").
    Reference turns carry no times, so the texts are aligned character by character (the same alignment as CER).
    None when the reference has fewer than two speakers (nothing to measure)."""
    import itertools
    r_chars, r_spk, h_chars, h_spk = [], [], [], []
    for seg in ref_segments:
        t = normalise(seg.get("text", "")).replace(" ", "")
        r_chars += list(t)
        r_spk += [seg.get("speaker")] * len(t)
    for seg in hyp_segments:
        t = normalise(seg.get("text", "")).replace(" ", "")
        h_chars += list(t)
        h_spk += [seg.get("speaker")] * len(t)
    speakers = sorted({s for s in r_spk if s})
    if len(speakers) < 2 or not h_chars or not r_chars:
        return None
    n, m = len(r_chars), len(h_chars)
    prev = list(range(m + 1))
    back = []
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        row = bytearray(m + 1)
        ri = r_chars[i - 1]
        for j in range(1, m + 1):
            d, l, u = prev[j - 1] + (ri != h_chars[j - 1]), cur[j - 1] + 1, prev[j] + 1
            if d <= l and d <= u:
                cur[j], row[j] = d, 0
            elif u <= l:
                cur[j], row[j] = u, 1
            else:
                cur[j], row[j] = l, 2
        back.append(row)
        prev = cur
    pairs, i, j = [], n, m
    while i > 0 and j > 0:
        op = back[i - 1][j]
        if op == 0:
            pairs.append((r_spk[i - 1], h_spk[j - 1]))
            i, j = i - 1, j - 1
        elif op == 1:
            i -= 1
        else:
            j -= 1
    if not pairs:
        return None
    labels = sorted({h for _, h in pairs if h}, key=str)
    best = 0
    slots = speakers + [None] * max(0, len(labels) - len(speakers))
    for perm in itertools.permutations(slots, len(labels)):
        mp = dict(zip(labels, perm))
        best = max(best, sum(1 for r, h in pairs if mp.get(h) == r))
    return round(best / len(pairs), 4)


def self_test() -> tuple[bool, str]:
    checks = [
        cer("नमस्कार मंडळी", "नमस्कार मंडळी") == 0.0,
        cer("abc", "abd") == 1 / 3,
        wer("a b c", "a x c") == 1 / 3,
        cer("हा हा।", "हा हा") == 0.0,
    ]
    return all(checks), f"{sum(checks)}/{len(checks)} metric checks passed"
