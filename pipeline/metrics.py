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


# Spelling-variant folding for the "normalised" score (OIWER-like; documented, applied to BOTH sides):
#   nukta dropped (ज़ -> ज), chandrabindu -> anusvara (ँ -> ं), candra vowels -> plain (ॅ -> े, ॉ -> ो, ऑ -> ओ, ॲ -> अ),
#   digits -> number words (library/numbers/<lang>.toml). "loose" also merges short/long i and u (ि/ी, ु/ू).
_FOLD = str.maketrans({"\u093c": None, "\u0901": "\u0902", "\u0945": "\u0947", "\u0949": "\u094b",
                       "\u0911": "\u0913", "\u0972": "\u0905"})
_LOOSE = str.maketrans({"\u0940": "\u093f", "\u0942": "\u0941", "\u0908": "\u0907", "\u090a": "\u0909"})


def normalise_level(text: str, lang: str = "", level: str = "strict") -> str:
    """level: "strict" (= normalise), "norm" (spelling variants folded, digits as words), "loose" (+ i/u length)."""
    t = normalise(text)
    if level == "strict":
        return t
    t = unicodedata.normalize("NFD", t).translate(_FOLD)
    t = unicodedata.normalize("NFC", t)
    if lang:
        try:
            from .numbers import digits_to_words
            t = digits_to_words(t, lang)[0].replace(" टक्के", " टक्के")
        except Exception:  # noqa: BLE001 — scoring must never crash on a number table problem
            pass
    if level == "loose":
        t = t.translate(_LOOSE)
    return _SPACE.sub(" ", t).strip()


def cer_level(reference: str, hypothesis: str, lang: str = "", level: str = "strict") -> float:
    ref = normalise_level(reference, lang, level).replace(" ", "")
    hyp = normalise_level(hypothesis, lang, level).replace(" ", "")
    if not ref:
        return 0.0 if not hyp else 1.0
    return _edit_distance(ref, hyp) / len(ref)


def turn_alignment(ref_segments: list[dict], hyp_segments: list[dict]) -> list[dict]:
    """Align hypothesis segments to reference turns (character alignment, like CER). For each reference turn:
    the hypothesis text that landed on it, its speakers, and the turn's own CER. Missing turns get text ""."""
    r_chars, r_turn, h_chars, h_seg = [], [], [], []
    for i, seg in enumerate(ref_segments):
        t = normalise(seg.get("text", "")).replace(" ", "")
        r_chars += list(t)
        r_turn += [i] * len(t)
    for j, seg in enumerate(hyp_segments):
        t = normalise(seg.get("text", "")).replace(" ", "")
        h_chars += list(t)
        h_seg += [j] * len(t)
    votes: dict = {}
    for ri, hj in _align_pairs(r_chars, h_chars):
        votes.setdefault(h_seg[hj], []).append(r_turn[ri])
    owner, last = {}, 0
    for j in range(len(hyp_segments)):
        v = votes.get(j)
        owner[j] = max(set(v), key=v.count) if v else last
        last = owner[j]
    rows = []
    for i, seg in enumerate(ref_segments):
        hyp = [hyp_segments[j] for j in range(len(hyp_segments)) if owner[j] == i]
        text = " ".join(h.get("text", "") for h in hyp).strip()
        rows.append({"turn": i + 1, "ref_speaker": seg.get("speaker"), "ref": seg.get("text", ""), "hyp": text,
                     "hyp_speakers": sorted({h.get("speaker") for h in hyp if h.get("speaker")}),
                     "cer": round(cer(seg.get("text", ""), text), 3) if text else 1.0})
    return rows


def _align_pairs(a: list, b: list) -> list[tuple[int, int]]:
    """Index pairs (i, j) of a Levenshtein alignment of two sequences (matches and substitutions)."""
    n, m = len(a), len(b)
    prev = list(range(m + 1))
    back = []
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        row = bytearray(m + 1)
        ai = a[i - 1]
        for j in range(1, m + 1):
            d, l, u = prev[j - 1] + (ai != b[j - 1]), cur[j - 1] + 1, prev[j] + 1
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
            pairs.append((i - 1, j - 1))
            i, j = i - 1, j - 1
        elif op == 1:
            i -= 1
        else:
            j -= 1
    return pairs[::-1]


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
