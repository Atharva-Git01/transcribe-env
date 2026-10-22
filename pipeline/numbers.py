"""Numbers as words (owner convention 2026-10-22) for Marathi and Hindi, from library/numbers/<lang>.toml.

Integers in the Indian system (thousand, lakh, crore) and percentages are converted; decimals, times and other
digit patterns are left as they are and reported, so nothing is guessed. ASCII and Devanagari digits both count.
"""
from __future__ import annotations

import re
from functools import lru_cache

from .common import TRX_HOME, load_toml

DEV_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")
# a whole number not glued to other digits, separators or letters; optional % right after
INTEGER = re.compile(r"(?<![\d.,:/])(\d{1,9})(?![\d.,:/])(\s?%)?")
OTHER_NUMBER = re.compile(r"\d")


@lru_cache(maxsize=4)
def table(lang: str) -> dict | None:
    p = TRX_HOME / "library" / "numbers" / f"{lang}.toml"
    if not p.is_file():
        return None
    t = load_toml(p)
    assert len(t["words"]) == 100, f"{p}: words must list 0..99"
    return t


def to_words(n: int, lang: str) -> str:
    t = table(lang)
    if t is None:
        raise ValueError(f"no number table for {lang}")
    w = t["words"]
    if n < 100:
        return w[n]
    parts = []
    for unit, size in (("crore", 10 ** 7), ("lakh", 10 ** 5), ("thousand", 1000)):
        if n >= size:
            parts.append(f"{to_words(n // size, lang)} {t[unit]}")
            n %= size
    if n >= 100:
        h, n = divmod(n, 100)
        if n == 0 and not parts and h == 1:
            parts.append(t["hundred_exact"])
        else:
            parts.append(t["hundreds_prefix"][h])
    if n:
        parts.append(w[n])
    return " ".join(parts)


def digits_to_words(text: str, lang: str) -> tuple[str, int, list[str]]:
    """Returns (text with integers/percentages as words, how many were converted, digit patterns left as is)."""
    t = table(lang)
    if t is None:
        return text, 0, []
    s = text.translate(DEV_DIGITS)
    count = 0

    def rep(m):
        nonlocal count
        count += 1
        word = to_words(int(m.group(1)), lang)
        return word + (f" {t['percent']}" if m.group(2) else "")
    out = INTEGER.sub(rep, s)
    left = sorted(set(re.findall(r"[\d.,:/]*\d[\d.,:/]*", out)))
    if not count and not left:
        return text, 0, []          # untouched (keeps the original digit script if nothing to do)
    return out, count, left
