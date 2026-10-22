"""Word n-gram language model + lexicon for one language (runtime side: offline, numpy only, memory-mapped).

Built once by `setup/build_lm.py` from a large free text corpus (IndicCorp v2, CC-0) into
toolchain/models/lm-<lang>-v1/:  vocab.txt (one word per line, id = line number, sorted by frequency),
unigram.npy (counts by id), bigram_keys.npy / bigram_counts.npy, trigram_keys.npy / trigram_counts.npy
(keys = word ids packed 20 bits each into uint64, sorted -> exact lookups by binary search), meta.json.

Scoring is "stupid backoff" (Brants et al. 2007): it ranks candidate sentences well, which is all the
correction step needs (it compares a few candidates of one segment; it never generates text).
"""
from __future__ import annotations

import json
import math
import re
import unicodedata
from functools import lru_cache
from pathlib import Path

import numpy as np

from .common import TRX_HOME

BITS = 20
MAX_VOCAB = (1 << BITS) - 2          # ids 0..MAX_VOCAB-1 are words; MAX_VOCAB+1 = <unk>
UNK = (1 << BITS) - 1
BOS = 0                              # id 0 is reserved for <s> in vocab.txt
_PUNCT = re.compile(r"[\.,!?;:\"'()\[\]{}<>«»“”‘’…।॥\-–—/\\|*#@&%+=_~`^]+")


def model_dir(lang: str) -> Path:
    return TRX_HOME / "toolchain" / "models" / f"lm-{lang}-v1"


def tokens(text: str) -> list[str]:
    """The ONE tokenisation used for building and scoring: NFC, punctuation -> space, Latin lower-case."""
    t = unicodedata.normalize("NFC", text)
    t = _PUNCT.sub(" ", t)
    return [w.lower() for w in t.split() if w]


def pack2(a, b):
    return (np.asarray(a, dtype=np.uint64) << np.uint64(BITS)) | np.asarray(b, dtype=np.uint64)


def pack3(a, b, c):
    return (((np.asarray(a, dtype=np.uint64) << np.uint64(BITS)) | np.asarray(b, dtype=np.uint64))
            << np.uint64(BITS)) | np.asarray(c, dtype=np.uint64)


class NgramLM:
    def __init__(self, lang: str, alpha: float = 0.4):
        d = model_dir(lang)
        if not (d / "meta.json").is_file():
            raise FileNotFoundError(f"no language model for '{lang}' in {d}")
        self.lang = lang
        self.alpha = alpha
        self.meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        words = (d / "vocab.txt").read_text(encoding="utf-8").split("\n")
        self.vocab = {w: i for i, w in enumerate(words) if w}
        self.words = words
        self.uni = np.load(d / "unigram.npy", mmap_mode="r")
        self.bk, self.bc = np.load(d / "bigram_keys.npy", mmap_mode="r"), np.load(d / "bigram_counts.npy", mmap_mode="r")
        self.tk, self.tc = np.load(d / "trigram_keys.npy", mmap_mode="r"), np.load(d / "trigram_counts.npy", mmap_mode="r")
        self.total = float(self.meta["tokens"])
        self.v = float(len(self.vocab))

    def wid(self, w: str) -> int:
        return self.vocab.get(w, UNK)

    def count(self, w: str) -> int:
        i = self.vocab.get(w)
        return int(self.uni[i]) if i is not None else 0

    @staticmethod
    def _find(keys, counts, key) -> int:
        i = int(np.searchsorted(keys, np.uint64(key)))
        return int(counts[i]) if i < len(keys) and int(keys[i]) == int(key) else 0

    def _c1(self, i: int) -> int:
        return int(self.uni[i]) if i != UNK else 0

    def _c2(self, a: int, b: int) -> int:
        if UNK in (a, b):
            return 0
        return self._find(self.bk, self.bc, int(pack2(a, b)))

    def _c3(self, a: int, b: int, c: int) -> int:
        if UNK in (a, b, c):
            return 0
        return self._find(self.tk, self.tc, int(pack3(a, b, c)))

    def _p_uni(self, c: int) -> float:
        return (self._c1(c) + 0.5) / (self.total + 0.5 * self.v)   # <unk> gets 0.5 pseudo-count

    def log10_score(self, a: int, b: int, c: int) -> float:
        """log10 stupid-backoff score of word c after (a, b)."""
        c3 = self._c3(a, b, c)
        if c3:
            return math.log10(c3 / max(1, self._c2(a, b)))
        c2 = self._c2(b, c)
        if c2:   # unigram[0] holds the sentence count, so <s> works as a history word too
            return math.log10(self.alpha * c2 / max(1, self._c1(b)))
        return math.log10(self.alpha * self.alpha * self._p_uni(c))

    @lru_cache(maxsize=20000)
    def sentence_score(self, text: str) -> tuple[float, int]:
        """(total log10 score, number of words) for one sentence/segment."""
        ws = [self.wid(w) for w in tokens(text)]
        if not ws:
            return 0.0, 0
        hist = [BOS, BOS]
        total = 0.0
        for c in ws:
            total += self.log10_score(hist[-2], hist[-1], c)
            hist.append(c)
        return total, len(ws)

    def per_word(self, text: str) -> float:
        s, n = self.sentence_score(text)
        return s / n if n else 0.0


@lru_cache(maxsize=4)
def load(lang: str) -> NgramLM | None:
    try:
        return NgramLM(lang)
    except FileNotFoundError:
        return None


def installed(lang: str) -> bool:
    d = model_dir(lang)
    return (d / "meta.json").is_file() and (d / "SOURCE.json").is_file()
