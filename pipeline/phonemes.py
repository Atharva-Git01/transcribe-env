"""The phoneme judge (PLAN_v0.9 A3): which of two spellings of a line does the audio support?

A multilingual phoneme recogniser (wav2vec2-xlsr-53-espeak-cv-ft) hears the line as IPA phonemes. Each candidate
spelling of the line is turned into sounds by a small rule-based reader for Devanagari (Marathi/Hindi), both sides
are reduced to the same coarse sound classes, and the spelling whose sounds are closer (edit distance) wins. Only
the slot word differs between the two candidates, so the difference comes from that word. A tie keeps the standard
spelling: the judge never changes a word on no evidence.

Coarse classes (what both sides can tell apart reliably): vowels without length (ə, a, i, u, e, o), consonants by
place and manner, aspiration as its own mark "H", s vs sh, dental vs retroflex n/t/d/l. The recogniser's IPA and the
reader's output both map into them.
"""
from __future__ import annotations

import unicodedata

_CONS = {
    "क": "k", "ख": "kH", "ग": "g", "घ": "gH", "ङ": "n", "च": "c", "छ": "cH", "ज": "j", "झ": "jH", "ञ": "n",
    "ट": "T", "ठ": "TH", "ड": "D", "ढ": "DH", "ण": "N", "त": "t", "थ": "tH", "द": "d", "ध": "dH", "न": "n",
    "प": "p", "फ": "pH", "ब": "b", "भ": "bH", "म": "m", "य": "y", "र": "r", "ल": "l", "ळ": "L", "व": "v",
    "श": "S", "ष": "S", "स": "s", "ह": "h",
}
_VOWEL = {"अ": "@", "आ": "a", "इ": "i", "ई": "i", "उ": "u", "ऊ": "u", "ऋ": "ru", "ए": "e", "ऐ": "@i", "ओ": "o",
          "औ": "@u", "ऑ": "o", "ॲ": "e", "ऍ": "e"}
_MATRA = {"ा": "a", "ि": "i", "ी": "i", "ु": "u", "ू": "u", "ृ": "ru", "े": "e",
          "ै": "@i", "ो": "o", "ौ": "@u", "ॅ": "e", "ॉ": "o", "ॆ": "e", "ॊ": "o"}
_VIRAMA, _ANUSVARA, _CANDRA, _NUKTA, _VISARGA = "्", "ं", "ँ", "़", "ः"

# the recogniser's IPA symbols -> coarse classes (longest match first)
_IPA = {
    "t͡ʃʰ": "cH", "d͡ʒʱ": "jH", "t͡ʃ": "c", "d͡ʒ": "j", "t͡s": "c", "d͡z": "j", "tʃʰ": "cH", "dʒʱ": "jH", "tʃ": "c",
    "dʒ": "j", "ts": "c", "dz": "j", "kʰ": "kH", "ɡʱ": "gH", "gʱ": "gH", "ʈʰ": "TH", "ɖʱ": "DH", "t̪ʰ": "tH",
    "d̪ʱ": "dH", "tʰ": "tH", "dʱ": "dH", "pʰ": "pH", "bʱ": "bH", "t̪": "t", "d̪": "d",
    "k": "k", "ɡ": "g", "g": "g", "q": "k", "x": "kH", "ɣ": "g", "ʈ": "T", "ɖ": "D", "ɳ": "N", "t": "t", "d": "d",
    "n": "n", "ŋ": "n", "ɲ": "n", "p": "p", "f": "pH", "b": "b", "m": "m", "j": "y", "r": "r", "ɾ": "r", "ɹ": "r",
    "ɽ": "r", "l": "l", "ɭ": "L", "ʋ": "v", "v": "v", "w": "v", "β": "v", "ʃ": "S", "ʂ": "S", "ɕ": "S", "s": "s",
    "z": "j", "ʒ": "j", "h": "h", "ɦ": "h", "c": "c", "ɟ": "j",
    "ə": "@", "ɐ": "@", "ʌ": "@", "ɜ": "@", "a": "a", "ɑ": "a", "æ": "e", "ä": "a", "i": "i", "ɪ": "i", "y": "i",
    "u": "u", "ʊ": "u", "ɯ": "u", "e": "e", "ɛ": "e", "o": "o", "ɔ": "o", "ɒ": "o", "ø": "e", "œ": "e",
}
_DROP = {"ː", "ˑ", "ˈ", "ˌ", "̃", "͡", "‿", "|", "ʔ"}


def read_devanagari(text: str) -> list[str]:
    """Coarse sound classes of Devanagari text: consonants carry their inherent vowel unless a vowel sign or virama
    follows; a word-final inherent vowel is not spoken. Other characters are skipped."""
    out: list[str] = []
    for word in unicodedata.normalize("NFC", text).split():
        snd: list[str] = []
        chars = [c for c in word if c != _NUKTA]
        for k, c in enumerate(chars):
            nxt = chars[k + 1] if k + 1 < len(chars) else ""
            if c in _CONS:
                snd += _split(_CONS[c])
                if nxt not in _MATRA and nxt != _VIRAMA:
                    snd.append("@")
            elif c in _MATRA:
                if snd and snd[-1] == "@":
                    snd.pop()
                snd += _split(_MATRA[c])
            elif c in _VOWEL:
                snd += _split(_VOWEL[c])
            elif c in (_ANUSVARA, _CANDRA):
                snd.append("n")
            elif c == _VISARGA:
                snd.append("h")
        if len(snd) > 1 and snd[-1] == "@" and snd[-2] not in ("@", "a", "i", "u", "e", "o"):
            snd.pop()                                   # word-final inherent vowel is silent
        out += snd
    return out


def _split(code: str) -> list[str]:
    return ["@" if ch == "@" else ch for ch in code]


def read_ipa(phonemes: str) -> list[str]:
    """Coarse sound classes of the recogniser's IPA output (symbols separated by spaces or not)."""
    s = unicodedata.normalize("NFD", phonemes)
    s = "".join(ch for ch in s if ch not in _DROP and unicodedata.category(ch) != "Mn" or ch in ("̪",))
    s = unicodedata.normalize("NFC", s).replace("̪", "")
    out, i = [], 0
    keys = sorted(_IPA, key=len, reverse=True)
    s = s.replace(" ", "")
    while i < len(s):
        for k in keys:
            if s.startswith(k, i):
                out += _split(_IPA[k])
                i += len(k)
                break
        else:
            i += 1
    return out


def distance(a: list[str], b: list[str]) -> int:
    from .metrics import _edit_distance
    return _edit_distance(a, b)


def judge(heard: str, candidates: list[str]) -> tuple[int | None, list[int]]:
    """Index of the candidate line whose sounds are closest to what the recogniser heard (None on a tie for the
    best), and every candidate's distance. heard = the recogniser's IPA for the line."""
    h = read_ipa(heard)
    d = [distance(h, read_devanagari(c)) for c in candidates]
    best = min(d)
    return (d.index(best) if d.count(best) == 1 else None), d
