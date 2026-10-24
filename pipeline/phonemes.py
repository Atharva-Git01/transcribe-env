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
    # symbols of other languages the recogniser may still use, mapped to the nearest Marathi class
    "ɻ": "L", "ʐ": "S", "ʑ": "j", "ʝ": "y", "ɨ": "i", "ᵻ": "i", "ʉ": "u", "ɚ": "@", "ɵ": "o", "ɫ": "l", "ʎ": "l",
    "ɬ": "l", "θ": "t", "ð": "d", "χ": "kH", "X": "kH", "ħ": "h", "ʕ": "h", "ʁ": "r", "ɴ": "n", "N": "n", "ɸ": "pH",
}
_DROP = {"ː", "ˑ", "ˈ", "ˌ", "̃", "͡", "‿", "|", "ʔ", "ʲ"}


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


# ---- version 2: the recogniser's own likelihood of each spelling (CTC), not an edit distance of its best guess ----
# Devanagari -> the recogniser's phoneme tokens (wav2vec2-xlsr-53-espeak-cv-ft). Its inventory has no breathy-voiced
# stops (gh, jh, dh, bh) and no voiced dental d: those become the plain stop + "h" and "d".
_CTC_CONS = {
    "क": "k", "ख": "kʰ", "ग": "ɡ", "घ": "ɡ h", "ङ": "ŋ", "च": "tʃ", "छ": "tʃʰ", "ज": "dʒ", "झ": "dʒ h", "ञ": "ɲ",
    "ट": "ʈ", "ठ": "ʈʰ", "ड": "ɖ", "ढ": "ɖ h", "ण": "ɳ", "त": "t̪", "थ": "tʰ", "द": "d", "ध": "d h", "न": "n",
    "प": "p", "फ": "pʰ", "ब": "b", "भ": "b h", "म": "m", "य": "j", "र": "ɾ", "ल": "l", "ळ": "ɭ", "व": "ʋ",
    "श": "ʃ", "ष": "ʂ", "स": "s", "ह": "h",
}
_CTC_VOWEL = {"अ": "ə", "आ": "aː", "इ": "i", "ई": "iː", "उ": "u", "ऊ": "uː", "ऋ": "ɾ u", "ए": "eː", "ऐ": "ə i",
              "ओ": "oː", "औ": "ə u", "ऑ": "ɔ", "ॲ": "ɛ", "ऍ": "ɛ"}
_CTC_MATRA = {"ा": "aː", "ि": "i", "ी": "iː", "ु": "u", "ू": "uː", "ृ": "ɾ u",
              "े": "eː", "ै": "ə i", "ो": "oː", "ौ": "ə u", "ॅ": "ɛ", "ॉ": "ɔ",
              "ॆ": "eː", "ॊ": "oː"}


def ctc_tokens(text: str) -> list[str]:
    """Devanagari text as the recogniser's phoneme tokens (inherent vowel ə; a word-final one is silent)."""
    out: list[str] = []
    for word in unicodedata.normalize("NFC", text).split():
        snd: list[str] = []
        chars = [c for c in word if c != _NUKTA]
        for k, c in enumerate(chars):
            nxt = chars[k + 1] if k + 1 < len(chars) else ""
            if c in _CTC_CONS:
                snd += _CTC_CONS[c].split()
                if nxt not in _CTC_MATRA and nxt != _VIRAMA:
                    snd.append("ə")
            elif c in _CTC_MATRA:
                if snd and snd[-1] == "ə":
                    snd.pop()
                snd += _CTC_MATRA[c].split()
            elif c in _CTC_VOWEL:
                snd += _CTC_VOWEL[c].split()
            elif c in (_ANUSVARA, _CANDRA):
                snd.append("n")
            elif c == _VISARGA:
                snd.append("h")
        if len(snd) > 1 and snd[-1] == "ə":
            snd.pop()
        out += snd
    return out


def ctc_nll(log_probs, text: str, vocab: dict, blank: int) -> float | None:
    """Negative log-likelihood of the text's phonemes under the recogniser's frame log-probabilities (T x V)."""
    import torch
    ids = [vocab[t] for t in ctc_tokens(text) if t in vocab]
    if not ids or len(ids) > log_probs.shape[0]:
        return None
    loss = torch.nn.functional.ctc_loss(log_probs[:, None, :], torch.tensor([ids]), torch.tensor([log_probs.shape[0]]),
                                        torch.tensor([len(ids)]), blank=blank, reduction="sum", zero_infinity=True)
    return float(loss)


def ctc_judge(log_probs, candidates: list[str], vocab: dict, blank: int, margin: float = 0.0):
    """Index of the most likely candidate line, or None when it is not ahead of the first candidate (the standard
    spelling) by more than `margin` nats; and every candidate's negative log-likelihood."""
    nll = [ctc_nll(log_probs, c, vocab, blank) for c in candidates]
    if any(x is None for x in nll):
        return None, nll
    best = min(range(len(nll)), key=nll.__getitem__)
    return (best if best == 0 or nll[0] - nll[best] > margin else None), nll
