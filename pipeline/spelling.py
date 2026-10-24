"""As-spoken spelling (PLAN_v0.9): the spoken-variant list (A2) and the spoken-form metric.

library/spoken_variants/<lang>.toml lists standard spellings with the forms people write when they write a word as it
is spoken (mined from IndicVoices by setup/build_spoken_variants.py; the owner reviews it, decision D3). Only rows
with use = "yes" count: an "ambiguous" spoken form is also an ordinary word of its own, "not here" is the owner's
"not said in my region".

spoken_forms() scores a transcript against a truth written as spoken (the owner's convention, 2026-10-23):
  spoken slot    a truth word in a listed spoken form   -> hit (the transcript wrote it that way too) or miss
  standard slot  a truth word (or two, "X aahe") in a standard form that has a listed spoken variant
                                                        -> kept, or switched (a spoken form the speaker did not say)
Each slot is compared with the transcript words that hold the same words once spelling is folded away (the "content"
level, match_blocks). A slot whose words the transcript does not have is "unmatched": misheard or lost words belong to
the content CER, not to the spelling. Same-sound spellings are one spelling here (ortho): short or long i and u, the
final anusvara, a nasal consonant + virama before a consonant (= anusvara), nukta, chandrabindu, and spacing.
"""
from __future__ import annotations

import difflib
import re

_VARIANTS: dict = {}
_NASAL = re.compile("[ङञणनम]्(?=[क-ह])")


def ortho(word: str, lang: str = "") -> str:
    """Words with every same-sound spelling difference folded away (module docstring)."""
    from .metrics import normalise_level
    return _ortho_of_loose(normalise_level(word, lang, "loose"))


def _ortho_of_loose(t: str) -> str:
    t = _NASAL.sub("ं", t.replace(" ", ""))
    return t[:-1] if t.endswith("ं") else t


def variants(lang: str) -> dict:
    """{"to_standard": ortho(spoken) -> standard text, "standard_words": ortho(standard word) -> {ortho(spoken)},
    "standard_phrases": (ortho(w1), ortho(w2)) -> {ortho(spoken)}, "pairs": n}; empty without a list."""
    if lang in _VARIANTS:
        return _VARIANTS[lang]
    import tomllib
    from .common import TRX_HOME
    p = TRX_HOME / "library" / "spoken_variants" / f"{lang}.toml"
    rows = []
    if lang and p.is_file():
        try:
            rows = [r for r in tomllib.loads(p.read_text(encoding="utf-8")).get("pairs", [])
                    if len(r) >= 6 and r[5] == "yes"]
        except (OSError, ValueError):
            rows = []
    out = {"to_standard": {}, "standard_words": {}, "standard_phrases": {}, "pairs": len(rows)}
    for std, spk, *_ in rows:
        key = ortho(spk, lang)
        out["to_standard"].setdefault(key, std)
        words = tuple(ortho(w, lang) for w in std.split())
        table = out["standard_words"] if len(words) == 1 else out["standard_phrases"]
        table.setdefault(words[0] if len(words) == 1 else words, set()).add(key)
    _VARIANTS[lang] = out
    return out


def standardise_loose(text: str, lang: str) -> str:
    """Loose-folded text with every listed spoken form replaced by its standard words (also loose-folded): the
    content level of metrics.normalise_level uses it, so a spoken form and its standard spelling are one word."""
    table = variants(lang)["to_standard"] if lang else {}
    if not table:
        return text
    from .metrics import normalise_level
    out = []
    for w in text.split():
        std = table.get(_ortho_of_loose(w))
        out.append(normalise_level(std, lang, "loose") if std else w)
    return " ".join(out)


def content_word(w: str, lang: str) -> str:
    """One word at the content level; "·" for a word that folds away (a filler, a bare "aahe")."""
    from .metrics import normalise_level
    return normalise_level(w, lang, "content").replace(" ", "") or "·"


def match_blocks(ref: list[str], hyp: list[str], lang: str) -> list[tuple[int, int, int, int]]:
    """Blocks (i1, i2, j1, j2): ref[i1:i2] and hyp[j1:j2] hold the same words once spelling is folded away — one word
    each, or a run written apart on one side and together on the other ("X la" / "Xla", "mahiti aahe" / "mahitiye").
    Words that fold away next to a block (a bare "aahe", a filler) join it."""
    rf, hf = [content_word(w, lang) for w in ref], [content_word(w, lang) for w in hyp]
    blocks: list[tuple[int, int, int, int]] = []
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, rf, hf, autojunk=False).get_opcodes():
        if op == "equal":
            blocks += [(i1 + k, i1 + k + 1, j1 + k, j1 + k + 1) for k in range(i2 - i1)]
        elif op == "replace" and "".join(rf[i1:i2]).replace("·", "") == "".join(hf[j1:j2]).replace("·", ""):
            blocks.append((i1, i2, j1, j2))
        elif op in ("insert", "delete") and blocks and blocks[-1][1] == i1 and blocks[-1][3] == j1 \
                and all(x == "·" for x in (hf[j1:j2] if op == "insert" else rf[i1:i2])):
            a1, _, b1, _ = blocks[-1]
            blocks[-1] = (a1, i2, b1, j2)
    return blocks


def match_words(ref: list[str], hyp: list[str], lang: str) -> dict[int, str]:
    """For each reference word, the hypothesis word(s) of its block, joined by spaces. Unmatched words are absent."""
    out = {}
    for i1, i2, j1, j2 in match_blocks(ref, hyp, lang):
        out.update({i: " ".join(hyp[j1:j2]) for i in range(i1, i2)})
    return out


def block_spelling(ref: list[str], hyp: list[str], block: tuple[int, int, int, int], lang: str) -> tuple[str, str]:
    """(ortho of the block's reference words, ortho of its hypothesis words)."""
    i1, i2, j1, j2 = block
    return ortho(" ".join(ref[i1:i2]), lang), ortho(" ".join(hyp[j1:j2]), lang)


def spoken_forms(reference: str, hypothesis: str, lang: str, detail: bool = False) -> dict | None:
    """Spoken-form recall and wrong switches of a hypothesis against a truth written as spoken (module docstring).
    None when the language has no variant list."""
    v = variants(lang)
    if not v["pairs"]:
        return None
    from .metrics import normalise
    ref, hyp = normalise(reference).split(), normalise(hypothesis).split()
    covering = {}
    for b in match_blocks(ref, hyp, lang):
        covering.update({i: b for i in range(b[0], b[1])})
    counts = {"spoken": 0, "hits": 0, "misses": 0, "spoken_unmatched": 0,
              "standard": 0, "kept": 0, "switched": 0, "other_spelling": 0, "standard_unmatched": 0}
    rows = []
    o_ref = [ortho(w, lang) for w in ref]
    i = 0
    while i < len(ref):
        o, span, spoken_of = o_ref[i], 1, None
        if o in v["to_standard"]:
            kind = "spoken"
        elif i + 1 < len(ref) and (o, o_ref[i + 1]) in v["standard_phrases"]:
            kind, span, spoken_of = "standard", 2, v["standard_phrases"][(o, o_ref[i + 1])]
        elif o in v["standard_words"]:
            kind, spoken_of = "standard", v["standard_words"][o]
        else:
            i += 1
            continue
        b = covering.get(i)
        tb, hb = block_spelling(ref, hyp, b, lang) if b else (None, None)
        if kind == "spoken":
            outcome = "spoken_unmatched" if b is None else "hits" if hb == tb else "misses"
        else:
            outcome = "standard_unmatched" if b is None else "kept" if hb == tb else \
                "switched" if hb in spoken_of else "other_spelling"
        counts[kind] += 1
        counts[outcome] += 1
        if detail:
            rows.append({"truth": " ".join(ref[i:i + span]), "hyp": " ".join(hyp[b[2]:b[3]]) if b else None,
                         "outcome": outcome})
        i += span
    matched_spoken = counts["hits"] + counts["misses"]
    matched_standard = counts["kept"] + counts["switched"]
    counts["recall"] = round(counts["hits"] / matched_spoken, 4) if matched_spoken else None
    counts["switch_rate"] = round(counts["switched"] / matched_standard, 4) if matched_standard else None
    if detail:
        counts["rows"] = rows
    return counts


def transfer(standard: str, verbatim: str, lang: str) -> tuple[str, list[tuple[str, str]]]:
    """Spelling transfer (PLAN_v0.9 section 6): the words of `standard` (the recogniser's transcript) with the
    spelling of `verbatim` (a model that writes as spoken) wherever both hold the same words once spelling is folded
    away. A word that differs is never taken from `verbatim`; fillers keep the recogniser's spelling. Returns the new
    text and every change as (standard words, as-spoken words)."""
    from .metrics import normalise
    std, ver = standard.split(), normalise(verbatim).split()
    std_n = [normalise(w) for w in std]
    out, changes, last = [], [], 0
    for i1, i2, j1, j2 in match_blocks(std_n, ver, lang):
        if all(content_word(w, lang) == "·" for w in std_n[i1:i2]):
            continue
        a, b = block_spelling(std_n, ver, (i1, i2, j1, j2), lang)
        if a != b:
            out += std[last:i1] + ver[j1:j2]
            changes.append((" ".join(std[i1:i2]), " ".join(ver[j1:j2])))
            last = i2
    out += std[last:]
    return " ".join(out), changes
