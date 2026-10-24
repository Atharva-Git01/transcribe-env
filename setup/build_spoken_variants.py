"""Build the spoken-variant list (PLAN_v0.9 A2): standard spelling -> the forms people write when they write a word
as it is spoken, mined from an IndicVoices slice (CC-BY-4.0), whose clips carry both texts: `normalized` (standard)
and `verbatim` (as spoken).

  python setup/build_spoken_variants.py --slice toolchain/data/indicvoices-mr-slice1 --lang mr

Writes library/spoken_variants/<lang>.toml (read by pipeline/spelling.py) and REVIEW_<lang>.md next to it (the same
list as a table for the owner, decision D3). Setup-time tool, offline. Only generic language data is written: word
pairs seen in a public dataset, with counts; never a clip's text.

Pairs that differ only in what the owner spells one way for one sound (2026-10-24: short or long i and u, the final
anusvara, nukta, chandrabindu) are not variants and are left out, and so are spacing-only differences (English
letters written apart or together).
"""
from __future__ import annotations

import argparse
import difflib
import json
import sys
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

HOME = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HOME))

from pipeline.metrics import _edit_distance, normalise, normalise_level  # noqa: E402

AAHE = {"आहे", "आहेत", "आहेस", "आहोत", "आहात"}   # contractions: "<word> आहे" spoken as one word
AMBIGUOUS_RATIO = 0.05   # a spoken form written at least this often (vs its standard form) is also a word of its own
ASPIRATED = dict(zip("खघछझठढथधफभ", "कगचजटडतदपब"))


def loose(w: str, lang: str) -> str:
    return normalise_level(w, lang, "loose")


def ortho(w: str, lang: str) -> str:
    """The spelling with every same-sound difference folded away (owner: one standard spelling for these)."""
    t = loose(w, lang).replace(" ", "")
    return t[:-1] if t.endswith("ं") else t


def similar(a: str, b: str) -> bool:
    return _edit_distance(a, b) <= max(1, round(0.4 * max(len(a), len(b))))


def use_of(std: str, spk: str, lang: str, counts) -> str:
    """"yes", or "ambiguous" when the spoken form is also an ordinary written word (corpus count >= AMBIGUOUS_RATIO
    of the standard form's, e.g. "kay" = "what" for "kahi" = "some") or a single letter: the metric and the content
    fold cannot tell which word was meant, so they leave such pairs alone."""
    if len(ortho(spk, lang)) <= 1:
        return "ambiguous"
    if counts is not None and " " not in std:
        if counts(spk) >= AMBIGUOUS_RATIO * max(1, counts(std)):
            return "ambiguous"
    return "yes"


def kind(std: str, spk: str, lang: str) -> str:
    if " " in std and " " not in spk:
        return "contraction"
    a, b = loose(std, lang), loose(spk, lang)
    reps = [(a[i1:i2], b[j1:j2]) for op, i1, i2, j1, j2 in
            difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes() if op != "equal"]
    for x, y in reps:
        if "ह" in x and "ह" not in y:
            return "h dropped"
        if ("श" in x or "ष" in x) and "स" in y or "स" in x and "श" in y:
            return "s / sh"
        if "ण" in x and "न" in y:
            return "dental n"
        if "ळ" in x and ("ल" in y or "ड" in y):
            return "retroflex l"
        if any(ASPIRATED.get(c) and ASPIRATED[c] in y for c in x):
            return "aspiration dropped"
    if all(not x for x, _ in reps) or all(not y for _, y in reps):
        return "sound added" if all(not x for x, _ in reps) else "sound dropped"
    return "vowel or other"


def pairs_of(normalized: str, verbatim: str, lang: str) -> list[tuple[str, str]]:
    """(standard, spoken) word pairs where the two texts of one clip differ by more than spelling."""
    n, v = normalise(normalized).split(), normalise(verbatim).split()
    out = []
    sm = difflib.SequenceMatcher(None, [loose(w, lang) for w in n], [loose(w, lang) for w in v], autojunk=False)
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op != "replace":
            continue
        if i2 - i1 == j2 - j1:
            for a, b in zip(n[i1:i2], v[j1:j2]):
                if ortho(a, lang) != ortho(b, lang) and similar(loose(a, lang), loose(b, lang)):
                    out.append((a, b))
        elif (i2 - i1, j2 - j1) == (2, 1) and n[i1 + 1] in AAHE:
            a, b = " ".join(n[i1:i2]), v[j1]
            if loose(b, lang).startswith(loose(n[i1], lang)[:-1] or "\0"):
                out.append((a, b))
    return [(a, b) for a, b in out if a.replace(" ", "") != b.replace(" ", "")]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slice", required=True)
    ap.add_argument("--lang", default="mr")
    ap.add_argument("--min-count", type=int, default=2)
    a = ap.parse_args()
    root = Path(a.slice) if Path(a.slice).is_absolute() else HOME / a.slice
    src = json.loads((root / "SOURCE.json").read_text(encoding="utf-8"))
    count, speakers = Counter(), defaultdict(set)
    clips = 0
    for line in (root / "manifest.jsonl").read_text(encoding="utf-8").splitlines():
        m = json.loads(line)
        clips += 1
        for p in pairs_of(m["normalized"], m["verbatim"], a.lang):
            count[p] += 1
            speakers[p].add(m["speaker_id"])
    kept = sorted((p for p, c in count.items() if c >= a.min_count), key=lambda p: (-count[p], p))
    kinds = Counter(kind(s, v, a.lang) for s, v in kept)
    out_dir = HOME / "library" / "spoken_variants"
    out_dir.mkdir(parents=True, exist_ok=True)
    from pipeline import lm
    model = lm.load(a.lang) if lm.installed(a.lang) else None
    counts = model.count if model is not None else None
    use = {p: use_of(*p, a.lang, counts) for p in kept}
    old = out_dir / f"{a.lang}.toml"
    if old.is_file():   # the owner's review marks survive a rebuild
        import tomllib
        for row in tomllib.loads(old.read_text(encoding="utf-8")).get("pairs", []):
            if len(row) > 5 and row[5] == "not here" and (row[0], row[1]) in use:
                use[(row[0], row[1])] = "not here"
    head = [f"# Spoken-variant list v1 for '{a.lang}' (PLAN_v0.9 A2), built {date.today().isoformat()} by "
            "setup/build_spoken_variants.py.",
            f"# Source: {src['dataset']} at revision {src['revision'][:8]} ({src['credit']}): {clips} clips of "
            f"{src['id']}, `normalized` (standard) vs `verbatim` (as spoken).",
            f"# A pair is listed when it was seen at least {a.min_count} times; spelling-only differences (short/long "
            "i and u, the final anusvara, nukta) are not variants (owner, 2026-10-24).",
            "# Columns: standard, spoken, kind, times seen, speakers, use. use = \"yes\" | \"ambiguous\" (the spoken form is",
            f"# also an ordinary word: corpus count >= {AMBIGUOUS_RATIO} x the standard form's, or one letter) | \"not here\"",
            "# (owner's review, decision D3: not said in the region). pipeline/spelling.py uses only \"yes\" rows.",
            f"# Kinds: {', '.join(f'{k} {n}' for k, n in kinds.most_common())}.", ""]
    rows = [f'  ["{s}", "{v}", "{kind(s, v, a.lang)}", {count[(s, v)]}, {len(speakers[(s, v)])}, "{use[(s, v)]}"],'
            for s, v in kept]
    (out_dir / f"{a.lang}.toml").write_text("\n".join(head + ["pairs = ["] + rows + ["]", ""]), encoding="utf-8")
    md = [f"# Spoken variants ({a.lang}), v1 — for the owner's review (decision D3)", "",
          f"Seen in {clips} IndicVoices clips (as spoken vs standard). Tell Claude the numbers of the rows whose "
          "spoken form is **not** said where your calls come from; they stay listed but are no longer used.", "",
          "Rows marked *ambiguous* are not used: their spoken form is also an ordinary word.", "",
          "| # | standard | spoken | kind | seen | speakers | use |", "|---|---|---|---|---|---|---|"]
    md += [f"| {i} | {s} | {v} | {kind(s, v, a.lang)} | {count[(s, v)]} | {len(speakers[(s, v)])} | {use[(s, v)]} |"
           for i, (s, v) in enumerate(kept, 1)]
    (out_dir / f"REVIEW_{a.lang}.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"{len(kept)} pairs (of {len(count)} seen; {sum(count.values())} occurrences in {clips} clips): "
          + ", ".join(f"{k} {n}" for k, n in kinds.most_common()) + f"; use: {dict(Counter(use.values()))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
