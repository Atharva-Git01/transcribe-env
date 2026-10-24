"""Names and rare words written in more than one way in one transcript (PLAN_v0.9 B4a, failure F17).

Each turn is decoded on its own, so the same name can come out in two or three spellings across a call. They are
listed in review_list.md so the reviewer can make them one; the transcript is never changed. Only rare words count
(seen fewer than RARE_COUNT times in the language's corpus): common words rightly take several forms. A case ending
("-la", "-cha" … the endings of library/scoring/<lang>.toml) is taken off before comparing, and two spellings belong
together when they sound the same (spelling.ortho) or, for longer words, differ in one letter. With --hotwords the
listed spelling is shown next to the group.
"""
from __future__ import annotations

from collections import defaultdict

RARE_COUNT = 50          # corpus count below which a word may be a name (the lexicon has ~60-80 M words)
MIN_LETTERS = 3
ONE_EDIT_MIN = 5         # spellings one letter apart are grouped only from this length on


def stem(word: str, endings: tuple) -> str:
    """The word without its longest case ending, when at least MIN_LETTERS letters remain."""
    for e in sorted(endings, key=len, reverse=True):
        if word.endswith(e) and len(word) - len(e) >= MIN_LETTERS:
            return word[:-len(e)]
    return word


def written_ways(segments: list[dict], lang: str, count, hotwords=(), endings: tuple = ()) -> list[dict]:
    """[{"spellings": [(stem, times, first start), …], "listed": hotword or None}] for every rare word that the
    transcript writes in at least two ways, most frequent first. count(word) is the corpus count (lexicon)."""
    from .lm import tokens
    from .metrics import _edit_distance
    from .spelling import ortho
    seen: dict[str, list] = defaultdict(list)
    for s in segments:
        for w in tokens(s.get("text_standard", s.get("text", ""))):
            if len(w) < MIN_LETTERS or not all("ऀ" <= c <= "ॿ" for c in w):
                continue
            st = stem(w, endings)
            if count(w) < RARE_COUNT and count(st) < RARE_COUNT:
                seen[st].append(s.get("start", 0.0))
    stems = sorted(seen)
    key = {st: ortho(st, lang) for st in stems}
    parent = {st: st for st in stems}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for i, a in enumerate(stems):
        for b in stems[i + 1:]:
            ka, kb = key[a], key[b]
            if ka == kb or (min(len(ka), len(kb)) >= ONE_EDIT_MIN and abs(len(ka) - len(kb)) <= 1
                            and _edit_distance(ka, kb) <= 1):
                parent[find(a)] = find(b)
    groups: dict[str, list] = defaultdict(list)
    for st in stems:
        groups[find(st)].append(st)
    hot = {ortho(h, lang): h for h in hotwords}
    out = []
    for members in groups.values():
        if len(members) < 2:
            continue
        spellings = sorted(((st, len(seen[st]), min(seen[st])) for st in members), key=lambda x: (-x[1], x[2]))
        listed = next((hot[key[st]] for st in members if key[st] in hot), None)
        out.append({"spellings": spellings, "listed": listed})
    return sorted(out, key=lambda g: (-sum(x[1] for x in g["spellings"]), g["spellings"][0][2]))
