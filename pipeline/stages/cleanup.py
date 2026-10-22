"""LLM cleanup (phase 3), called from polish when config/defaults.toml [llm_cleanup].enabled = true.

Local Gemma 3 suggests a corrected version of ONE segment at a time. Two layers decide (E17):
1. whole-suggestion gates: not empty, no "Here is…" preamble, no over-deletion, numbers/Latin names/tags kept;
2. a WORD gate: the suggestion is aligned word by word with what was heard, and a word may change only if the change
   is a small spelling fix (<= max_word_edit_ratio of its letters), punctuation, or a listed Hindi -> Marathi form
   (library/lang_forms/<lang>.toml). No word is added or removed, no other script appears, no Hindi form is
   introduced into Marathi. Everything else keeps the heard word, so a "fix" that changes meaning cannot get through.
The ASR text is always kept in `text_asr`; every suggestion and every word decision is logged in
work/llm_cleanup.json.
"""
from __future__ import annotations

import re
from functools import lru_cache

from ..common import TRX_HOME, load_config, load_toml, write_json
from ..context import Ctx
from ..llm.gemma3 import LocalLLM
from ..metrics import _edit_distance

BANNED_PREFIX = re.compile(r"^\s*(here is|here's|corrected|sure|okay,? here|the corrected|output|transcript)\b",
                           re.IGNORECASE)
ENTITY = re.compile(r"\d+|[A-Za-z][A-Za-z'\-]+")
TAG = re.compile(r"\[[^\]]*\]")
PUNCT = re.compile(r"[\.,!?;:\"'()…।॥]+")
LATIN = re.compile(r"[A-Za-z]")
FOREIGN = re.compile(r"[\u0980-\u0dff\u0600-\u06ff]")   # Bengali … Sinhala, Arabic: not expected in mr/hi text


def gates(original: str, suggestion: str, min_len: float) -> tuple[bool, str]:
    """Whole-suggestion checks (layer 1)."""
    s = suggestion.strip().strip('"').strip()
    if not s:
        return False, "gate_empty"
    if BANNED_PREFIX.match(s):
        return False, "gate_banned_prefix"
    if len(s) < min_len * len(original):
        return False, f"gate_over_deletion (length ratio {len(s) / max(1, len(original)):.2f})"
    lost = [e for e in ENTITY.findall(original) if e not in s]
    if lost:
        return False, f"gate_entity_guard (changed/missing: {', '.join(lost[:3])})"
    if TAG.findall(original) != TAG.findall(s):
        return False, "gate_entity_guard (tags changed)"
    return True, "passed"


@lru_cache(maxsize=8)
def lang_forms(lang: str) -> tuple[frozenset, frozenset]:
    p = TRX_HOME / "library" / "lang_forms" / f"{lang}.toml"
    if not p.is_file():
        return frozenset(), frozenset()
    d = load_toml(p)
    return frozenset(d.get("hindi_forms", [])), frozenset(tuple(x) for x in d.get("form_map", []))


def _bare(w: str) -> str:
    return PUNCT.sub("", w)


def _close(a: str, b: str, max_rel: float) -> bool:
    """Acceptance test for a spelling fix: at most max_rel of the heard word's letters change."""
    x, y = _bare(a), _bare(b)
    return _edit_distance(x, y) / max(1, len(x)) <= max_rel


def _alignable(a: str, b: str) -> bool:
    """Looser test used only to line words up (about a third of the letters may differ)."""
    x, y = _bare(a), _bare(b)
    return _edit_distance(x, y) <= max(1, len(x) // 3)


def align_words(a: list[str], b: list[str], max_rel: float) -> list[tuple[str | None, str | None]]:
    """Word-level Levenshtein alignment; a near-identical pair costs 0.5 so spelling fixes align as substitutions."""
    n, m = len(a), len(b)

    def sub(x, y):
        return 0.0 if x == y else (0.5 if _alignable(x, y) else 1.0)
    D = [[0.0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        D[i][0] = float(i)
    for j in range(m + 1):
        D[0][j] = float(j)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            D[i][j] = min(D[i - 1][j] + 1, D[i][j - 1] + 1, D[i - 1][j - 1] + sub(a[i - 1], b[j - 1]))
    i, j, out = n, m, []
    while i > 0 or j > 0:
        if i > 0 and j > 0 and D[i][j] == D[i - 1][j - 1] + sub(a[i - 1], b[j - 1]):
            out.append((a[i - 1], b[j - 1]))
            i, j = i - 1, j - 1
        elif i > 0 and D[i][j] == D[i - 1][j] + 1:
            out.append((a[i - 1], None))
            i -= 1
        else:
            out.append((None, b[j - 1]))
            j -= 1
    return out[::-1]


def word_gate(original: str, suggestion: str, lang: str, max_rel: float, use_lm: bool = True) -> tuple[str, list[list]]:
    """Layer 2: keep only the allowed word changes. Returns (final text, [[heard, suggested, decision], …]).
    With a language model for `lang` (toolchain/models/lm-<lang>-v1), each allowed change is applied one at a time
    and kept only if the line does not read as less natural <lang> (per-word LM score does not drop)."""
    hindi, form_map = lang_forms(lang)
    words, log, pending = [], [], []      # pending: (position, suggested word, decision text)
    for wa, wb in align_words(original.split(), suggestion.split(), max_rel):
        if wa is None:
            log.append([None, wb, "rejected: added word"])
            continue
        words.append(wa)
        if wb is None:
            log.append([wa, None, "rejected: removed word"])
            continue
        if wa == wb:
            continue
        ca, cb = _bare(wa), _bare(wb)
        if ca == cb:
            pending.append((len(words) - 1, wb, "accepted: punctuation"))
        elif LATIN.search(cb) and not LATIN.search(ca) or FOREIGN.search(cb) and not FOREIGN.search(ca):
            log.append([wa, wb, "rejected: another script"])
        elif lang == "mr" and cb in hindi and ca not in hindi:
            log.append([wa, wb, "rejected: introduces a Hindi form"])
        elif (ca, cb) in form_map:
            pending.append((len(words) - 1, wb, "accepted: Hindi form -> Marathi form"))
        elif _close(wa, wb, max_rel):
            pending.append((len(words) - 1, wb, "accepted: spelling"))
        else:
            log.append([wa, wb, "rejected: different word (meaning may change)"])
    lm = None
    if use_lm and pending:
        from .. import lm as lmmod
        lm = lmmod.load(lang) if lmmod.installed(lang) else None
    for pos, wb, decision in pending:
        heard = words[pos]
        if lm is not None and decision != "accepted: punctuation":
            trial = words[:pos] + [wb] + words[pos + 1:]
            if lm.per_word(" ".join(trial)) < lm.per_word(" ".join(words)) - 1e-9:
                log.append([heard, wb, "rejected: language model prefers the heard form"])
                continue
            decision += " (language model agrees)"
        words[pos] = wb
        log.append([heard, wb, decision])
    return " ".join(words), log


def run(ctx: Ctx, cfg: dict) -> None:
    thr = load_config("thresholds")["llm_cleanup"]
    langs = set(cfg.get("languages", ["mr", "hi"]))
    style = ctx.plan["output"].get("style", "verbatim")
    todo = [s for s in ctx.segments if s.get("language") in langs and s["text"].strip()]
    if not todo:
        ctx.log.add("polish", "LLM cleanup: nothing to do", f"No segments in languages {sorted(langs)}.")
        return
    llm = LocalLLM(cfg.get("model", "gemma3_4b_it_q4"), int(cfg.get("gpu_layers", 99)),
                   int(cfg.get("ctx_size", 4096)), int(load_config("defaults")["runtime"]["seed"]))
    records, accepted = [], 0
    try:
        for s in todo:
            p = load_toml(TRX_HOME / "prompts" / "gemma3_cleanup" / f"{s['language']}.toml")
            system = p["system"] + ("\n" + p.get("verbatim_rule", "") if style == "verbatim" else "")
            shots = []
            for ex in p.get("example", []):
                shots += [("user", ex["input"]), ("assistant", ex["output"])]
            sugg = llm.chat(system, shots + [("user", s["text"])], max_tokens=max(64, 3 * len(s["text"])))
            ok, why = gates(s["text"], sugg, float(thr["min_length_ratio"]))
            final, words = (word_gate(s["text"], sugg.strip().strip('"').strip(), s["language"],
                                      float(thr["max_word_edit_ratio"])) if ok else (s["text"], []))
            changed = final != s["text"]
            records.append({"id": s["id"], "asr": s["text"], "suggestion": sugg, "accepted": changed,
                            "gate": why if not ok else ("word gate: some edits kept" if changed else "word gate: no edit kept"),
                            "final": final, "words": words})
            if changed:
                s["text_asr"] = s["text"]
                s["text"] = final
                s.setdefault("edits", []).append({"by": cfg.get("model"), "gate": "word gate",
                                                  "words": [w for w in words if w[2].startswith("accepted")]})
                accepted += 1
    finally:
        llm.close()
    path = ctx.work / "llm_cleanup.json"
    write_json(path, {"model": cfg.get("model"), "records": records})
    ctx.custody.add(path, "derived", "polish")
    ctx.metrics["llm_edits_accepted"] = accepted
    ctx.metrics["llm_edits_rejected"] = sum(1 for r in records if not r["accepted"])
    ctx.log.add("polish", f"Gemma 3 cleanup: {accepted} edit(s) accepted, {ctx.metrics['llm_edits_rejected']} rejected",
                "The local LLM only suggests; whole-suggestion gates + a word gate decide; the ASR text is kept in text_asr.",
                model=cfg.get("model"))
