"""LLM cleanup (phase 3), called from polish when config/defaults.toml [llm_cleanup].enabled = true.

Local Gemma 3 suggests a corrected version of ONE segment at a time. The suggestion is accepted only if it passes
all 5 gates (FMS rules, config/thresholds.toml [llm_cleanup]). The ASR text is always kept in `text_asr`;
every suggestion (accepted or rejected, with the gate result) is logged in work/llm_cleanup.json.
"""
from __future__ import annotations

import re

from ..common import TRX_HOME, load_config, load_toml, write_json
from ..context import Ctx
from ..llm.gemma3 import LocalLLM
from ..metrics import _edit_distance

BANNED_PREFIX = re.compile(r"^\s*(here is|here's|corrected|sure|okay,? here|the corrected|output|transcript)\b",
                           re.IGNORECASE)
ENTITY = re.compile(r"\d+|[A-Za-z][A-Za-z'\-]+")
TAG = re.compile(r"\[[^\]]*\]")


def gates(original: str, suggestion: str, max_edit: float, min_len: float) -> tuple[bool, str]:
    s = suggestion.strip().strip('"').strip()
    if not s:
        return False, "gate2_empty"
    if BANNED_PREFIX.match(s):
        return False, "gate1_banned_prefix"
    if len(s) < min_len * len(original):
        return False, f"gate3_over_deletion (length ratio {len(s) / max(1, len(original)):.2f})"
    ratio = _edit_distance(original, s) / max(1, len(original))
    if ratio > max_edit:
        return False, f"gate4_edit_ratio {ratio:.2f} > {max_edit}"
    lost = [e for e in ENTITY.findall(original) if e not in s]
    if lost:
        return False, f"gate5_entity_guard (changed/missing: {', '.join(lost[:3])})"
    if TAG.findall(original) != TAG.findall(s):
        return False, "gate5_entity_guard (tags changed)"
    return True, f"accepted (edit ratio {ratio:.2f})"


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
            ok, why = gates(s["text"], sugg, float(thr["max_edit_ratio"]), float(thr["min_length_ratio"]))
            records.append({"id": s["id"], "asr": s["text"], "suggestion": sugg, "accepted": ok, "gate": why})
            if ok and sugg.strip() != s["text"]:
                s["text_asr"] = s["text"]
                s["text"] = sugg.strip().strip('"').strip()
                s.setdefault("edits", []).append({"by": cfg.get("model"), "gate": why})
                accepted += 1
    finally:
        llm.close()
    path = ctx.work / "llm_cleanup.json"
    write_json(path, {"model": cfg.get("model"), "records": records})
    ctx.custody.add(path, "derived", "polish")
    ctx.metrics["llm_edits_accepted"] = accepted
    ctx.metrics["llm_edits_rejected"] = sum(1 for r in records if not r["accepted"])
    ctx.log.add("polish", f"Gemma 3 cleanup: {accepted} edit(s) accepted, {ctx.metrics['llm_edits_rejected']} rejected",
                "The local LLM only suggests; 5 fixed gates decide; the ASR text is kept in text_asr.",
                model=cfg.get("model"))
