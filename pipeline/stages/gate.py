"""Stage: gate. Verdict PASS / REVIEW / FAIL from fixed thresholds (config/thresholds.toml [gate])."""
from __future__ import annotations

from ..common import load_config
from ..context import Ctx


def run(ctx: Ctx, params: dict) -> None:
    g = load_config("thresholds")[params.get("thresholds", "gate")]  # recipe names the thresholds.toml section
    soft = tuple(g.get("soft_flags", []))
    total = sum(s["end"] - s["start"] for s in ctx.segments) or 0.0
    flagged = sum(s["end"] - s["start"] for s in ctx.segments if s["flags"])
    hard = sum(s["end"] - s["start"] for s in ctx.segments if any(not f.startswith(soft) for f in s["flags"]))
    ratio = flagged / total if total else 0.0
    hard_ratio = hard / total if total else 0.0
    reasons = []
    verdict = "PASS"
    if not ctx.segments and g.get("fail_if_no_text", True):
        verdict = "FAIL"
        reasons.append("No text was recognised.")
    elif hard_ratio > g["fail_if_flagged_ratio_above"]:
        verdict = "FAIL"
        reasons.append(f"{hard_ratio:.0%} of transcribed time has hard flags (loops, low confidence, wrong script …; "
                       f"limit {g['fail_if_flagged_ratio_above']:.0%}).")
    else:
        if ratio > g["review_if_flagged_ratio_above"]:
            verdict = "REVIEW"
            reasons.append(f"{ratio:.0%} of transcribed time is flagged (review above "
                           f"{g['review_if_flagged_ratio_above']:.0%}).")
        if ctx.metrics.get("possible_omissions"):
            verdict = "REVIEW"
            reasons.append(f"{ctx.metrics['possible_omissions']} stretch(es) of speech produced no text.")
        amb = [w for w in ctx.passport.get("warnings", []) if w.startswith("language_ambiguous")]
        if amb and ctx.language_source != "forced":
            verdict = "REVIEW"
            reasons.append(amb[0])
    if verdict == "PASS":
        reasons.append(f"{len(ctx.segments)} segments; {ratio:.0%} of transcribed time flagged.")
    ctx.verdict, ctx.verdict_reasons = verdict, reasons
    ctx.metrics["flagged_time_ratio"] = round(ratio, 4)
    ctx.metrics["hard_flagged_time_ratio"] = round(hard_ratio, 4)
    ctx.log.add("gate", f"verdict {verdict}", "; ".join(reasons), flagged_time_ratio=round(ratio, 4),
                hard_flagged_time_ratio=round(hard_ratio, 4))
