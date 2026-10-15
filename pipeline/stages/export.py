"""Stage: export. transcript .txt .srt .vtt .json + review_list.md (+ readable variant when --style readable)."""
from __future__ import annotations

import re

from ..audio import fmt_ts
from ..common import env_version, load_config, write_json
from ..context import Ctx

FILLERS = {"अं", "हं", "अम्", "उम्", "umm", "um", "uh", "hmm", "हम्म"}


def _line_text(s: dict, review_tag: str) -> str:
    return s["text"] + (f" {review_tag}" if s["flags"] else "")


def _readable(text: str) -> str:
    toks = [t for t in text.split() if t.strip(".,?!।").lower() not in FILLERS]
    return re.sub(r"\s+", " ", " ".join(toks)).strip()


def run(ctx: Ctx, params: dict) -> None:
    out_cfg = ctx.plan["output"]
    style = out_cfg.get("style", "verbatim")
    tag = out_cfg.get("review_tag", "[review]")
    formats = out_cfg.get("formats", ["txt", "srt", "vtt", "json"]) if params.get("formats", "defaults") == "defaults" \
        else params["formats"]
    src = ctx.passport["source"]
    engines = ", ".join(f"{e['engine']} ({e['device']}/{e['compute_type']})" for e in ctx.engines_used)
    header = [
        f"Transcript — run {ctx.run_dir.name}",
        f"Source: {src['file_name']}  sha256 {src['sha256']}",
        f"Duration: {ctx.passport['container']['duration_sec']} s · Language: {ctx.language} ({ctx.language_source})",
        f"Recipe: {ctx.plan['recipe']['id']} v{ctx.plan['recipe']['version']} · Engine: {engines} · Style: {style}",
        f"Verdict: {ctx.verdict} · Environment {env_version()}",
        "Lines marked [review] are uncertain; see review_list.md. Speaker SPK? = not determined.",
        "",
    ]
    written = {}
    variants = [("transcript", lambda s: _line_text(s, tag))]
    if style == "readable":
        variants.append(("transcript_readable", lambda s: _readable(_line_text(s, tag))))
    for base, fn in variants:
        if "txt" in formats:
            p = ctx.out / f"{base}.txt"
            lines = [f"[{fmt_ts(s['start'], '.')} - {fmt_ts(s['end'], '.')}] {s['speaker']}: {fn(s)}" for s in ctx.segments]
            p.write_text("\n".join(header + lines) + "\n", encoding="utf-8")
            written[p.name] = p
        if "srt" in formats:
            p = ctx.out / f"{base}.srt"
            blocks = [f"{i}\n{fmt_ts(s['start'])} --> {fmt_ts(s['end'])}\n{s['speaker']}: {fn(s)}\n"
                      for i, s in enumerate(ctx.segments, 1)]
            p.write_text("\n".join(blocks), encoding="utf-8")
            written[p.name] = p
        if "vtt" in formats:
            p = ctx.out / f"{base}.vtt"
            blocks = ["WEBVTT\n"] + [f"{fmt_ts(s['start'], '.')} --> {fmt_ts(s['end'], '.')}\n<v {s['speaker']}>{fn(s)}\n"
                                     for s in ctx.segments]
            p.write_text("\n".join(blocks), encoding="utf-8")
            written[p.name] = p
    if "json" in formats:
        p = ctx.out / "transcript.json"
        write_json(p, {"schema": "trx.transcript/1", "run_id": ctx.run_dir.name, "env_version": env_version(),
                       "source": src, "language": ctx.language, "language_source": ctx.language_source,
                       "recipe": ctx.plan["recipe"], "engines": ctx.engines_used, "verdict": ctx.verdict,
                       "segments": ctx.segments})
        written[p.name] = p
    rl = ctx.out / "review_list.md"
    rows = ["# Review list", "", f"Run {ctx.run_dir.name} · verdict {ctx.verdict} · {len(ctx.review)} item(s)", ""]
    if ctx.review:
        rows += ["| Time | Speaker | Text | Why |", "|---|---|---|---|"]
        for r in ctx.review:
            text = (r["text"] or "—").replace("|", "/")
            rows.append(f"| {fmt_ts(r['start'], '.')}–{fmt_ts(r['end'], '.')} | {r['speaker']} | {text} | "
                        f"{'; '.join(r['reasons']).replace('|', '/')} |")
    else:
        rows.append("Nothing flagged.")
    rl.write_text("\n".join(rows) + "\n", encoding="utf-8")
    written[rl.name] = rl
    for p in written.values():
        ctx.custody.add(p, "output", "export")
    ctx.log.add("export", f"wrote {', '.join(sorted(written))}", "Deliverables in out/; every file hashed in custody.json.",
                style=style)
