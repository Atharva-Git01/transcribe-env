"""Stage: export. transcript .txt .srt .vtt .json + review_list.md (+ readable variant when --style readable).

Spelling (owner, 2026-10-23; PLAN_v0.9 A1): the main transcript is written as spoken and transcript_standard.txt keeps
the recogniser's standard spelling for searching and quoting; every segment carries both (`text`, `text_standard`).
Until an as-spoken spelling step is adopted the two are the same text, and the header says so.
review_list.md also lists names and rare words written in more than one way (PLAN_v0.9 B4a)."""
from __future__ import annotations

import re

from ..audio import fmt_ts
from ..common import env_version, load_config, write_json
from ..context import Ctx

FILLERS = {"अं", "हं", "अम्", "उम्", "umm", "um", "uh", "hmm", "हम्म"}


def _counted(flags: list[str]) -> list[str]:
    """Flags that make a line [review]; info flags (e.g. unknown_words) are listed in review_list.md only."""
    info = tuple(load_config("thresholds")["gate"].get("info_flags", []))
    return [f for f in flags if not (info and f.startswith(info))]


def _line_text(s: dict, review_tag: str) -> str:
    return s["text"] + (f" {review_tag}" if _counted(s["flags"]) else "")


def _spelling_line(spelling: str, respelled: int, step: str | None) -> str:
    if spelling != "as_spoken":
        return f"Spelling: {spelling}"
    if not step:
        return ("Spelling: the recogniser's standard spelling — no as-spoken spelling step is adopted yet (PLAN_v0.9); "
                "transcript_standard.txt holds the same text.")
    return (f"Spelling: as spoken ({step}); {respelled} line(s) differ from the recogniser's standard spelling, "
            "which transcript_standard.txt keeps.")


def _names_section(ctx: Ctx) -> list[str]:
    """review_list.md rows for names/rare words the transcript writes in several ways (needs the lexicon)."""
    import tomllib

    from .. import lm
    from ..common import TRX_HOME
    from ..names import written_ways
    lang = ctx.language or ""
    if not lang or not lm.installed(lang):
        return []
    f = TRX_HOME / "library" / "scoring" / f"{lang}.toml"
    endings = tuple(tomllib.loads(f.read_text(encoding="utf-8")).get("number_endings", [])) if f.is_file() else ()
    hot = ctx.stage_params("recognise").get("hotwords") or []
    groups = written_ways(ctx.segments, lang, lm.load(lang).count, hot, endings)
    ctx.metrics["names_written_several_ways"] = len(groups)
    if not groups:
        return []
    rows = ["", "## Names or rare words written in more than one way", "",
            "The same name may have come out in different spellings in different turns. Check which one is right; "
            "the transcript is not changed.", "", "| Spellings (times, first at) | Listed spelling (--hotwords) |",
            "|---|---|"]
    for g in groups[:40]:
        ways = " · ".join(f"{st} ({n}×, {fmt_ts(t, '.')})" for st, n, t in g["spellings"])
        rows.append(f"| {ways} | {g['listed'] or '—'} |")
    return rows


def _readable(text: str) -> str:
    toks = [t for t in text.split() if t.strip(".,?!।").lower() not in FILLERS]
    return re.sub(r"\s+", " ", " ".join(toks)).strip()


def run(ctx: Ctx, params: dict) -> None:
    out_cfg = ctx.plan["output"]
    style = out_cfg.get("style", "verbatim")
    spelling = out_cfg.get("spelling", "standard")
    for s in ctx.segments:                   # the standard copy: the recogniser's spelling (PLAN_v0.9 A1)
        s.setdefault("text_standard", s["text"])
    respelled = sum(1 for s in ctx.segments if s["text_standard"] != s["text"])
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
        _spelling_line(spelling, respelled, ctx.metrics.get("spelling_step")),
        "Lines marked [review] are uncertain; review_list.md lists them and also words the lexicon does not know. "
        "Speakers: SPK1, SPK2 … (by voice), SPK? = not determined.",
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
    if spelling == "as_spoken" and "txt" in formats:
        p = ctx.out / "transcript_standard.txt"
        lines = [f"[{fmt_ts(s['start'], '.')} - {fmt_ts(s['end'], '.')}] {s['speaker']}: {s['text_standard']}"
                 + (f" {tag}" if _counted(s["flags"]) else "") for s in ctx.segments]
        std_header = header[:1] + ["Standard-spelling copy of transcript.txt (the recogniser's own spelling), for "
                                   "searching and quoting. The main transcript is written as spoken."] + header[1:]
        p.write_text("\n".join(std_header + lines) + "\n", encoding="utf-8")
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
    rows += _names_section(ctx)
    rl.write_text("\n".join(rows) + "\n", encoding="utf-8")
    written[rl.name] = rl
    for p in written.values():
        ctx.custody.add(p, "output", "export")
    ctx.log.add("export", f"wrote {', '.join(sorted(written))}", "Deliverables in out/; every file hashed in custody.json.",
                style=style, spelling=spelling, lines_respelled=respelled,
                names_written_several_ways=ctx.metrics.get("names_written_several_ways", 0))
