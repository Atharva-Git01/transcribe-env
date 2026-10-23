"""trx eval: run the real pipeline on every fixture (offline), score CER/WER against human ground truth and compare
with evals/baseline.json. Eval runs go to evals/runs/ (never runs/, which is evidence).

  trx eval                         default config
  trx eval --engine <id>           force one ASR engine for all languages
  trx eval --recipe <id>           force one recipe
  trx eval --fixture <id>          only one fixture
  trx eval --save-baseline         write the result as the new baseline (maintainer, after an accepted change)
"""
from __future__ import annotations

import time

from .common import TRX_HOME, TrxError, env_version, load_config, read_json, utc_now, write_json
from .metrics import (TURN_KINDS, cer, cer_level, line_scores, meaning_errors, self_test, speaker_accuracy,
                      turn_alignment, turn_outcomes, wer)
from .stages import IMPLEMENTED

FIXTURES = TRX_HOME / "evals" / "fixtures"
EVAL_RUNS = TRX_HOME / "evals" / "runs"
RESULTS = TRX_HOME / "evals" / "results"              # pushed to GitHub: numbers only, never a fixture's words
COMPARE = TRX_HOME / "evals" / "compare"              # per-turn text (truth vs output): this PC only (git-ignored)
BASELINE = TRX_HOME / "evals" / "baseline.json"
NOISE_MARGIN = 0.01


def list_fixtures() -> list[dict]:
    rows = []
    for d in sorted(p for p in FIXTURES.iterdir() if p.is_dir()):
        row = {"id": d.name, "ok": False, "detail": "", "dir": d}
        try:
            truth = read_json(d / "truth.json")
            problems = []
            if truth.get("schema") != "trx.truth/1":
                problems.append("schema must be trx.truth/1")
            if not (d / truth.get("audio", "")).is_file():
                problems.append(f"audio file '{truth.get('audio')}' missing")
            if not truth.get("reference_text", "").strip():
                problems.append("empty reference_text")
            until = truth.get("scored_until_sec")
            if until is not None and not (isinstance(until, (int, float)) and until > 0):
                problems.append("scored_until_sec must be a positive number of seconds")
            row.update(ok=not problems, truth=truth, detail="; ".join(problems) or
                       f"{truth.get('language')} · {len(truth['reference_text'])} chars · "
                       f"{f'scored 0-{until:.0f} s · ' if until else ''}{truth.get('notes', '')[:60]}")
        except Exception as e:  # noqa: BLE001
            row["detail"] = f"truth.json unreadable: {e}"
        rows.append(row)
    return rows


def hypothesis_text(run_dir) -> str:
    p = run_dir / "out" / "transcript.json"
    if not p.is_file():
        return ""
    return " ".join(s["text"] for s in read_json(p)["segments"])


def turns_without_text(turns: list[dict]) -> list[dict]:
    """Turn rows as they may leave this PC: no words of the recording, and the truth's speaker names (often real
    people) become A, B, C … in order of first appearance."""
    names: dict = {}
    for t in turns:
        names.setdefault(t.get("ref_speaker"), chr(ord("A") + len(names)) if len(names) < 26 else f"S{len(names) + 1}")
    return [dict({k: v for k, v in t.items() if k not in ("ref", "hyp")}, ref_speaker=names[t.get("ref_speaker")],
                 missing=t["missing"] if "missing" in t else not t.get("hyp")) for t in turns]


def failure_monitor(truth: dict, hyp: str, hyp_segs: list[dict], lang: str, turns: list[dict]) -> dict:
    """HOW the transcript went wrong (E39), numbers only: the CER left once spelling conventions are folded away,
    what happened to each turn (also written into the turn rows), negations and numbers lost or added, short lines
    (< 2 s) against long ones, and how many of the bad lines (> 40 % wrong) carry a review flag."""
    ref = truth["reference_text"]
    mon = {"cer_content": round(cer_level(ref, hyp, lang, "content"), 4)}
    if truth.get("segments") and hyp_segs:
        kinds = turn_outcomes(truth["segments"], hyp_segs, lang)
        for t, k in zip(turns, kinds):
            t["what_happened"] = k
        mon["turn_outcomes"] = {k: kinds.count(k) for k in TURN_KINDS if k in kinds}
    mon.update(meaning_errors(ref, hyp, lang))
    lines = list(zip(line_scores(ref, hyp_segs, lang), hyp_segs)) if hyp_segs else []

    def cer_of(sel):
        n = sum(sc["ref"] for sc, _ in sel)
        return round(sum(sc["err"] for sc, _ in sel) / n, 4) if n else None
    mon["short_line_cer"] = cer_of([p for p in lines if p[1].get("end", 0) - p[1].get("start", 0) < 2.0])
    mon["long_line_cer"] = cer_of([p for p in lines if p[1].get("end", 0) - p[1].get("start", 0) >= 2.0])
    info = set(load_config("thresholds")["gate"].get("info_flags", []))
    bad = [s for sc, s in lines if sc["ref"] >= 8 and sc["err"] > 0.4 * sc["ref"]]
    mon["bad_lines"] = len(bad)
    mon["bad_lines_flagged"] = sum(1 for s in bad if any(f.split(":")[0].split(" (")[0] not in info
                                                         for f in s.get("flags", [])))
    return mon


def monitor_line(m: dict) -> str:
    """One line for the console and the comparison files."""
    t = m.get("turn_outcomes", {})
    turns = (f"turns: {t.get('correct', 0)} correct, {t.get('misheard', 0)} misheard, {t.get('merged', 0)} merged, "
             f"{t.get('lost', 0)} lost (+{t.get('lost_backchannel', 0)} backchannels), "
             f"{t.get('wrong_speaker', 0) + t.get('merged_other', 0)} wrong speaker | ") if t else ""
    return (f"content CER {m['cer_content']} | {turns}negations {m['negation_errors']} wrong of {m['negations']} | "
            f"numbers {m['number_errors']} wrong of {m['numbers']} | lines < 2 s CER {m['short_line_cer']}, "
            f"longer {m['long_line_cer']} | bad lines flagged {m['bad_lines_flagged']}/{m['bad_lines']}")


def _write_comparison(path, result: dict, text: bool) -> None:
    """Per-turn comparison for every fixture of this eval. text=True (evals/compare/, this PC only) shows the truth and
    the output side by side; text=False (evals/results/, pushed) keeps only speaker letters and per-turn CER."""
    lines = [f"# Eval {result['utc']} — environment {result['env_version']}", "",
             "| fixture | engine | CER | normalised | loose | std | WER | spk | missing turns | Hindi forms | RTF | verdict |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in result["rows"]:
        lines.append(f"| {r['fixture']} | {'+'.join(r['engines'])} | {r['cer']} | {r['cer_norm']} | {r['cer_loose']} | "
                     f"{r['cer_std'] if r['cer_std'] is not None else '-'} | {r['wer']} | "
                     f"{r['speaker_acc'] if r['speaker_acc'] is not None else '-'} | {r['missing_turns']}/{r['turns']} | "
                     f"{r['hindi_forms'] if r['hindi_forms'] is not None else '-'} | {r['rtf']} | {r['verdict']} |")
    if any(r.get("monitor") for r in result["rows"]):
        lines += ["", "## Failure monitor (how it went wrong, E39)", ""]
        lines += [f"- **{r['fixture']}**: {monitor_line(r['monitor'])}" for r in result["rows"] if r.get("monitor")]
    for r in result["rows"]:
        if not r.get("turn_rows"):
            continue
        if text:
            lines += ["", f"## {r['fixture']}", "",
                      "| # | truth speaker | truth | output | output speakers | turn CER | what happened |",
                      "|---|---|---|---|---|---|---|"]
            for t in r["turn_rows"]:
                lines.append(f"| {t['turn']} | {t['ref_speaker']} | {t['ref']} | {t['hyp'] or '— missing —'} | "
                             f"{', '.join(t['hyp_speakers']) or '-'} | {t['cer']} | {t.get('what_happened', '-')} |")
        else:
            lines += ["", f"## {r['fixture']}", "", "| # | truth speaker | output speakers | turn CER | what happened |",
                      "|---|---|---|---|---|"]
            for t in r["turn_rows"]:
                missing = t["missing"] if "missing" in t else not t.get("hyp")
                lines.append(f"| {t['turn']} | {t['ref_speaker']} | "
                             f"{'— missing —' if missing else ', '.join(t['hyp_speakers']) or '-'} | {t['cer']} | "
                             f"{t.get('what_happened', '-')} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_eval(engine: str | None = None, recipe: str | None = None, fixture: str | None = None,
             save_baseline: bool = False, lang: str | None = None) -> dict:
    ok, detail = self_test()
    if not ok:
        raise TrxError("INTERNAL_ERROR", f"metric self-test failed: {detail}")
    fixtures = [f for f in list_fixtures() if not fixture or f["id"] == fixture]
    if not fixtures:
        raise TrxError("USAGE", f"no fixture '{fixture}'")
    bad = [f for f in fixtures if not f["ok"]]
    if bad:
        raise TrxError("CONFIG_INVALID", "; ".join(f"{f['id']}: {f['detail']}" for f in bad))
    if "recognise" not in IMPLEMENTED:
        raise TrxError("STAGE_NOT_INSTALLED", "scoring needs the recognise stage (phase 1)")
    from .run import transcribe
    EVAL_RUNS.mkdir(parents=True, exist_ok=True)
    rows = []
    for f in fixtures:
        truth = f["truth"]
        t0 = time.time()
        name, info = transcribe(str(f["dir"] / truth["audio"]), forced_recipe=recipe, case=f"eval_{f['id']}",
                                engine=engine, lang=lang, runs_dir=EVAL_RUNS)
        hyp = hypothesis_text(EVAL_RUNS / info["run_id"])
        report = read_json(EVAL_RUNS / info["run_id"] / "report.json")
        tj = EVAL_RUNS / info["run_id"] / "out" / "transcript.json"
        hyp_segs = read_json(tj)["segments"] if tj.is_file() else []
        until = truth.get("scored_until_sec")
        if until:   # the truth covers only the start of the recording (E39): score the lines that start before it
            hyp_segs = [s for s in hyp_segs if s["start"] < until]
            hyp = " ".join(s["text"] for s in hyp_segs)
        spk = speaker_accuracy(truth.get("segments", []), hyp_segs) if hyp_segs else None
        # the truth's language, for scoring only: `lang` stays the operator's override for every fixture (E30)
        t_lang = truth.get("language", "")
        turns = turn_alignment(truth.get("segments", []), hyp_segs) if truth.get("segments") else []
        from .stages.cleanup import lang_forms
        hindi = lang_forms("mr")[0] if t_lang == "mr" else frozenset()
        hindi_forms = sum(1 for w in hyp.split() if w.strip(".,?!।") in hindi)
        std_ref = truth.get("reference_text_standard")
        rows.append({
            "fixture": f["id"], "exit": name, "verdict": info["verdict"], "recipe": info["recipe"],
            "engines": report["metrics"].get("engines", []), "language": report["metrics"].get("language"),
            "cer": round(cer(truth["reference_text"], hyp), 4), "wer": round(wer(truth["reference_text"], hyp), 4),
            "seconds": round(time.time() - t0, 1), "run_id": info["run_id"],
            "flagged_time_ratio": report["metrics"].get("flagged_time_ratio"),
            "speaker_acc": spk, "speakers_found": report["metrics"].get("speakers_found"),
            "cer_norm": round(cer_level(truth["reference_text"], hyp, t_lang, "norm"), 4),
            "cer_loose": round(cer_level(truth["reference_text"], hyp, t_lang, "loose"), 4),
            "cer_std": round(cer_level(std_ref, hyp, t_lang, "norm"), 4) if std_ref else None,
            "missing_turns": sum(1 for t in turns if not t["hyp"]), "turns": len(turns), "scored_until_sec": until,
            "hindi_forms": hindi_forms if t_lang == "mr" else None,
            "rtf": round((time.time() - t0) / max(1e-6, float(truth.get("duration_sec") or 1)), 2),
            "monitor": failure_monitor(truth, hyp, hyp_segs, t_lang, turns),
            "turn_rows": turns,
        })
    result = {"schema": "trx.eval/1", "env_version": env_version(), "utc": utc_now(),
              "overrides": {"engine": engine, "recipe": recipe, "lang": lang}, "rows": rows}
    RESULTS.mkdir(parents=True, exist_ok=True)
    COMPARE.mkdir(parents=True, exist_ok=True)
    out = RESULTS / f"eval_{result['utc'].replace(':', '').replace('-', '')}_{engine or 'default'}.json"
    public = dict(result, rows=[dict(r, turn_rows=turns_without_text(r["turn_rows"])) for r in rows])
    write_json(out, public)
    _write_comparison(out.with_suffix(".md"), public, text=False)
    _write_comparison(COMPARE / out.with_suffix(".md").name, result, text=True)
    result["file"] = str(out.relative_to(TRX_HOME))
    base = read_json(BASELINE) if BASELINE.is_file() else {"scores": {}}
    regressions = []
    for r in rows:
        key = f"{r['fixture']}|{r['recipe']}|{'+'.join(r['engines'])}|{r['language']}"
        r["key"] = key
        prev = base.get("scores", {}).get(key)
        r["baseline_cer"] = prev["cer"] if prev else None
        if prev and r["cer"] > prev["cer"] + NOISE_MARGIN:
            regressions.append(f"{key}: CER {prev['cer']} -> {r['cer']}")
    result["regressions"] = regressions
    if save_baseline:
        base.setdefault("scores", {})
        for r in rows:
            if r["exit"] not in ("OK", "GATE_FAILED") or not r["engines"]:   # a run that stopped is no baseline
                r["baseline_saved"] = False
                continue
            base["scores"][r["key"]] = {"cer": r["cer"], "wer": r["wer"], "speaker_acc": r.get("speaker_acc"),
                                        "env_version": env_version(), "utc": result["utc"]}
        base["env_version"] = env_version()
        write_json(BASELINE, base)
    return result
