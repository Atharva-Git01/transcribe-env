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

from .common import TRX_HOME, TrxError, env_version, read_json, utc_now, write_json
from .metrics import cer, cer_level, self_test, speaker_accuracy, turn_alignment, wer
from .stages import IMPLEMENTED

FIXTURES = TRX_HOME / "evals" / "fixtures"
EVAL_RUNS = TRX_HOME / "evals" / "runs"
RESULTS = TRX_HOME / "evals" / "results"
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
            row.update(ok=not problems, truth=truth, detail="; ".join(problems) or
                       f"{truth.get('language')} · {len(truth['reference_text'])} chars · {truth.get('notes', '')[:60]}")
        except Exception as e:  # noqa: BLE001
            row["detail"] = f"truth.json unreadable: {e}"
        rows.append(row)
    return rows


def hypothesis_text(run_dir) -> str:
    p = run_dir / "out" / "transcript.json"
    if not p.is_file():
        return ""
    return " ".join(s["text"] for s in read_json(p)["segments"])


def _write_comparison(path, result: dict) -> None:
    """Human-readable per-turn comparison (truth vs output) for every fixture of this eval."""
    lines = [f"# Eval {result['utc']} — environment {result['env_version']}", "",
             "| fixture | engine | CER | normalised | loose | std | WER | spk | missing turns | Hindi forms | RTF | verdict |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in result["rows"]:
        lines.append(f"| {r['fixture']} | {'+'.join(r['engines'])} | {r['cer']} | {r['cer_norm']} | {r['cer_loose']} | "
                     f"{r['cer_std'] if r['cer_std'] is not None else '-'} | {r['wer']} | "
                     f"{r['speaker_acc'] if r['speaker_acc'] is not None else '-'} | {r['missing_turns']}/{r['turns']} | "
                     f"{r['hindi_forms'] if r['hindi_forms'] is not None else '-'} | {r['rtf']} | {r['verdict']} |")
    for r in result["rows"]:
        if not r.get("turn_rows"):
            continue
        lines += ["", f"## {r['fixture']}", "", "| # | truth speaker | truth | output | output speakers | turn CER |",
                  "|---|---|---|---|---|---|"]
        for t in r["turn_rows"]:
            lines.append(f"| {t['turn']} | {t['ref_speaker']} | {t['ref']} | {t['hyp'] or '— missing —'} | "
                         f"{', '.join(t['hyp_speakers']) or '-'} | {t['cer']} |")
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
        spk = speaker_accuracy(truth.get("segments", []), hyp_segs) if hyp_segs else None
        lang = truth.get("language", "")
        turns = turn_alignment(truth.get("segments", []), hyp_segs) if truth.get("segments") else []
        from .stages.cleanup import lang_forms
        hindi = lang_forms("mr")[0] if lang == "mr" else frozenset()
        hindi_forms = sum(1 for w in hyp.split() if w.strip(".,?!।") in hindi)
        std_ref = truth.get("reference_text_standard")
        rows.append({
            "fixture": f["id"], "exit": name, "verdict": info["verdict"], "recipe": info["recipe"],
            "engines": report["metrics"].get("engines", []), "language": report["metrics"].get("language"),
            "cer": round(cer(truth["reference_text"], hyp), 4), "wer": round(wer(truth["reference_text"], hyp), 4),
            "seconds": round(time.time() - t0, 1), "run_id": info["run_id"],
            "flagged_time_ratio": report["metrics"].get("flagged_time_ratio"),
            "speaker_acc": spk, "speakers_found": report["metrics"].get("speakers_found"),
            "cer_norm": round(cer_level(truth["reference_text"], hyp, lang, "norm"), 4),
            "cer_loose": round(cer_level(truth["reference_text"], hyp, lang, "loose"), 4),
            "cer_std": round(cer_level(std_ref, hyp, lang, "norm"), 4) if std_ref else None,
            "missing_turns": sum(1 for t in turns if not t["hyp"]), "turns": len(turns),
            "hindi_forms": hindi_forms if lang == "mr" else None,
            "rtf": round((time.time() - t0) / max(1e-6, float(truth.get("duration_sec") or 1)), 2),
            "turn_rows": turns,
        })
    result = {"schema": "trx.eval/1", "env_version": env_version(), "utc": utc_now(),
              "overrides": {"engine": engine, "recipe": recipe, "lang": lang}, "rows": rows}
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / f"eval_{result['utc'].replace(':', '').replace('-', '')}_{engine or 'default'}.json"
    write_json(out, result)
    _write_comparison(out.with_suffix(".md"), result)
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
            base["scores"][r["key"]] = {"cer": r["cer"], "wer": r["wer"], "speaker_acc": r.get("speaker_acc"),
                                        "env_version": env_version(), "utc": result["utc"]}
        base["env_version"] = env_version()
        write_json(BASELINE, base)
    return result
