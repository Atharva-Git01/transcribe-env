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
from .metrics import cer, self_test, speaker_accuracy, wer
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
        spk = speaker_accuracy(truth.get("segments", []), read_json(tj)["segments"]) if tj.is_file() else None
        rows.append({
            "fixture": f["id"], "exit": name, "verdict": info["verdict"], "recipe": info["recipe"],
            "engines": report["metrics"].get("engines", []), "language": report["metrics"].get("language"),
            "cer": round(cer(truth["reference_text"], hyp), 4), "wer": round(wer(truth["reference_text"], hyp), 4),
            "seconds": round(time.time() - t0, 1), "run_id": info["run_id"],
            "flagged_time_ratio": report["metrics"].get("flagged_time_ratio"),
            "speaker_acc": spk, "speakers_found": report["metrics"].get("speakers_found"),
        })
    result = {"schema": "trx.eval/1", "env_version": env_version(), "utc": utc_now(),
              "overrides": {"engine": engine, "recipe": recipe, "lang": lang}, "rows": rows}
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / f"eval_{result['utc'].replace(':', '').replace('-', '')}_{engine or 'default'}.json"
    write_json(out, result)
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
