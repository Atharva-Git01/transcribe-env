"""trx eval: score the pipeline on evals/fixtures against human ground truth, compare with evals/baseline.json.

Fixture layout: evals/fixtures/<id>/truth.json (schema trx.truth/1) + the audio file it names.
In phase 0 the fixtures and the metric are checked; scoring needs the recognise stage (phase 1).
"""
from __future__ import annotations

from .common import TRX_HOME, TrxError, read_json
from .metrics import self_test
from .stages import IMPLEMENTED

FIXTURES = TRX_HOME / "evals" / "fixtures"


def list_fixtures() -> list[dict]:
    rows = []
    for d in sorted(p for p in FIXTURES.iterdir() if p.is_dir()):
        row = {"id": d.name, "ok": False, "detail": ""}
        truth_path = d / "truth.json"
        try:
            truth = read_json(truth_path)
            problems = []
            if truth.get("schema") != "trx.truth/1":
                problems.append("schema must be trx.truth/1")
            if not (d / truth.get("audio", "")).is_file():
                problems.append(f"audio file '{truth.get('audio')}' missing")
            if not truth.get("reference_text", "").strip():
                problems.append("empty reference_text")
            row.update(ok=not problems, detail="; ".join(problems) or
                       f"{truth.get('language')} · {len(truth['reference_text'])} chars · {truth.get('notes', '')[:60]}")
        except Exception as e:  # noqa: BLE001
            row["detail"] = f"truth.json unreadable: {e}"
        rows.append(row)
    return rows


def run_eval() -> dict:
    ok, detail = self_test()
    if not ok:
        raise TrxError("INTERNAL_ERROR", f"metric self-test failed: {detail}")
    fixtures = list_fixtures()
    bad = [f for f in fixtures if not f["ok"]]
    if bad:
        raise TrxError("CONFIG_INVALID", "; ".join(f"{f['id']}: {f['detail']}" for f in bad))
    if "recognise" not in IMPLEMENTED:
        raise TrxError("STAGE_NOT_INSTALLED", f"{len(fixtures)} fixture(s) valid, metric self-test passed; "
                       "scoring needs the recognise stage (phase 1)", fixtures=[f["id"] for f in fixtures])
    raise TrxError("INTERNAL_ERROR", "scoring loop not implemented yet")  # replaced in phase 1
