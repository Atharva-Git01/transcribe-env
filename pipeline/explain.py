"""trx explain <run>: turn a run's records into plain sentences anyone (or any model) can repeat accurately."""
from __future__ import annotations

from .common import RUNS_DIR, TrxError, read_json


def explain(run_id: str) -> list[str]:
    run_id = run_id.replace("\\", "/").rstrip("/").split("/")[-1]
    run = RUNS_DIR / run_id
    if not run.is_dir():
        raise TrxError("RUN_NOT_FOUND", run_id)
    lines = [f"Run {run_id}"]
    if (run / "report.json").is_file():
        rep = read_json(run / "report.json")
        lines.append(f"Verdict: {rep['verdict']} (exit {rep['exit_name']}, environment {rep['env_version']}).")
        lines += [f"  Reason: {r}" for r in rep["reasons"]]
    if (run / "passport.json").is_file():
        p = read_json(run / "passport.json")
        lines.append(
            f"The input '{p['source']['file_name']}' is {p['container']['duration_sec']} s of {p['stream']['codec']} "
            f"audio at {p['stream']['sample_rate']} Hz; channel layout: {p['channels']['layout_class']}; "
            f"bandwidth: {p['bandwidth']['class']} (95% of energy below {p['bandwidth']['rolloff95_hz_median']:.0f} Hz); "
            f"estimated SNR {p['noise']['snr_db_estimate']} dB; loudness {p['levels']['integrated_lufs']} LUFS.")
        lines += [f"  Warning: {w}" for w in p["warnings"]]
    if (run / "decisions.json").is_file():
        lines.append("What happened, in order:")
        for d in read_json(run / "decisions.json"):
            lines.append(f"  {d['seq']}. [{d['stage']}] {d['action']} — because: {d['reason']}")
    if (run / "custody.json").is_file():
        c = read_json(run / "custody.json")
        lines.append(f"Original sha256 {c['original']['sha256']}; {len(c['entries'])} files hashed in custody.json.")
    return lines
