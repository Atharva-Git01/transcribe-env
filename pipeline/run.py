"""The runtime orchestrator: one input -> one run folder. Deterministic, offline, no AI steering.

Order: preflight -> ingest (read-only copy + hash) -> probe -> select -> plan -> stages from the plan -> report.
Every step is logged in decisions.json and every file is hashed into custody.json.
"""
from __future__ import annotations

import json
import os
import shutil
import stat
import subprocess
from pathlib import Path

from .common import (RUNS_DIR, TrxError, env_version, load_config, local_stamp, safe_case_name, sha256_file,
                     utc_now, write_json)
from .context import Ctx
from .guards import offline
from .recipes import build_plan
from .records import Custody, Decisions
from .schema_check import validate
from .stages import IMPLEMENTED
from .stages.probe import _URL, probe
from .stages.select import select

RUN_GITIGNORE = "input/\nwork/\n"


def preflight() -> dict:
    ok, detail = offline.self_test()
    if not ok:
        raise TrxError("OFFLINE_GUARD_FAILED", detail)
    min_gb = float(load_config("defaults")["runtime"]["min_free_gb"])
    free_gb = shutil.disk_usage(RUNS_DIR).free / 1e9
    if free_gb < min_gb:
        raise TrxError("DISK_LOW", f"{free_gb:.1f} GB free, need {min_gb} GB")
    return {"offline": detail, "free_gb": round(free_gb, 1)}


def _validated_write(path: Path, data, schema: str) -> None:
    errors = validate(data, schema)
    if errors:
        raise TrxError("SCHEMA_INVALID", f"{path.name}: " + "; ".join(errors[:5]))
    write_json(path, data)


def _git_commit(run_dir: Path, message: str) -> str:
    def git(*args):
        return subprocess.run(["git", *args], cwd=run_dir, capture_output=True, text=True, timeout=60)
    try:
        if not (run_dir / ".git").exists():
            git("init", "-q")
        git("add", "-A")
        res = git("commit", "-q", "-m", message)
        return "committed" if res.returncode == 0 else f"not committed: {res.stderr.strip()[:120]}"
    except Exception as e:  # noqa: BLE001
        return f"not committed: {e}"


def _env_commit() -> str:
    try:
        r = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=RUNS_DIR.parent, capture_output=True,
                           text=True, timeout=20)
        dirty = subprocess.run(["git", "status", "--porcelain", "--", "config", "recipes", "pipeline", "prompts",
                                "library"], cwd=RUNS_DIR.parent, capture_output=True, text=True, timeout=20).stdout
        return (r.stdout.strip() or "unknown") + ("+uncommitted-changes" if dirty.strip() else "")
    except Exception:  # noqa: BLE001
        return "unknown"


def _write_report(run_dir: Path, run_id: str, verdict: str, reasons: list, exit_name: str, plan: dict | None,
                  passport: dict | None, custody: Custody | None, run_metrics: dict | None = None) -> None:
    stages = [{"name": s["name"], "status": s["status"], "available_from_phase": s["available_from_phase"]}
              for s in (plan or {}).get("stages", [])]
    outputs = {p.name: f"out/{p.name}" for p in sorted((run_dir / "out").glob("*")) if p.is_file()}
    metrics = {} if passport is None else {
        "duration_sec": passport["container"]["duration_sec"],
        "snr_db_estimate": passport["noise"]["snr_db_estimate"],
        "bandwidth": passport["bandwidth"]["class"],
        "layout": passport["channels"]["layout_class"],
    }
    metrics.update(run_metrics or {})
    metrics["env_commit"] = _env_commit()
    report = {
        "schema": "trx.report/1", "env_version": env_version(), "run_id": run_id, "verdict": verdict,
        "reasons": reasons, "exit_name": exit_name,
        "recipe": (plan or {}).get("recipe", {}).get("id", "none"),
        "stages": stages, "outputs": outputs, "metrics": metrics,
    }
    _validated_write(run_dir / "report.json", report, "report")
    lines = [f"# Run {run_id}", "", f"**Verdict: {verdict}**  (exit: {exit_name}, env {env_version()})", ""]
    lines += ["## Why"] + [f"- {r}" for r in reasons] + [""]
    if passport:
        lines += ["## The audio", f"- {passport['container']['duration_sec']} s, {passport['stream']['codec']}, "
                  f"{passport['stream']['sample_rate']} Hz, layout: {passport['channels']['layout_class']}",
                  f"- band: {passport['bandwidth']['class']}, SNR estimate: {passport['noise']['snr_db_estimate']} dB, "
                  f"loudness: {passport['levels']['integrated_lufs']} LUFS"]
        lines += [f"- warning: {w}" for w in passport["warnings"]] + [""]
    if plan:
        lines += ["## Plan", f"- recipe `{plan['recipe']['id']}` (rule {plan['selected_by']['rule_id']}: "
                  f"{plan['selected_by']['reason']})"]
        lines += [f"- {s['name']}: {s['status']}" + (f" (arrives in phase {s['available_from_phase']})"
                  if s["status"] == "not_installed" else "") for s in plan["stages"]] + [""]
    lines += ["## Records", "- passport.json, plan.json, decisions.json, custody.json (sha256 of every file)", ""]
    (run_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    if custody:
        custody.add(run_dir / "report.json", "record", "report")


class RunLock:
    """Held for as long as a run executes. The OS releases it when the process ends (also on a crash or power cut),
    so a run folder with state 'running' and a free lock is an interrupted run that can be resumed."""

    def __init__(self, run_dir: Path):
        self.path = run_dir / "work" / ".lock"
        self.fh = None

    def acquire(self) -> bool:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fh = open(self.path, "a+")
        try:
            if os.name == "nt":
                import msvcrt
                fh.seek(0)
                msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            fh.close()
            return False
        self.fh = fh
        return True

    def release(self) -> None:
        if self.fh:
            try:
                if os.name == "nt":
                    import msvcrt
                    self.fh.seek(0)
                    msvcrt.locking(self.fh.fileno(), msvcrt.LK_UNLCK, 1)
            except OSError:
                pass
            self.fh.close()
            self.fh = None


def _state_path(run_dir: Path) -> Path:
    return run_dir / "work" / "run_state.json"


def _write_state(run_dir: Path, **kv) -> None:
    p = _state_path(run_dir)
    data = json.loads(p.read_text(encoding="utf-8")) if p.is_file() else {}
    data.update(kv)
    write_json(p, data)


def find_unfinished(input_path: str, overrides: dict, runs_dir: Path = RUNS_DIR) -> Path | None:
    """An interrupted run of the same file with the same options (language: the run's own, unless one is given)."""
    src = Path(input_path).expanduser()
    if not src.is_file():
        return None
    sha = None
    for st in sorted(runs_dir.glob("*/work/run_state.json"), reverse=True):
        try:
            data = json.loads(st.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        if data.get("state") != "running":
            continue
        sha = sha or sha256_file(src)
        if data.get("input_sha256") != sha:
            continue
        theirs, mine = dict(data.get("overrides", {})), dict(overrides)
        if "lang" not in mine:
            theirs.pop("lang", None)
        if theirs != mine:
            continue
        run_dir = st.parent.parent
        if not (run_dir / "plan.json").is_file():   # stopped before planning: nothing to resume, start afresh
            continue
        lock = RunLock(run_dir)
        if lock.acquire():   # nobody holds it: the process that ran it is gone
            lock.release()
            return run_dir
    return None


def transcribe(input_path: str, forced_recipe: str | None = None, style: str | None = None,
               lang: str | None = None, case: str | None = None, engine: str | None = None,
               runs_dir: Path = RUNS_DIR, lang_how: str | None = None, engine_how: str | None = None,
               hotwords: list[str] | None = None) -> tuple[str, dict]:
    pre = preflight()
    if _URL.match(str(input_path)):
        raise TrxError("INPUT_UNREADABLE", "URLs are not allowed; give a local file path")
    src = Path(input_path).expanduser()
    if not src.is_file():
        raise TrxError("INPUT_NOT_FOUND", str(src))
    if style and style not in ("verbatim", "readable"):
        raise TrxError("USAGE", "--style must be verbatim or readable")

    run_id = f"{safe_case_name(case or src.stem)}-{local_stamp()}"
    run_dir = runs_dir / run_id
    for sub in ("input", "work", "out"):
        (run_dir / sub).mkdir(parents=True, exist_ok=False)
    (run_dir / ".gitignore").write_text(RUN_GITIGNORE, encoding="utf-8")
    lock = RunLock(run_dir)
    lock.acquire()
    log = Decisions(run_dir)
    log.add("preflight", "checked offline guard and disk", "A run starts only when it is provably offline "
            "and has disk space.", **pre)

    original_hash = sha256_file(src)
    overrides = {k: v for k, v in {"recipe": forced_recipe, "style": style, "lang": lang, "engine": engine,
                                   "hotwords": hotwords}.items() if v}
    _write_state(run_dir, state="running", input_sha256=original_hash, overrides=overrides, started_utc=utc_now(),
                 env_version=env_version())
    custody = Custody(run_dir, run_id, src.resolve(), original_hash, src.stat().st_size)
    copy = run_dir / "input" / src.name
    shutil.copy2(src, copy)
    if sha256_file(copy) != original_hash:
        raise TrxError("INTERNAL_ERROR", "input copy hash differs from the original")
    os.chmod(copy, stat.S_IREAD)
    custody.add(copy, "input_copy", "ingest")
    log.add("ingest", "copied the input read-only and verified its sha256",
            "The original is never touched; all work happens on a verified copy.",
            original=str(src.resolve()), sha256=original_hash)
    if lang_how:
        log.add("language", f"language {lang or 'auto'}: {lang_how}",
                "Owner rule: the measured language is shown and confirmed before transcribing (pipeline/langcheck.py).")
    if engine_how:
        log.add("engine", f"engine override {engine}: {engine_how}", "Chosen by the operator for this file only.")
    if hotwords:
        log.add("hotwords", f"{len(hotwords)} case term(s) given to the recogniser",
                "Names/places of this case help the engine spell them; they never add words that were not said.",
                terms=hotwords)

    plan = passport = None
    try:
        from .langcheck import cache_path, cached_passport
        passport = cached_passport(original_hash)
        if passport:  # measured during the language check on a verified copy with the same sha256
            cache_path(original_hash).unlink(missing_ok=True)
            how = "reused the measurement from the language check (verified copy, same sha256)"
        else:
            passport = probe(copy)
            how = "measured the audio"
        _validated_write(run_dir / "passport.json", passport, "passport")
        custody.add(run_dir / "passport.json", "record", "probe")
        log.add("probe", how, "Recipe choice is based on measurements, not guesses.",
                layout=passport["channels"]["layout_class"], band=passport["bandwidth"]["class"],
                snr_db=passport["noise"]["snr_db_estimate"], warnings=passport["warnings"])

        selection = select(passport, forced_recipe)
        log.add("select", f"chose recipe '{selection['recipe']}'", selection["reason"], rule=selection["rule_id"])

        plan = build_plan(run_id, selection, overrides)
        _validated_write(run_dir / "plan.json", plan, "plan")
        custody.add(run_dir / "plan.json", "record", "plan")
        log.add("plan", "wrote plan.json", "plan.json is the only source of parameters for every stage.",
                stages=[s["name"] for s in plan["stages"]])
    except Exception as e:  # noqa: BLE001
        return _finish(run_dir, run_id, lock, log, custody, plan, passport, *_failure(e, run_dir, log), {})
    return _execute(run_dir, run_id, lock, log, custody, plan, passport, copy)


def resume(run_dir: Path) -> tuple[str, dict]:
    """Continue an interrupted run in its own folder with its own plan.json (never re-planned). Stages re-run from the
    start (they are deterministic and seeded); recognition skips every chunk already saved in work/recognise_chunks/."""
    pre = preflight()
    lock = RunLock(run_dir)
    if not lock.acquire():
        raise TrxError("USAGE", f"run {run_dir.name} is still running in another window")
    run_id = run_dir.name
    log = Decisions.load(run_dir)
    custody = Custody.load(run_dir)
    plan = json.loads((run_dir / "plan.json").read_text(encoding="utf-8"))
    passport = json.loads((run_dir / "passport.json").read_text(encoding="utf-8"))
    for st in plan["stages"]:
        if st["status"] in ("done", "skipped"):
            st["status"] = "pending"
    copy = next(p for p in (run_dir / "input").iterdir() if p.is_file())
    if sha256_file(copy) != custody.data["original"]["sha256"]:
        lock.release()
        raise TrxError("INTERNAL_ERROR", "the run's input copy no longer matches the original sha256")
    _write_state(run_dir, state="running", resumed_utc=utc_now())
    log.add("resume", "resumed an interrupted run", "The previous process stopped before finishing; saved work "
            "(denoising, diarization, recognition chunks) is reused, everything else is redone exactly as planned.", **pre)
    return _execute(run_dir, run_id, lock, log, custody, plan, passport, copy)


def _failure(e: Exception, run_dir: Path, log: Decisions) -> tuple[str, str, list]:
    if isinstance(e, offline.NetworkBlocked):
        return "NETWORK_BLOCKED", "FAIL", [str(e)]
    if isinstance(e, TrxError):
        log.add("run", "failed", e.detail or e.name)
        return e.name, "FAIL", [f"{e.name}: {e.detail}"]
    import traceback
    (run_dir / "work" / "traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
    log.add("run", "crashed", f"{type(e).__name__}: {e}", traceback="work/traceback.txt")
    return "INTERNAL_ERROR", "FAIL", [f"{type(e).__name__}: {e}"]


def _execute(run_dir, run_id, lock, log, custody, plan, passport, copy) -> tuple[str, dict]:
    run_metrics: dict = {}
    try:
        ctx = Ctx(run_dir=run_dir, input_file=copy, passport=passport, plan=plan, log=log, custody=custody)
        exit_name, reasons = "OK", []
        for st in plan["stages"]:
            if st["status"] == "not_installed" and st["params"].get("required", True) is False:
                st["status"] = "skipped"
                log.add(st["name"], "skipped", "Optional stage (required = false) not installed in this phase; "
                        "the run continues without it.", available_from_phase=st["available_from_phase"])
                continue
            if st["status"] == "not_installed":
                exit_name = "STAGE_NOT_INSTALLED"
                reasons.append(f"Stopped at stage '{st['name']}': it arrives in phase {st['available_from_phase']}.")
                log.add(st["name"], "stopped", "This stage is not installed in the current environment phase.",
                        available_from_phase=st["available_from_phase"])
                break
            IMPLEMENTED[st["name"]](ctx, st["params"])
            st["status"] = "done"
        _validated_write(run_dir / "plan.json", plan, "plan")
        if exit_name == "OK":
            verdict, reasons = ctx.verdict, list(ctx.verdict_reasons)
            if verdict == "FAIL":
                exit_name = "GATE_FAILED"
        else:
            verdict = "FAIL"
        run_metrics.update(ctx.metrics, language=ctx.language, engines=[e["engine"] for e in ctx.engines_used])
    except Exception as e:  # noqa: BLE001 — never lose the run record on an unexpected error
        exit_name, verdict, reasons = _failure(e, run_dir, log)
    return _finish(run_dir, run_id, lock, log, custody, plan, passport, exit_name, verdict, reasons, run_metrics)


def _finish(run_dir, run_id, lock, log, custody, plan, passport, exit_name, verdict, reasons, run_metrics):
    _write_report(run_dir, run_id, verdict, reasons, exit_name, plan, passport, custody, run_metrics)
    errors = validate(log.items, "decisions")
    if errors:
        exit_name = "SCHEMA_INVALID"
    _write_state(run_dir, state="finished", finished_utc=utc_now(), exit=exit_name, verdict=verdict)
    lock.release()
    git_state = _git_commit(run_dir, f"trx {env_version()}: {verdict} ({exit_name})")
    return exit_name, {"run_id": run_id, "run_dir": str(run_dir), "verdict": verdict,
                       "recipe": (plan or {}).get("recipe", {}).get("id"), "reasons": reasons, "git": git_state}
