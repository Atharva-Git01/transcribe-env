#!/usr/bin/env python
"""trx — the only entry point of the transcription environment. Run `trx help`.

Every command prints human-readable output, then ONE JSON status line (see config/exit_codes.toml).
Runtime commands install the offline guard first; nothing here ever reaches the internet.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
for stream in (sys.stdout, sys.stderr):
    try:
        stream.reconfigure(encoding="utf-8", errors="backslashreplace")
    except Exception:  # noqa: BLE001
        pass

from pipeline.common import TrxError, env_version, out, status_line, write_json  # noqa: E402
from pipeline.guards import offline  # noqa: E402

offline.install()

HELP = """trx {v} — local, offline transcription environment

  trx transcribe <file> [--recipe ID] [--style verbatim|readable] [--lang mr|hi|en] [--case NAME]
                                   full run: copy + hash, probe, choose recipe, plan, run stages, report
  trx watch [--once] [--interval SECONDS]
                                   process every new file dropped into inbox/
  trx probe <file> [--out FILE]    measure the audio only (writes a passport)
  trx explain <run-id>             plain-language account of a run
  trx eval                         score the pipeline on evals/fixtures (maintainers)
  trx doctor [--offline] [--brief] readiness + offline proof
  trx version | trx help

Exit codes and the exact next step for each: config/exit_codes.toml. Operator checklist: OPERATOR.md."""


class Parser(argparse.ArgumentParser):
    def error(self, message):
        raise TrxError("USAGE", message)


def build_parser() -> Parser:
    p = Parser(prog="trx", add_help=False)
    sub = p.add_subparsers(dest="cmd")
    t = sub.add_parser("transcribe", add_help=False)
    t.add_argument("file")
    t.add_argument("--recipe")
    t.add_argument("--style", choices=["verbatim", "readable"])
    t.add_argument("--lang")
    t.add_argument("--case")
    w = sub.add_parser("watch", add_help=False)
    w.add_argument("--once", action="store_true")
    w.add_argument("--interval", type=int, default=10)
    pr = sub.add_parser("probe", add_help=False)
    pr.add_argument("file")
    pr.add_argument("--out")
    e = sub.add_parser("explain", add_help=False)
    e.add_argument("run_id")
    sub.add_parser("eval", add_help=False)
    d = sub.add_parser("doctor", add_help=False)
    d.add_argument("--offline", action="store_true")
    d.add_argument("--brief", action="store_true")
    sub.add_parser("version", add_help=False)
    sub.add_parser("help", add_help=False)
    return p


def main(argv: list[str]) -> int:
    cmd = argv[0] if argv else "help"
    try:
        args = build_parser().parse_args(argv or ["help"])
        cmd = args.cmd or "help"

        if cmd == "help":
            out(HELP.format(v=env_version()))
            return status_line(cmd, "OK")

        if cmd == "version":
            out(env_version())
            return status_line(cmd, "OK")

        if cmd == "doctor":
            from pipeline.doctor import run_checks, summary
            checks = run_checks(offline_mode=args.offline)
            ok, req_fail, opt_fail = summary(checks)
            if not args.brief:
                for c in checks:
                    tag = "ok  " if c["ok"] else ("FAIL" if c["required"] else "info")
                    out(f"[{tag}] {c['id']:<36} {c['detail']}")
            out(f"doctor: {'READY' if ok else 'NOT READY'} — {len(checks)} checks, {req_fail} required failed, "
                f"{opt_fail} optional not ready")
            return status_line(cmd, "OK" if ok else "DOCTOR_FAILED", checks=len(checks),
                               required_failed=req_fail, optional_not_ready=opt_fail, offline=args.offline)

        if cmd == "probe":
            from pipeline.schema_check import validate
            from pipeline.stages.probe import probe
            from pipeline.stages.select import select
            passport = probe(args.file)
            errors = validate(passport, "passport")
            if errors:
                raise TrxError("SCHEMA_INVALID", "; ".join(errors[:5]))
            sel = select(passport)
            if args.out:
                write_json(Path(args.out), passport)
            ch, bw, nz = passport["channels"], passport["bandwidth"], passport["noise"]
            out(f"{passport['source']['file_name']}: {passport['container']['duration_sec']} s, "
                f"{passport['stream']['codec']} {passport['stream']['sample_rate']} Hz, {ch['count']} ch -> {ch['layout_class']}")
            out(f"  band {bw['class']} (rolloff95 {bw['rolloff95_hz_median']:.0f} Hz) · SNR~{nz['snr_db_estimate']} dB · "
                f"{passport['levels']['integrated_lufs']} LUFS · active {passport['activity']['active_ratio']:.0%}")
            for wmsg in passport["warnings"]:
                out(f"  warning: {wmsg}")
            out(f"  recipe: {sel['recipe']} ({sel['rule_id']}: {sel['reason']})")
            return status_line(cmd, "OK", recipe=sel["recipe"], rule=sel["rule_id"],
                               passport=args.out or None)

        if cmd == "transcribe":
            from pipeline.run import transcribe
            name, info = transcribe(args.file, args.recipe, args.style, args.lang, args.case)
            out(f"run {info['run_id']}: {info['verdict']} — recipe {info['recipe']}")
            for r in info["reasons"]:
                out(f"  {r}")
            out(f"  folder: {info['run_dir']}")
            return status_line(cmd, name, run_id=info["run_id"], verdict=info["verdict"], recipe=info["recipe"])

        if cmd == "explain":
            from pipeline.explain import explain
            for line in explain(args.run_id):
                out(line)
            return status_line(cmd, "OK", run_id=args.run_id)

        if cmd == "eval":
            from pipeline.evaluate import list_fixtures, run_eval
            for f in list_fixtures():
                out(f"  [{'ok' if f['ok'] else 'BAD'}] {f['id']}: {f['detail']}")
            run_eval()
            return status_line(cmd, "OK")

        if cmd == "watch":
            from pipeline.watch import watch
            done = watch(args.interval, args.once)
            out(f"processed {len(done)} new file(s)")
            return status_line(cmd, "OK", processed=len(done))

        raise TrxError("USAGE", f"unknown command '{cmd}'")
    except TrxError as e:
        if e.detail:
            out(f"error: {e.detail}")
        return status_line(cmd, e.name, e.detail, **e.extra)
    except offline.NetworkBlocked as e:
        return status_line(cmd, "NETWORK_BLOCKED", str(e))
    except Exception as e:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        return status_line(cmd, "INTERNAL_ERROR", f"{type(e).__name__}: {e}")


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
