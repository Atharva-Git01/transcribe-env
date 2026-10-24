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

  trx transcribe <file> [--lang mr|hi|en|auto] [--recipe ID] [--style verbatim|readable] [--case NAME]
                                   full run: copy + hash, probe, choose recipe, plan, run stages, report.
                                   Without --lang it shows the detected language and asks first.
                                   --hotwords FILE = names/places/terms of this case, one per line
                                   An interrupted run of the same file continues where it stopped.
  trx watch [--once] [--interval SECONDS]
                                   process every new file dropped into inbox/
  trx probe <file> [--out FILE]    measure the audio only (writes a passport)
  trx explain <run-id>             plain-language account of a run
  trx eval [--engine ID] [--recipe ID] [--fixture ID] [--save-baseline]
                                   score the pipeline on evals/fixtures (maintainers)
  trx screen build|run|report|transfer|rescore|listen|listen-score [--candidate ID] [--base ID] [--set v1]
                                   PLAN_v0.9 screening harness: every candidate model on the same fixed
                                   private lines, scored per challenge set (maintainers)
  trx doctor [--offline] [--brief] [--deep]
                                   readiness + offline proof (--deep: sha256 of every model)
  trx test                         unit tests of the deterministic parts (maintainers)
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
    t.add_argument("--fast", action="store_true")
    t.add_argument("--hotwords")
    w = sub.add_parser("watch", add_help=False)
    w.add_argument("--once", action="store_true")
    w.add_argument("--interval", type=int, default=10)
    pr = sub.add_parser("probe", add_help=False)
    pr.add_argument("file")
    pr.add_argument("--out")
    e = sub.add_parser("explain", add_help=False)
    e.add_argument("run_id")
    ev = sub.add_parser("eval", add_help=False)
    ev.add_argument("--engine")
    ev.add_argument("--recipe")
    ev.add_argument("--fixture")
    ev.add_argument("--lang")
    ev.add_argument("--save-baseline", action="store_true")
    sc = sub.add_parser("screen", add_help=False)
    sc.add_argument("action", choices=["build", "run", "report", "transfer", "rescore", "listen", "listen-score"])
    sc.add_argument("--n", type=int, default=20)
    sc.add_argument("--answers")
    sc.add_argument("--candidate")
    sc.add_argument("--base", default="indic_transcribe")
    sc.add_argument("--set", default="v1")
    d = sub.add_parser("doctor", add_help=False)
    d.add_argument("--offline", action="store_true")
    d.add_argument("--brief", action="store_true")
    d.add_argument("--deep", action="store_true")
    sub.add_parser("version", add_help=False)
    sub.add_parser("test", add_help=False)
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
            checks = run_checks(offline_mode=args.offline, deep=args.deep)
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
            if args.fast:   # removed in 0.7.2 (owner): the default is the fastest good option since the GPU switch
                raise TrxError("USAGE", "--fast was removed in 0.7.2: the default run is now the fastest good option "
                                        "(Indic-Transcribe on the GPU). Run the same command without --fast.")
            from pipeline.langcheck import confirm
            from pipeline.run import find_unfinished, resume, transcribe
            engine = None
            from pipeline.hotwords import read_hotwords
            hot = read_hotwords(args.hotwords) if args.hotwords else None
            same = {k: v for k, v in {"recipe": args.recipe, "style": args.style, "engine": engine, "hotwords": hot,
                                      "lang": None if args.lang in (None, "auto") else args.lang}.items() if v}
            unfinished = find_unfinished(args.file, same)
            if unfinished:  # same file + same options, interrupted earlier: continue it instead of starting over
                out(f"resuming interrupted run {unfinished.name} (saved work is reused)")
                name, info = resume(unfinished)
            else:
                lang, lang_how = confirm(args.file, args.lang)
                name, info = transcribe(args.file, args.recipe, args.style, lang, args.case, engine=engine,
                                        lang_how=lang_how, hotwords=hot)
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
            res = run_eval(args.engine, args.recipe, args.fixture, args.save_baseline, args.lang)
            out(f"\n{'fixture':<28} {'recipe':<22} {'engine':<30} {'CER':>6} {'nCER':>6} {'WER':>6} {'base':>6} "
                f"{'spk':>5} {'miss':>5} {'RTF':>5} {'verdict':<7}")
            for r in res["rows"]:
                base = "-" if r["baseline_cer"] is None else f"{r['baseline_cer']:.4f}"
                spk = "-" if r.get("speaker_acc") is None else f"{r['speaker_acc']:.3f}"
                out(f"{r['fixture']:<28} {r['recipe'] or '-':<22} {'+'.join(r['engines'])[:30] or '-':<30} "
                    f"{r['cer']:>6.3f} {r['cer_norm']:>6.3f} {r['wer']:>6.3f} {base:>6} {spk:>5} "
                    f"{str(r['missing_turns']) + '/' + str(r['turns']):>5} {r['rtf']:>5} {r['verdict']:<7}")
                if r.get("monitor"):
                    from pipeline.evaluate import monitor_line
                    out(f"  how: {monitor_line(r['monitor'])}"
                        + ("  [truth written as spoken: judged by the content CER]" if r.get("spelling") == "as_spoken" else ""))
            for reg in res["regressions"]:
                out(f"REGRESSION {reg}")
            out(f"saved {res['file']} (+ .md; truth vs output per turn: evals/compare/, this PC only)" + (" and updated evals/baseline.json" if args.save_baseline else ""))
            return status_line(cmd, "OK" if not res["regressions"] else "EVAL_REGRESSION",
                               results=res["file"], regressions=len(res["regressions"]))

        if cmd == "screen":
            from pipeline import screen
            if args.action == "build":
                doc = screen.build(args.set)
                c = doc["counts"]
                out(f"screening set {args.set}: {c['lines']} lines, {c['audio_s']} s of audio; "
                    + ", ".join(f"{t} {c[t]['lines']} lines ({c[t]['audio_s']} s)" for t in screen.TAGS)
                    + f"; {c['spoken_slots']} spoken and {c['standard_slots']} standard slots")
                return status_line(cmd, "OK", set=args.set, lines=c["lines"])
            if args.action == "run":
                if not args.candidate:
                    raise TrxError("USAGE", "trx screen run needs --candidate ID (evals/screen_candidates.toml: "
                                            + ", ".join(sorted(screen.candidates())) + ")")
                r = screen.run(args.candidate, args.set)
                out(screen.summary_line(r))
                out(f"saved {r['file']} (texts: evals/screening/{args.set}/out/, evals/compare/; this PC only)")
                return status_line(cmd, "OK", candidate=args.candidate, results=r["file"])
            if args.action == "listen":
                from pipeline import listening
                if not args.candidate:
                    raise TrxError("USAGE", "trx screen listen needs --candidate ID (compared with --base)")
                r = listening.page(args.candidate, args.base, args.set, args.n)
                out(f"listening check: {r['lines']} of {r['differing_lines']} differing lines -> {r['file']} "
                    "(private: open it on this PC only)")
                return status_line(cmd, "OK", lines=r["lines"])
            if args.action == "listen-score":
                from pipeline import listening
                if not args.answers:
                    raise TrxError("USAGE", "trx screen listen-score needs --answers <downloaded json>")
                r = listening.score(args.answers)
                out(f"owner's choices: {r['counts']}")
                return status_line(cmd, "OK", **{k.replace('-', '_'): v for k, v in r["counts"].items()})
            if args.action == "rescore":
                if not args.candidate:
                    raise TrxError("USAGE", "trx screen rescore needs --candidate ID")
                r = screen.rescore(args.candidate, args.set)
                out(screen.summary_line(r))
                return status_line(cmd, "OK", candidate=args.candidate, results=r["file"])
            if args.action == "transfer":
                if not args.candidate:
                    raise TrxError("USAGE", "trx screen transfer needs --candidate ID (the model whose spelling is used)")
                r = screen.transfer_run(args.candidate, args.base, args.set)
                out(screen.summary_line(r) + f" | {r['spelling_changes']} spelling changes")
                out(f"saved {r['file']} (texts: evals/screening/{args.set}/out/, evals/compare/; this PC only)")
                return status_line(cmd, "OK", candidate=r["candidate"], results=r["file"])
            rows = screen.report(args.set)
            for r in rows:
                out(screen.summary_line(r))
            out(f"saved evals/results/screen_{args.set}_summary.md ({len(rows)} candidate(s))")
            return status_line(cmd, "OK", candidates=len(rows))

        if cmd == "test":
            import unittest
            suite = unittest.defaultTestLoader.discover(str(Path(__file__).resolve().parent / "tests"))
            res = unittest.TextTestRunner(verbosity=1, stream=sys.stdout).run(suite)
            ok = res.wasSuccessful()
            return status_line(cmd, "OK" if ok else "DOCTOR_FAILED", tests=res.testsRun,
                               failures=len(res.failures) + len(res.errors))

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
