"""trx doctor: is this environment ready, and is it provably offline? Every check: id, ok, required, detail."""
from __future__ import annotations

import importlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

from .common import TRX_HOME, load_config, load_toml
from .guards import offline
from .metrics import self_test as metrics_self_test
from .recipes import all_recipe_ids, check_recipe
from .schema_check import load_schema, schema_names
from .stages import IMPLEMENTED
from .stages.select import check_rules

REQUIRED_DIRS = ["config", "recipes", "pipeline", "prompts", "library", "references", "knowledge", "evals",
                 "inbox", "runs", "toolchain", "setup"]
REQUIRED_FILES = ["README.md", "OPERATOR.md", "MAINTAINER.md", "VERSION", "manifest.toml", "knowledge/ERRORS.md",
                  "knowledge/CHANGELOG.md"]


def _run(cmd):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
        return r.returncode == 0, (r.stdout or r.stderr).strip().splitlines()[0] if (r.stdout or r.stderr) else ""
    except Exception as e:  # noqa: BLE001
        return False, str(e)


def run_checks(offline_mode: bool = False, deep: bool = False) -> list[dict]:
    checks: list[dict] = []

    def add(cid, ok, detail, required=True):
        checks.append({"id": cid, "ok": bool(ok), "required": required, "detail": detail})

    add("python", sys.version_info >= (3, 11), f"{sys.version.split()[0]} at {sys.executable}")
    for mod in ("numpy",):
        try:
            m = importlib.import_module(mod)
            add(f"py:{mod}", True, getattr(m, "__version__", ""))
        except ImportError:
            add(f"py:{mod}", False, "not importable")
    for tool in ("ffmpeg", "ffprobe", "git"):
        ok, line = _run([tool, "-version" if tool != "git" else "--version"])
        add(f"bin:{tool}", ok, line[:80])
    ok, line = _run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"])
    add("gpu", ok, line or "no NVIDIA GPU visible (CPU engines only)", required=False)

    for d in REQUIRED_DIRS:
        add(f"dir:{d}", (TRX_HOME / d).is_dir(), d)
    for f in REQUIRED_FILES:
        add(f"file:{f}", (TRX_HOME / f).is_file(), f)

    # configs
    for name in ("defaults", "thresholds", "selection_rules", "exit_codes"):
        try:
            load_config(name)
            add(f"config:{name}", True, "loads")
        except Exception as e:  # noqa: BLE001
            add(f"config:{name}", False, str(e))
    try:
        probs = check_rules(load_config("selection_rules").get("rule", []))
        add("config:selection_rules:logic", not probs, "; ".join(probs) or "ordered, recipes exist, ends with fallback")
    except Exception as e:  # noqa: BLE001
        add("config:selection_rules:logic", False, str(e))
    try:
        codes = load_config("exit_codes")
        bad = [k for k, v in codes.items() if not {"code", "meaning", "next"} <= set(v)]
        dup = len({v["code"] for v in codes.values()}) != len(codes)
        add("config:exit_codes:logic", not bad and not dup and codes.get("OK", {}).get("code") == 0,
            f"{len(codes)} codes" + (f"; incomplete: {bad}" if bad else "") + ("; duplicate numbers" if dup else ""))
    except Exception as e:  # noqa: BLE001
        add("config:exit_codes:logic", False, str(e))

    ids = all_recipe_ids()
    for rid in ids:
        try:
            data = load_toml(TRX_HOME / "recipes" / f"{rid}.toml")
            probs = check_recipe(data, rid)
            template = data.get("recognise", {}).get("initial_prompt_file", "")
            langs = load_config("defaults")["languages"]["priority"] if "{lang}" in template else [None]
            for lang in langs:
                rel = template.format(lang=lang) if lang else template
                if rel and not (TRX_HOME / rel).is_file():
                    probs.append(f"prompt file missing: {rel}")
            add(f"recipe:{rid}", not probs, "; ".join(probs) or "valid, prompt files present")
        except Exception as e:  # noqa: BLE001
            add(f"recipe:{rid}", False, str(e))

    for name in schema_names():
        try:
            load_schema(name)
            add(f"schema:{name}", True, "loads")
        except Exception as e:  # noqa: BLE001
            add(f"schema:{name}", False, str(e))

    ok, detail = metrics_self_test()
    add("metrics:self_test", ok, detail)

    # library sanity: vocab variants that are themselves vocab keys rewrite real words into other words
    for vf in sorted((TRX_HOME / "library" / "vocab").glob("*.json")):
        try:
            vocab = json.loads(vf.read_text(encoding="utf-8"))
            keys = set(vocab)
            clashes = sorted({v for vs in vocab.values() for v in vs if v in keys})
            add(f"vocab:{vf.name}", not clashes,
                f"{len(vocab)} entries" + (f"; {len(clashes)} variants are real headwords (e.g. {clashes[:3]}) — "
                                           f"keep disabled until reviewed" if clashes else ""), required=False)
        except Exception as e:  # noqa: BLE001
            add(f"vocab:{vf.name}", False, str(e), required=False)

    # models from the manifest (informational until their phase lands)
    try:
        manifest = load_toml(TRX_HOME / "manifest.toml")
        from .models import installed
        for comp in manifest.get("component", []):
            if comp.get("kind") != "model":
                continue
            ok = installed(comp["id"])
            where = ""
            if ok:
                p = TRX_HOME / comp["path"]
                where = " (on D: USB via junction)" if str(p.resolve()).upper().startswith("D:") else ""
                src = json.loads((p / "SOURCE.json").read_text(encoding="utf-8"))
                bad = [n for n, v in src.get("files", {}).items()
                       if not (p / n).is_file() or (p / n).stat().st_size != v["size"]]
                ok = not bad
                where += f"; size mismatch: {bad}" if bad else ""
                if deep and ok:
                    from .common import sha256_file
                    mism = [n for n, v in src["files"].items() if sha256_file(p / n) != v["sha256"]]
                    ok = not mism
                    where += f"; SHA256 MISMATCH: {mism}" if mism else "; sha256 verified"
            status = comp.get("status", "")
            why = {"excluded": " — excluded (licence/policy, see manifest.toml)",
                   "not_installed": " — owner decision pending (QUESTIONS.md)"}.get(status, "")
            add(f"model:{comp['id']}", ok, f"phase {comp.get('phase')} — "
                + ("installed" + where if ok else ("not installed" + where + why)), required=False)
    except Exception as e:  # noqa: BLE001
        add("manifest", False, str(e))

    # Phase 1+ runtime: venv packages exactly as locked, configured engines installed, binaries present
    lock = TRX_HOME / "setup" / "requirements.lock"
    if lock.is_file():
        from importlib import metadata
        want = {}
        for f in (lock, TRX_HOME / "setup" / "requirements-nodeps.lock", TRX_HOME / "setup" / "requirements-ml.lock"):
            if f.is_file():
                for ln in f.read_text(encoding="utf-8").splitlines():
                    if "==" in ln and not ln.startswith("#"):
                        name, ver = ln.split()[0].split("==")
                        want[name.lower()] = ver
        wrong = []
        for name, ver in want.items():
            try:
                have = metadata.version(name)
                if have != ver:
                    wrong.append(f"{name} {have}!={ver}")
            except metadata.PackageNotFoundError:
                wrong.append(f"{name} missing")
        in_venv = ".venv" in sys.executable
        add("venv:locked_packages", in_venv and not wrong,
            (f"{len(want)} packages match setup/requirements.lock" if not wrong else "; ".join(wrong[:6]))
            + ("" if in_venv else " (not running from toolchain/python/.venv)"))
        try:
            from .models import installed as _inst
            d = load_config("defaults")
            from .stages.recognise import engine_choices
            prim = d["engines"]["primary"]
            none_ok = [k for k in prim if not any(_inst(e) for e in engine_choices(prim, k))]
            fallback = {k: next(e for e in engine_choices(prim, k) if _inst(e)) for k in prim
                        if k not in none_ok and not _inst(engine_choices(prim, k)[0])}
            add("engines:primary_installed", not none_ok,
                (f"no installed engine for: {none_ok}" if none_ok else "an installed engine for every language")
                + (f"; first choice missing, using {fallback}" if fallback else ""))
            if d["engines"]["cross_check"].get("enabled"):
                cc = {k: v for k, v in d["engines"]["cross_check"].items() if k != "enabled"}
                cm = [k for k in cc if engine_choices(cc, k) and not any(_inst(e) for e in engine_choices(cc, k))]
                add("engines:cross_check_installed", not cm, "cross-check engines installed" if not cm
                    else f"no installed cross-check engine for: {cm} (verify is skipped there)", required=False)
            lid = [e for e in d["languages"].get("lid_engines", []) if _inst(e)]
            add("engines:lid", bool(lid), f"LID engine: {lid[0]}" if lid else "no LID engine installed")
            if d["llm_cleanup"]["enabled"]:
                from .llm.gemma3 import gguf_path, server_exe
                ok = bool(server_exe()) and bool(gguf_path(d["llm_cleanup"]["model"]))
                # required only if the engine each language actually gets is one whose output is cleaned (E26)
                fams = list(d["llm_cleanup"].get("engines", []))
                firsts = [next(e for e in engine_choices(prim, k) if _inst(e)) for k in prim if k not in none_ok]
                needed = not fams or any(e.startswith(f) for e in firsts for f in fams)
                add("llm_cleanup:ready", ok, ("llama-server + Gemma 3 GGUF present" if ok else
                    "LLM cleanup enabled but llama-server or the Gemma 3 model is missing (see toolchain/models/)")
                    + ("" if needed else "; only Whisper output is cleaned (--fast, fallback), so runs go on without"),
                    required=needed)
        except Exception as e:  # noqa: BLE001
            add("engines", False, str(e))
        add("bin:deep-filter", (TRX_HOME / "toolchain/bin/deep-filter/deep-filter.exe").is_file(), "DeepFilterNet3 binary")
        add("bin:llama-server", (TRX_HOME / "toolchain/bin/llama.cpp/llama-server.exe").is_file(),
            "llama.cpp (Vulkan) for Gemma 3 cleanup", required=False)
        try:
            from . import gpu
            n = gpu.cuda_device_count()
            add("gpu:cuda_usable", n > 0, f"{n} CUDA device(s) usable with the environment's cuBLAS/cuDNN"
                if n else "no usable CUDA device: engines run on CPU (slow)", required=False)
        except Exception as e:  # noqa: BLE001
            add("gpu:cuda_usable", False, str(e), required=False)
        try:
            from .stages.diarize_sherpa import AVAILABLE as sh
            add("diarization:sherpa_onnx", sh, "speaker diarization available" if sh else "not available",
                required=False)
        except Exception as e:  # noqa: BLE001
            add("diarization:sherpa_onnx", False, str(e), required=False)

    min_gb = float(load_config("defaults")["runtime"]["min_free_gb"])
    free = shutil.disk_usage(TRX_HOME).free / 1e9
    add("disk", free >= min_gb, f"{free:.1f} GB free (min {min_gb})")

    ok, detail = offline.self_test()
    add("offline:guard", ok, detail, required=True)
    if offline_mode:
        try:
            defaults = load_config("defaults")
            urls = [v for v in json.dumps(defaults).split('"') if "://" in v]
            add("offline:no_urls_in_config", not urls, "no URLs in runtime config" if not urls else str(urls))
        except Exception as e:  # noqa: BLE001
            add("offline:no_urls_in_config", False, str(e))

    add("stages:implemented", True, ", ".join(sorted(IMPLEMENTED)) or "none yet (phase 0: probe + select + plan only)",
        required=False)
    return checks


def summary(checks: list[dict]) -> tuple[bool, int, int]:
    req_fail = [c for c in checks if c["required"] and not c["ok"]]
    return not req_fail, len(req_fail), sum(1 for c in checks if not c["required"] and not c["ok"])


__all__ = ["run_checks", "summary", "Path"]
