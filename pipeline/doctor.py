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


def run_checks(offline_mode: bool = False) -> list[dict]:
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
            langs = data.get("lid", {}).get("candidates", []) if "{lang}" in template else [None]
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
        for comp in manifest.get("component", []):
            if comp.get("kind") != "model":
                continue
            present = (TRX_HOME / comp["path"]).exists() if comp.get("path") else False
            add(f"model:{comp['id']}", present, f"phase {comp.get('phase')} — "
                + ("present" if present else "not installed"), required=False)
    except Exception as e:  # noqa: BLE001
        add("manifest", False, str(e))

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
