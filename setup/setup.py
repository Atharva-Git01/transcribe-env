"""setup.py — the ONLY part of the environment that uses the internet. Nothing happens without --approve.

  python setup/setup.py plan    [--phase N]         what would be installed: sizes, licences, status, disk
  python setup/setup.py install --phase N --approve install one phase (venv from requirements.lock, models, binaries)
  python setup/setup.py verify                       sha256 of every installed model/binary vs setup/sources.toml
  python setup/setup.py write-sources                record SOURCE.json next to every installed component

Exact specs (URL, sha256, destination) live in setup/sources.toml; Python packages in setup/requirements.lock.
Run it with the system Python 3.12 (it creates the venv) — never during a transcription run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tomllib
import urllib.request
import zipfile
from pathlib import Path

HOME = Path(__file__).resolve().parents[1]
SOURCES = HOME / "setup" / "sources.toml"
LOCK = HOME / "setup" / "requirements.lock"
LOCK_NODEPS = HOME / "setup" / "requirements-nodeps.lock"
VENV = HOME / "toolchain" / "python" / ".venv"
VPY = VENV / "Scripts" / "python.exe"


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def load() -> list[dict]:
    with open(SOURCES, "rb") as f:
        return tomllib.load(f)["item"]


def present(it: dict) -> bool:
    if it["kind"] == "venv":
        return VPY.is_file()
    return all((HOME / f["dest"]).is_file() for f in it.get("files", []))


def size_mb(it: dict) -> float:
    return sum(f.get("size", 0) for f in it.get("files", [])) / 1e6 + it.get("extra_mb", 0)


def plan(phase: int | None) -> None:
    items = [i for i in load() if phase is None or i["phase"] == phase]
    need = 0.0
    print(f"{'id':<34} {'ph':>2} {'MB':>8} {'licence':<28} status")
    for it in items:
        ok = present(it)
        mb = size_mb(it)
        need += 0 if ok else mb
        print(f"{it['id']:<34} {it['phase']:>2} {mb:>8.0f} {it.get('licence', ''):<28} {'installed' if ok else 'missing'}")
    free = shutil.disk_usage(HOME).free / 1e9
    print(f"\nto download: {need / 1e3:.2f} GB · free now: {free:.1f} GB · keep >= 3 GB free after install")


def _download(url: str, dest: Path, expect: str | None) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"  downloading {url}")
    with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f, 1 << 20)
    got = sha256(tmp)
    if expect and got != expect:
        tmp.unlink()
        raise SystemExit(f"sha256 mismatch for {dest.name}: {got} != {expect}")
    tmp.replace(dest)


def _download_range(url: str, dest: Path, nbytes: int | None, expect: str | None) -> None:
    """Like _download, but only the first nbytes (HTTP Range) — for corpora that are larger than we need."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"Range": f"bytes=0-{nbytes - 1}"} if nbytes else {})
    print(f"  downloading {url}" + (f" (first {nbytes:,} bytes)" if nbytes else ""))
    with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f, 1 << 20)
    got = sha256(tmp)
    if expect and got != expect:
        tmp.unlink()
        raise SystemExit(f"sha256 mismatch for {dest.name}: {got} != {expect}")
    tmp.replace(dest)


def install(phase: int) -> None:
    for it in [i for i in load() if i["phase"] == phase]:
        if present(it):
            print(f"[skip] {it['id']} already installed")
            continue
        print(f"[install] {it['id']}")
        kind = it["kind"]
        if kind == "venv":
            subprocess.run([sys.executable, "-m", "venv", str(VENV)], check=True)
            subprocess.run([str(VPY), "-m", "pip", "install", "--no-cache-dir", "--require-hashes", "-r", str(LOCK)],
                           check=True)
            if LOCK_NODEPS.is_file():
                subprocess.run([str(VPY), "-m", "pip", "install", "--no-cache-dir", "--no-deps", "--require-hashes",
                                "-r", str(LOCK_NODEPS)], check=True)
        elif kind == "file":
            for f in it["files"]:
                _download(f["url"], HOME / f["dest"], f.get("sha256"))
        elif kind in ("zip", "tar.bz2"):
            arc = HOME / "toolchain" / "downloads" / Path(it["url"]).name
            _download(it["url"], arc, it.get("archive_sha256"))
            target = HOME / it["extract_to"]
            target.mkdir(parents=True, exist_ok=True)
            if kind == "zip":
                zipfile.ZipFile(arc).extractall(target)
            else:
                tarfile.open(arc).extractall(target, filter="data")
        elif kind == "hardlink":
            for f in it["files"]:
                src = Path(os.path.expandvars(f["from"]))
                src = next(iter(sorted(Path(src.parent).glob(src.name))), src) if "*" in str(src) else src
                dst = HOME / f["dest"]
                dst.parent.mkdir(parents=True, exist_ok=True)
                if not src.is_file():
                    raise SystemExit(f"{it['id']}: source {src} not found; see setup/README.md for the alternative")
                os.link(src, dst)
        elif kind == "build":
            # download exact byte ranges (sha256-checked) into a fresh folder, run the builder, drop the raw input
            work = HOME / "toolchain" / "downloads" / f"{it['id']}-build"
            work.mkdir(parents=True, exist_ok=True)
            names = {}
            for d in it["downloads"]:
                dest = work / d["as"]
                _download_range(d["url"], dest, d.get("bytes"), d.get("sha256"))
                names[d["as"].split(".")[0]] = str(dest)
            cmd = [str(VPY), "-X", "utf8", "-I"] + [a.format(home=HOME, **names) for a in it["build"]]
            subprocess.run(cmd, check=True, cwd=HOME)
            if it.get("delete_downloads", True):
                shutil.rmtree(work, ignore_errors=True)
        else:
            raise SystemExit(f"unknown kind {kind}")
    write_sources()
    print("done — now run: bin\\trx doctor --offline")


def verify() -> int:
    bad = 0
    for it in load():
        for f in it.get("files", []):
            p = HOME / f["dest"]
            if not p.is_file():
                continue
            if f.get("sha256") and sha256(p) != f["sha256"]:
                print(f"MISMATCH {f['dest']}")
                bad += 1
    print("all hashes match" if not bad else f"{bad} mismatch(es)")
    return 1 if bad else 0


def write_sources() -> None:
    for it in load():
        if it["kind"] in ("venv", "build") or not present(it):   # builders write their own SOURCE.json
            continue
        dest_dir = HOME / it.get("source_json_dir", Path(it["files"][0]["dest"]).parent.as_posix())
        rec = {"id": it["id"], "licence": it.get("licence"), "source": it.get("url") or it.get("page"),
               "files": {Path(f["dest"]).name: {"sha256": f.get("sha256") or sha256(HOME / f["dest"]),
                                                "size": (HOME / f["dest"]).stat().st_size} for f in it["files"]}}
        mb = [v["sha256"] for k, v in rec["files"].items() if k in ("model.bin",) or k.endswith(".gguf")]
        if mb:
            rec["model_sha256"] = mb[0]
        (dest_dir / "SOURCE.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["plan", "install", "verify", "write-sources"])
    ap.add_argument("--phase", type=int)
    ap.add_argument("--approve", action="store_true")
    a = ap.parse_args()
    if a.cmd == "plan":
        plan(a.phase)
    elif a.cmd == "install":
        if a.phase is None or not a.approve:
            plan(a.phase)
            print("\nnothing installed: add --phase N --approve (only after the owner's OK)")
            return 2
        install(a.phase)
    elif a.cmd == "verify":
        return verify()
    else:
        write_sources()
    return 0


if __name__ == "__main__":
    sys.exit(main())
