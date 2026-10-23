"""Turn one of the owner's recordings + its transcript into an eval fixture. Setup-time tool. The fixture stays on
this PC only (evals/fixtures/*/ is git-ignored): never push a private recording or its words.

  python setup/make_fixture.py --id call_20261101 --audio "D:/rec/call.m4a" --text "D:/rec/call.txt" --lang mr

The transcript is a UTF-8 text file with one turn per line, `speaker: text`, in the order spoken. Blank lines and
lines starting with # are skipped. Write it in the output convention (standard spelling, numbers as words, English
words in Devanagari). The audio is copied unchanged (any format ffmpeg reads); its sha256 goes into truth.json.
If the transcript stops before the recording ends, add `--scored-until <seconds>` (where it stops): otherwise every
line after that point counts as an error (E39).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
from datetime import date
from pathlib import Path

HOME = Path(__file__).resolve().parents[1]
LINE = re.compile(r"^\s*([^:]{1,40}?)\s*:\s*(.+?)\s*$")
LANGS = {"mr", "hi", "en"}


def parse(text: str) -> tuple[list[dict], list[int]]:
    """`speaker: text` lines -> turns; also returns the numbers of lines that are not in that form."""
    segs, bad = [], []
    for n, line in enumerate(text.splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        m = LINE.match(line)
        if not m:
            bad.append(n)
            continue
        segs.append({"speaker": m.group(1).strip(), "text": m.group(2)})
    return segs, bad


def duration(path: Path) -> float:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1",
                          str(path)], capture_output=True, text=True, check=True).stdout.strip()
    return round(float(out), 3)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", required=True, help="fixture id: lower-case letters, digits and _")
    ap.add_argument("--audio", required=True)
    ap.add_argument("--text", required=True, help="UTF-8 file, one `speaker: text` line per turn")
    ap.add_argument("--lang", required=True, choices=sorted(LANGS))
    ap.add_argument("--notes", default="")
    ap.add_argument("--scored-until", type=float, default=None, metavar="SECONDS",
                    help="the transcript covers only the start of the recording: score output lines starting before this")
    a = ap.parse_args()
    if not re.fullmatch(r"[a-z0-9_]{3,60}", a.id):
        print("--id: use 3-60 lower-case letters, digits or _")
        return 2
    audio, text = Path(a.audio), Path(a.text)
    segs, bad = parse(text.read_text(encoding="utf-8-sig"))
    if bad:
        print(f"{text.name}: lines {bad[:10]} are not `speaker: text`; fix them and run again")
        return 1
    if not segs:
        print(f"{text.name}: no turns found")
        return 1
    out = HOME / "evals" / "fixtures" / a.id
    if out.exists():
        print(f"{out} already exists; choose another --id (fixtures are never overwritten)")
        return 1
    out.mkdir(parents=True)
    dst = out / f"audio{audio.suffix.lower()}"
    shutil.copy2(audio, dst)
    speakers = list(dict.fromkeys(s["speaker"] for s in segs))
    truth = {
        "schema": "trx.truth/1", "id": a.id, "audio": dst.name,
        "audio_sha256": hashlib.sha256(dst.read_bytes()).hexdigest(), "language": a.lang,
        "duration_sec": duration(dst), "speakers": speakers,
        "reference_text": " ".join(s["text"] for s in segs), "segments": segs,
        "source": f"owner recording {audio.name}, manual transcript {text.name} (added {date.today().isoformat()})",
        "notes": a.notes or f"{len(speakers)} speakers, {len(segs)} turns",
    }
    if a.scored_until:
        truth["scored_until_sec"] = a.scored_until
    (out / "truth.json").write_text(json.dumps(truth, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{a.id}: {len(segs)} turns, speakers {speakers}, {truth['duration_sec']} s -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
