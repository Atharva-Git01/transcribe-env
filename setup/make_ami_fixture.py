"""Make a multi-speaker eval fixture from the AMI Meeting Corpus (CC-BY-4.0): a window of one meeting's mixed
headset audio + speaker turns built from the manual word-level annotations. Setup-time tool.

  python setup/make_ami_fixture.py --meeting ES2004a --audio .../ES2004a.Mix-Headset.wav \
      --annotations .../ami_public_manual_1.6.2.zip --dur 180

The window is chosen automatically: the one with the most distinct speakers and the most speaker changes.
Annotation XML is read from the zip in memory (no paths from the archive are written).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

HOME = Path(__file__).resolve().parents[1]


def words(zpath: Path, meeting: str) -> list[dict]:
    out = []
    with zipfile.ZipFile(zpath) as z:
        for name in z.namelist():
            m = re.fullmatch(rf"words/{meeting}\.([A-Z])\.words\.xml", name)
            if not m:
                continue
            root = ET.fromstring(z.read(name))
            for el in root.iter():
                if el.tag.split("}")[-1] != "w" or el.get("punc") == "true" or not (el.text or "").strip():
                    continue
                st, en = el.get("starttime"), el.get("endtime")
                if st is None or en is None:
                    continue
                out.append({"start": float(st), "end": float(en), "text": el.text.strip(), "speaker": m.group(1)})
    return sorted(out, key=lambda w: w["start"])


def turns(ws: list[dict], gap: float = 1.0) -> list[dict]:
    per: dict = {}
    for w in ws:
        cur = per.get(w["speaker"])
        if cur and w["start"] - cur[-1]["end"] <= gap:
            cur[-1]["end"] = w["end"]
            cur[-1]["text"] += " " + w["text"]
        else:
            per.setdefault(w["speaker"], []).append(dict(w))
    return sorted((t for ts in per.values() for t in ts), key=lambda t: t["start"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--meeting", required=True)
    ap.add_argument("--audio", required=True)
    ap.add_argument("--annotations", required=True)
    ap.add_argument("--dur", type=float, default=180.0)
    a = ap.parse_args()
    ts = turns(words(Path(a.annotations), a.meeting))
    if not ts:
        print("no words found")
        return 1
    best = None
    t0 = 0.0
    while t0 + a.dur <= ts[-1]["end"]:
        inside = [t for t in ts if t["start"] >= t0 and t["end"] <= t0 + a.dur]
        spk = len({t["speaker"] for t in inside})
        changes = sum(1 for x, y in zip(inside, inside[1:]) if x["speaker"] != y["speaker"])
        score = (spk, changes)
        if best is None or score > best[0]:
            best = (score, t0, inside)
        t0 += 30.0
    (_, start, inside) = best
    fid = f"ami_{a.meeting.lower()}_{int(start)}s"
    out = HOME / "evals" / "fixtures" / fid
    out.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-ss", f"{start:.2f}", "-t", f"{a.dur:.2f}", "-i", a.audio,
                    "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(out / "audio.wav")], check=True)
    segs = [{"speaker": t["speaker"], "text": t["text"]} for t in inside]
    truth = {
        "schema": "trx.truth/1", "id": fid, "audio": "audio.wav",
        "audio_sha256": hashlib.sha256((out / "audio.wav").read_bytes()).hexdigest(), "language": "en",
        "duration_sec": a.dur, "speakers": sorted({s["speaker"] for s in segs}),
        "reference_text": " ".join(s["text"] for s in segs), "segments": segs,
        "source": f"AMI Meeting Corpus, meeting {a.meeting}, Mix-Headset audio {start:.0f}-{start + a.dur:.0f} s, manual word "
                  "annotations v1.6.2 (CC-BY-4.0, https://groups.inf.ed.ac.uk/ami/corpus/)",
        "notes": f"{len({s['speaker'] for s in segs})} speakers, {len(segs)} turns, English meeting — validates the speaker "
                 "count estimate and speaker accuracy on 3+ speakers (overlaps exist; turns ordered by start time).",
    }
    (out / "truth.json").write_text(json.dumps(truth, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{fid}: {len(segs)} turns, speakers {truth['speakers']}, {len(truth['reference_text'])} chars")
    return 0


if __name__ == "__main__":
    sys.exit(main())
