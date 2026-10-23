"""Turn finished runs of the owner's test recordings into draft transcripts for the owner to correct, in the
`speaker: text` format setup/make_fixture.py reads. Maintainer tool. The drafts hold the words of private recordings:
they are written next to the audio (evals/test_audio/drafts/), which is git-ignored.

  python setup/make_drafts.py --manifest evals/test_audio/manifest.json

For each recording in the manifest, the newest run whose case is its id (runs/<id>-<time>/) becomes drafts/<id>.txt:
one line per speaker turn (consecutive segments of one speaker merged), a `# mm:ss-mm:ss` comment with its time, and a
`# review:` comment where the run flagged something. drafts/SUMMARY.md lists language, speakers and verdict per file.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HOME = Path(__file__).resolve().parents[1]


def mmss(t: float) -> str:
    return f"{int(t // 60):02d}:{t % 60:04.1f}"


def turns(segments: list[dict]) -> list[dict]:
    out: list[dict] = []
    for s in segments:
        if not s.get("text", "").strip():
            continue
        if out and out[-1]["speaker"] == s.get("speaker"):
            out[-1]["end"] = s["end"]
            out[-1]["text"] += " " + s["text"].strip()
            out[-1]["flags"] += s.get("flags", [])
        else:
            out.append({"speaker": s.get("speaker") or "?", "start": s["start"], "end": s["end"],
                        "text": s["text"].strip(), "flags": list(s.get("flags", []))})
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    a = ap.parse_args()
    manifest = Path(a.manifest)
    rec = json.loads(manifest.read_text(encoding="utf-8"))["files"]
    drafts = manifest.parent / "drafts"
    drafts.mkdir(exist_ok=True)
    rows = ["| id | length | language | recipe | speakers | verdict | flagged | run |", "|---|---|---|---|---|---|---|---|"]
    for r in rec:
        runs = sorted((HOME / "runs").glob(f"{r['id']}-*/out/transcript.json"))
        if not runs:
            rows.append(f"| {r['id']} | {r['duration_sec']:.0f} s | – | – | – | no run | – | – |")
            continue
        run = runs[-1].parents[1]
        tj = json.loads(runs[-1].read_text(encoding="utf-8"))
        rep = json.loads((run / "report.json").read_text(encoding="utf-8"))
        m = rep.get("metrics", {})
        ts = turns(tj["segments"])
        recipe = tj.get("recipe", {})
        recipe = recipe.get("id") if isinstance(recipe, dict) else recipe
        lines = [f"# {r['id']} - {r['file']} ({r['duration_sec']:.0f} s)",
                 f"# draft from run {run.name}: language {tj.get('language')}, recipe {recipe}, "
                 f"verdict {tj.get('verdict')}",
                 "# Correct each line: keep `speaker: text`; rename SPK1/SPK2 to names if you like; standard spelling,",
                 "# numbers as words, English words in Devanagari. Add a missing turn as a new line where it was spoken.",
                 ""]
        for t in ts:
            lines.append(f"# {mmss(t['start'])}-{mmss(t['end'])}")
            if t["flags"]:
                lines.append("# review: " + "; ".join(sorted({f.split(' (')[0] for f in t["flags"]})))
            lines.append(f"{t['speaker']}: {t['text']}")
        (drafts / f"{r['id']}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        flagged = m.get("flagged_time_ratio")
        rows.append(f"| {r['id']} | {r['duration_sec']:.0f} s | {tj.get('language')} | {recipe} | "
                    f"{m.get('speakers_found', '–')} | {tj.get('verdict')} | "
                    f"{'–' if flagged is None else f'{flagged:.0%}'} | {run.name} |")
    (drafts / "SUMMARY.md").write_text("# Test recordings: first pass\n\n" + "\n".join(rows) + "\n", encoding="utf-8")
    print(f"{len(rec)} recordings -> {drafts}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
