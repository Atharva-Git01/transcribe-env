"""Cut a small TRAINING slice from IndicVoices Marathi (ai4bharat/IndicVoices, AI4Bharat, CC-BY-4.0): spontaneous
speech (Conversation, Extempore) with its Level 1 text written as spoken (column `verbatim`; `normalized` holds the
standard spelling of the same words, e.g. as-spoken "nay" vs standard "nahi").

  python setup/make_indicvoices_slice.py --out toolchain/data/indicvoices-mr-slice1 --hours 2

Setup-time tool (internet once). The dataset is gated: the owner accepts its terms on the dataset page and logs in
with `hf auth login`; Claude never types or reads tokens. Only the parquet row groups that are used are read, by
HTTP range requests (about 100 MB each), never whole shards; one row group from each of --shards shards spread over
the 72, so the slice holds many speakers and districts. Writes into an empty --out:
  audio/<id>.flac  the dataset's own FLAC bytes (header checked: 16 kHz mono, sample count; its names never used)
  manifest.jsonl   one clip per line: verbatim (as spoken), normalized (standard), unsanitized_verbatim (with
                   [noise]-style tags), duration, speaker fields, district, scenario, source row, sha256
  SOURCE.json      written last (marks a complete slice): pinned revision, rows used, rules, licence, credit, totals
The slice is training data only, never a test set (PLAN_v0.9 section 4.2).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from datetime import date
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem

HOME = Path(__file__).resolve().parents[1]
REPO = "ai4bharat/IndicVoices"
REVISION = "c96f9088f138cf89d419da7e8e643e1f05c00a87"
N_SHARDS = 72
CREDIT = ("IndicVoices, AI4Bharat (Javed et al., 2024: IndicVoices: Towards building an Inclusive Multilingual "
          "Speech Dataset for Indian Languages), CC-BY-4.0, https://huggingface.co/datasets/ai4bharat/IndicVoices")
META = ["duration", "samples", "verbatim", "normalized", "unsanitized_verbatim", "speaker_id", "gender", "age_group",
        "area", "district", "scenario", "task_name"]
CONVERSATION_SHARE = 0.7    # our calls are conversations; Extempore fills the rest and adds speakers
MIN_S, MAX_S = 0.5, 30.0    # short replies are kept on purpose: they are where the recognizer invents words
SPEAKER_CAP_S = 180.0       # at most 3 minutes per speaker over the whole slice
NOT_SPOKEN = re.compile(r"[0-9A-Za-z\[\]<>{}]")   # digits, Latin letters, tags: not the as-spoken Devanagari style


def flac_info(data: bytes) -> tuple[int, int, int] | None:
    """(sample rate, channels, total samples) from the FLAC STREAMINFO header; None if not a FLAC stream."""
    if len(data) < 42 or data[:4] != b"fLaC" or data[4] & 0x7F != 0:
        return None
    b = int.from_bytes(data[18:26], "big")
    return b >> 44, ((b >> 41) & 0x7) + 1, b & ((1 << 36) - 1)


def pick(rows: list[dict], budget_s: float, used: dict[str, float]) -> list[int]:
    """Indexes of the clips kept from one row group: Conversation up to its share of the budget, then Extempore up
    to the whole budget, in a fixed pseudo-random order, with the per-speaker cap counted over the whole slice."""
    order = sorted(range(len(rows)), key=lambda i: hashlib.sha256(f"{rows[i]['speaker_id']}|{i}".encode()).digest())
    keep, spent = [], 0.0
    for scenario, cap in (("Conversation", budget_s * CONVERSATION_SHARE), ("Extempore", budget_s)):
        for i in order:
            r = rows[i]
            text = (r["verbatim"] or "").strip()
            if r["scenario"] != scenario or not MIN_S <= r["duration"] <= MAX_S or not text or NOT_SPOKEN.search(text):
                continue
            if spent + r["duration"] > cap or used.get(r["speaker_id"], 0.0) + r["duration"] > SPEAKER_CAP_S:
                continue
            keep.append(i)
            spent += r["duration"]
            used[r["speaker_id"]] = used.get(r["speaker_id"], 0.0) + r["duration"]
    return sorted(keep)


def read_group(pf: pq.ParquetFile, g: int, columns: list[str]):
    for attempt in range(3):
        try:
            return pf.read_row_group(g, columns=columns)
        except OSError as e:
            if attempt == 2:
                raise
            print(f"  read failed ({e}); retrying")
            time.sleep(10 * (attempt + 1))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--hours", type=float, default=2.0)
    ap.add_argument("--shards", type=int, default=8)
    a = ap.parse_args()
    out = Path(a.out) if Path(a.out).is_absolute() else HOME / a.out
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} is not empty: choose a new --out (nothing was changed)")
    (out / "audio").mkdir(parents=True, exist_ok=True)
    fs = HfFileSystem()
    budget = a.hours * 3600 / a.shards
    used: dict[str, float] = {}
    manifest, parts, fetched = [], [], 0
    for k, shard in enumerate(round(j * N_SHARDS / a.shards) for j in range(a.shards)):
        name = f"marathi/train-{shard:05d}-of-{N_SHARDS:05d}.parquet"
        pf = pq.ParquetFile(fs.open(f"datasets/{REPO}@{REVISION}/{name}", "rb", block_size=16 << 20))
        g = k % pf.metadata.num_row_groups
        rows = read_group(pf, g, META).to_pylist()
        keep = pick(rows, budget, used)
        if not keep:
            print(f"{name} group {g}: nothing usable")
            continue
        audio = read_group(pf, g, ["audio_filepath"]).column(0)
        rg = pf.metadata.row_group(g)
        fetched += sum(rg.column(c).total_compressed_size for c in range(rg.num_columns)
                       if rg.column(c).path_in_schema.split(".")[0] in {*META, "audio_filepath"})
        bad = 0
        for i in keep:
            r, data = rows[i], audio[i].as_py()["bytes"]
            if flac_info(data) != (16000, 1, r["samples"]):
                bad += 1
                continue
            cid = f"ivmr_s{shard:02d}_g{g}_r{i:04d}"
            (out / "audio" / f"{cid}.flac").write_bytes(data)
            manifest.append({"id": cid, "audio": f"audio/{cid}.flac", "sha256": hashlib.sha256(data).hexdigest(),
                             **{c: r[c] for c in META}, "source": {"file": name, "row_group": g, "row": i}})
        mins = sum(rows[i]["duration"] for i in keep) / 60
        parts.append({"file": name, "row_group": g, "rows_in_group": len(rows), "clips": len(keep) - bad})
        print(f"{name} group {g}: {len(keep) - bad} clips, {mins:.1f} min" + (f", {bad} bad audio skipped" if bad else ""))
    if not manifest:
        raise SystemExit("no clips selected")
    with open(out / "manifest.jsonl", "w", encoding="utf-8", newline="\n") as f:
        for m in manifest:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")
    hours = sum(m["duration"] for m in manifest) / 3600
    by_scenario = {s: round(sum(m["duration"] for m in manifest if m["scenario"] == s) / 3600, 3)
                   for s in ("Conversation", "Extempore")}
    totals = {"clips": len(manifest), "hours": round(hours, 3), "hours_by_scenario": by_scenario,
              "speakers": len({m["speaker_id"] for m in manifest}),
              "districts": sorted({m["district"] for m in manifest}),
              "clips_with_spoken_forms": sum(m["verbatim"] != m["normalized"] for m in manifest),
              "fetched_mb": round(fetched / 1e6), "stored_mb": round(sum((out / m["audio"]).stat().st_size
                                                                         for m in manifest) / 1e6, 1)}
    (out / "SOURCE.json").write_text(json.dumps({
        "id": out.name, "dataset": REPO, "subset": "marathi (train split)", "revision": REVISION,
        "licence": "CC-BY-4.0", "credit": CREDIT, "made": date.today().isoformat(),
        "tool": "setup/make_indicvoices_slice.py", "role": "training data only, never a test set (PLAN_v0.9 4.2)",
        "text_levels": {"verbatim": "Level 1, as spoken (training target)", "normalized": "Level 2, standard spelling",
                        "unsanitized_verbatim": "Level 1 with [noise]-style tags"},
        "rules": {"scenarios": {"Conversation": CONVERSATION_SHARE, "Extempore": round(1 - CONVERSATION_SHARE, 2)},
                  "duration_s": [MIN_S, MAX_S], "speaker_cap_s": SPEAKER_CAP_S,
                  "verbatim_excludes": NOT_SPOKEN.pattern, "hours": a.hours, "shards": a.shards},
        "parts": parts, "totals": totals}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(totals, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
