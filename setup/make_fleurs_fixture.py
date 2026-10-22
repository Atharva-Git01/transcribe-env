"""Turn part of a downloaded FLEURS test split (google/fleurs, CC-BY-4.0) into ONE eval fixture: N clips (one per
sentence, different sentences) joined with 1 s of silence, plus truth.json in the trx.truth/1 format.

  python setup/make_fleurs_fixture.py --lang mr --tsv .../mr_in_test.tsv --tar .../mr_in_test_part.tar.gz --n 10

Setup-time tool. The tar.gz may be a byte-range prefix: only complete WAV members are used. Members are read
in memory by base name (no paths from the archive are ever written), so a hostile archive cannot escape.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import subprocess
import sys
import tarfile
import wave
from pathlib import Path

import numpy as np

HOME = Path(__file__).resolve().parents[1]
csv.field_size_limit(10**7)


def members(tar_gz: Path):
    """Yield (base name, bytes) of complete regular-file members; stop quietly at a truncated end."""
    with open(tar_gz, "rb") as raw:
        try:
            with tarfile.open(fileobj=gzip.GzipFile(fileobj=raw), mode="r|") as tf:
                for m in tf:
                    if not m.isfile() or not m.name.lower().endswith(".wav"):
                        continue
                    f = tf.extractfile(m)
                    data = f.read() if f else b""
                    if len(data) == m.size:
                        yield Path(m.name).name, data
        except (EOFError, tarfile.ReadError, OSError, gzip.BadGzipFile):
            return


def read_wav_bytes(data: bytes, sr: int = 16000) -> tuple[np.ndarray, int]:
    """Any WAV (FLEURS uses 32-bit float) -> mono float32 at sr, through ffmpeg (E12: decode only via ffmpeg)."""
    res = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", "pipe:0", "-ac", "1", "-ar", str(sr),
                          "-f", "f32le", "-"], input=data, capture_output=True, timeout=120)
    if res.returncode != 0:
        raise RuntimeError(res.stderr.decode("utf-8", "replace")[-200:])
    return np.frombuffer(res.stdout, dtype=np.float32), sr


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", required=True)
    ap.add_argument("--tsv", required=True)
    ap.add_argument("--tar", required=True)
    ap.add_argument("--n", type=int, default=10)
    a = ap.parse_args()
    rows = {}
    with open(a.tsv, encoding="utf-8", newline="") as f:
        for r in csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
            if len(r) >= 3:
                rows[r[1]] = {"sid": r[0], "text": r[2]}
    picked, seen_sid, audio, sr0 = [], set(), [], None
    for name, data in members(Path(a.tar)):
        r = rows.get(name)
        if not r or r["sid"] in seen_sid:   # one recording per sentence: different sentences only
            continue
        x, sr = read_wav_bytes(data)
        if sr0 and sr != sr0:
            continue
        sr0 = sr
        seen_sid.add(r["sid"])
        picked.append({"file": name, "sentence_id": r["sid"], "text": r["text"], "sec": round(len(x) / sr, 2)})
        audio += [x, np.zeros(int(sr), dtype=np.float32)]
        if len(picked) >= a.n:
            break
    if not picked:
        print("no complete clips found")
        return 1
    fid = f"fleurs_{a.lang}_test{len(picked)}"
    out = HOME / "evals" / "fixtures" / fid
    out.mkdir(parents=True, exist_ok=True)
    pcm = (np.clip(np.concatenate(audio[:-1]), -1, 1) * 32767).astype(np.int16)
    with wave.open(str(out / "audio.wav"), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr0)
        w.writeframes(pcm.tobytes())
    sha = hashlib.sha256((out / "audio.wav").read_bytes()).hexdigest()
    truth = {
        "schema": "trx.truth/1", "id": fid, "audio": "audio.wav", "audio_sha256": sha, "language": a.lang,
        "duration_sec": round(len(pcm) / sr0, 2), "speakers": ["reader"],
        "reference_text": " ".join(p["text"] for p in picked),
        "segments": [{"speaker": "reader", "text": p["text"]} for p in picked],
        "source": "Google FLEURS test split (https://huggingface.co/datasets/google/fleurs), CC-BY-4.0; clips: "
                  + ", ".join(p["file"] for p in picked),
        "notes": f"{len(picked)} read Wikipedia sentences (clean studio-like audio, NOT phone calls), joined with 1 s "
                 f"silence; possibly several readers. Use to check that changes do not hurt clean speech.",
    }
    (out / "truth.json").write_text(json.dumps(truth, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{fid}: {len(picked)} clips, {truth['duration_sec']} s, {len(truth['reference_text'])} chars -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
