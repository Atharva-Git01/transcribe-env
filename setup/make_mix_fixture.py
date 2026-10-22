"""Make a "speech with music behind it" eval fixture: an existing fixture's audio + a music file at a chosen
speech-to-music ratio. The truth (text, speakers) is the base fixture's, unchanged. Setup-time tool.

  python setup/make_mix_fixture.py --base fleurs_mr_test10 --music <file> --snr-db 5 \
      --music-credit "Scott Joplin, Sunflower Slow Drag, US Marine Band (public domain, Wikimedia Commons)"
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np

HOME = Path(__file__).resolve().parents[1]


def decode(path: Path, sr: int = 16000) -> np.ndarray:
    res = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", str(path), "-ac", "1", "-ar", str(sr), "-f", "f32le",
                          "-"], capture_output=True, timeout=600)
    if res.returncode != 0:
        raise SystemExit(res.stderr.decode("utf-8", "replace")[-300:])
    return np.frombuffer(res.stdout, dtype=np.float32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--music", required=True)
    ap.add_argument("--snr-db", type=float, required=True)
    ap.add_argument("--music-credit", required=True)
    a = ap.parse_args()
    base = HOME / "evals" / "fixtures" / a.base
    truth = json.loads((base / "truth.json").read_text(encoding="utf-8"))
    speech = decode(base / truth["audio"])
    music = decode(Path(a.music))
    reps = int(np.ceil(len(speech) / len(music)))
    music = np.tile(music, reps)[: len(speech)]
    def rms(x): return float(np.sqrt(np.mean(np.square(x, dtype=np.float64)) + 1e-12))
    music *= rms(speech) / rms(music) / (10 ** (a.snr_db / 20))
    mix = speech + music
    mix /= max(1.0, float(np.max(np.abs(mix))) / 0.98)          # no clipping
    fid = f"{a.base}_music{int(a.snr_db)}db"
    out = HOME / "evals" / "fixtures" / fid
    out.mkdir(parents=True, exist_ok=True)
    with wave.open(str(out / "audio.wav"), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes((np.clip(mix, -1, 1) * 32767).astype(np.int16).tobytes())
    t = dict(truth)
    t.update(id=fid, audio="audio.wav", audio_sha256=hashlib.sha256((out / "audio.wav").read_bytes()).hexdigest(),
             source=truth["source"] + f" + background music at speech/music {a.snr_db:g} dB: {a.music_credit}",
             notes=f"{truth.get('notes', '')} Background music mixed in at {a.snr_db:g} dB below the speech "
                   f"(tests the Demucs separate stage).")
    (out / "truth.json").write_text(json.dumps(t, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{fid}: {len(mix) / 16000:.1f} s, speech/music {a.snr_db:g} dB -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
