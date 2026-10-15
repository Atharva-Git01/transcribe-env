"""Audio helpers: ffmpeg calls (local paths only) and 16-bit WAV read/write with numpy."""
from __future__ import annotations

import subprocess
import wave
from pathlib import Path

import numpy as np

from .common import TrxError


def ffmpeg(args: list[str], timeout: int = 7200) -> None:
    cmd = ["ffmpeg", "-nostdin", "-hide_banner", "-v", "error", "-y", *args]
    res = subprocess.run(cmd, capture_output=True, timeout=timeout)
    if res.returncode != 0:
        raise TrxError("INTERNAL_ERROR", "ffmpeg failed: " + res.stderr.decode("utf-8", "replace")[-400:])


def decode_mono(path: Path, sr: int = 16000) -> np.ndarray:
    """Any local media file -> mono float32 at `sr`, via ffmpeg (no PyAV dependency)."""
    cmd = ["ffmpeg", "-nostdin", "-v", "error", "-i", str(path), "-map", "0:a:0", "-ac", "1", "-ar", str(sr),
           "-f", "f32le", "-"]
    res = subprocess.run(cmd, capture_output=True, timeout=7200)
    if res.returncode != 0:
        raise TrxError("INPUT_UNREADABLE", res.stderr.decode("utf-8", "replace")[-300:])
    return np.frombuffer(res.stdout, dtype=np.float32).copy()


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    """Mono 16-bit PCM WAV -> float32 in [-1, 1]."""
    with wave.open(str(path), "rb") as w:
        if w.getsampwidth() != 2:
            raise TrxError("INTERNAL_ERROR", f"{path.name}: expected 16-bit PCM")
        sr, ch, n = w.getframerate(), w.getnchannels(), w.getnframes()
        data = np.frombuffer(w.readframes(n), dtype=np.int16).astype(np.float32) / 32768.0
    if ch > 1:
        data = data.reshape(-1, ch).mean(axis=1)
    return data, sr


def write_wav(path: Path, audio: np.ndarray, sr: int) -> None:
    pcm = np.clip(audio, -1.0, 1.0)
    pcm = (pcm * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())


def rms_dbfs(audio: np.ndarray) -> float:
    if audio.size == 0:
        return -120.0
    return float(10.0 * np.log10(np.mean(np.square(audio, dtype=np.float64)) + 1e-12))


def fmt_ts(sec: float, sep: str = ",") -> str:
    ms = int(round(sec * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"
