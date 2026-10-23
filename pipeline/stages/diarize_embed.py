"""Diarization by speaker embeddings on sliding windows (mode "embed_cluster", E19).

1. Windows of `window_s` (1.0 s) every `hop_s` (0.25 s) inside the VAD speech regions.
2. A speaker embedding per window: 3D-Speaker ERes2NetV2 (Apache-2.0) through sherpa-onnx (no torch).
3. Clustering (cosine k-means, farthest-point start):
   - `num_speakers` > 0: exactly that many (phone calls: 2 parties);
   - 0: estimated by splitting clusters while the two halves are less similar than `split_cosine` (at most
     `max_speakers`; every speaker needs `min_speaker_s` of speech). Marked "estimated" — not validated yet.
4. Viterbi smoothing with a switching penalty (`switch_penalty`): speakers do not change every quarter second.
5. Turns = runs of one speaker; turns shorter than `min_turn_s` join their neighbour.
Measured on the Pappa call (2 parties, mono narrowband): 94.5 % of words get the right speaker
(pyannote-segmentation pipeline: <= 71 %, everything-one-speaker: 54 %).
"""
from __future__ import annotations

import hashlib
import json

import numpy as np

from ..audio import read_wav
from ..common import TRX_HOME, env_version, sha256_file, write_json
from ..context import Ctx

MODEL_DIR = TRX_HOME / "toolchain" / "models" / "speaker-eres2netv2"
MODEL = MODEL_DIR / "3dspeaker_speech_eres2netv2_sv_zh-cn_16k-common.onnx"

try:
    import sherpa_onnx  # noqa: F401
    AVAILABLE = MODEL.is_file() and (MODEL_DIR / "SOURCE.json").is_file()
except ImportError:
    AVAILABLE = False


def windows(speech, win: float, hop: float, min_len: float = 0.5) -> list[tuple[float, float]]:
    out = []
    for a, b in speech:
        if b - a < min_len:          # too short for a stable embedding: labelled later from its neighbours
            continue
        s = a
        while True:
            e = min(s + win, b)
            if e - s >= min_len:
                out.append((round(s, 3), round(e, 3)))
            if e >= b:
                break
            s += hop
    return out


def embed(audio: np.ndarray, wins, sr: int = 16000) -> np.ndarray:
    import sherpa_onnx
    ext = sherpa_onnx.SpeakerEmbeddingExtractor(sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=str(MODEL),
                                                                                          num_threads=4))
    X = []
    for s, e in wins:
        st = ext.create_stream()
        st.accept_waveform(sr, audio[int(s * sr):int(e * sr)].astype(np.float32))
        st.input_finished()
        v = np.asarray(ext.compute(st), dtype=np.float64)
        X.append(v / (np.linalg.norm(v) + 1e-9))
    return np.stack(X) if X else np.zeros((0, ext.dim))


def kmeans(X: np.ndarray, k: int, iters: int = 100) -> tuple[np.ndarray, np.ndarray]:
    """Cosine k-means with deterministic farthest-point initialisation."""
    C = [X[int(np.argmin(X @ X.mean(0)))]]          # the window least like the average
    while len(C) < k:
        sims = np.max(X @ np.stack(C).T, axis=1)
        C.append(X[int(np.argmin(sims))])
    C = np.stack(C)
    lab = np.zeros(len(X), dtype=int)
    for _ in range(iters):
        lab = (X @ C.T).argmax(1)
        newC = np.stack([X[lab == j].mean(0) if (lab == j).any() else C[j] for j in range(k)])
        newC /= np.linalg.norm(newC, axis=1, keepdims=True) + 1e-9
        if np.allclose(newC, C):
            break
        C = newC
    return lab, C


def estimate(X: np.ndarray, hop: float, max_k: int, split_cos: float, min_speaker_s: float) -> np.ndarray:
    """Divisive clustering: split the cluster whose best 2-split is least similar, while below split_cos."""
    lab = np.zeros(len(X), dtype=int)
    min_win = max(2, int(min_speaker_s / hop))
    while lab.max() + 1 < max_k:
        best = None
        for c in range(lab.max() + 1):
            idx = np.where(lab == c)[0]
            if len(idx) < 2 * min_win:
                continue
            sub, C = kmeans(X[idx], 2)
            if min(np.bincount(sub, minlength=2)) < min_win:
                continue
            cos = float(C[0] @ C[1])
            if cos < split_cos and (best is None or cos < best[0]):
                best = (cos, idx, sub)
        if best is None:
            break
        _, idx, sub = best
        lab[idx[sub == 1]] = lab.max() + 1
    return lab


def viterbi(S: np.ndarray, penalty: float) -> np.ndarray:
    T, K = S.shape
    D, B = S[0].copy(), np.zeros((T, K), dtype=int)
    for t in range(1, T):
        cand = D[:, None] - penalty * (1 - np.eye(K))      # [from, to]
        B[t] = cand.argmax(0)
        D = cand.max(0) + S[t]
    path = [int(D.argmax())]
    for t in range(T - 1, 0, -1):
        path.append(int(B[t, path[-1]]))
    return np.array(path[::-1])


def turns_from(wins, labels, speech, hop: float, min_turn: float) -> list[dict]:
    """Each window owns the hop-long slice around its centre; slices -> turns, clipped to speech regions."""
    pieces = []
    for (s, e), lab in zip(wins, labels):
        c = (s + e) / 2
        pieces.append([c - hop / 2, c + hop / 2, int(lab)])
    # cover every speech region fully: region parts without a window take the nearest window's speaker
    turns = []
    for a, b in speech:
        inside = [p for p in pieces if p[1] > a and p[0] < b]
        if not inside:
            if pieces:
                near = min(pieces, key=lambda p: min(abs(p[0] - b), abs(p[1] - a)))
                turns.append({"start": a, "end": b, "speaker": near[2]})
            continue
        inside[0][0], inside[-1][1] = a, b
        for p in inside:
            p0, p1 = max(p[0], a), min(p[1], b)
            if turns and turns[-1]["speaker"] == p[2] and p0 - turns[-1]["end"] <= 1.0:
                turns[-1]["end"] = p1
            else:
                turns.append({"start": p0, "end": p1, "speaker": p[2]})
    changed = True
    while changed and len(turns) > 1:
        changed = False
        for i, t in enumerate(turns):
            if t["end"] - t["start"] < min_turn:
                j = i - 1 if i > 0 else i + 1
                turns[j]["start"] = min(turns[j]["start"], t["start"])
                turns[j]["end"] = max(turns[j]["end"], t["end"])
                turns.pop(i)
                changed = True
                break
    merged = []
    for t in turns:
        if merged and merged[-1]["speaker"] == t["speaker"] and t["start"] - merged[-1]["end"] <= 1.0:
            merged[-1]["end"] = t["end"]
        else:
            merged.append(dict(t))
    return [{"start": round(t["start"], 3), "end": round(t["end"], 3), "speaker": t["speaker"]} for t in merged]


def diarize(audio: np.ndarray, speech, params: dict) -> tuple[list[dict], dict]:
    win, hop = float(params.get("window_s", 1.0)), float(params.get("hop_s", 0.25))
    wins = windows(speech, win, hop)
    info = {"windows": len(wins)}
    if len(wins) < 2:
        return ([{"start": a, "end": b, "speaker": 0} for a, b in speech], dict(info, speakers=1, how="too little speech"))
    X = embed(audio, wins)
    k = int(params.get("num_speakers", 0))
    if k > 0:
        lab, C = kmeans(X, min(k, len(X)))
        how = f"fixed: {k} speakers (recipe)"
    else:
        lab = estimate(X, hop, int(params.get("max_speakers", 4)), float(params.get("split_cosine", 0.68)),
                       float(params.get("min_speaker_s", 5.0)))
        k = int(lab.max()) + 1
        C = np.stack([X[lab == j].mean(0) for j in range(k)])
        C /= np.linalg.norm(C, axis=1, keepdims=True) + 1e-9
        how = f"estimated: {k} speaker(s) (not validated on labelled multi-speaker audio yet)"
    if k > 1:
        lab = viterbi(X @ C.T, float(params.get("switch_penalty", 0.1)))
    turns = turns_from(wins, lab, speech, hop, float(params.get("min_turn_s", 0.3)))
    cos = float(np.min(C @ C.T)) if k > 1 else 1.0
    return turns, dict(info, speakers=k, how=how, min_centroid_cosine=round(cos, 3))


def run(ctx: Ctx, params: dict) -> None:
    tr = ctx.tracks[0]
    path = ctx.work / "diarization.json"
    key = hashlib.sha256(json.dumps({"audio": sha256_file(tr.wav16), "speech": tr.speech, "params": params,
                                     "env": env_version(), "model": MODEL.name}, sort_keys=True).encode()).hexdigest()
    if path.is_file():  # resumed run: same audio, same settings -> reuse the saved result
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("key") == key:
            ctx.turns = data["turns"]
            ctx.metrics.update(diarization="embed_cluster", speakers_found=data["info"]["speakers"])
            ctx.log.add("diarize", f"reused saved diarization: {data['info']['how']}",
                        "Same audio and settings as before the interruption.")
            return
    audio, _ = read_wav(tr.wav16)
    raw, info = diarize(audio, tr.speech, params)
    order = []
    for t in raw:  # SPK1, SPK2 … in order of first appearance
        if t["speaker"] not in order:
            order.append(t["speaker"])
    ctx.turns = [{"start": t["start"], "end": t["end"], "speaker": f"SPK{order.index(t['speaker']) + 1}"} for t in raw]
    write_json(path, {"key": key, "engine": f"embed_cluster + {MODEL.stem}", "info": info, "turns": ctx.turns})
    ctx.custody.add(path, "derived", "diarize")
    ctx.metrics.update(diarization="embed_cluster", speakers_found=info["speakers"],
                       speakers_how="fixed" if info["how"].startswith("fixed") else "estimated")
    ctx.log.add("diarize", f"{info['how']}; {len(ctx.turns)} turns",
                "Speakers separated by voice: speaker embeddings on 1 s windows, clustered and smoothed (E19).",
                **info)
