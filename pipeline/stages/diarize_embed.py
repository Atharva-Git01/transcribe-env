"""Diarization by speaker embeddings on sliding windows (mode "embed_cluster", E19).

1. Windows of `window_s` (1.0 s) every `hop_s` (0.25 s) inside the VAD speech regions.
2. A speaker embedding per window: 3D-Speaker ERes2NetV2 (Apache-2.0). GPU first (PLAN_v0.9 B6): with
   onnxruntime-gpu the windows run in batches on the GPU, on features computed exactly as sherpa-onnx computes them
   (`_fbank`, identical to 0.0002); otherwise one by one through sherpa-onnx on the CPU, as before.
3. Clustering (cosine k-means, farthest-point start):
   - `num_speakers` > 0: exactly that many (phone calls: 2 parties);
   - 0: estimated by splitting clusters while the two halves are less similar than `split_cosine` (at most
     `max_speakers`; every speaker needs `min_speaker_s` of speech). Marked "estimated" — not validated yet.
4. Viterbi smoothing with a switching penalty (`switch_penalty`): speakers do not change every quarter second.
   Across a pause of at least `pause_min_s` the cost is `switch_penalty_pause` (lower: turns change at pauses, E39).
5. Turns = runs of one speaker; turns shorter than `min_turn_s` join their neighbour.
6. Second pass (PLAN_v0.9 B2, off unless the recipe sets it): `short_regions = "own_voice"` gives a speech region too
   short for a window (< 0.5 s, mostly replies) a padded voice sample of its own, labelled by the nearest speaker
   centroid, instead of the nearest window's speaker; `rescan_long_turns_s` > 0 re-scans every turn at least that
   long with `rescan_window_s` windows every `rescan_hop_s` and splits off a run of at least `rescan_min_run_s` where
   the other speaker's centroid is closer by more than `rescan_margin` (a reply under the other speaker's turn).
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


_MEL: dict = {}


def _fbank(x: np.ndarray) -> np.ndarray:
    """The model's input for one window, exactly as sherpa-onnx computes it (Kaldi fbank as configured by sherpa for
    this model): 25 ms frames every 10 ms, centred, edges reflected (snip_edges = false), DC removed, pre-emphasis
    0.97, Povey window, 512-point power spectrum, 80 mel bands 20-7600 Hz, log; then minus the mean over the
    window's frames (the model's `feature_normalize_type` = global-mean)."""
    if "W" not in _MEL:
        mel = lambda f: 1127.0 * np.log(1.0 + f / 700.0)  # noqa: E731
        lo, hi = mel(20.0), mel(7600.0)
        d = (hi - lo) / 81
        m = mel((16000 / 512) * np.arange(256))
        W = np.zeros((80, 256))
        for b in range(80):
            left, c, r = lo + b * d, lo + (b + 1) * d, lo + (b + 2) * d
            W[b] = np.where((m > left) & (m < r), np.minimum((m - left) / (c - left), (r - m) / (r - c)), 0.0)
        _MEL["W"] = W.T
        _MEL["win"] = (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(400) / 399)) ** 0.85
    x = x.astype(np.float64)
    n = len(x)
    idx = (160 * np.arange((n + 80) // 160) - 120)[:, None] + np.arange(400)[None, :]
    idx = np.where(idx < 0, -idx - 1, idx)
    idx = np.where(idx >= n, 2 * n - 1 - idx, idx)
    fr = x[idx]
    fr = fr - fr.mean(1, keepdims=True)
    pre = fr.copy()
    pre[:, 1:] = fr[:, 1:] - 0.97 * fr[:, :-1]
    pre[:, 0] = fr[:, 0] - 0.97 * fr[:, 0]
    spec = np.abs(np.fft.rfft(pre * _MEL["win"], 512)) ** 2
    f = np.log(np.maximum(spec[:, :256] @ _MEL["W"], np.finfo(np.float32).eps))
    return (f - f.mean(0, keepdims=True)).astype(np.float32)


_SESSION: dict = {}


def _gpu_session():
    """An onnxruntime CUDA session of the model (one per process), or None (no onnxruntime-gpu, no usable GPU, or
    the CPU forced)."""
    if "s" not in _SESSION:
        _SESSION["s"] = _new_gpu_session()
    return _SESSION["s"]


def _new_gpu_session():
    from ..common import load_config
    if load_config("defaults")["runtime"].get("device", "auto") == "cpu":
        return None
    try:
        import onnxruntime as ort
        if "CUDAExecutionProvider" not in ort.get_available_providers():
            return None
        from .. import gpu
        gpu.prepare()
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        sess = ort.InferenceSession(str(MODEL), sess_options=opts,
                                    providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        return sess if sess.get_providers()[0] == "CUDAExecutionProvider" else None
    except Exception:  # noqa: BLE001 — the CPU path is always there
        return None


def embed(audio: np.ndarray, wins, sr: int = 16000, batch: int = 32) -> np.ndarray:
    """Unit-length embeddings of the windows; on the GPU in batches of equal-length windows when possible."""
    sess = _gpu_session() if sr == 16000 else None
    if sess is not None:
        feats = [_fbank(audio[int(s * sr):int(e * sr)]) for s, e in wins]
        X = np.zeros((len(wins), 192))
        by_len: dict = {}
        for i, f in enumerate(feats):
            by_len.setdefault(f.shape[0], []).append(i)
        for idxs in by_len.values():
            for k in range(0, len(idxs), batch):
                part = idxs[k:k + batch]
                X[part] = sess.run(None, {"x": np.stack([feats[i] for i in part])})[0]
        embed.last_device = "cuda"
        return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    embed.last_device = "cpu"
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


def penalties(wins, speech, inside: float, at_pause: float, pause_min: float) -> np.ndarray:
    """Cost of a speaker change before each window: `at_pause` when the window starts a speech region that follows
    at least `pause_min` s of silence (people take turns at pauses: a short reply between two pauses keeps its own
    speaker, E39), else `inside`."""
    pen, k, prev = np.full(len(wins), float(inside)), 0, None
    for t, (s, _) in enumerate(wins):
        while k + 1 < len(speech) and s >= speech[k + 1][0] - 1e-6:
            k += 1
        if prev is not None and k != prev and speech[k][0] - speech[prev][1] >= pause_min:
            pen[t] = at_pause
        prev = k
    return pen


def viterbi(S: np.ndarray, penalty) -> np.ndarray:
    """penalty: cost of a speaker change between neighbouring windows; one number or one per window (index t = the
    change between windows t-1 and t)."""
    T, K = S.shape
    pen = np.broadcast_to(np.asarray(penalty, dtype=float), (T,))
    D, B = S[0].copy(), np.zeros((T, K), dtype=int)
    for t in range(1, T):
        cand = D[:, None] - pen[t] * (1 - np.eye(K))      # [from, to]
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
    info["embeddings_on"] = getattr(embed, "last_device", "cpu")
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
        inside = float(params.get("switch_penalty", 0.1))
        lab = viterbi(X @ C.T, penalties(wins, speech, inside, float(params.get("switch_penalty_pause", inside)),
                                         float(params.get("pause_min_s", 0.2))))
    if k > 1 and params.get("short_regions", "nearest") == "own_voice":
        extra = short_windows(speech, win, 0.5)
        if extra:
            E = embed(audio, [w for w, _ in extra])
            wins = list(wins) + [region for _, region in extra]
            lab = np.concatenate([lab, (E @ C.T).argmax(1)])
            order = np.argsort([(a + b) / 2 for a, b in wins], kind="stable")
            wins, lab = [wins[i] for i in order], lab[order]
            info["short_regions_own_voice"] = len(extra)
    turns = turns_from(wins, lab, speech, hop, float(params.get("min_turn_s", 0.3)))
    if k > 1 and float(params.get("rescan_long_turns_s", 0)) > 0:
        turns, n = rescan(audio, turns, C, params)
        info["rescan_splits"] = n
    cos = float(np.min(C @ C.T)) if k > 1 else 1.0
    return turns, dict(info, speakers=k, how=how, min_centroid_cosine=round(cos, 3))


def short_windows(speech, win: float, min_len: float) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """[(padded window, region)] for every speech region shorter than min_len: the region in the middle of a window
    of `min_len` (the silence around it adds no voice)."""
    out = []
    for a, b in speech:
        if b - a < min_len:
            pad = (min_len - (b - a)) / 2
            out.append(((max(0.0, a - pad), b + pad), (a, b)))
    return out


def rescan(audio: np.ndarray, turns: list[dict], C: np.ndarray, params: dict) -> tuple[list[dict], int]:
    """Split off the other speaker's runs inside long turns (module docstring, step 6)."""
    min_s = float(params["rescan_long_turns_s"])
    w, h = float(params.get("rescan_window_s", 0.6)), float(params.get("rescan_hop_s", 0.1))
    margin, min_run = float(params.get("rescan_margin", 0.1)), float(params.get("rescan_min_run_s", 0.5))
    out, splits = [], 0
    grid = {i: np.arange(t["start"], t["end"] - w + 1e-6, h) for i, t in enumerate(turns) if t["end"] - t["start"] >= min_s}
    flat = [(float(s), float(s) + w) for i in grid for s in grid[i]]
    Xall = embed(audio, flat) if flat else np.zeros((0, C.shape[1]))
    at, rows = 0, {}
    for i in grid:
        rows[i] = Xall[at:at + len(grid[i])]
        at += len(grid[i])
    for ti, t in enumerate(turns):
        if ti not in grid:
            out.append(t)
            continue
        starts = grid[ti]
        sc = rows[ti] @ C.T
        own = int(t["speaker"])
        other = np.where(np.arange(sc.shape[1]) == own, -9, sc).argmax(1)
        ahead = sc[np.arange(len(sc)), other] - sc[:, own] > margin
        cur, runs = None, []
        for i, on in enumerate(list(ahead) + [False]):
            if on and cur is None:
                cur = i
            elif not on and cur is not None:
                a, b = starts[cur] + w / 2 - h / 2, starts[i - 1] + w / 2 + h / 2
                if b - a >= min_run:
                    runs.append((max(t["start"], a), min(t["end"], b), int(np.bincount(other[cur:i]).argmax())))
                cur = None
        pos = t["start"]
        for a, b, spk in runs:
            if a - pos > 0.05:
                out.append({"start": round(pos, 3), "end": round(a, 3), "speaker": own})
            out.append({"start": round(a, 3), "end": round(b, 3), "speaker": spk})
            pos = b
            splits += 1
        if t["end"] - pos > 0.05:
            out.append({"start": round(pos, 3), "end": round(t["end"], 3), "speaker": own})
    return out, splits


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
