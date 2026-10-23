"""Language identification helpers (Whisper encoder LID, restricted to the candidate languages).

Used by probe (file-level top-3 + code-switch risk, BEFORE the recipe is chosen) and by the lid stage
(per-segment decisions for code-switched audio).
"""
from __future__ import annotations

import re

import numpy as np

from .common import TRX_HOME, load_config
from .models import first_installed

SR = 16000
INDIC = {"mr", "hi"}


def lid_engine_id() -> str | None:
    langs = load_config("defaults")["languages"]
    return first_installed(langs.get("lid_engines", []))


def windows_from_speech(speech: list[tuple[float, float]], duration: float, n: int, win: float) -> list[tuple[float, float]]:
    """Up to n windows of `win` seconds, centred on speech, spread evenly over the file."""
    if not speech:
        return [(0.0, min(win, duration))]
    centres = []
    total = sum(e - s for s, e in speech)
    for k in range(n):
        target = total * (k + 0.5) / n
        acc = 0.0
        for s, e in speech:
            if acc + (e - s) >= target:
                centres.append(s + (target - acc))
                break
            acc += e - s
    out = []
    for c in centres:
        a = max(0.0, min(c - win / 2, duration - win))
        w = (round(a, 2), round(min(duration, a + win), 2))
        if w not in out:
            out.append(w)
    return out


LATIN = re.compile(r"^[A-Za-z][A-Za-z'\-]*$")
TOKEN = re.compile(r"[^\s\.,!?।॥;:\"'()\-–—…]+")


def _markers(lang: str) -> set[str]:
    p = TRX_HOME / "library" / "lid_markers" / f"{lang}.txt"
    if not p.is_file():
        return set()
    return {w.strip() for w in p.read_text(encoding="utf-8").splitlines() if w.strip() and not w.startswith("#")}


def text_evidence(texts: list[str]) -> dict:
    """DISTINCT Marathi vs Hindi marker words (distinct, so a decoder loop cannot inflate a count) + Latin share."""
    mr, hi = _markers("mr"), _markers("hi")
    toks = [t for txt in texts for t in TOKEN.findall(txt)]
    n_lat = sum(1 for t in toks if LATIN.match(t))
    return {"tokens": len(toks), "mr_markers": len(set(toks) & mr), "hi_markers": len(set(toks) & hi),
            "latin_words": n_lat, "latin_ratio": round(n_lat / len(toks), 3) if toks else 0.0}


def speech_groups(speech: list[tuple[float, float]], max_len: float = 15.0, max_gap: float = 1.0, take: int = 8):
    groups, cur = [], None
    for s, e in speech:
        if cur and s - cur[1] <= max_gap and e - cur[0] <= max_len:
            cur[1] = e
        else:
            if cur:
                groups.append(tuple(cur))
            cur = [s, e]
    if cur:
        groups.append(tuple(cur))
    groups = [g for g in groups if g[1] - g[0] >= 1.5]
    if len(groups) > take:  # evenly spread, deterministic
        step = len(groups) / take
        groups = [groups[int(i * step)] for i in range(take)]
    return groups


def acoustic_decision(acoustic: dict, engine_id: str) -> dict:
    """Step 2 for an engine whose own language ID tells Marathi from Hindi (`lid_trusted`, Indic-Transcribe: right on
    5/5 fixtures with P = 1.0): the acoustic decision stands and no forced decoding runs (it took 28 s per file with
    Whisper, E34). Same result shape as refine(); weak evidence still falls back to the owner's priority language."""
    cfg = load_config("defaults")["languages"]
    windows = acoustic.get("per_window", [])
    top = acoustic["top3"][0]
    agree = sum(1 for w in windows if w["top"] == top["lang"])
    en_windows = sum(1 for w in windows if w["top"] == "en")
    decision, conf = top["lang"], float(top["prob"])
    basis = f"acoustic language ID of {engine_id}: {decision} in {agree}/{len(windows)} windows (mean P {conf:.2f})"
    priority = [lang for lang in cfg.get("priority", []) if lang in INDIC]
    if decision in INDIC and priority and conf < float(cfg.get("text_lid_min_confidence", 0.65)) \
            and decision != priority[0]:
        basis = f"{basis}; evidence weak -> priority language {priority[0]}"
        decision = priority[0]
    risk = "high" if decision in INDIC and en_windows >= int(cfg.get("code_switch_min_en_windows", 3)) else "low"
    return {"decision": decision, "basis": basis, "code_switch_risk": risk, "confidence": round(conf, 3),
            "evidence": {"windows": len(windows), "agree": agree, "en_windows": en_windows}}


def refine(engine, audio: np.ndarray, speech: list[tuple[float, float]], acoustic: dict) -> dict:
    """Step 2 of LID (E10). The same short speech groups are decoded twice, forced to Marathi and forced to Hindi
    (beam 5, temperature 0). Looping segments (compression ratio > 2.4) are dropped. Decision:
      1) duration-weighted mean log-probability: the language the model is more confident in, if the gap is
         >= text_lid_ll_margin;
      2) otherwise distinct marker words (library/lid_markers) from both decodes;
      3) otherwise the acoustic decision, with a warning.
    English / code-switching stays acoustic (forced Indic decoding transliterates English)."""
    cfg = load_config("defaults")["languages"]
    groups = speech_groups(speech)
    q = {"beam_size": 5, "best_of": 1, "temperature": [0.0], "condition_on_previous_text": False}
    ll, texts = {}, {"mr": [], "hi": []}
    for lang in ("mr", "hi"):
        num = den = 0.0
        for a, b in groups:
            for s in engine.transcribe(audio[int(a * SR):int(b * SR)], lang, q, None, None, False):
                if s["compression_ratio"] <= 2.4 and s["text"]:
                    d = max(0.1, s["end"] - s["start"])
                    num += s["avg_logprob"] * d
                    den += d
                    texts[lang].append(s["text"])
        ll[lang] = round(num / den, 4) if den else None
    # Symmetric "survival" evidence: Marathi words that appear even when the decoder is FORCED to Hindi, versus
    # Hindi words that appear even when it is FORCED to Marathi. Each forced decode pulls toward its own language,
    # so only the cross-language survivors are evidence about the audio.
    ev_mr_in_hi = text_evidence(texts["hi"])
    ev_hi_in_mr = text_evidence(texts["mr"])
    ev = {"tokens": ev_mr_in_hi["tokens"] + ev_hi_in_mr["tokens"],
          "mr_markers": ev_mr_in_hi["mr_markers"],   # Marathi markers surviving Hindi-forced decoding
          "hi_markers": ev_hi_in_mr["hi_markers"],   # Hindi markers surviving Marathi-forced decoding
          "latin_ratio": round((ev_mr_in_hi["latin_words"] + ev_hi_in_mr["latin_words"]) /
                               max(1, ev_mr_in_hi["tokens"] + ev_hi_in_mr["tokens"]), 3)}
    acoustic_top = acoustic["top3"][0]["lang"] if acoustic.get("top3") else None
    en_windows = sum(1 for w in acoustic.get("per_window", []) if w["top"] == "en")
    n_windows = len(acoustic.get("per_window", []))
    margin = float(cfg.get("text_lid_ll_margin", 0.03))
    conf = None
    if acoustic_top == "en" and en_windows >= max(1, int(0.6 * n_windows)):
        decision, basis = "en", f"acoustic: English in {en_windows}/{n_windows} windows"
    elif ll["mr"] is not None and ll["hi"] is not None and abs(ll["mr"] - ll["hi"]) >= margin:
        decision = "mr" if ll["mr"] > ll["hi"] else "hi"
        basis = f"forced-decode log-prob mr {ll['mr']} vs hi {ll['hi']}"
        conf = min(1.0, 0.5 + abs(ll["mr"] - ll["hi"]) * 5)
    elif ev["mr_markers"] + ev["hi_markers"] >= int(cfg.get("text_lid_min_markers", 3)):
        decision = "mr" if ev["mr_markers"] >= ev["hi_markers"] else "hi"
        n = ev["mr_markers"] + ev["hi_markers"]
        basis = (f"cross-forced survivors: Marathi words under Hindi decoding {ev['mr_markers']} vs Hindi words "
                 f"under Marathi decoding {ev['hi_markers']} (log-prob gap < {margin})")
        conf = max(ev["mr_markers"], ev["hi_markers"]) / n
    else:
        decision, basis = acoustic_top, "acoustic only (forced decoding inconclusive)"
    # Weak Marathi-vs-Hindi evidence: the owner's priority language wins, and the run is marked REVIEW
    # (probe adds a language_ambiguous warning; the gate turns that into REVIEW). Whisper is Hindi-biased, so
    # without this rule Marathi audio would mostly be decoded as Hindi (E10).
    priority = [l for l in cfg.get("priority", []) if l in INDIC]
    min_conf = float(cfg.get("text_lid_min_confidence", 0.65))
    if decision in INDIC and priority and (conf is None or conf < min_conf) and decision != priority[0]:
        basis = f"{basis}; evidence weak (confidence {round(conf, 2) if conf is not None else None}) -> priority language {priority[0]}"
        decision = priority[0]
    indic = decision in INDIC
    risk = "high" if indic and en_windows >= int(cfg.get("code_switch_min_en_windows", 3)) else "low"
    ev.update(ll_mr=ll["mr"], ll_hi=ll["hi"], groups=len(groups), en_windows=en_windows)
    return {"decision": decision, "basis": basis, "evidence": ev, "code_switch_risk": risk,
            "confidence": round(conf, 3) if conf is not None else None}


def detect(engine, audio: np.ndarray, windows: list[tuple[float, float]], candidates: list[str]) -> dict:
    per_window = []
    for a, b in windows:
        chunk = audio[int(a * SR):int(b * SR)]
        if chunk.size < SR:  # < 1 s: not enough for LID
            continue
        per_window.append({"window": [a, b], "probs": engine.detect_language(chunk, candidates)})
    if not per_window:
        return {"top3": None, "per_window": [], "code_switch_risk": None}
    mean = {c: float(np.mean([w["probs"][c] for w in per_window])) for c in candidates}
    ranked = sorted(mean.items(), key=lambda kv: -kv[1])
    cfg = load_config("defaults")["languages"]
    en_min = float(cfg.get("code_switch_en_min", 0.20))
    tops = {max(w["probs"], key=w["probs"].get) for w in per_window}
    indic_strong = any(mean.get(l, 0) >= 0.30 for l in INDIC)
    risk = "high" if (("en" in tops and tops & INDIC) or (mean.get("en", 0) >= en_min and indic_strong)) else "low"
    return {
        "top3": [{"lang": l, "prob": round(p, 3)} for l, p in ranked[:3]],
        "per_window": [{"window": w["window"], "top": max(w["probs"], key=w["probs"].get),
                        "probs": {k: round(v, 3) for k, v in w["probs"].items()}} for w in per_window],
        "code_switch_risk": risk,
    }
