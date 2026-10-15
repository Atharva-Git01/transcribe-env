"""Stage: polish. Flags (never deletes) suspicious text; optional vocab and Gemma 3 cleanup (phase 3).

Flags: low_confidence · repetitive · loop · prompt_copy · known_hallucination_phrase · wrong_script ·
hindi_forms_in_marathi · possible_omission (speech heard, no text). Real doubled words are exempt from loop checks.
"""
from __future__ import annotations

import re

from ..common import TRX_HOME, load_config
from ..context import Ctx

ARABIC = re.compile(r"[؀-ۿ]")
# letters that are neither Devanagari nor Latin (e.g. Bengali, Gujarati, Arabic) in a mr/hi segment = wrong script
FOREIGN_LETTER = re.compile(r"[^ऀ-ॿ᳐-᳿꣠-ꣿA-Za-z0-9\s\.,!?;:'\"()\-–—…।॥%/&+@#*\[\]]")
CHAR_RUN = re.compile(r"(.)\1{7,}")  # the same character 8+ times in a row (decoder loop inside one "word")
HINDI_IN_MARATHI = {"नहीं", "क्या", "मैं", "हैं", "है", "हूँ", "कहाँ"}
WORD = re.compile(r"[^\s\.,!?।॥;:\"'()]+")


def _wordlist(path) -> set[str]:
    if not path.is_file():
        return set()
    return {ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip() and not ln.startswith("#")}


def find_loop(tokens: list[str], doubled: set[str]) -> str | None:
    # same token 3+ times in a row (4+ if it is a known real doubled word)
    run = 1
    for i in range(1, len(tokens)):
        run = run + 1 if tokens[i] == tokens[i - 1] else 1
        limit = 4 if tokens[i] in doubled else 3
        if run >= limit:
            return f"'{tokens[i]}' repeated {run}x"
    # an n-gram (2..6 words) repeated 3+ times back to back
    for n in range(2, 7):
        for i in range(0, len(tokens) - 3 * n + 1):
            gram = tokens[i:i + n]
            if tokens[i + n:i + 2 * n] == gram and tokens[i + 2 * n:i + 3 * n] == gram:
                if n == 2 and gram[0] == gram[1] and gram[0] in doubled:
                    continue
                return f"'{' '.join(gram)}' repeated 3x"
    return None


def _prompt_shingles(text: str, k: int = 4) -> set[tuple]:
    toks = WORD.findall(text)
    return {tuple(toks[i:i + k]) for i in range(len(toks) - k + 1)}


def run(ctx: Ctx, params: dict) -> None:
    thr = load_config("thresholds")["verify"]
    lib = TRX_HOME / "library"
    rec = ctx.stage_params("recognise")
    prompt_tpl = rec.get("initial_prompt_file", "")
    flagged = 0
    for s in ctx.segments:
        lang = s.get("language") or ctx.language or ""
        doubled = _wordlist(lib / "doubled_words" / f"{lang}.txt")
        phrases = _wordlist(lib / "hallucination_phrases" / f"{lang}.txt") | _wordlist(lib / "hallucination_phrases" / "any.txt")
        prompt_path = TRX_HOME / prompt_tpl.format(lang=lang) if prompt_tpl else None
        shingles = _prompt_shingles(prompt_path.read_text(encoding="utf-8")) if prompt_path and prompt_path.is_file() else set()
        tokens = WORD.findall(s["text"])
        flags = list(s.get("flags", []))  # keep flags from earlier stages (verify)
        if s.get("avg_logprob") is not None and s["avg_logprob"] < thr["min_avg_logprob"]:
            flags.append(f"low_confidence (avg_logprob {s['avg_logprob']})")
        if s.get("no_speech_prob") is not None and s["no_speech_prob"] > thr["max_no_speech_prob"]:
            flags.append(f"low_confidence (no_speech_prob {s['no_speech_prob']})")
        if s.get("compression_ratio") and s["compression_ratio"] > thr["max_compression_ratio"]:
            flags.append(f"repetitive (compression ratio {s['compression_ratio']})")
        if params.get("doubled_words_guard", True):
            loop = find_loop(tokens, doubled)
            if loop:
                flags.append(f"loop: {loop}")
        run = CHAR_RUN.search(s["text"])
        if run:
            flags.append(f"loop: character '{run.group(1)}' repeated {len(run.group(0))}x")
        if shingles and _prompt_shingles(s["text"]) & shingles:
            flags.append("prompt_copy: text matches the initial prompt")
        low = s["text"].lower()
        hit = [p for p in phrases if p.lower() in low]
        if hit:
            flags.append(f"known_hallucination_phrase: {hit[0]}")
        if lang in ("mr", "hi"):
            foreign = sorted(set(FOREIGN_LETTER.findall(s["text"])))
            if foreign:
                kind = "Arabic/Urdu" if ARABIC.search(s["text"]) else "non-Devanagari"
                flags.append(f"wrong_script: {kind} characters ({''.join(foreign[:5])})")
        if lang == "mr":
            hindi = sorted(set(tokens) & HINDI_IN_MARATHI)
            if hindi:
                flags.append(f"hindi_forms_in_marathi: {', '.join(hindi)}")
        s["flags"] = flags
        if flags:
            flagged += 1
            ctx.review.append({"start": s["start"], "end": s["end"], "speaker": s["speaker"], "text": s["text"],
                               "reasons": flags})

    # speaker turns from diarization (phase 4): each segment gets the speaker with the most overlap
    if ctx.turns:
        for s in ctx.segments:
            share = {}
            for t in ctx.turns:
                ov = max(0.0, min(s["end"], t["end"]) - max(s["start"], t["start"]))
                if ov > 0:
                    share[t["speaker"]] = share.get(t["speaker"], 0.0) + ov
            if share:
                best = max(share, key=share.get)
                s["speaker"] = best
                dur = max(1e-6, s["end"] - s["start"])
                minority = sum(v for k, v in share.items() if k != best) / dur
                if minority > 0.3:
                    s["flags"].append(f"speaker_change_inside_segment ({minority:.0%} by another speaker)")
                    if len(s["flags"]) == 1:
                        flagged += 1
                        ctx.review.append({"start": s["start"], "end": s["end"], "speaker": s["speaker"],
                                           "text": s["text"], "reasons": s["flags"]})

    # true stereo: a segment much quieter on its own channel than on the other one is crosstalk (bleed)
    if ctx.metrics.get("diarization") == "channel" and len(ctx.tracks) == 2:
        from ..audio import read_wav, rms_dbfs
        limit = float(ctx.stage_params("diarize").get("crosstalk_suppression_db", 12.0))
        audio = [read_wav(t.wav16)[0] for t in ctx.tracks]
        for s in ctx.segments:
            i = 0 if s["track"] == ctx.tracks[0].label else 1
            a, b = int(s["start"] * 16000), int(s["end"] * 16000)
            own, other = rms_dbfs(audio[i][a:b]), rms_dbfs(audio[1 - i][a:b])
            if other - own > limit:
                s["flags"].append(f"crosstalk: {other - own:.0f} dB louder on the other channel")
                if len(s["flags"]) == 1:
                    flagged += 1
                    ctx.review.append({"start": s["start"], "end": s["end"], "speaker": s["speaker"],
                                       "text": s["text"], "reasons": s["flags"]})

    # possible omissions: VAD speech with (almost) no recognised words
    min_sec = float(thr["omission_min_speech_sec"])
    for tr in ctx.tracks:
        segs = [s for s in ctx.segments if s["track"] == tr.label]
        for a, b in tr.speech:
            if b - a < min_sec:
                continue
            covered = sum(max(0.0, min(b, s["end"]) - max(a, s["start"])) for s in segs)
            if covered < 0.2 * (b - a):
                ctx.review.append({"start": a, "end": b, "speaker": tr.label, "text": "",
                                   "reasons": [f"possible_omission: {b - a:.1f} s of speech with no recognised text"]})
    ctx.review.sort(key=lambda r: r["start"])
    ctx.metrics["segments"] = len(ctx.segments)
    ctx.metrics["flagged_segments"] = flagged
    ctx.metrics["possible_omissions"] = sum(1 for r in ctx.review if r["reasons"][0].startswith("possible_omission"))
    ctx.log.add("polish", f"flagged {flagged} of {len(ctx.segments)} segments; "
                f"{ctx.metrics['possible_omissions']} possible omissions",
                "Doubtful text is kept and marked for review, never silently deleted (E02).")

    defaults = load_config("defaults")
    if defaults["llm_cleanup"]["enabled"]:
        from . import cleanup
        cleanup.run(ctx, defaults["llm_cleanup"])
    else:
        ctx.log.add("polish", "LLM cleanup off", "config/defaults.toml [llm_cleanup].enabled = false.")
