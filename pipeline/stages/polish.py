"""Stage: polish. Flags (never deletes) suspicious text; optional vocab and Gemma 3 cleanup (phase 3).

Flags: low_confidence · uncertain · repetitive · loop · prompt_copy · known_hallucination_phrase · wrong_script ·
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
WORD = re.compile(r"[^\s\.,!?।॥;:\"'()]+")


LATIN_OR_DIGIT = re.compile(r"[A-Za-z0-9]")
LATIN_WORD = re.compile(r"[A-Za-z][A-Za-z'\-]*")


def uncertain(s: dict, lang: str, thr: dict) -> str | None:
    """Soft flag when the recogniser itself was unsure of a line (E40): the mean log-probability of the tokens it
    chose (Indic-Transcribe, `token_logprob_mean`) below `min_token_logprob_mean`. A line of acknowledgements only
    ("हं", "हो" …) is left alone: the model hesitates between their spellings, not between meanings."""
    conf = s.get("token_logprob_mean")
    if conf is None or conf >= float(thr.get("min_token_logprob_mean", float("-inf"))):
        return None
    from ..metrics import normalise, scoring_words
    words = normalise(s.get("text", "")).split()
    if words and all(w in scoring_words(lang)["backchannels"] for w in words):
        return None
    return f"uncertain: the recogniser was unsure of these words (mean token log-prob {conf})"


def _lexicon(lang: str):
    """The language model's vocabulary (millions of real word forms from IndicCorp v2), if installed."""
    from .. import lm
    return lm.load(lang) if lang and lm.installed(lang) else None


def lm_tokens(text: str) -> list[str]:
    from ..lm import tokens
    return tokens(text)


def _wordlist(path) -> set[str]:
    if not path.is_file():
        return set()
    return {ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip() and not ln.startswith("#")}


def repetition(text: str, ratio: float | None, thr: dict) -> str | None:
    """'repetitive' means the same phrases again and again (E37). Natural speech, human-written: at most 18 % of a
    line's word triples are repeats; a phrase said twice gives 50 %, decoder loops 60-85 %. The compression ratio is
    only a backstop for repetition without words, with a length-aware limit no natural text reached (long Devanagari
    lines compress past the old fixed 2.4 by length alone)."""
    words = WORD.findall(text)
    tri = [" ".join(words[i:i + 3]) for i in range(len(words) - 2)]
    if len(tri) >= int(thr.get("repeat_min_triples", 6)):
        share = 1.0 - len(set(tri)) / len(tri)
        if share >= float(thr.get("repeat_share_max", 0.35)):
            return f"{share:.0%} of its word triples repeat"
    if ratio:
        limit = max(float(thr["max_compression_ratio"]), float(thr.get("compression_base", 2.25))
                    + float(thr.get("compression_per_char", 0.0016)) * len(text)) + float(thr.get("compression_margin", 0.35))
        if ratio > limit:
            return f"compression ratio {ratio} above {limit:.2f} for {len(text)} characters"
    return None


def find_loop(tokens: list[str], doubled: set[str]) -> str | None:
    # a whole short turn of one listed backchannel ("हं हं हं हं") is how people acknowledge, not a loop (E37)
    if 2 <= len(tokens) <= 5 and len(set(tokens)) == 1 and tokens[0] in doubled:
        return None
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


def _turn_at(turns: list[dict], t: float) -> str | None:
    hit = [x for x in turns if x["start"] <= t <= x["end"]]
    if hit:
        return hit[0]["speaker"]
    near = min(turns, key=lambda x: min(abs(x["start"] - t), abs(x["end"] - t))) if turns else None
    return near["speaker"] if near and min(abs(near["start"] - t), abs(near["end"] - t)) <= 1.0 else None


def split_by_turns(segments: list[dict], turns: list[dict]) -> list[dict]:
    """Cut a segment where the speaker changes (each word goes to the diarization turn at its midpoint).
    Text is unchanged — only line boundaries move — so every transcript line has one speaker."""
    if not turns:
        return segments
    out = []
    for seg in segments:
        words = seg.get("words") or []
        labels = [_turn_at(turns, (w["start"] + w["end"]) / 2) for w in words]
        known = [lab for lab in labels if lab]
        if len(words) < 2 or len(set(known)) < 2:
            out.append(seg)
            continue
        for i, lab in enumerate(labels):  # unknown words join the previous (or next) speaker
            labels[i] = lab or (labels[i - 1] if i else next(x for x in labels if x))
        pieces, cur = [], [0]
        for i in range(1, len(words)):
            if labels[i] != labels[cur[0]]:
                pieces.append(cur)
                cur = []
            cur.append(i)
        pieces.append(cur)
        from .recognise import text_compression_ratio
        for idx in pieces:
            ws = [words[i] for i in idx]
            text = "".join(w["word"] for w in ws).strip()
            out.append(dict(seg, start=ws[0]["start"], end=ws[-1]["end"], text=text, words=ws,
                            compression_ratio=text_compression_ratio(text),
                            split_from=seg.get("split_from") or [seg["start"], seg["end"]], split_reason="speaker change"))
    for i, s in enumerate(out, 1):
        s["id"] = i
    return out


def cleanup_decision(llm: dict, primary: list[str], gemma_ready) -> str:
    """'run' | 'engine' (not for this engine family, E26) | 'missing' (Gemma not available, e.g. on an unplugged USB
    stick: the text is kept, the run goes on) | 'off'. gemma_ready is called only when the cleanup would run."""
    if not llm.get("enabled"):
        return "off"
    families = list(llm.get("engines", []))
    if families and not any(e.startswith(f) for e in primary for f in families):
        return "engine"
    return "run" if gemma_ready() else "missing"


def run(ctx: Ctx, params: dict) -> None:
    thr = load_config("thresholds")["verify"]
    lib = TRX_HOME / "library"
    rec = ctx.stage_params("recognise")
    prompt_tpl = rec.get("initial_prompt_file", "")
    if ctx.turns:
        before = len(ctx.segments)
        ctx.segments = split_by_turns(ctx.segments, ctx.turns)
        if len(ctx.segments) != before:
            ctx.log.add("polish", f"split {len(ctx.segments) - before} line(s) at speaker changes",
                        "Every transcript line gets one speaker; words are not changed, only where lines break.")
    out_cfg = load_config("defaults")["output"]
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
        unsure = uncertain(s, lang, thr)
        if unsure:
            flags.append(unsure)
        rep = repetition(s["text"], s.get("compression_ratio"), thr)
        if rep:
            flags.append(f"repetitive ({rep})")
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
        if s.get("recovered"):
            flags.append("recovered_turn: decoded on its own after the first pass missed it")
        if any(q["track"] == s.get("track") and min(q["end"], s["end"]) - max(q["start"], s["start"]) > 0.1
               for q in ctx.quiet_regions):
            flags.append("quiet_speech: found only by the second, more sensitive speech check")
        # unknown words are looked up in the standard-spelling copy: a spoken form is not a misrecognition
        lex = _lexicon(lang) if out_cfg.get("spelling", "standard") in ("standard", "as_spoken") else None
        if lex is not None:
            hot = set(rec.get("hotwords") or [])
            unknown = [w for w in lm_tokens(s.get("text_standard", s["text"])) if len(w) > 1
                       and not LATIN_OR_DIGIT.search(w)
                       and w not in hot and lex.count(w) < int(thr.get("lexicon_min_count", 3))]
            if unknown:
                flags.append(f"unknown_words: {', '.join(dict.fromkeys(unknown))} (not in the {lang} lexicon)")
        if lang in ("mr", "hi") and out_cfg.get("english_words") == "devanagari":
            latin = [w for w in WORD.findall(s["text"]) if LATIN_WORD.fullmatch(w)]
            if latin:
                flags.append(f"latin_script: {', '.join(latin[:5])} (convention: English words in Devanagari)")
        if lang == "mr":
            from .cleanup import lang_forms
            hindi = sorted(set(tokens) & lang_forms("mr")[0])
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
        # review items made before the speakers were known still carry the track label "SPK?" (E35)
        speaker_at = {(s["start"], s["end"]): s["speaker"] for s in ctx.segments}
        for r in ctx.review:
            r["speaker"] = speaker_at.get((r["start"], r["end"]), r["speaker"])

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
    llm = defaults["llm_cleanup"]
    primary = [e["engine"] for e in ctx.engines_used if e.get("role") != "cross_check"]
    families = list(llm.get("engines", []))
    from ..llm.gemma3 import gguf_path, server_exe
    decision = cleanup_decision(llm, primary, lambda: bool(server_exe()) and bool(gguf_path(llm.get("model", ""))))
    if decision == "run":
        from . import cleanup
        cleanup.run(ctx, llm)
    elif decision == "engine":
        ctx.log.add("polish", f"LLM cleanup skipped for {', '.join(primary) or 'this engine'}",
                    f"config/defaults.toml [llm_cleanup].engines = {families}: with this engine the cleanup changed "
                    "nothing in eval and only cost time (E26).")
    elif decision == "missing":
        ctx.metrics["llm_cleanup"] = "skipped: Gemma 3 not available"
        ctx.log.add("polish", "LLM cleanup skipped: Gemma 3 is not available",
                    "llama-server or the Gemma 3 model folder is missing (it may live on the USB stick, see "
                    "toolchain/models/); the text is kept exactly as recognised.")
    else:
        ctx.log.add("polish", "LLM cleanup off", "config/defaults.toml [llm_cleanup].enabled = false.")

    if out_cfg.get("numbers") == "words":
        from ..numbers import digits_to_words
        converted, left_over = 0, set()
        for seg in ctx.segments:
            lang = seg.get("language") or ctx.language or ""
            new, n, left = digits_to_words(seg["text"], lang)
            if n:
                seg.setdefault("text_asr", seg["text"])
                seg["text"] = new
                seg.setdefault("edits", []).append({"by": "numbers as words", "count": n})
                converted += n
            left_over |= set(left)
        if converted or left_over:
            ctx.log.add("polish", f"numbers as words: {converted} converted"
                        + (f"; left as written: {sorted(left_over)}" if left_over else ""),
                        "Owner convention: numbers are written as words (decimals, times and similar are left as "
                        "recognised rather than guessed).")
