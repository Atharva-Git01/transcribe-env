"""Stage: lid. Decide the language: forced (--lang or recipe) > passport measurement > fresh detection.
With per_segment = true (code-switched audio), every speech group gets its own language."""
from __future__ import annotations

from .. import gpu
from ..audio import read_wav
from ..common import TrxError, load_config
from ..context import Ctx
from ..engines.faster_whisper_engine import make
from ..lid import detect, lid_engine_id


def _groups(speech, max_len: float, max_gap: float = 1.0):
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
    return groups


def segment_language(probs: dict, main: str, priority: list[str], en_min: float) -> str:
    """One speech group: English only on strong evidence (P(en) >= en_min against every other candidate);
    otherwise the main language. A wrong "en" makes Whisper TRANSLATE Marathi into English (E20)."""
    if main != "en":
        return "en" if probs.get("en", 0.0) >= en_min else main
    indic = 1.0 - probs.get("en", 1.0)
    return priority[0] if probs and indic >= en_min else "en"


def run(ctx: Ctx, params: dict) -> None:
    rec = ctx.stage_params("recognise")
    forced = ctx.plan["overrides"].get("lang") or (rec.get("language") if rec.get("language") not in ("auto", "per_segment") else None)
    if params.get("mode") == "forced" and not forced:
        raise TrxError("CONFIG_INVALID", "lid.mode = forced but no language given")
    if forced:
        ctx.language, ctx.language_source = forced, "forced"
    else:
        lang_info = ctx.passport["language"]
        dec = lang_info.get("decision") or {}
        top = lang_info.get("top3")
        if dec.get("lang"):
            ctx.language, ctx.language_source = dec["lang"], f"passport decision ({dec.get('basis')})"
        elif top:
            ctx.language, ctx.language_source = top[0]["lang"], "passport (acoustic LID only)"
        else:
            ctx.language, ctx.language_source = None, "not measured"

    if params.get("per_segment"):
        # Per speech group only "main language vs English": Marathi and Hindi cannot be told apart reliably on one
        # short group (E10), and the main language is the confirmed (or measured) one. English groups are decoded
        # as English, so code-switched speech keeps its English words in Latin script.
        langs_cfg = load_config("defaults")["languages"]
        priority = langs_cfg["priority"]
        main = ctx.language or priority[0]
        en_min = float(langs_cfg.get("segment_en_min", 0.8))
        # Whisper spreads Indic speech over mr AND hi (E10), so English must beat ALL candidate languages together
        # and reach en_min; otherwise the group stays in the main language (a wrong "en" makes Whisper translate).
        cand = list(dict.fromkeys([main] + [c for c in priority if c != "en"] + ["en"]))
        eng_id = lid_engine_id()
        if not eng_id:
            raise TrxError("MODEL_MISSING", "no installed LID engine (config [languages].lid_engines)")
        eng = make(eng_id)
        eng.load(gpu.choose_device(), rec.get("faster_whisper", {}))
        max_len = float(ctx.stage_params("vad").get("max_segment_s", 15.0))
        try:
            for tr in ctx.tracks:
                audio, _ = read_wav(tr.wav16)
                for a, b in _groups(tr.speech, max_len):
                    if b - a < 2.0:
                        lang, probs = main, {}
                    else:
                        res = detect(eng, audio, [(a, b)], cand)
                        p = res["per_window"][0]["probs"] if res["per_window"] else {}
                        lang = segment_language(p, main, priority, en_min)
                        probs = {"acoustic": p}
                    ctx.segment_languages.append({"track": tr.label, "start": a, "end": b, "language": lang,
                                                  "probs": probs})
        finally:
            eng.unload()
        counts = {c: sum(1 for g in ctx.segment_languages if g["language"] == c) for c in (main, "en")}
        ctx.log.add("lid", f"per-group language ({main} vs English) for {len(ctx.segment_languages)} speech groups: "
                    f"{counts}", "Code-switched audio: English groups are recognised as English, the rest in the "
                    "main language (Marathi vs Hindi is never decided per group, E10).", engine=eng_id)
    if not ctx.language:
        ctx.language = load_config("defaults")["languages"]["priority"][0]
        ctx.language_source = "default (no measurement available)"
    for tr in ctx.tracks:
        tr.language = ctx.language
    ctx.log.add("lid", f"file language = {ctx.language}", f"Source: {ctx.language_source}.",
                passport_top3=ctx.passport["language"].get("top3"))
