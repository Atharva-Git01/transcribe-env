"""Stage: lid. Decide the language: forced (--lang or recipe) > passport measurement > fresh detection.
With per_segment = true (code-switched audio), every speech group gets its own language."""
from __future__ import annotations

from .. import gpu
from ..audio import read_wav
from ..common import TrxError, load_config
from ..context import Ctx
from ..engines.faster_whisper_engine import make
from ..lid import detect, lid_engine_id, refine


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


def run(ctx: Ctx, params: dict) -> None:
    rec = ctx.stage_params("recognise")
    candidates = list(params.get("candidates", ["mr", "hi", "en"]))
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

    if params.get("per_segment") and not forced:
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
                        lang, probs = ctx.language, {}
                    else:
                        res = detect(eng, audio, [(a, b)], candidates)
                        ref = refine(eng, audio, [(a, b)], res) if res["top3"] else None  # one group
                        lang = (ref["decision"] if ref and ref["decision"] else
                                (res["top3"][0]["lang"] if res["top3"] else ctx.language))
                        probs = {"acoustic": res["per_window"][0]["probs"] if res["per_window"] else {},
                                 "text": ref["evidence"] if ref else {}}
                    ctx.segment_languages.append({"track": tr.label, "start": a, "end": b, "language": lang,
                                                  "probs": probs})
        finally:
            eng.unload()
        langs = sorted({g["language"] for g in ctx.segment_languages if g["language"]})
        ctx.log.add("lid", f"per-segment language for {len(ctx.segment_languages)} speech groups: {langs}",
                    "Mixed-language audio: each speech group is recognised in its own language.", engine=eng_id)
    if not ctx.language:
        ctx.language = load_config("defaults")["languages"]["priority"][0]
        ctx.language_source = "default (no measurement available)"
    for tr in ctx.tracks:
        tr.language = ctx.language
    ctx.log.add("lid", f"file language = {ctx.language}", f"Source: {ctx.language_source}.",
                passport_top3=ctx.passport["language"].get("top3"))
