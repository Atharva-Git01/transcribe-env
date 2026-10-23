"""Stage: verify (phase 2). A second, different local engine transcribes the same speech; per segment the two
transcripts are compared. Low agreement = the segment is flagged and the other engine's text is attached as an
alternative for the reviewer. Nothing is overwritten (model log-probs are not comparable across models).
"""
from __future__ import annotations

from .. import gpu
from ..audio import read_wav
from ..common import load_config, write_json
from ..context import Ctx
from ..engines.faster_whisper_engine import make
from ..metrics import cer
from ..models import installed
from .recognise import _decode_track, engine_choices


def _cross_engine(ctx: Ctx, table: dict) -> str | None:
    key = "mixed" if ctx.segment_languages else (ctx.language or "unknown")
    primary = {e["engine"] for e in ctx.engines_used}
    return next((e for e in engine_choices(table, key) if installed(e) and e not in primary), None)


def run(ctx: Ctx, params: dict) -> None:
    rec = ctx.stage_params("recognise")
    table = rec.get("engine_table_cross_check", {})
    if not table.get("enabled", False):
        ctx.log.add("verify", "skipped", "Cross-check disabled in config/defaults.toml [engines.cross_check].")
        return
    eng_id = _cross_engine(ctx, table)
    if not eng_id:
        ctx.log.add("verify", "skipped", "No installed cross-check engine different from the primary engine.")
        return
    min_agree = float(load_config("thresholds")["verify"]["min_agreement"])
    from .recognise import engine_params
    fw = engine_params(rec, eng_id)
    fw.pop("segmentation", None)
    vad = ctx.stage_params("vad") or None
    eng = make(eng_id)
    info = eng.load(gpu.choose_device(), fw)
    alt_all = []
    try:
        for tr in ctx.tracks:
            audio, _ = read_wav(tr.wav16)
            # same clips as the primary engine (E14), so segments line up for the agreement score
            segs = _decode_track(eng, audio, tr, ctx, dict(rec, word_timestamps=True), vad, fw, ctx.language,
                                 rec.get("initial_prompt_file", ""))
            for s in segs:
                s["track"] = tr.label
            alt_all += segs
    finally:
        eng.unload()
    disagree = 0
    for s in ctx.segments:
        # the other engine's words whose midpoint falls inside this segment: works for any segmentation on either side
        other = "".join(w["word"] for a in alt_all if a["track"] == s["track"] for w in (a.get("words") or [])
                        if s["start"] <= (w["start"] + w["end"]) / 2 <= s["end"]).strip()
        agreement = round(1.0 - min(1.0, cer(s["text"], other)), 3) if other else 0.0
        s["agreement"] = agreement
        s["alternative"] = {"engine": eng_id, "text": other}
        if agreement < min_agree:
            disagree += 1
            s.setdefault("flags", []).append(f"engines_disagree (agreement {agreement:.2f} with {eng_id})")
    path = ctx.work / "asr_crosscheck.json"
    write_json(path, {"engine": info, "segments": alt_all})
    ctx.custody.add(path, "derived", "verify")
    ctx.engines_used.append(dict(info, role="cross_check"))
    ctx.metrics["cross_check_engine"] = eng_id
    ctx.metrics["engine_disagreements"] = disagree
    ctx.log.add("verify", f"cross-checked with {eng_id}: {disagree} of {len(ctx.segments)} segments disagree",
                f"Two different models agreeing is strong evidence; disagreement (< {min_agree}) goes to review.",
                **{k: v for k, v in info.items() if k != "engine"})
