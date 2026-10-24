"""Stage: merge (PLAN_v0.9 section 6): the spelling transfer.

The recogniser's words stay. Indic-Transcribe with a decoder LoRA trained to write as spoken (setup/train_it_lora.py)
reads every line again, and wherever both readings hold the same words once spelling is folded away, its spelling is
used (spelling.transfer). A word that differs is never taken, so the content of the transcript cannot change; the
standard copy keeps the recogniser's spelling (`text_standard`), and every changed word is logged in decisions.json.

Recipe: [merge] mode = "spelling_transfer", lora = "<adapter folder>". Off (mode = "off") until an adapter passes the
screening, `trx eval` and the owner's listening check (PLAN_v0.9 section 5). A missing adapter skips the stage with a
log line; the transcript is then in the recogniser's spelling, and the header says so.
"""
from __future__ import annotations

from .. import gpu
from ..audio import read_wav
from ..common import TRX_HOME
from ..context import Ctx
from ..engines.faster_whisper_engine import make


def run(ctx: Ctx, params: dict) -> None:
    mode = params.get("mode", "off")
    if mode != "spelling_transfer":
        ctx.log.add("merge", "skipped", f"[merge] mode = {mode!r}: no spelling step is adopted (PLAN_v0.9).")
        return
    from pathlib import Path

    from ..spelling import transfer
    from .recognise import engine_params
    lora = Path(params.get("lora", ""))
    lora = lora if lora.is_absolute() else TRX_HOME / lora
    primary = next((e["engine"] for e in ctx.engines_used if e.get("role") not in ("cross_check", "meaning_check")), "")
    if not primary.startswith("indic_transcribe") or not (lora / "adapter_config.json").is_file():
        ctx.log.add("merge", "spelling transfer skipped", f"Needs Indic-Transcribe as the recogniser (here: "
                    f"{primary or 'none'}) and the adapter {lora.name} (missing: it may live on the USB stick).")
        return
    rec = ctx.stage_params("recognise")
    fw = dict(engine_params(rec, primary), lora=str(lora))
    eng = make(primary)
    info = eng.load(gpu.choose_device(), fw)
    changed_lines, changes = 0, []
    try:
        audio_of = {tr.label: read_wav(tr.wav16)[0] for tr in ctx.tracks}
        for s in ctx.segments:
            audio = audio_of.get(s.get("track"))
            if audio is None or not s.get("text"):
                continue
            lang = s.get("language") or ctx.language or ""
            spoken = " ".join(x["text"] for x in eng.transcribe(audio, lang, fw, None, None, False,
                                                                 clip_timestamps=[s["start"], s["end"]])).strip()
            new, ch = transfer(s["text"], spoken, lang)
            if ch:
                s["text_standard"] = s["text"]
                s["text"] = new
                s["spelling_changes"] = [{"standard": a, "as_spoken": b} for a, b in ch]
                changed_lines += 1
                changes += [{"start": s["start"], "standard": a, "as_spoken": b} for a, b in ch]
    finally:
        eng.unload()
    ctx.metrics["spelling_step"] = f"spelling transfer, {lora.name}"
    ctx.metrics["lines_respelled"] = changed_lines
    ctx.log.add("merge", f"spelling transfer: {len(changes)} word(s) respelled as spoken in {changed_lines} line(s)",
                "The recogniser's words are kept; only their spelling follows the as-spoken model where both hold the "
                "same words (PLAN_v0.9 section 6). transcript_standard.txt keeps the recogniser's spelling.",
                adapter=lora.name, device=info.get("device"), changes=changes)
