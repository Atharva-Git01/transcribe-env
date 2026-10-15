"""Stage: recognise. Local ASR engine per language (engine table copied into plan.json), VAD segmentation with the
vad stage's parameters, then the fixed fallback ladder (E04) if a track with speech produced no text."""
from __future__ import annotations

import zlib

from .. import gpu
from ..audio import read_wav
from ..common import TRX_HOME, TrxError, load_config, write_json
from ..context import Ctx
from ..engines.faster_whisper_engine import make
from ..models import installed


def _prompt(template: str, lang: str) -> str | None:
    if not template:
        return None
    p = TRX_HOME / template.format(lang=lang)
    return p.read_text(encoding="utf-8").strip() if p.is_file() else None


def engine_choices(table: dict, lang: str) -> list[str]:
    """An engine-table entry is one engine id or an ordered list (first installed wins, e.g. when D: is unplugged)."""
    v = table.get(lang) or table.get("unknown") or []
    return [v] if isinstance(v, str) else [e for e in v if e]


def _engine_for(table: dict, lang: str) -> tuple[str, list[str]]:
    choices = engine_choices(table, lang)
    if not choices:
        raise TrxError("CONFIG_INVALID", f"no engine configured for language '{lang}'")
    for i, eng in enumerate(choices):
        if installed(eng):
            return eng, choices[:i]
    raise TrxError("MODEL_MISSING", f"none of the engines configured for '{lang}' is installed: {choices} "
                                    f"(large-v3 and Gemma live on the D: USB drive — is it connected?)")


def _decode_track(eng, audio, tr, ctx, params, vad, fw, lang, prompt_tpl):
    word_ts = bool(params.get("word_timestamps", True))
    groups = [g for g in ctx.segment_languages if g["track"] == tr.label]
    mode = params.get("segmentation", "vad_concat")
    segs = []
    if groups:  # per-segment languages (code-switched): each LID group decoded in its own language
        for g in groups:
            glang = g["language"] or lang
            part = eng.transcribe(audio, glang, fw, None, _prompt(prompt_tpl, glang), word_ts,
                                  clip_timestamps=[g["start"], g["end"]])
            for s in part:
                s["language"] = glang
            segs += part
    elif mode == "vad_concat":
        # faster-whisper's VAD (same parameters as the vad stage) joins speech into <= 30 s windows: most context.
        # Seeded eval (E14, E18): large-v3 CER 0.531 vs 0.577 with clips; turbo 0.793 vs 0.813.
        if not tr.speech:
            return []
        segs = eng.transcribe(audio, lang, fw, vad, _prompt(prompt_tpl, lang), word_ts)
    elif mode == "clips":
        # the vad stage's regions merged into groups of <= group_max_s, each decoded alone (less context)
        clips = speech_groups(tr.speech, float(params.get("group_max_s", 20.0)), float(params.get("group_max_gap_s", 0.8)))
        if not clips:
            return []
        segs = eng.transcribe(audio, lang, fw, None, _prompt(prompt_tpl, lang), word_ts,
                              clip_timestamps=[x for c in clips for x in c])
    else:
        raise TrxError("CONFIG_INVALID", f"recognise.segmentation '{mode}' (use 'vad_concat' or 'clips')")
    for s in segs:
        s.setdefault("language", lang)
        # faster-whisper gives every segment its 30 s decode WINDOW's compression ratio, so one loop marked the
        # whole window "repetitive". Measure each segment's own text; keep the window value for the record.
        s["window_compression_ratio"] = s.get("compression_ratio")
        s["compression_ratio"] = text_compression_ratio(s["text"])
    max_s = float(params.get("split_max_s", 0) or 0)
    if max_s and word_ts:
        segs = [p for s in segs for p in split_segment(s, max_s)]
    return segs


def text_compression_ratio(text: str) -> float:
    raw = text.encode("utf-8")
    return round(len(raw) / max(1, len(zlib.compress(raw))), 3) if raw else 0.0


SENTENCE_END = (".", "?", "!", "।", "॥")


def split_segment(seg: dict, max_s: float, min_s: float = 2.0, gap_s: float = 0.6) -> list[dict]:
    """Cut a long segment into pieces at word boundaries (sentence end, pause, or max length). Text is unchanged —
    only the boundaries move — so a loop or a doubtful stretch is flagged on its own few seconds (E14)."""
    words = seg.get("words") or []
    if len(words) < 2 or seg["end"] - seg["start"] <= max_s:
        return [seg]
    pieces, cur = [], []
    for w in words:
        if cur:
            dur = cur[-1]["end"] - cur[0]["start"]
            if (w["end"] - cur[0]["start"] > max_s
                    or (dur >= min_s and w["start"] - cur[-1]["end"] >= gap_s)
                    or (dur >= min_s and cur[-1]["word"].strip().endswith(SENTENCE_END))):
                pieces.append(cur)
                cur = []
        cur.append(w)
    pieces.append(cur)
    out = []
    for p in pieces:
        text = "".join(w["word"] for w in p).strip()
        out.append(dict(seg, start=p[0]["start"], end=p[-1]["end"], text=text, words=p,
                        compression_ratio=text_compression_ratio(text), split_from=[seg["start"], seg["end"]]))
    return out


def speech_groups(speech, max_len: float, max_gap: float) -> list[tuple[float, float]]:
    groups, cur = [], None
    for s, e in speech:
        if cur and s - cur[1] <= max_gap and e - cur[0] <= max_len:
            cur[1] = e
        else:
            if cur:
                groups.append((round(cur[0], 3), round(cur[1], 3)))
            cur = [s, e]
    if cur:
        groups.append((round(cur[0], 3), round(cur[1], 3)))
    return groups


def run(ctx: Ctx, params: dict) -> None:
    lang = ctx.language
    table = params["engine_table_primary"]
    per_segment = bool(ctx.segment_languages)
    eng_id, skipped = _engine_for(table, "mixed" if per_segment else lang)
    fw = dict(params.get("faster_whisper", {}))
    vad = ctx.stage_params("vad") or None
    eng = make(eng_id)
    info = eng.load(gpu.choose_device(), fw)
    ctx.engines_used.append(info)
    ctx.log.add("recognise", f"loaded {eng_id} on {info['device']} ({info['compute_type']})",
                f"Engine for language '{'mixed' if per_segment else lang}' from config/defaults.toml [engines.primary]"
                + (f"; preferred {skipped} not installed, so the next choice was used." if skipped else "."),
                skipped_not_installed=skipped, segmentation=params.get("segmentation", "vad_concat"), **info)
    ladder = load_config("defaults").get("fallback_ladder", {})
    all_segments = []
    try:
        for tr in ctx.tracks:
            audio, _ = read_wav(tr.wav16)
            speech_sec = sum(e - s for s, e in tr.speech)
            segs = _decode_track(eng, audio, tr, ctx, params, vad, fw, lang, params.get("initial_prompt_file", ""))
            step = "primary"
            if not any(s["text"] for s in segs) and speech_sec >= float(ladder.get("min_speech_sec", 1.0)):
                # E04 fallback ladder: relax silence checks -> no VAD -> (other engine is tried below)
                relaxed = dict(fw, no_speech_threshold=float(ladder.get("relaxed_no_speech_threshold", 0.9)),
                               log_prob_threshold=float(ladder.get("relaxed_log_prob_threshold", -2.0)))
                for step, use_vad in (("relaxed_thresholds", vad), ("vad_off", None)):
                    segs = eng.transcribe(audio, lang, relaxed, use_vad, _prompt(params.get("initial_prompt_file", ""), lang),
                                          bool(params.get("word_timestamps", True)))
                    for s in segs:
                        s["language"] = lang
                    ctx.log.add("recognise", f"track {tr.label}: fallback ladder step '{step}' -> {len(segs)} segments",
                                "Speech was detected but no text came back; never return empty silently (E04).")
                    if any(s["text"] for s in segs):
                        break
            for s in segs:
                s.update(track=tr.label, speaker=tr.label, engine=eng_id, ladder_step=step, flags=[])
            all_segments += [s for s in segs if s["text"]]
            ctx.log.add("recognise", f"track {tr.label}: {len(segs)} segments", "Recognition finished for this track.",
                        speech_sec=round(speech_sec, 2), ladder_step=step)
    finally:
        eng.unload()
    all_segments.sort(key=lambda s: (s["start"], s["track"]))
    for i, s in enumerate(all_segments, 1):
        s["id"] = i
    ctx.segments = all_segments
    raw = ctx.work / "asr_raw.json"
    write_json(raw, {"engine": info, "language": lang, "segments": all_segments})
    ctx.custody.add(raw, "derived", "recognise")
