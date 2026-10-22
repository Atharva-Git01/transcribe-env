"""Stage: recognise. Local ASR engine per language (engine table copied into plan.json), VAD segmentation with the
vad stage's parameters, then the fixed fallback ladder (E04) if a track with speech produced no text."""
from __future__ import annotations

import json
import zlib

from .. import gpu
from ..audio import read_wav
from ..common import TRX_HOME, TrxError, env_version, load_config, write_json
from ..context import Ctx
from ..engines.faster_whisper_engine import make
from ..models import installed


def _prompt(template: str, lang: str) -> str | None:
    if not template:
        return None
    p = TRX_HOME / template.format(lang=lang)
    return p.read_text(encoding="utf-8").strip() if p.is_file() else None


def engine_choices(table: dict, lang: str) -> list[str]:
    """An engine-table entry is one engine id or an ordered list (first installed wins, e.g. when a model folder is missing)."""
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
                                    f"(check toolchain/models/ and run `trx doctor`)")


def _decode_track(eng, audio, tr, ctx, params, vad, fw, lang, prompt_tpl):
    groups = [g for g in ctx.segment_languages if g["track"] == tr.label]
    return _decode(eng, audio, tr.speech, groups, params, vad, fw, lang, prompt_tpl)


def _decode(eng, audio, speech, groups, params, vad, fw, lang, prompt_tpl):
    """Decode one stretch of audio. `speech` and `groups` are in this audio's own time (0 = its first sample)."""
    word_ts = bool(params.get("word_timestamps", True))
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
        if not speech:
            return []
        segs = eng.transcribe(audio, lang, fw, vad, _prompt(prompt_tpl, lang), word_ts)
    elif mode == "clips":
        # the vad stage's regions merged into groups of <= group_max_s, each decoded alone (less context)
        clips = speech_groups(speech, float(params.get("group_max_s", 20.0)), float(params.get("group_max_gap_s", 0.8)))
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


def chunk_spans(speech, chunk_s: float) -> list[tuple[float, float | None]]:
    """Cut a track into spans of about chunk_s seconds, always in the middle of a silence between VAD regions,
    so no word is ever split between two chunks. chunk_s = 0 -> one span (the whole track)."""
    if not chunk_s or not speech:
        return [(0.0, None)]
    spans, start = [], 0.0
    for i in range(len(speech) - 1):
        if speech[i][1] - start >= chunk_s:
            cut = round((speech[i][1] + speech[i + 1][0]) / 2, 3)
            spans.append((start, cut))
            start = cut
    spans.append((start, None))
    return spans


def _shift(segs: list[dict], offset: float) -> list[dict]:
    for s in segs:
        s["start"], s["end"] = round(s["start"] + offset, 3), round(s["end"] + offset, 3)
        if "split_from" in s:
            s["split_from"] = [round(x + offset, 3) for x in s["split_from"]]
        for w in s.get("words") or []:
            w["start"], w["end"] = round(w["start"] + offset, 3), round(w["end"] + offset, 3)
    return segs


def _chunk_key(**parts) -> str:
    import hashlib
    import json
    return hashlib.sha256(json.dumps(parts, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")).hexdigest()


def turn_units(turns: list[dict], speech, max_s: float = 30.0, max_gap: float = 0.5) -> list[dict]:
    """Decode units for segmentation = "turns": adjacent same-speaker turns merged (gap <= max_gap, length <= max_s);
    a longer turn is cut at the speech-region boundaries inside it, or at max_s."""
    units = []
    for t in sorted(turns, key=lambda x: x["start"]):
        if units and units[-1]["speaker"] == t["speaker"] and t["start"] - units[-1]["end"] <= max_gap \
                and t["end"] - units[-1]["start"] <= max_s:
            units[-1]["end"] = t["end"]
        else:
            units.append({"start": t["start"], "end": t["end"], "speaker": t["speaker"]})
    out = []
    for u in units:
        a = u["start"]
        while u["end"] - a > max_s:
            cuts = [e for s0, e in speech if a + 1.0 < e < a + max_s]
            b = max(cuts) if cuts else a + max_s
            out.append(dict(u, start=a, end=b))
            a = b
        out.append(dict(u, start=a))
    return out


def _plausible(seg: dict, params: dict) -> bool:
    """A recovered (short, decoded alone) segment is kept only if it does not look invented: no loop or repeated
    words, no character runs, decent confidence. Whisper tends to invent repeats on short clips."""
    from .polish import CHAR_RUN, WORD, find_loop
    toks = WORD.findall(seg["text"])
    if not toks or find_loop(toks, set()) or CHAR_RUN.search(seg["text"]):
        return False
    if len(toks) >= 2 and len(set(toks)) <= len(toks) // 2:          # e.g. "x x", "a b b b"
        return False
    if seg.get("avg_logprob") is not None and seg["avg_logprob"] < float(params.get("recover_min_logprob", -0.8)):
        return False
    return not (seg.get("no_speech_prob") is not None and seg["no_speech_prob"] > 0.5)


def _words_in(segs: list[dict], a: float, b: float) -> int:
    return sum(1 for sg in segs for w in (sg.get("words") or []) if a <= (w["start"] + w["end"]) / 2 <= b)


def run(ctx: Ctx, params: dict) -> None:
    lang = ctx.language
    table = params["engine_table_primary"]
    per_segment = bool(ctx.segment_languages)
    eng_id, skipped = _engine_for(table, "mixed" if per_segment else lang)
    fw = dict(params.get("faster_whisper", {}))
    if params.get("hotwords"):
        fw["hotwords"] = " ".join(params["hotwords"])
    vad = ctx.stage_params("vad") or None
    prompt_tpl = params.get("initial_prompt_file", "")
    held = {"eng": None, "info": None}

    def engine():  # loaded only when something actually has to be decoded (a resumed run may need nothing)
        if held["eng"] is None:
            e = make(eng_id)
            held["info"] = e.load(gpu.choose_device(), fw)
            held["eng"] = e
            ctx.engines_used.append(held["info"])
            ctx.log.add("recognise", f"loaded {eng_id} on {held['info']['device']} ({held['info']['compute_type']})",
                        f"Engine for language '{'mixed' if per_segment else lang}' from config/defaults.toml "
                        "[engines.primary]" + (f"; preferred {skipped} not installed, so the next choice was used."
                                               if skipped else "."),
                        skipped_not_installed=skipped, segmentation=params.get("segmentation", "vad_concat"),
                        **held["info"])
        return held["eng"]

    chunk_s = float(params.get("chunk_minutes", 0) or 0) * 60.0
    ckpt_dir = ctx.work / "recognise_chunks"
    ladder = load_config("defaults").get("fallback_ladder", {})
    all_segments = []
    try:
        for tr in ctx.tracks:
            audio, _ = read_wav(tr.wav16)
            speech_sec = sum(e - s for s, e in tr.speech)
            groups = [g for g in ctx.segment_languages if g["track"] == tr.label]
            if params.get("segmentation") == "turns" and ctx.turns and len(ctx.tracks) == 1:
                # one decode per speaker turn: each turn's text belongs to that turn (PLAN P3/P5)
                segs = []
                for u in turn_units(ctx.turns, tr.speech):
                    a, b = max(0.0, u["start"] - 0.1), min(len(audio) / 16000.0, u["end"] + 0.1)
                    part = engine().transcribe(audio[int(a * 16000):int(b * 16000)], lang, fw, None,
                                               _prompt(prompt_tpl, lang), bool(params.get("word_timestamps", True)))
                    for sg in _shift(part, a):
                        sg.update(language=lang, turn_speaker=u["speaker"])
                        sg["window_compression_ratio"] = sg.get("compression_ratio")
                        sg["compression_ratio"] = text_compression_ratio(sg["text"])
                        segs.append(sg)
            elif not chunk_s:
                segs = _decode(engine(), audio, tr.speech, groups, params, vad, fw, lang, prompt_tpl)
            else:
                segs = []
                spans = chunk_spans(tr.speech, chunk_s)
                reused = 0
                for k, (a, b) in enumerate(spans, 1):
                    end = b if b is not None else len(audio) / 16000.0
                    sp = [(max(s0, a) - a, min(e0, end) - a) for s0, e0 in tr.speech if e0 > a and s0 < end]
                    gr = [dict(g, start=max(g["start"], a) - a, end=min(g["end"], end) - a)
                          for g in groups if g["end"] > a and g["start"] < end]
                    key = _chunk_key(engine=eng_id, lang=lang, track=tr.label, span=[a, end], speech=sp, groups=gr,
                                     params={k2: v for k2, v in params.items() if not k2.startswith("engine_table")},
                                     fw=fw, vad=vad, env=env_version(), seed=load_config("defaults")["runtime"].get("seed"))
                    path = ckpt_dir / f"track{tr.index + 1}_chunk{k:04d}.json"
                    if path.is_file():
                        data = json.loads(path.read_text(encoding="utf-8"))
                        if data.get("key") == key:
                            segs += data["segments"]
                            reused += 1
                            held["info"] = held["info"] or data.get("engine_info")
                            continue
                    part = _decode(engine(), audio[int(a * 16000):int(end * 16000)], sp, gr, params, vad, fw, lang,
                                   prompt_tpl)
                    part = _shift(part, a)
                    write_json(path, {"key": key, "span": [a, end], "engine_info": held["info"], "segments": part})
                    segs += part
                    ctx.log.add("recognise", f"track {tr.label}: chunk {k}/{len(spans)} saved "
                                f"({a / 60:.1f}-{end / 60:.1f} min, {len(part)} segments)",
                                "Long recordings are recognised in chunks cut at silences; each finished chunk is "
                                "saved so an interrupted run continues from the next one.")
                if reused:
                    ctx.log.add("recognise", f"track {tr.label}: reused {reused} of {len(spans)} saved chunk(s)",
                                "These chunks were finished before the interruption and match the plan exactly.")
            if params.get("recover_missed_turns", True) and segs:
                # a speaker turn (or speech region) with no recognised words is decoded again on its own (PLAN P3)
                units = ([t for t in ctx.turns] if ctx.turns and len(ctx.tracks) == 1 else
                         [{"start": a0, "end": b0, "speaker": tr.label} for a0, b0 in tr.speech])
                min_s = float(params.get("recover_min_s", 0.3))
                recovered = []
                for u in units:
                    if u["end"] - u["start"] < min_s or _words_in(segs, u["start"], u["end"]) > 0:
                        continue
                    a, b = max(0.0, u["start"] - 0.1), min(len(audio) / 16000.0, u["end"] + 0.1)
                    part = engine().transcribe(audio[int(a * 16000):int(b * 16000)], lang, fw, None, None,
                                               bool(params.get("word_timestamps", True)))
                    for sg in _shift(part, a):
                        if sg["text"].strip() and _plausible(sg, params):
                            sg.update(language=lang, recovered=True)
                            sg["window_compression_ratio"] = sg.get("compression_ratio")
                            sg["compression_ratio"] = text_compression_ratio(sg["text"])
                            recovered.append(sg)
                if recovered:
                    segs += recovered
                    ctx.log.add("recognise", f"track {tr.label}: recovered {len(recovered)} segment(s) from turns "
                                "the first pass left empty", "Short replies inside long windows get skipped by "
                                "Whisper; each empty turn was decoded again on its own (flagged for review).",
                                recovered=[(round(x["start"], 2), round(x["end"], 2), x["text"]) for x in recovered])
            step = "primary"
            if not any(s["text"] for s in segs) and speech_sec >= float(ladder.get("min_speech_sec", 1.0)):
                # E04 fallback ladder: relax silence checks -> no VAD -> (other engine is tried below)
                relaxed = dict(fw, no_speech_threshold=float(ladder.get("relaxed_no_speech_threshold", 0.9)),
                               log_prob_threshold=float(ladder.get("relaxed_log_prob_threshold", -2.0)))
                for step, use_vad in (("relaxed_thresholds", vad), ("vad_off", None)):
                    segs = engine().transcribe(audio, lang, relaxed, use_vad, _prompt(prompt_tpl, lang),
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
        if held["eng"] is not None:
            held["eng"].unload()
    if held["eng"] is None and held["info"]:  # everything came from saved chunks
        ctx.engines_used.append(dict(held["info"], note="from saved chunks"))
    info = held["info"] or {"engine": eng_id}
    all_segments.sort(key=lambda s: (s["start"], s["track"]))
    for i, s in enumerate(all_segments, 1):
        s["id"] = i
    ctx.segments = all_segments
    raw = ctx.work / "asr_raw.json"
    write_json(raw, {"engine": info, "language": lang, "segments": all_segments})
    ctx.custody.add(raw, "derived", "recognise")
