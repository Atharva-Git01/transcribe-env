"""trx screen: the PLAN_v0.9 screening harness. Every candidate model decodes the same fixed lines (one model at a
time, on the GPU when it can) and is scored per challenge set in minutes. Maintainers only; the lines are private.

  trx screen build [--set v1]                   cut the lines once: evals/screening/<set>/ (git-ignored)
  trx screen run --candidate ID [--set v1]      decode every line with one candidate (evals/screen_candidates.toml)
  trx screen report [--set v1]                  one table of every candidate's latest result
  trx screen rescore --candidate ID           score a candidate's stored output again (after a scorer change)
  trx screen transfer --candidate ID [--base indic_transcribe]
                                                the spelling transfer of PLAN_v0.9 section 6, from stored outputs: the
                                                base's words with the candidate's spelling (no model is run)

Lines: the scored part of every fixture whose truth is written as spoken, cut where the baseline transcript (the
latest eval run of that fixture) put its lines; a line's truth is the truth words aligned to it (a word the baseline
did not have goes to the line before it). The set is frozen: build never overwrites it. Challenge sets (section 5):
  S  spelling       spoken slots (a word the owner wrote as spoken where the baseline wrote it standard) and standard
                    slots (a word both wrote the same way that has a known spoken variant): spoken-form recall, wrong
                    switches. The list-based measure of `trx eval` (pipeline/spelling.py) is reported too.
  R  short replies  lines under 2 s: content CER, words with no counterpart in the truth
  N  meaning        lines with a negation or a number word: negation and number errors
  G  general        the rest: content CER and strict CER
Every run reports the load time, the real-time factor and the peak GPU memory (nvidia-smi, any framework).
A candidate with `judge = true` (the phoneme recogniser) writes no transcript: it is scored on the S slots only, by
which spelling of each line its sounds support (pipeline/phonemes.py): spoken slots it gets right (recall) and
standard slots it would wrongly switch to a listed or observed spoken variant.
Outputs: evals/screening/<set>/out/<candidate>.json and evals/compare/screen_<set>_<candidate>.md (texts, this PC
only) and evals/results/screen_<set>_<candidate>_<utc>.json (numbers only, pushed).
"""
from __future__ import annotations

import subprocess
import threading
import time
from collections import Counter, defaultdict

from .common import TRX_HOME, TrxError, env_version, load_toml, read_json, utc_now, write_json

SCREENING = TRX_HOME / "evals" / "screening"
CANDIDATES = TRX_HOME / "evals" / "screen_candidates.toml"
RESULTS = TRX_HOME / "evals" / "results"
COMPARE = TRX_HOME / "evals" / "compare"
PAD_S = 0.15                    # audio kept around each line, so its edge words are whole
SHORT_S = 2.0                   # R: lines shorter than this
TAGS = ("S", "R", "N", "G")


def candidates() -> dict:
    return load_toml(CANDIDATES) if CANDIDATES.is_file() else {}


def _baseline_run(fixture_id: str):
    runs = [r for r in sorted((TRX_HOME / "evals" / "runs").glob(f"eval_{fixture_id}-*"))
            if (r / "out" / "transcript.json").is_file() and (r / "work" / "track1_16k.wav").is_file()]
    return runs[-1] if runs else None


def _owners(words: list[str], segs: list[dict]) -> list[int]:
    """The baseline line each truth word belongs to (character alignment, as the eval aligns turns)."""
    from .metrics import _align_pairs, normalise
    r_chars, r_word, h_chars, h_seg = [], [], [], []
    for k, w in enumerate(words):
        r_chars += list(w)
        r_word += [k] * len(w)
    for j, s in enumerate(segs):
        t = normalise(s["text"]).replace(" ", "")
        h_chars += list(t)
        h_seg += [j] * len(t)
    votes: dict = defaultdict(Counter)
    for i, j in _align_pairs(r_chars, h_chars):
        votes[r_word[i]][h_seg[j]] += 1
    owner, last = [], 0
    for k in range(len(words)):
        if votes.get(k):
            last = max(last, votes[k].most_common(1)[0][0])
        owner.append(last)
    return owner


def _slots(truth_words: list[str], base_words: list[str], lang: str, standard_forms: set) -> list[dict]:
    """Spelling slots of one line: a block of truth words spelled otherwise than the baseline wrote the same words
    (spoken), or one word spelled alike that has a known spoken variant (standard). Fillers are not slots."""
    from .spelling import block_spelling, content_word, match_blocks, variants
    listed = variants(lang)["standard_words"]
    out = []
    for b in match_blocks(truth_words, base_words, lang):
        i1, i2, j1, j2 = b
        if all(content_word(w, lang) == "·" for w in truth_words[i1:i2]):
            continue
        tb, hb = block_spelling(truth_words, base_words, b, lang)
        if tb != hb:
            out.append({"i": i1, "n": i2 - i1, "kind": "spoken", "truth": " ".join(truth_words[i1:i2]),
                        "standard": " ".join(base_words[j1:j2])})
        elif i2 - i1 == 1 and (tb in listed or tb in standard_forms):
            out.append({"i": i1, "n": 1, "kind": "standard", "truth": truth_words[i1]})
    return out


def build(set_id: str = "v1") -> dict:
    from .audio import read_wav, write_wav
    from .metrics import meaning_errors, normalise
    from .spelling import ortho
    d = SCREENING / set_id
    if (d / "lines.json").is_file():
        raise TrxError("USAGE", f"screening set '{set_id}' exists and is frozen; delete evals/screening/{set_id} "
                                "yourself to rebuild it, or choose a new --set")
    fixtures = TRX_HOME / "evals" / "fixtures"
    lines, sources, skipped = [], {}, []
    work = []
    for fx in sorted(p for p in fixtures.iterdir() if (p / "truth.json").is_file()):
        truth = read_json(fx / "truth.json")
        if truth.get("spelling") != "as_spoken":
            continue
        run = _baseline_run(fx.name)
        if run is None:
            skipped.append(fx.name)
            continue
        segs = read_json(run / "out" / "transcript.json")["segments"]
        until = truth.get("scored_until_sec")
        segs = [s for s in segs if not until or s["start"] < until]
        words = normalise(truth["reference_text"]).split()
        owner = _owners(words, segs)
        work.append((fx, truth, run, segs, words, owner))
        sources[fx.name] = {"baseline_run": run.name, "scored_until_sec": until}
    # standard forms the owner was seen to say as spoken anywhere in the set: their standard-written words are slots
    spoken_std = set()
    for fx, truth, run, segs, words, owner in work:
        lang = truth.get("language", "")
        for j, s in enumerate(segs):
            tw = [w for w, o in zip(words, owner) if o == j]
            spoken_std |= {ortho(x["standard"], lang) for x in _slots(tw, normalise(s["text"]).split(), lang, set())
                           if x["kind"] == "spoken" and " " not in x["standard"]}
    (d / "audio").mkdir(parents=True, exist_ok=True)
    for fx, truth, run, segs, words, owner in work:
        lang = truth.get("language", "")
        audio, sr = read_wav(run / "work" / "track1_16k.wav")
        for j, s in enumerate(segs):
            tw = [w for w, o in zip(words, owner) if o == j]
            if not tw:
                continue
            a, b = max(0.0, s["start"] - PAD_S), min(len(audio) / sr, s["end"] + PAD_S)
            lid = f"{fx.name}_{j:03d}"
            write_wav(d / "audio" / f"{lid}.wav", audio[int(a * sr):int(b * sr)], sr)
            text = " ".join(tw)
            slots = _slots(tw, normalise(s["text"]).split(), lang, spoken_std)
            me = meaning_errors(text, text, lang)
            tags = [t for t, on in (("S", bool(slots)), ("R", b - a - 2 * PAD_S < SHORT_S),
                                    ("N", me["negations"] + me["numbers"] > 0)) if on] or ["G"]
            lines.append({"id": lid, "fixture": fx.name, "language": lang, "start": round(a, 3), "end": round(b, 3),
                          "duration": round(b - a, 3), "text": text, "baseline": s["text"], "slots": slots,
                          "tags": tags})
    if not lines:
        raise TrxError("USAGE", "no fixture written as spoken has an eval run to cut lines from: run `trx eval` first")
    doc = {"schema": "trx.screen/1", "set": set_id, "made_utc": utc_now(), "env_version": env_version(),
           "pad_s": PAD_S, "short_s": SHORT_S, "sources": sources, "skipped": skipped, "segments": lines,
           "counts": _set_counts(lines)}
    write_json(d / "lines.json", doc)
    return doc


def _set_counts(lines: list[dict]) -> dict:
    c = {"lines": len(lines), "audio_s": round(sum(x["duration"] for x in lines), 1)}
    for t in TAGS:
        sel = [x for x in lines if t in x["tags"]]
        c[t] = {"lines": len(sel), "audio_s": round(sum(x["duration"] for x in sel), 1)}
    c["spoken_slots"] = sum(1 for x in lines for s in x["slots"] if s["kind"] == "spoken")
    c["standard_slots"] = sum(1 for x in lines for s in x["slots"] if s["kind"] == "standard")
    return c


class GpuPeak:
    """Peak GPU memory in use while a candidate loads and decodes, minus what was in use before (nvidia-smi every
    0.5 s, so it works for PyTorch, CTranslate2 and onnxruntime alike). None without an NVIDIA GPU."""

    def __init__(self):
        self.base = self.peak = self._used()
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._poll, daemon=True)

    @staticmethod
    def _used():
        try:
            r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                               capture_output=True, text=True, timeout=10)
            return int(r.stdout.split()[0])
        except (OSError, ValueError, IndexError, subprocess.SubprocessError):
            return None

    def _poll(self):
        while not self._stop.wait(0.5):
            u = self._used()
            if u is not None and (self.peak is None or u > self.peak):
                self.peak = u

    def __enter__(self):
        self._t.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._t.join(5)

    @property
    def mb(self):
        return None if self.base is None or self.peak is None else self.peak - self.base


def _errs(ref: str, hyp: str, lang: str, level: str) -> tuple[int, int]:
    from .metrics import _edit_distance, normalise_level
    r = normalise_level(ref, lang, level).replace(" ", "")
    h = normalise_level(hyp, lang, level).replace(" ", "")
    return _edit_distance(r, h), len(r)


def _ratio(a: int, b: int):
    return round(a / b, 4) if b else None


def score(lines: list[dict], hyps: dict) -> dict:
    """Numbers per challenge set, over all lines and per fixture (= recording: a model trained on one recording is
    judged on the others, PLAN_v0.9 section 5). hyps: line id -> text."""
    res = _score_sets(lines, hyps)
    res["by_fixture"] = {f: _score_sets([x for x in lines if x["fixture"] == f], hyps)
                         for f in sorted({x["fixture"] for x in lines})}
    return res


def _score_sets(lines: list[dict], hyps: dict) -> dict:
    from .metrics import meaning_errors, normalise
    from .spelling import block_spelling, match_blocks, match_words, spoken_forms

    def block(sel):
        if not sel:
            return None
        out = {"lines": len(sel), "audio_s": round(sum(x["duration"] for x in sel), 1)}
        for level, key in (("strict", "cer"), ("content", "cer_content")):
            e = n = 0
            for x in sel:
                a, b = _errs(x["text"], hyps.get(x["id"], ""), x["language"], level)
                e, n = e + a, n + b
            out[key] = _ratio(e, n)
        return out
    res = {"all": block(lines)}
    for tag in TAGS:
        res[tag] = block([x for x in lines if tag in x["tags"]])
    if res["S"]:
        c = Counter()
        lst = Counter()
        for x in lines:
            if not x["slots"]:
                continue
            lang = x["language"]
            tw, hw = x["text"].split(), normalise(hyps.get(x["id"], "")).split()
            covering = {}
            for b in match_blocks(tw, hw, lang):
                covering.update({i: b for i in range(b[0], b[1])})
            for s in x["slots"]:
                b = covering.get(s["i"])
                same = b is not None and len(set(block_spelling(tw, hw, b, lang))) == 1
                if s["kind"] == "spoken":
                    c["unmatched" if b is None else "hits" if same else "misses"] += 1
                else:
                    c["standard_unmatched" if b is None else "kept" if same else "switched"] += 1
            sf = spoken_forms(x["text"], hyps.get(x["id"], ""), lang) or {}
            lst.update({k: v for k, v in sf.items() if isinstance(v, int)})
        res["S"].update({"spoken_slots": c["hits"] + c["misses"] + c["unmatched"], "hits": c["hits"],
                         "misses": c["misses"], "unmatched": c["unmatched"],
                         "recall": _ratio(c["hits"], c["hits"] + c["misses"]),
                         "standard_slots": c["kept"] + c["switched"] + c["standard_unmatched"],
                         "switched": c["switched"], "switch_rate": _ratio(c["switched"], c["kept"] + c["switched"]),
                         "list_recall": _ratio(lst["hits"], lst["hits"] + lst["misses"]),
                         "list_switch_rate": _ratio(lst["switched"], lst["kept"] + lst["switched"])})
    if res["R"]:
        invented = truth_words = 0
        for x in lines:
            if "R" not in x["tags"]:
                continue
            hw = normalise(hyps.get(x["id"], "")).split()
            matched = match_words(hw, x["text"].split(), x["language"])   # hypothesis words found in the truth
            invented += len(hw) - len(matched)
            truth_words += len(x["text"].split())
        res["R"].update({"words_not_in_truth": invented, "per_100_truth_words": _ratio(100 * invented, truth_words)})
    if res["N"]:
        m = Counter()
        for x in lines:
            if "N" in x["tags"]:
                m.update(meaning_errors(x["text"], hyps.get(x["id"], ""), x["language"]))
        res["N"].update(dict(m))
    return res


def judge_scores(lines: list[dict], heard: dict) -> dict:
    """The phoneme judge on the S slots (module docstring). heard: line id -> the recogniser's IPA."""
    from .phonemes import judge
    from .spelling import ortho, variants
    observed: dict = defaultdict(set)              # standard word -> spoken spellings the owner used in this set
    for x in lines:
        for sl in x["slots"]:
            if sl["kind"] == "spoken" and " " not in sl["standard"]:
                observed[ortho(sl["standard"], x["language"])].add(sl["truth"])
    c = Counter()
    for x in lines:
        if not x["slots"] or x["id"] not in heard:
            continue
        words, lang = x["text"].split(), x["language"]
        for sl in x["slots"]:
            i, n = sl["i"], sl.get("n", 1)

            def line_with(alt, i=i, n=n):
                return " ".join(words[:i] + alt.split() + words[i + n:])
            if sl["kind"] == "spoken":
                win, _ = judge(heard[x["id"]], [line_with(sl["standard"]), x["text"]])
                c["spoken"] += 1
                c["tie" if win is None else "right" if win == 1 else "wrong"] += 1
            else:
                o = ortho(sl["truth"], lang)
                alts = sorted(variants(lang)["standard_words"].get(o, set()) | observed.get(o, set()))
                if not alts:
                    continue
                win, _ = judge(heard[x["id"]], [x["text"]] + [line_with(a) for a in alts])
                c["standard"] += 1
                c["kept" if win in (0, None) else "switched"] += 1
    return {"spoken_slots": c["spoken"], "prefers_spoken": c["right"], "ties": c["tie"],
            "recall": _ratio(c["right"], c["spoken"]), "standard_slots": c["standard"],
            "false_switches": c["switched"], "false_switch_rate": _ratio(c["switched"], c["standard"])}


JUDGE_MARGINS = (0.0, 2.0, 5.0)   # nats a spoken spelling must be ahead of the standard one (judge = "ctc")


def judge_scores_ctc(lines: list[dict], logp: dict, vocab: dict, blank: int) -> dict:
    """The phoneme judge, version 2: the recogniser's own CTC likelihood of each spelling of the line; a spoken
    spelling wins only when it is ahead of the standard one by more than the margin. One block per margin."""
    from .phonemes import ctc_judge
    from .spelling import ortho, variants
    observed: dict = defaultdict(set)
    for x in lines:
        for sl in x["slots"]:
            if sl["kind"] == "spoken" and " " not in sl["standard"]:
                observed[ortho(sl["standard"], x["language"])].add(sl["truth"])
    out = {}
    for margin in JUDGE_MARGINS:
        c = Counter()
        for x in lines:
            if x["id"] not in logp:
                continue
            words, lang = x["text"].split(), x["language"]
            for sl in x["slots"]:
                i, n = sl["i"], sl.get("n", 1)

                def line_with(alt, i=i, n=n):
                    return " ".join(words[:i] + alt.split() + words[i + n:])
                if sl["kind"] == "spoken":
                    win, _ = ctc_judge(logp[x["id"]], [line_with(sl["standard"]), x["text"]], vocab, blank, margin)
                    c["spoken"] += 1
                    c["right" if win == 1 else "tie" if win is None else "wrong"] += 1
                else:
                    o = ortho(sl["truth"], lang)
                    alts = sorted(variants(lang)["standard_words"].get(o, set()) | observed.get(o, set()))
                    if not alts:
                        continue
                    win, _ = ctc_judge(logp[x["id"]], [x["text"]] + [line_with(a) for a in alts], vocab, blank, margin)
                    c["standard"] += 1
                    c["kept" if win in (0, None) else "switched"] += 1
        out[f"margin_{margin:g}"] = {"spoken_slots": c["spoken"], "prefers_spoken": c["right"], "ties": c["tie"],
                                     "recall": _ratio(c["right"], c["spoken"]), "standard_slots": c["standard"],
                                     "false_switches": c["switched"],
                                     "false_switch_rate": _ratio(c["switched"], c["standard"])}
    best = max(out.values(), key=lambda b: (b["false_switch_rate"] or 0) <= 0.05 and (b["recall"] or 0))
    return dict(best, margins=out)


def run(candidate: str, set_id: str = "v1") -> dict:
    from .audio import read_wav
    from .engines.faster_whisper_engine import make
    from .gpu import choose_device
    cfg = candidates().get(candidate)
    if cfg is None:
        raise TrxError("USAGE", f"no candidate '{candidate}' in evals/screen_candidates.toml "
                                f"(have: {', '.join(sorted(candidates())) or 'none'})")
    d = SCREENING / set_id
    if not (d / "lines.json").is_file():
        raise TrxError("USAGE", f"no screening set '{set_id}': run `trx screen build` first")
    lines = read_json(d / "lines.json")["segments"]
    params = dict(cfg.get("params", {}))
    prompt = (TRX_HOME / cfg["prompt"]).read_text(encoding="utf-8").strip() if cfg.get("prompt") else None
    eng = make(cfg["engine"])
    hyps: dict = {}
    with GpuPeak() as gp:
        t0 = time.time()
        info = eng.load(choose_device(), params)
        load_s = time.time() - t0
        if hasattr(eng, "configure"):
            eng.configure(params)
        t1 = time.time()
        logp, table = {}, None
        for x in lines:
            audio, sr = read_wav(d / "audio" / f"{x['id']}.wav")
            if cfg.get("judge") == "ctc":            # the judge's likelihoods, only where there is a spelling slot
                if x["slots"]:
                    logp[x["id"]], vocab, blank = eng.log_probs(audio)
                    table = (vocab, blank)
                continue
            segs = eng.transcribe(audio, x["language"], dict(cfg.get("fw", {})), None, prompt, False,
                                  clip_timestamps=[0.0, len(audio) / sr])
            hyps[x["id"]] = " ".join(s["text"].strip() for s in segs).strip()
        decode_s = time.time() - t1
        eng.unload()
    audio_s = sum(x["duration"] for x in lines)
    if cfg.get("judge") == "ctc":
        scores = {"judge": judge_scores_ctc(lines, logp, *table)}
    else:
        scores = {"judge": judge_scores(lines, hyps)} if cfg.get("judge") else score(lines, hyps)
    stamp = utc_now()
    result = {"schema": "trx.screen_result/1", "set": set_id, "candidate": candidate, "engine": cfg["engine"],
              "env_version": env_version(), "utc": stamp, "device": info.get("device"),
              "compute_type": info.get("compute_type"), "fallback_reason": info.get("fallback_reason"),
              "load_s": round(load_s, 1), "decode_s": round(decode_s, 1), "audio_s": round(audio_s, 1),
              "rtf": round(decode_s / max(audio_s, 1e-6), 3), "gpu_peak_mb": gp.mb, "scores": scores}
    (d / "out").mkdir(exist_ok=True)
    if hyps:
        write_json(d / "out" / f"{candidate}.json", {"candidate": candidate, "utc": stamp, "info": info,
                                                     "segments": [{"line": x["id"], "text": hyps[x["id"]]}
                                                                  for x in lines]})
        _write_compare(set_id, candidate, lines, hyps, result)
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / f"screen_{set_id}_{candidate}_{stamp.replace(':', '').replace('-', '')}.json"
    write_json(out, result)
    result["file"] = str(out.relative_to(TRX_HOME))
    return result


def rescore(candidate: str, set_id: str = "v1") -> dict:
    """The candidate's latest result with its scores recomputed from its stored output (no model is run)."""
    d = SCREENING / set_id
    f = d / "out" / f"{candidate}.json"
    prev = [read_json(p) for p in sorted(RESULTS.glob(f"screen_{set_id}_{candidate}_*.json"))]
    prev = [r for r in prev if r.get("candidate") == candidate]
    if not f.is_file() or not prev:
        raise TrxError("USAGE", f"no stored output and result of '{candidate}' on set {set_id}")
    lines = read_json(d / "lines.json")["segments"]
    hyps = {s["line"]: s["text"] for s in read_json(f)["segments"]}
    stamp = utc_now()
    result = dict(prev[-1], utc=stamp, env_version=env_version(), rescored_from=prev[-1]["utc"],
                  scores=score(lines, hyps))
    out = RESULTS / f"screen_{set_id}_{candidate}_{stamp.replace(':', '').replace('-', '')}.json"
    write_json(out, result)
    result["file"] = str(out.relative_to(TRX_HOME))
    return result


def transfer_run(spelling: str, base: str = "indic_transcribe", set_id: str = "v1") -> dict:
    """Score the spelling transfer from the stored outputs of two candidates (both must have run on this set)."""
    from .spelling import transfer
    d = SCREENING / set_id
    lines = read_json(d / "lines.json")["segments"]
    outs = {}
    for c in (base, spelling):
        f = d / "out" / f"{c}.json"
        if not f.is_file():
            raise TrxError("USAGE", f"no stored output of '{c}' on set {set_id}: run `trx screen run --candidate {c}`")
        outs[c] = {s["line"]: s["text"] for s in read_json(f)["segments"]}
    hyps, changes = {}, 0
    for x in lines:
        hyps[x["id"]], ch = transfer(outs[base].get(x["id"], ""), outs[spelling].get(x["id"], ""), x["language"])
        changes += len(ch)
    latest = {r["candidate"]: r for r in (read_json(p) for p in sorted(RESULTS.glob(f"screen_{set_id}_*.json")))
              if r.get("schema") == "trx.screen_result/1"}
    rb, rs = latest.get(base, {}), latest.get(spelling, {})
    name, stamp = f"transfer_{spelling}", utc_now()
    result = {"schema": "trx.screen_result/1", "set": set_id, "candidate": name,
              "engine": f"{base} words + {spelling} spelling", "env_version": env_version(), "utc": stamp,
              "device": rs.get("device"), "compute_type": "spelling transfer", "fallback_reason": None,
              "load_s": round((rb.get("load_s") or 0) + (rs.get("load_s") or 0), 1),
              "decode_s": round((rb.get("decode_s") or 0) + (rs.get("decode_s") or 0), 1),
              "audio_s": rb.get("audio_s"), "rtf": round((rb.get("rtf") or 0) + (rs.get("rtf") or 0), 3),
              "gpu_peak_mb": max(rb.get("gpu_peak_mb") or 0, rs.get("gpu_peak_mb") or 0) or None,
              "spelling_changes": changes, "scores": score(lines, hyps)}
    write_json(d / "out" / f"{name}.json", {"candidate": name, "utc": stamp,
                                            "segments": [{"line": x["id"], "text": hyps[x["id"]]} for x in lines]})
    _write_compare(set_id, name, lines, hyps, result)
    out = RESULTS / f"screen_{set_id}_{name}_{stamp.replace(':', '').replace('-', '')}.json"
    write_json(out, result)
    result["file"] = str(out.relative_to(TRX_HOME))
    return result


def _write_compare(set_id: str, candidate: str, lines: list[dict], hyps: dict, result: dict) -> None:
    COMPARE.mkdir(parents=True, exist_ok=True)
    rows = [f"# Screening {set_id} — {candidate} ({result['utc']}, env {result['env_version']})", "",
            summary_line(result), "", "| line | tags | truth | output | content CER |", "|---|---|---|---|---|"]
    for x in lines:
        e, n = _errs(x["text"], hyps.get(x["id"], ""), x["language"], "content")
        rows.append(f"| {x['id']} | {''.join(x['tags'])} | {x['text']} | {hyps.get(x['id']) or '—'} | "
                    f"{_ratio(e, n)} |")
    (COMPARE / f"screen_{set_id}_{candidate}.md").write_text("\n".join(rows) + "\n", encoding="utf-8")


def summary_line(r: dict) -> str:
    s = r["scores"]
    if "judge" in s:
        j = s["judge"]
        return (f"{r['candidate']} (judge): spoken slots heard as spoken {j['prefers_spoken']}/{j['spoken_slots']} "
                f"(recall {j['recall']}, {j['ties']} ties) | standard slots it would switch {j['false_switches']}/"
                f"{j['standard_slots']} ({j['false_switch_rate']}) | {r['device']} {r['compute_type']}, RTF {r['rtf']}, "
                f"load {r['load_s']} s, GPU {r['gpu_peak_mb']} MB")

    def g(tag, key):
        return (s.get(tag) or {}).get(key)
    sp = s.get("S") or {}
    return (f"{r['candidate']}: content CER {g('all', 'cer_content')} (strict {g('all', 'cer')}) | S spoken forms "
            f"{sp.get('hits', 0)}/{sp.get('hits', 0) + sp.get('misses', 0)} (recall {sp.get('recall')}), switched "
            f"{sp.get('switched', 0)}/{sp.get('standard_slots', 0)} | R content CER {g('R', 'cer_content')}, "
            f"{g('R', 'words_not_in_truth')} words not in the truth | N negations {g('N', 'negation_errors')} wrong "
            f"of {g('N', 'negations')}, numbers {g('N', 'number_errors')} of {g('N', 'numbers')} | G content CER "
            f"{g('G', 'cer_content')} | {r['device']} {r['compute_type']}, RTF {r['rtf']}, load {r['load_s']} s, "
            f"GPU {r['gpu_peak_mb']} MB")


def report(set_id: str = "v1") -> list[dict]:
    """The latest result of every candidate on this set, best content CER first."""
    latest: dict = {}
    for p in sorted(RESULTS.glob(f"screen_{set_id}_*.json")):
        r = read_json(p)
        if r.get("schema") == "trx.screen_result/1" and r.get("set") == set_id:
            latest[r["candidate"]] = r
    rows = sorted(latest.values(), key=lambda r: ((r["scores"].get("all") or {}).get("cer_content") or 9))
    md = [f"# Screening set {set_id}: latest result per candidate", "",
          "| candidate | device | content CER | strict CER | S recall | S switched | R content CER | R words not in "
          "truth | N neg. wrong | N num. wrong | G content CER | RTF | load s | GPU MB |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        s = r["scores"]
        if "judge" in s:
            j = s["judge"]
            md.append(f"| {r['candidate']} (judge) | {r['device']} {r['compute_type']} | — | — | {j['recall']} | "
                      f"{j['false_switches']}/{j['standard_slots']} | — | — | — | — | — | {r['rtf']} | {r['load_s']} | "
                      f"{r['gpu_peak_mb']} |")
            continue

        def g(tag, key, s=s):
            return (s.get(tag) or {}).get(key)
        md.append(f"| {r['candidate']} | {r['device']} {r['compute_type']} | {g('all', 'cer_content')} | "
                  f"{g('all', 'cer')} | {g('S', 'recall')} | {g('S', 'switched')}/{g('S', 'standard_slots')} | "
                  f"{g('R', 'cer_content')} | {g('R', 'words_not_in_truth')} | {g('N', 'negation_errors')}/"
                  f"{g('N', 'negations')} | {g('N', 'number_errors')}/{g('N', 'numbers')} | {g('G', 'cer_content')} | "
                  f"{r['rtf']} | {r['load_s']} | {r['gpu_peak_mb']} |")
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"screen_{set_id}_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return rows
