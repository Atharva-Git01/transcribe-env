"""Scoring: character and word error rate with one fixed normalisation.

Normalisation (applied identically to reference and hypothesis): Unicode NFC, lowercase, remove punctuation
(incl. Devanagari danda । and double danda ॥), collapse whitespace. Bracketed tags like [inaudible …] are removed.
Distances are bit-parallel and alignments use numpy rows when numpy is there (pure-Python loops otherwise, same
result): a 20-minute call scores in seconds (E39). Alignment memory is O(len(ref) * len(hyp)): fine for eval clips
(minutes), not for hour-long files.

The failure monitor (E39) says HOW a transcript went wrong, not only how much: turn outcomes (`turn_outcomes`),
negations and numbers lost or added (`meaning_errors`), errors per output line (`line_scores`: short lines vs long,
and whether the bad lines carry a review flag) and the "content" level, which folds spelling conventions away.
"""
from __future__ import annotations

import re
import unicodedata

_TAGS = re.compile(r"\[[^\]]*\]")
_PUNCT = re.compile(r"[।॥.,!?;:\"'()\-–—…“”‘’/\\]")
_SPACE = re.compile(r"\s+")


def normalise(text: str) -> str:
    t = unicodedata.normalize("NFC", text)
    t = _TAGS.sub(" ", t).lower()
    t = _PUNCT.sub(" ", t)
    return _SPACE.sub(" ", t).strip()


# Spelling-variant folding for the "normalised" score (OIWER-like; documented, applied to BOTH sides):
#   nukta dropped (ज़ -> ज), chandrabindu -> anusvara (ँ -> ं), candra vowels -> plain (ॅ -> े, ॉ -> ो, ऑ -> ओ, ॲ -> अ),
#   digits -> number words (library/numbers/<lang>.toml). "loose" also merges short/long i and u (ि/ी, ु/ू).
_FOLD = str.maketrans({"\u093c": None, "\u0901": "\u0902", "\u0945": "\u0947", "\u0949": "\u094b",
                       "\u0911": "\u0913", "\u0972": "\u0905"})
_LOOSE = str.maketrans({"\u0940": "\u093f", "\u0942": "\u0941", "\u0908": "\u0907", "\u090a": "\u0909"})
# "content" = loose + hesitation fillers removed + (Marathi) colloquial and standard spellings of the same word
# folded together, so that what is left is misheard, missing or extra words (E39: on the owner's 20-minute call these
# conventions were 14 % of the CER). Applied to BOTH sides; a diagnostic level, never the official CER.
_FILLERS = re.compile(r"(?<!\S)(?:\u0905+|\u0905\u0902|\u0939\u0902+|\u0939\u093e\u0902|\u0939\u093e\u0901|\u0939\u092e\u094d\u092e*|\u0909\u0902+|\u0909\u092e\u094d\u092e*|uh+|um+|mm+|hmm+|mhm|er|ah)(?!\S)")
_CONS = "\u0915\u0916\u0917\u0918\u0919\u091a\u091b\u091c\u091d\u091e\u091f\u0920\u0921\u0922\u0923\u0924\u0925\u0926\u0927\u0928\u092a\u092b\u092c\u092d\u092e\u092f\u0930\u0932\u0935\u0936\u0937\u0938\u0939\u0933"
_MR_FOLDS = [
    (re.compile(r"\u0902(?=\s|$)"), ""),                      # word-final anusvara dropped
    (re.compile(f"[\u0919\u091e\u0923\u0928\u092e]\u094d(?=[{_CONS}])"), "\u0902"),        # nasal consonant = anusvara before a consonant
    (re.compile("\u0936"), "\u0938"), (re.compile("\u0923"), "\u0928"),      # colloquial s for sh, dental n for retroflex n
    (re.compile(r"(?<!\S)\u0928\u093e\u092f\u093f?(?!\S)"), "\u0928\u093e\u0939\u093f"),        # colloquial negation (after the i/u fold)
    (re.compile(r"(?<=\S\S)\u092f\u0947(?=\s|$)"), ""),            # contraction: "<word>-ye" = "<word> aahe"
    (re.compile(r"(?<!\S)\u0906\u0939\u0947(?!\S)"), ""),
]


def normalise_level(text: str, lang: str = "", level: str = "strict") -> str:
    """level: "strict" (= normalise), "norm" (spelling variants folded, digits as words), "loose" (+ i/u length),
    "content" (+ fillers removed and Marathi spelling conventions folded, see _MR_FOLDS)."""
    t = normalise(text)
    if level == "strict":
        return t
    t = unicodedata.normalize("NFD", t).translate(_FOLD)
    t = unicodedata.normalize("NFC", t)
    if lang:
        try:
            from .numbers import digits_to_words
            t = digits_to_words(t, lang)[0].replace(" टक्के", " टक्के")
        except Exception:  # noqa: BLE001 — scoring must never crash on a number table problem
            pass
    if level in ("loose", "content"):
        t = t.translate(_LOOSE)
    if level == "content":
        t = _FILLERS.sub(" ", t)
        if lang == "mr":
            for rx, rep in _MR_FOLDS:
                t = rx.sub(rep, t)
    return _SPACE.sub(" ", t).strip()


def cer_level(reference: str, hypothesis: str, lang: str = "", level: str = "strict") -> float:
    ref = normalise_level(reference, lang, level).replace(" ", "")
    hyp = normalise_level(hypothesis, lang, level).replace(" ", "")
    if not ref:
        return 0.0 if not hyp else 1.0
    return _edit_distance(ref, hyp) / len(ref)


def turn_alignment(ref_segments: list[dict], hyp_segments: list[dict]) -> list[dict]:
    """Align hypothesis segments to reference turns (character alignment, like CER). For each reference turn:
    the hypothesis text that landed on it, its speakers, and the turn's own CER. Missing turns get text ""."""
    r_chars, r_turn, h_chars, h_seg = [], [], [], []
    for i, seg in enumerate(ref_segments):
        t = normalise(seg.get("text", "")).replace(" ", "")
        r_chars += list(t)
        r_turn += [i] * len(t)
    for j, seg in enumerate(hyp_segments):
        t = normalise(seg.get("text", "")).replace(" ", "")
        h_chars += list(t)
        h_seg += [j] * len(t)
    votes: dict = {}
    for ri, hj in _align_pairs(r_chars, h_chars):
        votes.setdefault(h_seg[hj], []).append(r_turn[ri])
    owner, last = {}, 0
    for j in range(len(hyp_segments)):
        v = votes.get(j)
        owner[j] = max(set(v), key=v.count) if v else last
        last = owner[j]
    rows = []
    for i, seg in enumerate(ref_segments):
        hyp = [hyp_segments[j] for j in range(len(hyp_segments)) if owner[j] == i]
        text = " ".join(h.get("text", "") for h in hyp).strip()
        rows.append({"turn": i + 1, "ref_speaker": seg.get("speaker"), "ref": seg.get("text", ""), "hyp": text,
                     "hyp_speakers": sorted({h.get("speaker") for h in hyp if h.get("speaker")}),
                     "cer": round(cer(seg.get("text", ""), text), 3) if text else 1.0})
    return rows


def _ops_python(a, b) -> list:
    """Backtrace table of the Levenshtein alignment: 0 diagonal, 1 skip a[i], 2 skip b[j]. Ties prefer the diagonal,
    then skipping a[i]."""
    n, m = len(a), len(b)
    prev = list(range(m + 1))
    back = []
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        row = bytearray(m + 1)
        ai = a[i - 1]
        for j in range(1, m + 1):
            d, l, u = prev[j - 1] + (ai != b[j - 1]), cur[j - 1] + 1, prev[j] + 1
            if d <= l and d <= u:
                cur[j], row[j] = d, 0
            elif u <= l:
                cur[j], row[j] = u, 1
            else:
                cur[j], row[j] = l, 2
        back.append(row)
        prev = cur
    return back


def _ops_numpy(a, b, np):
    """The same table, one numpy row at a time: cur[j] = min(B[j], cur[j-1] + 1) with B = min(diagonal, skip a[i])
    is a running minimum of B[j] - j, so a row needs no Python loop. Identical choices, ~20x faster."""
    n, m = len(a), len(b)
    sym: dict = {}
    av = np.array([sym.setdefault(x, len(sym)) for x in a], dtype=np.int64)
    bv = np.array([sym.setdefault(x, len(sym)) for x in b], dtype=np.int64)
    ar = np.arange(m + 1, dtype=np.int64)
    prev = ar.copy()
    back = np.zeros((n, m + 1), dtype=np.uint8)
    for i in range(1, n + 1):
        d = prev[:-1] + (bv != av[i - 1])
        u = prev[1:] + 1
        best = np.minimum(d, u)
        c = np.empty(m + 1, dtype=np.int64)
        c[0] = i
        c[1:] = best - ar[1:]
        c = np.minimum.accumulate(c)
        back[i - 1, 1:] = np.where(best - ar[1:] <= c[:-1], np.where(d <= u, 0, 1), 2)
        prev = c + ar
    return back


def _align_pairs(a: list, b: list) -> list[tuple[int, int]]:
    """Index pairs (i, j) of a Levenshtein alignment of two sequences (matches and substitutions)."""
    n, m = len(a), len(b)
    if not n or not m:
        return []
    try:
        import numpy as np
        back = _ops_numpy(a, b, np)
    except ImportError:
        back = _ops_python(a, b)
    pairs, i, j = [], n, m
    while i > 0 and j > 0:
        op = back[i - 1][j]
        if op == 0:
            pairs.append((i - 1, j - 1))
            i, j = i - 1, j - 1
        elif op == 1:
            i -= 1
        else:
            j -= 1
    return pairs[::-1]


def _edit_distance(a, b) -> int:
    """Levenshtein distance of two sequences (characters or words): Myers/Hyyrö bit-parallel, exact."""
    if len(a) < len(b):
        a, b = b, a
    if not b:
        return len(a)
    peq: dict = {}
    for i, c in enumerate(b):
        peq[c] = peq.get(c, 0) | (1 << i)
    m = len(b)
    mask, top = (1 << m) - 1, 1 << (m - 1)
    pv, mv, score = mask, 0, m
    for c in a:
        eq = peq.get(c, 0)
        xv = eq | mv
        xh = (((eq & pv) + pv) ^ pv) | eq
        ph = mv | (~(xh | pv) & mask)
        mh = pv & xh
        if ph & top:
            score += 1
        elif mh & top:
            score -= 1
        ph = ((ph << 1) | 1) & mask
        mh = (mh << 1) & mask
        pv = mh | (~(xv | ph) & mask)
        mv = ph & xv
    return score


def _edit_distance_python(a, b) -> int:
    """The plain dynamic programme, kept for the self-test of the bit-parallel version."""
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def cer(reference: str, hypothesis: str) -> float:
    ref, hyp = normalise(reference).replace(" ", ""), normalise(hypothesis).replace(" ", "")
    if not ref:
        return 0.0 if not hyp else 1.0
    return _edit_distance(ref, hyp) / len(ref)


def wer(reference: str, hypothesis: str) -> float:
    ref, hyp = normalise(reference).split(), normalise(hypothesis).split()
    if not ref:
        return 0.0 if not hyp else 1.0
    return _edit_distance(ref, hyp) / len(ref)



def speaker_accuracy(ref_segments: list[dict], hyp_segments: list[dict]) -> float | None:
    """Share of aligned characters whose hypothesis speaker matches the reference speaker, under the best one-to-one
    mapping of hypothesis labels to reference speakers (labels are arbitrary: SPK1 may be "B").
    Reference turns carry no times, so the texts are aligned character by character (the same alignment as CER).
    None when the reference has fewer than two speakers (nothing to measure)."""
    import itertools
    r_chars, r_spk, h_chars, h_spk = [], [], [], []
    for seg in ref_segments:
        t = normalise(seg.get("text", "")).replace(" ", "")
        r_chars += list(t)
        r_spk += [seg.get("speaker")] * len(t)
    for seg in hyp_segments:
        t = normalise(seg.get("text", "")).replace(" ", "")
        h_chars += list(t)
        h_spk += [seg.get("speaker")] * len(t)
    speakers = sorted({s for s in r_spk if s})
    if len(speakers) < 2 or not h_chars or not r_chars:
        return None
    pairs = [(r_spk[i], h_spk[j]) for i, j in _align_pairs(r_chars, h_chars)]
    if not pairs:
        return None
    labels = sorted({h for _, h in pairs if h}, key=str)
    best = 0
    slots = speakers + [None] * max(0, len(labels) - len(speakers))
    for perm in itertools.permutations(slots, len(labels)):
        mp = dict(zip(labels, perm))
        best = max(best, sum(1 for r, h in pairs if mp.get(h) == r))
    return round(best / len(pairs), 4)


TURN_KINDS = ("correct", "misheard", "lost", "lost_backchannel", "merged", "merged_other", "wrong_speaker",
              "nothing_to_score")
_SCORING: dict = {}


def scoring_words(lang: str) -> dict:
    """library/scoring/<lang>.toml, the failure monitor's word lists (generic language data), in the "loose" fold.
    An unknown language gets the acknowledgements of every language and nothing else."""
    if lang in _SCORING:
        return _SCORING[lang]
    import tomllib
    from .common import TRX_HOME
    files = sorted((TRX_HOME / "library" / "scoring").glob(f"{lang or '*'}.toml"))
    raw: dict = {}
    for p in files:
        try:
            d = tomllib.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        for k, v in d.items():
            raw.setdefault(k, []).extend(v)
    if not lang:
        raw = {"backchannels": raw.get("backchannels", [])}

    def loose(ws):
        return [normalise_level(w, lang, "loose") for w in ws]
    _SCORING[lang] = {
        "backchannels": {normalise(w) for w in raw.get("backchannels", [])},
        "negations": set(loose(raw.get("negations", []))),
        "negation_stems": tuple(loose(raw.get("negation_stems", []))),
        "joined": [tuple(loose(pair)) for pair in raw.get("joined", [])],
        "not_negation": set(loose(raw.get("not_negation", []))),
        "number_endings": tuple(loose(raw.get("number_endings", []))),
    }
    return _SCORING[lang]


def _content_fold(text: str, lang: str, fillers: bool = True) -> str:
    """The "content" level; fillers=False keeps acknowledgements (so a lost "हं" turn can still be seen)."""
    if fillers:
        return normalise_level(text, lang, "content")
    t = normalise_level(text, lang, "loose")
    if lang == "mr":
        for rx, rep in _MR_FOLDS:
            t = rx.sub(rep, t)
    return _SPACE.sub(" ", t).strip()


def turn_outcomes(ref_segments: list[dict], hyp_segments: list[dict], lang: str = "") -> list[str]:
    """What happened to each reference turn, compared at the content level (spelling conventions do not count):
      correct          >= 85 % of its characters right
      misheard         partly right
      lost             < 30 % right; lost_backchannel when the turn is only an acknowledgement ("हं", "हो" …)
      merged           its words sit in a line that belongs to a neighbouring turn of the SAME speaker: the other
                       person's reply between them was lost (the old "missing turns" count called this missing)
      merged_other     its words sit inside the OTHER speaker's line
      wrong_speaker    its words are there under the other speaker's label (needs >= 2 reference speakers)
      nothing_to_score only fillers."""
    from collections import Counter, defaultdict
    r_chars, r_turn, h_chars, h_seg = [], [], [], []
    for i, s in enumerate(ref_segments):
        t = _content_fold(s.get("text", ""), lang, fillers=False).replace(" ", "")
        r_chars += list(t)
        r_turn += [i] * len(t)
    for j, s in enumerate(hyp_segments):
        t = _content_fold(s.get("text", ""), lang, fillers=False).replace(" ", "")
        h_chars += list(t)
        h_seg += [j] * len(t)
    pairs = _align_pairs(r_chars, h_chars)
    two = len({s.get("speaker") for s in ref_segments if s.get("speaker")}) >= 2
    spk_map: dict = {}
    for (r, h), _ in Counter((ref_segments[r_turn[i]].get("speaker"), hyp_segments[h_seg[j]].get("speaker"))
                             for i, j in pairs).most_common():
        if r not in spk_map and h not in spk_map.values():
            spk_map[r] = h
    of_turn, owner_votes = defaultdict(list), defaultdict(Counter)
    for i, j in pairs:
        of_turn[r_turn[i]].append((i, j))
        owner_votes[h_seg[j]][r_turn[i]] += 1
    owner = {j: c.most_common(1)[0][0] for j, c in owner_votes.items()}
    size = Counter(r_turn)
    acks = scoring_words(lang)["backchannels"]
    kinds = []
    for ti, seg in enumerate(ref_segments):
        n, al = size.get(ti, 0), of_turn.get(ti, [])
        same = sum(1 for i, j in al if r_chars[i] == h_chars[j])
        segs = {h_seg[j] for _, j in al}
        spks = Counter(hyp_segments[h_seg[j]].get("speaker") for _, j in al)
        words = normalise(seg.get("text", "")).split()
        ack = 0 < len(words) <= 3 and all(w in acks or _FILLERS.fullmatch(w) for w in words)
        want = spk_map.get(seg.get("speaker"))
        if n == 0:
            kind = "nothing_to_score"
        elif same < 0.3 * n:
            kind = "lost_backchannel" if ack else "lost"
        elif two and want is not None and spks.most_common(1)[0][0] != want:
            kind = "wrong_speaker"
        elif all(owner.get(j) != ti for j in segs):
            same_spk = all(ref_segments[owner[j]].get("speaker") == seg.get("speaker") for j in segs)
            kind = "merged" if same_spk or not two else "merged_other"
        elif same >= 0.85 * n:
            kind = "correct"
        else:
            kind = "misheard"
        kinds.append(kind)
    return kinds


def _joined(words: list[str], pairs: list[tuple]) -> list[str]:
    """Join the listed word pairs (two words that are one non-negative word, e.g. Marathi "otherwise"; E39: four
    false alarms on the 20-minute call came from its two spellings)."""
    out: list[str] = []
    for w in words:
        if out and (out[-1], w) in pairs:
            out[-1] += w
        else:
            out.append(w)
    return out


def _number_words(lang: str) -> set[str]:
    from .numbers import table
    t = table(lang) or {}
    listed = list(t.get("words", []))
    words = set(listed[2:]) | set(listed[:1]) | set(t.get("hundreds_prefix", [])) | \
        {t.get("hundred_exact", ""), t.get("thousand", ""), t.get("lakh", ""), t.get("crore", "")}
    words.discard("")                  # listed[1], "one", is left out: it is mostly an article ("a man")
    return {normalise_level(w, lang, "loose") for w in words}


def _number_core(w: str, nums: set, endings: tuple) -> str | None:
    """The number word inside w: a number with a case ending is the same number (FLEURS false alarm, E39)."""
    if w in nums:
        return w
    return next((w[:-len(e)] for e in endings if w.endswith(e) and w[:-len(e)] in nums), None)


def meaning_errors(reference: str, hypothesis: str, lang: str = "") -> dict:
    """Errors that change the meaning while costing few characters (E39): a negation lost or added (a "will" that
    became "will not"), a number word lost, changed or added. Words are compared after the "loose" fold; a word
    counts as kept when the same kind of word stands within two positions of where the alignment puts it. Word
    lists: library/scoring/<lang>.toml."""
    sw = scoring_words(lang)
    ref = _joined(normalise_level(reference, lang, "loose").split(), sw["joined"])
    hyp = _joined(normalise_level(hypothesis, lang, "loose").split(), sw["joined"])
    nums = _number_words(lang) if lang in ("mr", "hi") else set()
    pairs = _align_pairs(ref, hyp)
    to_h, to_r = dict(pairs), {j: i for i, j in pairs}

    def anchors(n, mapping):           # aligned partner, or the position after the previous aligned partner
        out, last = [], -1
        for k in range(n):
            if k in mapping:
                last = mapping[k]
                out.append(last)
            else:
                out.append(last + 1)
        return out
    ah, ar = anchors(len(ref), to_h), anchors(len(hyp), to_r)

    def near(seq, k):
        return seq[max(0, k - 2):k + 3]

    def neg(w):
        return w in sw["negations"] or (bool(sw["negation_stems"]) and w.startswith(sw["negation_stems"])
                                        and w not in sw["not_negation"])
    neg_err = sum(1 for i, w in enumerate(ref) if neg(w) and not any(neg(x) for x in near(hyp, ah[i]))) + \
        sum(1 for j, w in enumerate(hyp) if neg(w) and not any(neg(x) for x in near(ref, ar[j])))
    r_num = [_number_core(w, nums, sw["number_endings"]) for w in ref]
    h_num = [_number_core(w, nums, sw["number_endings"]) for w in hyp]
    num_err = sum(1 for i, c in enumerate(r_num) if c and c not in near(h_num, ah[i])) + \
        sum(1 for j, c in enumerate(h_num) if c and c not in near(r_num, ar[j]))
    return {"negations": sum(1 for w in ref if neg(w)), "negation_errors": neg_err,
            "numbers": sum(1 for c in r_num if c), "number_errors": num_err}


def line_scores(reference: str, hyp_segments: list[dict], lang: str = "") -> list[dict]:
    """Per output line, at the content level: the reference characters it covers and its errors (substituted and
    extra characters, plus missing reference characters charged to the line before them)."""
    r = list(normalise_level(reference, lang, "content").replace(" ", ""))
    h_chars, h_seg = [], []
    for j, s in enumerate(hyp_segments):
        t = normalise_level(s.get("text", ""), lang, "content").replace(" ", "")
        h_chars += list(t)
        h_seg += [j] * len(t)
    pairs = _align_pairs(r, h_chars)
    ref_n, err = [0] * len(hyp_segments), [0] * len(hyp_segments)
    seen_h, last = set(), None
    to_h = dict(pairs)
    for i, j in pairs:
        seen_h.add(j)
        ref_n[h_seg[j]] += 1
        err[h_seg[j]] += r[i] != h_chars[j]
    for j in range(len(h_chars)):
        if j not in seen_h:
            err[h_seg[j]] += 1
    for i in range(len(r)):
        if i in to_h:
            last = h_seg[to_h[i]]
        elif last is not None:
            ref_n[last] += 1
            err[last] += 1
    return [{"ref": ref_n[j], "err": err[j]} for j in range(len(hyp_segments))]


def self_test() -> tuple[bool, str]:
    checks = [
        cer("नमस्कार मंडळी", "नमस्कार मंडळी") == 0.0,
        cer("abc", "abd") == 1 / 3,
        wer("a b c", "a x c") == 1 / 3,
        cer("हा हा।", "हा हा") == 0.0,
    ]
    import random
    rnd = random.Random(5)
    for _ in range(60):   # the fast distance and alignment must equal the plain loops they replaced (E39)
        a = [rnd.choice("कखगी ं") for _ in range(rnd.randint(0, 25))]
        b = [rnd.choice("कखगी ं") for _ in range(rnd.randint(0, 25))]
        ops = _ops_python(a, b) if a and b else None
        slow, i, j = [], len(a), len(b)
        while ops is not None and i > 0 and j > 0:
            op = ops[i - 1][j]
            if op == 0:
                slow.append((i - 1, j - 1))
            i, j = i - (op != 2), j - (op != 1)
        checks.append(_edit_distance(a, b) == _edit_distance_python(a, b) and _align_pairs(a, b) == slow[::-1])
    checks += [   # invented sentences (never words of a case)
        cer_level("हं त्याचं घर शाळेजवळये", "त्याच घर साळेजवळ आहे", "mr", "content") == 0.0,
        meaning_errors("पाऊस उद्या पडणार आहे", "पाऊस उद्या पडणार नाही", "mr")["negation_errors"] == 1,
        meaning_errors("नाहीतर पाऊस पडेल", "नाही तर पाऊस पडेल", "mr")["negation_errors"] == 0,
        meaning_errors("चाळीस किलो तांदूळ आणला", "चौदा किलो तांदूळ आणला", "mr")["number_errors"] == 2,
        meaning_errors("पन्नास रुपयांचा आंबा", "पन्नासचा आंबा", "mr")["number_errors"] == 0,
        turn_outcomes([{"speaker": "A", "text": "आज बाजारात जाऊ"}, {"speaker": "B", "text": "हं"},
                       {"speaker": "A", "text": "भाजी आणि फळे घेऊन येऊ"}],
                      [{"speaker": "S1", "text": "आज बाजारात जाऊ भाजी आणि फळे घेऊन येऊ"}], "mr")
        == ["merged", "lost_backchannel", "correct"],   # the line belongs to the turn with more characters
    ]
    return all(checks), f"{sum(checks)}/{len(checks)} metric checks passed"
