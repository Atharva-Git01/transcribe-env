"""Build a word n-gram language model + lexicon (toolchain/models/lm-<lang>-v1/) from a corpus with one sentence
per line. Setup-time tool: runs offline on an already downloaded file; the runtime only reads the result.

  python setup/build_lm.py --lang mr --corpus toolchain/downloads/.../mr_first1GB.txt \
      --source "IndicCorp v2 (AI4Bharat), data/mr.txt, first 1 GiB" --licence CC-0

Pass 1 counts words -> vocabulary (words seen >= --min-word-count times, at most 2^20-2, by frequency).
Pass 2 counts bigrams/trigrams of vocabulary words in bounded-memory chunks, then drops n-grams seen fewer than
--min-count times. See pipeline/lm.py for the file layout and the scoring.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

HOME = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HOME))
from pipeline.lm import BOS, MAX_VOCAB, UNK, model_dir, pack2, pack3, tokens  # noqa: E402


def lines(path: Path, max_lines: int | None):
    """Complete lines only (a byte-range prefix ends mid-line: the last partial line is dropped)."""
    with open(path, "rb") as f:
        n = 0
        prev = None
        for raw in f:
            if prev is not None:
                yield prev
                n += 1
                if max_lines and n >= max_lines:
                    return
            prev = raw.decode("utf-8", "ignore").strip() if raw.endswith(b"\n") else None
        # the final line is yielded only if it ended with a newline
        if prev is not None and (not max_lines or n < max_lines):
            yield prev


def merge(keys: list, counts: list):
    k = np.concatenate(keys)
    c = np.concatenate(counts).astype(np.int64)
    order = np.argsort(k, kind="stable")
    k, c = k[order], c[order]
    uniq, start = np.unique(k, return_index=True)
    return uniq, np.add.reduceat(c, start)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--licence", required=True)
    ap.add_argument("--max-lines", type=int, default=None)
    ap.add_argument("--min-word-count", type=int, default=2)
    ap.add_argument("--min-count", type=int, default=2)
    ap.add_argument("--chunk-lines", type=int, default=200_000)
    a = ap.parse_args()
    corpus = Path(a.corpus)
    out = model_dir(a.lang)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # ---- pass 1: words
    wc: Counter = Counter()
    n_sent = n_tok = 0
    for ln in lines(corpus, a.max_lines):
        ws = tokens(ln)
        if ws:
            wc.update(ws)
            n_sent += 1
            n_tok += len(ws)
        if n_sent and n_sent % 1_000_000 == 0:
            print(f"  pass 1: {n_sent:,} sentences, {len(wc):,} word types ({time.time() - t0:.0f}s)", flush=True)
    vocab = ["<s>"] + [w for w, c in wc.most_common(MAX_VOCAB - 1) if c >= a.min_word_count]
    ids = {w: i for i, w in enumerate(vocab)}
    uni = np.zeros(len(vocab), dtype=np.uint32)
    uni[0] = n_sent
    for w, i in ids.items():
        if i:
            uni[i] = wc[w]
    del wc
    print(f"pass 1 done: {n_sent:,} sentences, {n_tok:,} tokens, vocab {len(vocab):,} ({time.time() - t0:.0f}s)", flush=True)

    # ---- pass 2: bigrams + trigrams in chunks
    bk, bc, tk, tc = [], [], [], []

    def flush(seqs):
        b_keys, t_keys = [], []
        for s in seqs:
            if len(s) < 3:
                continue
            arr = np.array(s, dtype=np.uint64)
            x, y, z = arr[:-2], arr[1:-1], arr[2:]
            ok3 = (x != UNK) & (y != UNK) & (z != UNK)
            t_keys.append(pack3(x[ok3], y[ok3], z[ok3]))
            p, q = arr[:-1], arr[1:]
            ok2 = (p != UNK) & (q != UNK)
            b_keys.append(pack2(p[ok2], q[ok2]))
        for store_k, store_c, parts in ((bk, bc, b_keys), (tk, tc, t_keys)):
            if parts:
                u, c = np.unique(np.concatenate(parts), return_counts=True)
                store_k.append(u)
                store_c.append(c.astype(np.int64))
            if len(store_k) >= 12:   # keep memory bounded: fold partial counts together
                u, c = merge(store_k, store_c)
                store_k[:] = [u]
                store_c[:] = [c]

    seqs, n = [], 0
    for ln in lines(corpus, a.max_lines):
        ws = tokens(ln)
        if not ws:
            continue
        seqs.append([BOS, BOS] + [ids.get(w, UNK) for w in ws])
        n += 1
        if len(seqs) >= a.chunk_lines:
            flush(seqs)
            seqs = []
            print(f"  pass 2: {n:,} sentences ({time.time() - t0:.0f}s)", flush=True)
    flush(seqs)
    b_keys, b_counts = merge(bk, bc)
    t_keys, t_counts = merge(tk, tc)
    kb, kt = b_counts >= a.min_count, t_counts >= a.min_count
    b_keys, b_counts = b_keys[kb], np.minimum(b_counts[kb], 2**32 - 1).astype(np.uint32)
    t_keys, t_counts = t_keys[kt], np.minimum(t_counts[kt], 2**32 - 1).astype(np.uint32)
    print(f"pass 2 done: {len(b_keys):,} bigrams, {len(t_keys):,} trigrams kept (min count {a.min_count}) "
          f"({time.time() - t0:.0f}s)", flush=True)

    (out / "vocab.txt").write_text("\n".join(vocab), encoding="utf-8")
    np.save(out / "unigram.npy", uni)
    np.save(out / "bigram_keys.npy", b_keys)
    np.save(out / "bigram_counts.npy", b_counts)
    np.save(out / "trigram_keys.npy", t_keys)
    np.save(out / "trigram_counts.npy", t_counts)
    corpus_sha = sha256(corpus)
    meta = {"lang": a.lang, "sentences": n_sent, "tokens": n_tok, "vocab": len(vocab), "bigrams": int(len(b_keys)),
            "trigrams": int(len(t_keys)), "min_word_count": a.min_word_count, "min_count": a.min_count,
            "scoring": "stupid backoff (alpha 0.4)", "corpus_bytes": corpus.stat().st_size,
            "corpus_sha256": corpus_sha, "built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    files = {p.name: {"sha256": sha256(p), "size": p.stat().st_size} for p in sorted(out.iterdir())
             if p.is_file() and p.name != "SOURCE.json"}
    src = {"id": f"lm_{a.lang}_v1", "licence": a.licence, "source": a.source,
           "note": f"built by setup/build_lm.py from {corpus.name} (sha256 {corpus_sha[:16]}…); the raw corpus is not kept",
           "model_sha256": files["trigram_keys.npy"]["sha256"], "files": files}
    (out / "SOURCE.json").write_text(json.dumps(src, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: meta[k] for k in ("sentences", "tokens", "vocab", "bigrams", "trigrams")}),
          f"-> {out} ({sum(v['size'] for v in files.values()) / 1e6:.0f} MB, {time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
