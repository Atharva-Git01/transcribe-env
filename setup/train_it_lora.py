"""Train small LoRA adapters on Indic-Transcribe's DECODER so that it writes words as spoken (PLAN_v0.9 section 4.2).

  python setup/train_it_lora.py --out toolchain/models/it-lora-v1-test-call --test-fixture call_240509_20min
         [--slice toolchain/data/indicvoices-mr-slice1] [--set v1] [--epochs 2] [--rank 16] [--lr 2e-4]

Setup-time tool, offline, on the GPU (owner's rule). The encoder stays frozen: its output for every training clip is
computed once and kept, so each pass runs only the decoder — minutes per run, little GPU memory. Training data:
- the IndicVoices slice; its wide-band clips are first squeezed to the phone band (300-3400 Hz, 8 kHz G.711 mu-law
  and back) so that they sound like calls. Target (--targets): "controlled" (default) = the standard text
  (`normalized`) with only the words whose spoken form in `verbatim` is a listed pair of the spoken-variant list
  (use = "yes"); "verbatim" = the transcribers' whole as-spoken text. Run v1 used "verbatim" and learned the
  transcribers' habits too, e.g. "nahi" -> "na" (a different word): negation errors on the call 10 -> 22 (E42);
- the owner's as-spoken lines of the screening set, EXCEPT those of --test-fixture: fold by recording, never train
  and test on the same recording. They are repeated --owner-weight times (they are few and closest to the calls).
5 % of the IndicVoices clips are held out to watch the loss. Writes the adapter (peft) and SOURCE.json (data, fold,
settings, base model, hashes). The adapter is trained on private lines: it stays on this PC (toolchain/ is
git-ignored) and is never published.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HOME = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HOME))

from pipeline.audio import decode_mono, read_wav  # noqa: E402

TARGETS = ["query_net", "key_net", "value_net", "out_projection"]   # decoder attention only (the encoder's are linear_*)
CACHE_VERSION = 1          # bump when the clip preparation changes (phone band, decoding)
WIDE_BAND = 0.005          # share of energy above 4 kHz from which a clip counts as wide band (calls: ~0)


def high_band_share(x: np.ndarray) -> float:
    p = np.abs(np.fft.rfft(x)) ** 2
    f = np.fft.rfftfreq(len(x), 1 / 16000)
    return float(p[f > 4000].sum() / max(p[f > 100].sum(), 1e-12))


def phone_band(x: np.ndarray) -> np.ndarray:
    """300-3400 Hz, 8 kHz, G.711 mu-law, back to 16 kHz: what a phone line does to a voice (ffmpeg only, E12)."""
    a = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-f", "f32le", "-ar", "16000", "-ac", "1", "-i", "pipe:0",
                        "-af", "highpass=f=300,lowpass=f=3400", "-ar", "8000", "-c:a", "pcm_mulaw", "-f", "wav",
                        "pipe:1"], input=x.astype(np.float32).tobytes(), capture_output=True, check=True).stdout
    b = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", "pipe:0", "-ar", "16000", "-ac", "1", "-f", "f32le",
                        "pipe:1"], input=a, capture_output=True, check=True).stdout
    return np.frombuffer(b, dtype=np.float32).copy()


def controlled(normalized: str, verbatim: str, lang: str = "mr") -> str:
    """The standard text with only the listed spoken forms of the verbatim text put in (pairs of the spoken-variant
    list with use = "yes"): a dropped or reduced word the list does not hold keeps its standard form."""
    import difflib

    from pipeline.metrics import normalise, normalise_level
    from pipeline.spelling import ortho, variants
    v = variants(lang)
    n, w = normalise(normalized).split(), normalise(verbatim).split()
    out = list(n)

    def listed(std, spk):
        o = ortho(spk, lang)
        if " " in std:
            return o in v["standard_phrases"].get(tuple(ortho(x, lang) for x in std.split()), set())
        return o in v["standard_words"].get(ortho(std, lang), set())
    loose = [normalise_level(x, lang, "loose") for x in n], [normalise_level(x, lang, "loose") for x in w]
    for op, i1, i2, j1, j2 in reversed(difflib.SequenceMatcher(None, *loose, autojunk=False).get_opcodes()):
        if op != "replace":
            continue
        if i2 - i1 == j2 - j1:
            for k in range(i2 - i1):
                if listed(n[i1 + k], w[j1 + k]):
                    out[i1 + k] = w[j1 + k]
        elif (i2 - i1, j2 - j1) == (2, 1) and listed(" ".join(n[i1:i2]), w[j1]):
            out[i1:i2] = [w[j1]]
    return " ".join(out)


def examples(slice_dir: Path, set_dir: Path, test_fixture: str, owner_weight: int, seed: int, targets: str):
    """[(audio, text, source)] for training and the held-out IndicVoices clips, plus counts."""
    rnd = random.Random(seed)
    iv = [json.loads(x) for x in (slice_dir / "manifest.jsonl").read_text(encoding="utf-8").splitlines()]
    rnd.shuffle(iv)
    n_val = max(20, len(iv) // 20)
    train, val, squeezed, changed = [], [], 0, 0
    for k, m in enumerate(iv):
        x = decode_mono(slice_dir / m["audio"])
        if high_band_share(x) > WIDE_BAND:
            x = phone_band(x)
            squeezed += 1
        text = m["verbatim"] if targets == "verbatim" else controlled(m["normalized"], m["verbatim"])
        changed += text != m["normalized"]
        (val if k < n_val else train).append((x, text, "indicvoices", m["id"]))
    lines = json.loads((set_dir / "lines.json").read_text(encoding="utf-8"))["segments"]
    own = [x for x in lines if x["fixture"] != test_fixture]
    for x in own:
        audio, _ = read_wav(set_dir / "audio" / f"{x['id']}.wav")
        train += [(audio, x["text"], "owner", x["id"])] * owner_weight
    counts = {"targets": targets, "indicvoices_with_spoken_forms": changed,
              "indicvoices_train": len(iv) - n_val, "indicvoices_heldout": n_val, "phone_band_squeezed": squeezed,
              "owner_lines": len(own), "owner_weight": owner_weight,
              "owner_fixtures": sorted({x["fixture"] for x in own}), "test_fixture": test_fixture}
    return train, val, counts


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--test-fixture", required=True)
    ap.add_argument("--slice", default="toolchain/data/indicvoices-mr-slice1")
    ap.add_argument("--set", default="v1")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--owner-weight", type=int, default=3)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--targets", choices=["controlled", "verbatim"], default="controlled")
    ap.add_argument("--cache", default="toolchain/data/it-encoder-cache-v1.pt",
                    help="encoder outputs of the IndicVoices clips, kept between runs (same base model, same clips)")
    a = ap.parse_args()
    out = Path(a.out) if Path(a.out).is_absolute() else HOME / a.out
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} is not empty (nothing was changed)")
    import torch
    from peft import LoraConfig, get_peft_model
    from pipeline.engines.indic_transcribe_engine import IndicTranscribeEngine
    random.seed(a.seed)
    torch.manual_seed(a.seed)
    t0 = time.time()
    train, val, counts = examples(HOME / a.slice, HOME / "evals" / "screening" / a.set, a.test_fixture,
                                  a.owner_weight, a.seed, a.targets)
    print(f"data: {len(train)} training clips, {len(val)} held out ({counts}); {time.time() - t0:.0f} s")
    eng = IndicTranscribeEngine("indic_transcribe_core")
    info = eng.load("cuda", {"device": "auto", "dtype": "bfloat16"})
    if info["device"] != "cuda":
        raise SystemExit("the GPU is not usable: training runs on the GPU only (owner's rule)")
    asr = eng.asr
    model, tok = asr.model, asr.tokenizer
    prompt = tok.encode_prompt("mr")
    mod = sys.modules[type(model).__module__]

    cache_path = HOME / a.cache
    key = f"{CACHE_VERSION}|{info.get('model_sha256')}"
    cache = torch.load(cache_path, weights_only=True) if cache_path.is_file() else {}
    if cache.get("__key__") != key:
        cache = {"__key__": key}
    fresh = 0

    def encode(items):
        nonlocal fresh
        enc = []
        for x, text, src, cid in items:
            if src == "indicvoices" and cid in cache:
                e = cache[cid]
            else:
                wav = asr._load(np.ascontiguousarray(x, dtype=np.float32))
                feats, mask = asr._features(wav)
                with torch.inference_mode():
                    e = model.model.encoder(feats, attention_mask=mask).last_hidden_state[0].to(torch.bfloat16).cpu()
                if src == "indicvoices":
                    cache[cid] = e
                    fresh += 1
            ids = [i + tok.spl_size for i in tok.multi.encode(text, out_type=int)] + [tok.eos_id]
            enc.append((e, prompt + ids, src))
        return enc
    t1 = time.time()
    tr, va = encode(train), encode(val)
    if fresh:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(cache, cache_path)
    print(f"encoder outputs: {len(tr) + len(va)} clips ({fresh} computed, the rest from {a.cache}), "
          f"{time.time() - t1:.0f} s")
    for p in model.parameters():
        p.requires_grad_(False)
    cfg = LoraConfig(r=a.rank, lora_alpha=2 * a.rank, lora_dropout=0.05, target_modules=TARGETS, bias="none")
    pm = get_peft_model(model, cfg)
    params = [p for p in pm.parameters() if p.requires_grad]
    n_params = sum(p.numel() for p in params)
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    steps = a.epochs * math.ceil(len(tr) / a.batch)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 50) * max(0.05, 1 - s / steps))

    def batch_loss(items):
        T = max(e.shape[0] for e, _, _ in items)
        L = max(len(s) for _, s, _ in items) - 1
        enc = torch.zeros(len(items), T, items[0][0].shape[1], dtype=torch.bfloat16)
        lengths = torch.tensor([e.shape[0] for e, _, _ in items])
        inp = torch.full((len(items), L), tok.pad_id if hasattr(tok, "pad_id") else 2, dtype=torch.long)
        lab = torch.full((len(items), L), -100, dtype=torch.long)
        for b, (e, s, _) in enumerate(items):
            enc[b, :e.shape[0]] = e
            inp[b, :len(s) - 1] = torch.tensor(s[:-1])
            lab[b, len(prompt) - 1:len(s) - 1] = torch.tensor(s[len(prompt):])
        eo = mod.IndicCanaryEncoderOutput(last_hidden_state=enc.cuda(), lengths=lengths.cuda())
        logits = pm(encoder_outputs=eo, decoder_input_ids=inp.cuda(), use_cache=False).logits
        return torch.nn.functional.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), lab.cuda().reshape(-1),
                                                 ignore_index=-100)

    def evaluate():
        pm.eval()
        with torch.no_grad():
            losses = [float(batch_loss(va[i:i + a.batch])) for i in range(0, len(va), a.batch)]
        pm.train()
        return round(sum(losses) / len(losses), 4)
    history = [{"epoch": 0, "heldout_loss": evaluate()}]
    print(f"held-out loss before training: {history[0]['heldout_loss']} ({n_params:,} trainable parameters)")
    pm.train()
    step, t2 = 0, time.time()
    for ep in range(1, a.epochs + 1):
        order = sorted(range(len(tr)), key=lambda i: (tr[i][0].shape[0] // 25, random.random()))
        batches = [order[i:i + a.batch] for i in range(0, len(order), a.batch)]
        random.shuffle(batches)
        run = 0.0
        for k, bidx in enumerate(batches, 1):
            loss = batch_loss([tr[i] for i in bidx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            step += 1
            run += float(loss)
            if k % 50 == 0:
                print(f"  epoch {ep} batch {k}/{len(batches)} loss {run / 50:.4f}")
                run = 0.0
        history.append({"epoch": ep, "heldout_loss": evaluate()})
        print(f"epoch {ep}: held-out loss {history[-1]['heldout_loss']} ({time.time() - t2:.0f} s)")
    out.mkdir(parents=True, exist_ok=True)
    pm.save_pretrained(str(out))
    files = {p.name: {"sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "size": p.stat().st_size}
             for p in sorted(out.iterdir()) if p.is_file()}
    (out / "SOURCE.json").write_text(json.dumps({
        "id": out.name, "kind": "LoRA adapter for indic_transcribe_core (decoder attention)", "private": True,
        "base_model_sha256": info.get("model_sha256"), "made_by": "setup/train_it_lora.py",
        "data": counts, "settings": {"rank": a.rank, "alpha": 2 * a.rank, "dropout": 0.05, "targets": TARGETS,
                                     "lr": a.lr, "epochs": a.epochs, "batch": a.batch, "seed": a.seed,
                                     "trainable_parameters": n_params, "wide_band_share": WIDE_BAND},
        "history": history, "minutes": round((time.time() - t0) / 60, 1), "files": files}, indent=2), encoding="utf-8")
    print(f"saved {out} ({(time.time() - t0) / 60:.1f} min)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
