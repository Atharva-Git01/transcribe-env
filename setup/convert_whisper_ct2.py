"""Convert a Hugging Face Whisper checkpoint to CTranslate2, so the faster-whisper engine runs it on the GPU
(PLAN_v0.9: IndicWhisper Marathi from AI4Bharat Vistaar, MIT).

  python setup/convert_whisper_ct2.py --src toolchain/models/indicwhisper-mr-hf --out toolchain/models/indicwhisper-mr-ct2
                                      --source "<where the checkpoint came from>" --licence MIT

Setup-time tool, offline. --src may be the extracted archive: the folder holding config.json is found inside it. A
tokenizer.json is written first when the checkpoint has only the slow tokenizer files (faster-whisper would otherwise
try to fetch one). Weights are kept in float16; the engine picks int8_float16 on the GPU. Writes SOURCE.json with
every output file's sha256 (`model_sha256` = model.bin).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

HOME = Path(__file__).resolve().parents[1]
NEEDED = ("config.json", "generation_config.json", "pytorch_model.bin", "model.safetensors", "preprocessor_config.json",
          "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "added_tokens.json",
          "special_tokens_map.json", "normalizer.json")


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--source", default="")
    ap.add_argument("--licence", default="")
    a = ap.parse_args()
    src = Path(a.src) if Path(a.src).is_absolute() else HOME / a.src
    out = Path(a.out) if Path(a.out).is_absolute() else HOME / a.out
    found = sorted(src.rglob("config.json"), key=lambda p: len(p.parts))
    if not found:
        raise SystemExit(f"no config.json under {src}")
    ckpt = found[0].parent
    if out.exists() and any(p for p in out.iterdir() if not p.name.startswith(".")):
        raise SystemExit(f"{out} is not empty (nothing was changed)")
    from ctranslate2.converters import TransformersConverter
    from transformers import WhisperTokenizerFast
    # staging folders inside --out: --out may be a junction (to the USB), which the converter must not delete
    out.mkdir(parents=True, exist_ok=True)
    work, staged = out / ".checkpoint", out / ".converted"
    work.mkdir(exist_ok=True)
    for name in NEEDED:                       # only what the converter reads (no optimizer or RNG states)
        if (ckpt / name).is_file():
            shutil.copy2(ckpt / name, work / name)
    if not (work / "tokenizer.json").is_file():
        WhisperTokenizerFast.from_pretrained(str(work)).save_pretrained(str(work))
    copy = [n for n in ("tokenizer.json", "preprocessor_config.json") if (work / n).is_file()]
    TransformersConverter(str(work), copy_files=copy, load_as_float16=True).convert(
        str(staged), quantization="float16", force=True)
    for f in staged.iterdir():
        f.replace(out / f.name)
    shutil.rmtree(staged)
    shutil.rmtree(work)
    files = {p.name: {"sha256": sha256(p), "size": p.stat().st_size} for p in sorted(out.iterdir()) if p.is_file()}
    (out / "SOURCE.json").write_text(json.dumps({
        "id": out.name, "licence": a.licence, "source": a.source, "converted_from": str(ckpt.relative_to(src.parent)),
        "tool": "setup/convert_whisper_ct2.py (CTranslate2 TransformersConverter, float16)",
        "model_sha256": files["model.bin"]["sha256"], "files": files}, indent=2), encoding="utf-8")
    print(f"converted {ckpt} -> {out}: " + ", ".join(f"{n} {v['size'] / 1e6:.0f} MB" for n, v in files.items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
