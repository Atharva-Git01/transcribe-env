"""GPU setup: make the environment's own CUDA 12 libraries visible (no system CUDA install needed).

- cuBLAS / cudart: toolchain/bin/cuda12 (hard links, see SOURCE.json there)
- cuDNN 9: the venv's nvidia-cudnn-cu12 wheel (site-packages/nvidia/cudnn/bin)
One GPU model at a time (config [runtime].gpu_one_model_at_a_time); callers unload before loading the next.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from .common import TRX_HOME, load_config

_done = False


def dll_dirs() -> list[Path]:
    dirs = [TRX_HOME / "toolchain" / "bin" / "cuda12"]
    for sp in sys.path:
        p = Path(sp) / "nvidia" / "cudnn" / "bin"
        if p.is_dir():
            dirs.append(p)
            break
    return [d for d in dirs if d.is_dir()]


def prepare() -> list[str]:
    global _done
    dirs = dll_dirs()
    if not _done:
        for d in dirs:
            try:
                os.add_dll_directory(str(d))
            except (AttributeError, OSError):
                pass
        os.environ["PATH"] = os.pathsep.join([str(d) for d in dirs] + [os.environ.get("PATH", "")])
        _done = True
    return [str(d) for d in dirs]


def cuda_device_count() -> int:
    prepare()
    try:
        import ctranslate2
        return int(ctranslate2.get_cuda_device_count())
    except Exception:  # noqa: BLE001
        return 0


def choose_device() -> str:
    want = load_config("defaults")["runtime"].get("device", "auto")
    if want == "cpu":
        return "cpu"
    if cuda_device_count() > 0:
        return "cuda"
    return "cpu"
