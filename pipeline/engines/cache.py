"""Engines a run has already loaded (E34). Indic-Transcribe identifies the language in the probe and then transcribes,
so it is loaded once per run instead of twice (~25 s). Only engines that ask for it (`keep_loaded`) are kept, and
only one at a time: asking for a different engine releases the held one first, so the GPU never holds two models
(E06). The memory check (E31) runs before every load. `release()` frees everything: the end of a run, the end of
recognition, or before a stage that needs the GPU memory.
"""
from __future__ import annotations

from ..common import TrxError, load_config

_held: dict = {}            # engine_id -> (engine, info)


def _check_memory(eng, device: str, params: dict) -> None:
    from .. import resources
    rt = load_config("defaults")["runtime"]
    mem = rt.get("memory_gb", {})
    on_gpu = getattr(eng, "uses_gpu", lambda *_: False)(device, params)
    need = float(mem.get(f"{eng.family}_gpu", mem.get(eng.family, 0.0)) if on_gpu else mem.get(eng.family, 0.0))
    have = resources.headroom_gb()
    if not resources.enough(need, have):
        raise TrxError("MEMORY_LOW", f"{have:.1f} GB of memory free (RAM + page file); {eng.id} needs about {need:g} GB")


def get(engine_id: str, device: str, params: dict):
    """(engine, load info). The info says reused=True when the engine was already loaded in this run."""
    from .faster_whisper_engine import make
    if engine_id in _held:
        eng, info = _held[engine_id]
        if hasattr(eng, "configure"):
            eng.configure(params)
        return eng, dict(info, reused=True)
    release()
    eng = make(engine_id)
    _check_memory(eng, device, params)
    info = eng.load(device, params)
    if getattr(eng, "keep_loaded", False):
        _held[engine_id] = (eng, info)
    return eng, info


def done(engine_id: str, eng) -> None:
    """The caller has finished with the engine: unload it unless the run keeps it for a later stage."""
    if engine_id not in _held:
        eng.unload()


def held() -> list[str]:
    return list(_held)


def release() -> None:
    for eng, _ in list(_held.values()):
        eng.unload()
    _held.clear()
