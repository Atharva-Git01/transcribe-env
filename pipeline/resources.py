"""Free memory before a speech model is loaded on the CPU (E31). Windows: the commit headroom (free RAM + free page
file, GlobalMemoryStatusEx). A system-managed page file can grow, but on 2026-10-23 it did not, with 5.5 GB free on C:,
so growth is not counted. Linux: MemAvailable + SwapFree. None when it cannot be measured.
"""
from __future__ import annotations

import ctypes
import sys
from pathlib import Path


def _windows_commit_free_gb() -> float | None:
    class MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
    st = MEMORYSTATUSEX()
    st.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):
        return None
    return st.ullAvailPageFile / 1e9


def headroom_gb() -> float | None:
    if sys.platform == "win32":
        commit = _windows_commit_free_gb()
        return None if commit is None else round(commit, 1)
    try:
        info = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines() if ":" in line)
        return round((float(info["MemAvailable"].split()[0]) + float(info["SwapFree"].split()[0])) / 1e6, 1)
    except (OSError, KeyError, ValueError):
        return None


def enough(need_gb: float, have_gb: float | None) -> bool:
    """Unknown headroom never blocks a run; a measured one must cover the model's need."""
    return have_gb is None or need_gb <= 0 or have_gb >= need_gb
