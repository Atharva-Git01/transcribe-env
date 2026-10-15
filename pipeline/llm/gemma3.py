"""Local Gemma 3 via llama.cpp `llama-server` on 127.0.0.1 (the offline guard allows only localhost).

No background service: the server is started for one run, used, and stopped. No auto-update, no downloads.
"""
from __future__ import annotations

import json
import socket
import subprocess
import time
import urllib.request

from ..common import TRX_HOME, TrxError
from ..models import model_dir

BIN_DIR = TRX_HOME / "toolchain" / "bin" / "llama.cpp"


def server_exe():
    for name in ("llama-server.exe", "llama-server"):
        p = BIN_DIR / name
        if p.is_file():
            return p
    return None


def gguf_path(model_id: str):
    d = model_dir(model_id)
    files = sorted(d.glob("*.gguf"))
    return files[0] if files else None


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


class LocalLLM:
    def __init__(self, model_id: str, gpu_layers: int = 99, ctx_size: int = 4096, seed: int = 1234):
        exe, gguf = server_exe(), gguf_path(model_id)
        if not exe or not gguf:
            raise TrxError("MODEL_MISSING", f"llama.cpp server or {model_id} GGUF not installed")
        self.port = _free_port()
        self.seed = seed
        log = open(TRX_HOME / "toolchain" / "llama_server_last.log", "w", encoding="utf-8")
        self.proc = subprocess.Popen(
            [str(exe), "-m", str(gguf), "--host", "127.0.0.1", "--port", str(self.port), "-ngl", str(gpu_layers),
             "-c", str(ctx_size), "--seed", str(seed), "--no-webui"],
            stdout=log, stderr=subprocess.STDOUT, cwd=str(BIN_DIR))
        deadline = time.time() + 180
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise TrxError("INTERNAL_ERROR", "llama-server exited during start (see toolchain/llama_server_last.log)")
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/health", timeout=2) as r:
                    if r.status == 200:
                        return
            except Exception:  # noqa: BLE001
                time.sleep(1)
        self.close()
        raise TrxError("INTERNAL_ERROR", "llama-server did not become ready in 180 s")

    def chat(self, system: str, messages: list[tuple[str, str]], max_tokens: int) -> str:
        msgs = [{"role": "system", "content": system}] + [{"role": r, "content": c} for r, c in messages]
        body = json.dumps({"messages": msgs, "temperature": 0.0, "top_k": 1, "seed": self.seed,
                           "max_tokens": max_tokens, "stream": False}).encode("utf-8")
        req = urllib.request.Request(f"http://127.0.0.1:{self.port}/v1/chat/completions", data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=300) as r:
            data = json.loads(r.read().decode("utf-8"))
        return data["choices"][0]["message"]["content"].strip()

    def close(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.proc.kill()
