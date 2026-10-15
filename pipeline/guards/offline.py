"""Offline guard (rule R2).

Installed at the start of every runtime command. Sets the offline flags model libraries respect and patches the
socket layer so ANY attempt to reach a non-local host raises NetworkBlocked before DNS or a packet is sent.

Subprocesses (ffmpeg, ffprobe, llama.cpp, engines) are not covered by this Python patch, so the runtime only ever
passes them local file paths; inputs that look like URLs are rejected in the probe stage.
"""
from __future__ import annotations

import os
import socket

OFFLINE_ENV = {
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "DO_NOT_TRACK": "1",
    "OLLAMA_NO_UPDATE": "1",
}
_LOCAL = {"localhost", "127.0.0.1", "::1", "0.0.0.0"}
_installed = False


class NetworkBlocked(ConnectionRefusedError):
    pass


def _is_local(host) -> bool:
    if host is None:
        return True
    if isinstance(host, bytes):
        host = host.decode("ascii", "ignore")
    host = str(host).strip("[]").lower()
    return host in _LOCAL or host.startswith("127.")


def _host_of(address):
    if isinstance(address, tuple) and address:
        return address[0]
    return None  # AF_UNIX path or unknown -> treated as local


def install() -> None:
    global _installed
    for k, v in OFFLINE_ENV.items():
        os.environ[k] = v
    if _installed:
        return

    orig_connect = socket.socket.connect
    orig_connect_ex = socket.socket.connect_ex
    orig_getaddrinfo = socket.getaddrinfo
    orig_create_connection = socket.create_connection

    def _check(host):
        if not _is_local(host):
            raise NetworkBlocked(f"trx offline guard blocked a network call to {host!r}")

    def connect(self, address):
        _check(_host_of(address))
        return orig_connect(self, address)

    def connect_ex(self, address):
        _check(_host_of(address))
        return orig_connect_ex(self, address)

    def getaddrinfo(host, *args, **kwargs):
        _check(host)
        return orig_getaddrinfo(host, *args, **kwargs)

    def create_connection(address, *args, **kwargs):
        _check(_host_of(address))
        return orig_create_connection(address, *args, **kwargs)

    socket.socket.connect = connect
    socket.socket.connect_ex = connect_ex
    socket.getaddrinfo = getaddrinfo
    socket.create_connection = create_connection
    _installed = True


def self_test() -> tuple[bool, str]:
    """Prove the guard works: a call to a public host must be refused before any DNS lookup."""
    install()
    missing = [k for k, v in OFFLINE_ENV.items() if os.environ.get(k) != v]
    if missing:
        return False, f"offline env flags not set: {missing}"
    try:
        socket.create_connection(("example.com", 443), timeout=1)
    except NetworkBlocked:
        return True, "public host blocked before DNS; offline flags set"
    except Exception as e:  # noqa: BLE001
        return False, f"guard did not intercept (got {type(e).__name__})"
    return False, "a public connection was allowed"
