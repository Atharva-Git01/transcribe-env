# toolchain/

Filled by `setup/` (after the owner's OK), never at runtime.

| Folder | Contents |
|---|---|
| `python/.venv` | the environment's own Python, installed from `requirements.lock` (phase 1) |
| `models/<id>/` | model weights + `SOURCE.json` (source URL, licence, card, sha256 of every file) |
| `bin/` | standalone binaries: deep-filter, llama.cpp, sherpa-onnx, whisper.cpp |

`trx doctor` reports what is present. Large folders are not tracked by git (`.gitignore`).
Models can live on another drive: replace `models/` with a directory junction (`mklink /J`) — paths stay the same.
