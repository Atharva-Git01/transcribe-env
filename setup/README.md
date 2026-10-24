# setup/ — the only part that may use the internet

Nothing here runs automatically, and nothing here runs during a transcription. Each phase is installed once, after
the owner approves it (name, source and size first). Run with the system Python 3.12; it creates the venv.

```
python setup/setup.py plan    [--phase N]          what would be installed: sizes, licences, status, disk
python setup/setup.py install --phase N --approve  install one phase (venv from the locks, models, binaries)
python setup/setup.py verify                       sha256 of every installed model/binary vs sources.toml
python setup/setup.py write-sources                record SOURCE.json next to every installed component
```

- `sources.toml`: every downloadable item (URL or Hugging Face repo at a pinned revision, sha256, destination).
  Gated Hugging Face models (IndicConformer, Indic-Transcribe) use the login the owner stored with
  `toolchain\python\.venv\Scripts\hf.exe auth login`. Claude never types or reads tokens.
- `requirements.lock`, `requirements-nodeps.lock`, `requirements-ml.lock` (CUDA 12.6 PyTorch, accelerate,
  transformers, sentencepiece, Demucs, …): pip `--require-hashes`. A PC without an NVIDIA GPU uses
  `requirements-ml-cpu.lock` instead (CPU build of PyTorch).
- `build_lm.py`: Marathi/Hindi word language models from IndicCorp v2 (CC-0).
- Fixture builders (the fixtures stay on this PC; `evals/fixtures/*/` is git-ignored):
  - `make_fleurs_fixture.py`: 10 FLEURS test sentences (CC-BY-4.0) joined into one file
  - `make_mix_fixture.py`: an existing fixture + a music file at a chosen speech-to-music ratio
  - `make_ami_fixture.py`: a 3-minute window of an AMI meeting (CC-BY-4.0) with speaker turns
  - `make_fixture.py`: one of the owner's recordings + a `speaker: text` transcript
- Training data (never a fixture or a test set; `toolchain/data/` is git-ignored, on this PC it is on the USB):
  - `make_indicvoices_slice.py`: 2 hours of IndicVoices Marathi conversation and extempore clips (CC-BY-4.0,
    gated like the models above) with their as-spoken text; reads only the parquet row groups it uses

After setup the runtime is offline again: `trx doctor --offline` must pass.
