# AI4Bharat IndicConformer 600M multilingual — candidate primary for mr / hi (phase 2)

- Built for all 22 official Indian languages (AI4Bharat). ~600M parameters: fits the 4 GB GPU.
- Conformer with CTC / RNNT decoding: non-autoregressive alignment, so it does not produce Whisper-style
  hallucination loops on silence. Usually faster than large-v3.
- Licence: listed as **MIT** on AIKosh and model directories — must be confirmed on the Hugging Face model card
  before download (manifest.toml `commercial_ok = "verify"`).
- Runtime: its loading code is vendored and pinned in `pipeline/engines/`; no `trust_remote_code` download at runtime.
- Expected weaknesses to check in eval: English words in code-switched speech, punctuation, word timestamps.

Sources: github.com/AI4Bharat/IndicConformerASR · aikosh.indiaai.gov.in (IndicConformer 600M listing).

## Measured here
None yet (phase 2).
