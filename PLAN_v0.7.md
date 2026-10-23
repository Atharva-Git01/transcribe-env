# Plan v0.7: fixing the Pappa-call errors and testing Indic engines (2026-10-22)

> **Status 2026-10-23: implemented in 0.7.0.** Results, decisions and what was rejected: `knowledge/CHANGELOG.md`
> 0.7.0 and `knowledge/ERRORS.md` E24–E29. Still open: the owner's recordings (phase 10), short replies merged by
> diarization, speed (CUDA PyTorch, QUESTIONS.md 20), deleting the fallback engine (QUESTIONS.md 21).

Written after comparing the 0.6.0 output of the Pappa call with the owner's ground truth (26 turns).
This plan assumes the owner provides what each step needs; section 6 lists those things.

---

## 1. What is wrong today, and why

Pappa call, 0.6.0: CER 0.513, WER 0.805. Speaker accuracy is 91.7 %: 21 of the 22 heard turns have the right speaker.

| # | Problem (evidence) | Root cause (measured) |
|---|---|---|
| P1 | **Wrong words in fast, colloquial Marathi.** About half the characters differ. Place words and verbs are misheard as sound-alike nonsense. | Whisper is weak at Marathi. Even clean read Marathi (FLEURS) gets CER 0.27. On this phone call it is worse. |
| P2 | **Hindi drift.** Hindi forms (है, में, करें, अच्छे, हाँ) appear where Marathi was spoken. | Whisper's training is Hindi-heavy (E10). The Gemma word gate fixes only some of these. |
| P3 | **Short replies lost.** Turns 9, 18 and 22 ("yes, yes"-type backchannels) are missing. | Not a VAD miss: every detected speech region produced text. Whisper skips short replies that sit inside a long multi-turn window. |
| P4 | **One full sentence lost (turn 24).** | The 51.7–52.7 s stretch has speech-level energy (−28 to −36 dBFS) but lies outside the VAD regions. It was quiet speech that VAD rejected. |
| P5 | **Last word ("bye") given to the wrong speaker.** | Diarization was right: the turn at 54.5–55.3 s is SPK1. Whisper's word timestamps were about 0.4 s early, so the word landed in the previous turn. |
| P6 | **CER penalises correct hearing.** | The truth uses colloquial spellings and digits (e.g. "२"). The system writes standard spellings and number words ("दोन"). Today's CER counts these as errors. |
| P7 | **Weak language decision.** On this call the marker evidence leaned Hindi (8 vs 5). | Whisper's language ID cannot see Marathi (E10). This is now covered by the owner confirming the language, but the guess itself is still poor. |
| P8 | **Too few test files.** There are 3 fixtures, and only one is a real phone call. None has 3+ speakers. | Every tuning decision so far rests on one 56-second call. Speaker-count estimation is unvalidated (E19). |
| P9 | **Slow.** About 5 minutes per minute of audio (large-v3 + diarization + Gemma). | Large-v3 loads and runs at roughly real time on the RTX 3050. Gemma adds 1–2 minutes. |
| P10 | **Open items from QUESTIONS.md.** | #3 IndicConformer (needs the HF login), #5 Demucs (after #3), #10 the owner's recordings. |

---

## 2. The two Indic engines, from their published facts

IndicConformer's own model card now recommends a newer model, **Indic-Transcribe-core**. Both will be tested.

| | **IndicConformer-600M** (`ai4bharat/indic-conformer-600m-multilingual`) | **Indic-Transcribe-core** (`bodhan-ai/indic-transcribe-core`) |
|---|---|---|
| Who | AI4Bharat | Bodhan AI with AI4Bharat (2026) |
| Architecture | Conformer, hybrid CTC + RNNT, 600M parameters | Canary: FastConformer encoder (811M) + Transformer decoder (419M), 1.2B parameters, fine-tuned from nvidia/canary-1b-v2 |
| Languages | 22 Indic languages (mr, hi …). **No English.** | 25 languages: 22 Indic + **English** + Bhojpuri + Bhili. Code-mixed speech is handled natively. |
| Output | Native script. Probably no punctuation (to verify). | Native script. English words inside Indic speech are **written in Devanagari** (as in the owner's transcript). Numbers stay as words. |
| Benchmark ("Voice of India", OIWER: word error rate counting accepted spelling variants as correct) | Marathi **11.6**, Hindi 6.5 | Marathi **5.7**, Hindi 3.5. In their table both beat Gemini 3 Pro (14.0 / 9.3). |
| Size | 2.56 GB (ONNX files + TorchScript preprocessor) | 4.9 GB `model.safetensors`. The duplicate `.nemo` copy (4.9 GB) is not needed. |
| Runtime | onnxruntime (already installed) plus **PyTorch** for `preprocessor.ts` | **PyTorch** + transformers + sentencepiece. The model code ships in the repo. |
| Timestamps | CTC frames give word timing (we compute it) | Text only. We decode per speaker turn, so turn times are enough. |
| Long audio | We cut at VAD/speaker turns | At most 45 s per pass. A bundled chunker cuts at pauses into pieces of 30 s or less, with a loop guard. |
| Built-in LID | No (language given) | Yes, but "Hindi-belt languages are often absorbed by Hindi". The language is passed whenever known. |
| Access | Gated "auto" (accept terms with an HF account) | Gated with a form: company, country, intended use, licence checkbox |
| Licence | **MIT** | **Indic Open Model License v1.0** (base model nvidia/canary-1b-v2 has its own terms) |

**Indic-Transcribe licence: what matters for FMS.**
- Free, including commercial use, when self-hosted (this environment qualifies).
- **Hosting it as an API or service that other people or companies call needs Bodhan AI's written approval first.** This matters if FMS becomes a SaaS.
- Credit is required ("Built with Indic-Transcribe-core from Bodhan AI / AI4Bharat").
- Modified versions you give to others must carry the same licence.
- Banned uses include mass surveillance and "automated decisions that affect someone's legal rights without human oversight". The environment's REVIEW/human-check rule must stay in place for forensic use.
- A separate commercial licence is needed above 500M monthly users or $250M revenue.

**What the engines should fix (to be proven on our audio):**
- **P1, P2:** trained on Indian conversational speech with language-specific output, so expected to fix the most.
- **P5:** decoding each speaker turn on its own ties text to the right turn.
- **P3:** decoding per turn includes short replies, if they are passed in as their own pieces.

**What they will not fix on their own:** P4 (VAD), P6 (scoring and conventions), P8 (fixtures), and the speaker-count estimate.

---

## 3. Fix plan, problem by problem

### P1 + P2: a better engine for Marathi and Hindi
- Add both engines as `pipeline/engines/indic_conformer_engine.py` and `indic_transcribe_engine.py`, behind the same interface as faster-whisper (`engines/base.py`).
- Test them head to head (section 4). The winner becomes `[engines.primary].mr/hi/mixed`. Whisper stays as the fallback and as the engine for English.
- Keep the lang_forms map, the Gemma word gate and the LM check. Re-measure them with the new engine and keep them only if eval still improves.

### P3 + P5: decode per speaker turn
- New segmentation mode `turns`: one decode per diarization turn, after merging adjacent same-speaker turns up to 30 s and splitting longer ones at pauses.
- A turn's text belongs to that turn, so there are no timestamp mix-ups.
- **Turn-level omission check:** a turn with speech (≥ 0.3 s) but no text is decoded again on its own. If it is still empty, it is flagged `possible_omission` (today only gaps of 1.5 s or more are checked).

### P4: quiet speech outside VAD
- **Gap audit:** measure the energy of every gap between VAD regions. A gap with speech-level energy is re-checked with a lower VAD threshold. If speech is found, it becomes its own turn and is decoded. If not, it is flagged `possible_missed_speech` with its times.
- Tune `vad.threshold` and `speech_pad_ms` of the phone recipe on the fixtures.
- Accept only if the number of missing truth turns drops and no fixture's CER gets worse.

### P6: fair scoring and an explicit transcript convention
- `trx eval` reports **strict CER** (as today) and **normalised CER**. Normalised CER is OIWER-like and uses documented rules:
  - nukta removed
  - chandrabindu treated as anusvara
  - ॉ/ॅ vowel signs unified
  - digits treated as number words (२ = दोन)
  - punctuation removed
- It also accepts `variants` per fixture: alternative spellings in truth.json, as multi-reference scoring.
- The owner chooses the **output convention** (section 6): spelled as spoken (colloquial) or standard; numbers as digits or words; English words in Devanagari or Latin. The convention goes into `config/defaults.toml [output]` and a post-processing step applies it.

### P7: language guess
- If Indic-Transcribe wins, use its `identify()` for the guess shown to the operator. Test it on all fixtures against the current Whisper 2-step LID; the owner still confirms.

### P8: fixtures
- New `setup/make_fixture.py` turns an owner recording plus a text file with `speaker: text` lines into a fixture, kept local only (git-ignored).
- With a 3+ speaker recording, tune diarization `split_cosine` / `min_speaker_s` and measure `spk`.

### P9: speed
- Measure the real-time factor of each engine on CPU, and on the GPU where it fits.
- Indic-Transcribe needs about 2.4 GB of VRAM in bf16 and might fit on the 4 GB RTX 3050. Otherwise it runs on the CPU (needs about 5 GB of RAM).
- If Gemma no longer improves results with the new engine, turning it off saves 1–2 minutes per file.

### P10: QUESTIONS.md items
- **#3** IndicConformer: covered above (needs the owner's HF login). Indic-Transcribe is added on the same login.
- **#5** Demucs:
  - `demucs` 4.1.0 plus the `htdemucs` weights (~80 MB, MIT), sharing the CPU PyTorch.
  - `separate` stage in `noisy_field_audio`.
  - Test it on an owner recording with music or TV in the background. Accept only if that recording improves and no clean fixture gets worse.
- **#10** owner recordings: section 6.

---

## 4. Test protocol: IndicConformer vs Indic-Transcribe vs large-v3

- **Same pipeline for all:** convert → denoise → VAD → diarization → decode per turn → polish → gate. Seeded, deterministic.
- **Variants:**
  - large-v3 (today)
  - IndicConformer CTC
  - IndicConformer RNNT
  - Indic-Transcribe with `lang` given
  - Indic-Transcribe with auto LID (LID accuracy only)
  - each with and without Gemma cleanup
- **Fixtures:** Pappa, FLEURS mr, FLEURS hi, plus every owner recording available by then.
- **Measured per run:**
  - strict CER, normalised CER, WER
  - truth turns missing
  - speaker accuracy
  - number of Hindi forms in Marathi output
  - real-time factor, peak RAM/VRAM
  - verdict
- **Report:** `evals/results/compare_v0.7.md`, with a per-turn comparison table for each fixture like the one shown in chat.
- **Decision rule:** an engine becomes primary for a language if it is at least **0.05 better in normalised CER** on the phone call(s), worsens no fixture by more than 0.01, and runs at an acceptable speed (at most 2× slower than real time on this PC, unless the owner accepts slower).

---

## 5. Implementation plan (phases)

| Phase | Work | Downloads | Done when |
|---|---|---|---|
| **0. Owner prerequisites** | Section 6 checklist | – | HF login done, terms accepted, conventions chosen |
| **1. ML toolchain** | Add `torch` 2.14 CPU (124 MB wheel) + `transformers` + `sentencepiece` + `soundfile` to a new hash-locked `setup/requirements-ml.lock`; check versions work together; doctor checks; never `trust_remote_code` at run time | ~150 MB | `trx doctor` READY, `trx test` green |
| **2. Fetch and review models** | Download both repos at a **pinned commit**; sha256 every file into `setup/sources.toml`; **read the shipped Python code before running any of it**; copy only the minimal inference code into `pipeline/engines/vendor/` with its licence notice; load from local files only (offline guard on) | 2.6 GB + 4.9 GB | Both load offline from `toolchain/models/`; doctor `--deep` passes |
| **3. Engine adapters** | `indic_conformer_engine.py`: ONNX sessions, TorchScript preprocessor, CTC + RNNT, word timing from CTC frames. `indic_transcribe_engine.py`: pieces of 30 s or less, loop guard, bf16 on GPU if it fits, else CPU. Both seeded and offline | – | Unit tests; one fixture transcribes end to end with each |
| **4. Per-turn decoding** | `segmentation = "turns"` in recognise; short-turn re-decode; turn-level `possible_omission`; tests | – | Pappa: missing turns 4 → ≤ 1 |
| **5. Gap audit / VAD tuning** | Energy check of VAD gaps, low-threshold re-check, `possible_missed_speech` flag; tune phone VAD on fixtures | – | Turn 24 recovered or flagged; no CER regression |
| **6. Eval upgrades** | Normalised CER (documented rules), fixture `variants`, per-turn comparison report, Hindi-form count, real-time factor | – | `trx eval` prints strict + normalised; report generated |
| **7. Head-to-head + decision** | Run the section 4 protocol; set engine tables by the decision rule; enable verify (Indic vs Whisper cross-check) if it adds signal | – | New baseline saved; CHANGELOG has the numbers |
| **8. Retune post-processing** | Gemma cleanup, word gate, LM check, `unknown_words` with the new engine; apply the owner's output convention (spelling, numbers, script) | – | Each kept step proven by eval |
| **9. Demucs** | Install `demucs` + `htdemucs` weights; `separate` stage; test on the owner's noisy recording | ~90 MB | Noisy fixture better, clean fixtures unchanged |
| **10. Owner fixtures + diarization validation** | `setup/make_fixture.py`; add all owner recordings; tune speaker-count estimation on the 3+ speaker one | – | `spk` measured on 3+ speakers; baseline re-saved |
| **11. Finish** | Remove the losing engine (frees 2.6–4.9 GB) if the owner agrees; manifest, sources, ERRORS, CHANGELOG 0.7.0, OPERATOR/README/MAINTAINER, licence notices (attribution line for Indic-Transcribe); tests; commit + push | – | Doctor READY, tests green, pushed |

Disk: about 7.7 GB of downloads. C: has 20 GB free, about 12 GB after downloads and 15–17 GB after removing the loser.
Order: 1 → 2 → 3 → 4 → 6 → 7 (decision) → 5 → 8 → 9 → 10 → 11. Phases 4–6 can start while the downloads run.

---

## 6. What the owner does before Phase 1

1. **Hugging Face access (you, in your browser):**
   - On the IndicConformer model page, accept the terms. Approval is automatic.
   - On the Indic-Transcribe-core page, fill in company, country and intended use, and accept the Indic Open Model License after reading it. Note the "no hosting for others without permission" condition.
2. **Login (you, in a terminal).** Create a *read* token at huggingface.co/settings/tokens and run `toolchain\python\.venv\Scripts\hf.exe auth login`. Claude must not type tokens or passwords for you, so do not paste the token into chat.
3. **Approve the downloads:**
   - PyTorch CPU + transformers + sentencepiece + soundfile: ~150 MB
   - IndicConformer: 2.6 GB
   - Indic-Transcribe-core: 4.9 GB
   - Demucs: ~90 MB
4. **Choose the output convention:**
   - spelled as spoken (colloquial) or standard spelling
   - numbers as digits or words
   - English words in Devanagari or Latin script
5. **Recordings with your transcripts (best 5–10, 1–3 min each).** Format: audio file + a .txt with one `speaker: text` line per turn. Ideally:
   - more Marathi phone calls
   - one Hindi call
   - one Hinglish/Marglish call
   - one with TV/music behind
   - one with 3+ speakers
6. **Later, after the speed test:** whether to install the CUDA build of PyTorch (~2.5 GB) if CPU speed is not acceptable.
