# Plan v0.9: transcripts as spoken, every candidate model tested on the GPU (2026-10-24)

> **Status: version 2, waiting for the owner's confirmation.** The owner's answers of 2026-10-24 are included;
> nothing is built yet. Every adopted change goes through `trx eval` (all fixtures, failure monitor).

**Changes since version 1 (2026-10-23):**
- **Testing.** Every candidate model is tested as it is, and every one that can learn from a *small* amount of audio
  is trained on that small amount. Tests use a short screening set on the GPU, so each one takes minutes.
- **The owner's answers:**
  - training data: own recordings plus a small IndicVoices slice;
  - downloads: approved once, as the list in section 8;
  - sounds that are the same: one standard spelling;
  - copies: as-spoken and standard both kept;
  - more as-spoken transcripts will follow over time.
- **New ideas:**
  - *spelling transfer:* a verbatim model decides only the spelling, Indic-Transcribe keeps the words;
  - fixed challenge sets;
  - decoder-only training on a frozen, pre-computed encoder (minutes per run);
  - a blind listening check by the owner;
  - every model on the GPU, including IndicConformer and the small helpers.

---

## 1. The goal: "as spoken"

| The speaker says | Today's transcript | As spoken |
|---|---|---|
| nay | नाही (nahi) | नाय |
| paishe | paise | paishe |
| mahitiye | mahiti aahe | mahitiye |
| kutha | kuthe | kutha |
| an | aani | an |

- **Follows the speaker:** every difference you can hear — a dropped h, contractions with "aahe", s or sh,
  dental or retroflex n, changed vowels, short forms, fillers as heard.
- **Same sound, one spelling (owner, 2026-10-24):** short or long i and u, and the final anusvara, always get one
  standard spelling.
- **Two copies (owner, 2026-10-24):** every run writes the as-spoken transcript (the main one) and a standard-spelling
  copy (the model's own spelling) for searching and quoting.
- **Unchanged:** numbers as spoken words (the engine's "native" mode already writes them as said), English words
  in Devanagari, speakers, timings, review flags.
- **Forensic rule:** never write a form the speaker did not say.
  - Every changed spelling is logged in `decisions.json`, with the standard form kept in `transcript.json`.
  - A change of *words* is never made silently.

---

## 2. What is known

| Question | Result |
|---|---|
| Can Indic-Transcribe write spoken forms? | **No.** Scored against the audio by its own decoder, its standard spelling won 54 of 54 times. |
| Can IndicConformer? | **Hardly:** 2 of 47 (CTC), 0 (RNNT). It is also less accurate on the call: CER 0.265 vs 0.232. |
| How do other systems do it? | Each writes like its training transcripts. Systems that write "as spoken" were trained on verbatim transcripts, or are prompted with examples: Sarvam's verbatim mode (cloud), OpenAI's prompt for gpt-4o-transcribe (cloud), CrisperWhisper (English and German). |
| Is there verbatim Marathi to learn from? | **IndicVoices** (AI4Bharat, CC-BY-4.0) transcribes at two levels. **Level 1 is "exactly as spoken"** (their example: "muje" for "mujhe"); Level 2 is textbook spelling. Our models probably learned Level 2. **Checked 2026-10-24: the released files hold both**: column `verbatim` is Level 1, `normalized` is Level 2 (e.g. *nay* vs *nahi*, *chaluye* vs *chalu aahe*). |
| What fails today (E39, E40) | Short replies invent words (28 % wrong vs 18 %); 9 acknowledgements lost; ~10 of 60 negations flipped; names in several spellings; single wrong words in long lines unflagged. |
| IndicConformer as a second opinion | Worse on every kind of line, but better on 8 of 20 uncertain lines. Its negation disagreements: 8 lines, 5 of them real. |

---

## 3. The problems

| # | Problem | Root cause |
|---|---|---|
| P1 | Standard spelling instead of what was said | Both local models learned standard written Marathi. |
| P2 | Short replies with invented words | A turn under 2 s is decoded alone, without context. |
| P3 | Replies lost or under the other speaker | A reply is shorter than the 1 s voice window; regions under 0.5 s get no voice sample. |
| P4 | Meaning flips (negations, amounts) | One misheard word costs few characters; nothing compares the meaning. |
| P5 | Names in several spellings | Names are outside the vocabulary, and each turn is decoded alone. |
| P6 | A confident wrong word inside a long line | The model is confident in wrong words too. |
| P7 | Little as-spoken truth (13 minutes) | More comes over time (owner); a small IndicVoices slice adds training data. |
| P8 | Extra passes cost time | Each check is a model pass. |

---

## 4. Candidate models — all run on the GPU (RTX 3050, 4 GB, one model at a time)

### 4.1 Tested as they are

| Candidate | What it is | Role if it wins | GPU memory | Licence | Status |
|---|---|---|---|---|---|
| Indic-Transcribe-core | 1.2B Canary (the current engine) | baseline | ~2.5 GB bf16 | Indic Open Model | installed |
| IndicConformer-600M | CTC + RNNT, 22 Indic languages | second reading, meaning check | ~1.5 GB | MIT | installed (USB). Today it runs on the CPU, so it gets moved to the GPU with `onnxruntime-gpu` |
| Whisper large-v3 + an as-spoken prompt | 1.55B; prompt with spoken spellings | spellings via prompt (its Marathi is weak) | ~3 GB int8/fp16 | MIT | installed (USB) |
| **IndicWhisper Marathi** (AI4Bharat Vistaar) | Whisper fine-tuned on 10.7k hours of Indian speech | transcriber or spelling source | ~1.5–3 GB fp16 (size of the base model checked on download) | MIT | download |
| **Omnilingual ASR CTC 300M / 1B** (Meta, 2025) | CTC, 1,600+ languages | transcriber or spelling source | ~2 / ~3 GB | Apache-2.0 | download (only if `fairseq2` installs on Windows) |
| **MMS-1b-all + Marathi adapter** (Meta) | CTC 1B with a 9 MB language adapter | base for small-data training | ~2 GB fp16 | **CC-BY-NC-4.0: tests only, not for paid work** | download |
| **wav2vec2-xlsr-53-espeak** (Meta) | phoneme recognizer (hears sounds) | acoustic judge of spellings | ~0.7 GB fp16 | Apache-2.0 | download (approved; restarts the interrupted one) |
| Allosaurus / ZIPA | phoneme recognizers | second judge | ? | ? | asked separately (size and licence not confirmed; ZIPA weights not found yet) |

**Left out:**
- too big for 4 GB: Omnilingual LLM 1B and larger (≥ 6 GB), its few-shot 7B model (~20 GB), Shuka 8B;
- cloud only: Sarvam Saaras, gpt-4o-transcribe.

A model that does not fit in fp16/bf16 is tried 8-bit or left out. It is **never moved to the CPU silently**.

### 4.2 Trained on a small amount of audio

| Candidate | How it is trained | Why it is quick |
|---|---|---|
| **Indic-Transcribe, decoder LoRA** | small adapters on the decoder only; the encoder stays frozen | The encoder's output for every training clip is computed **once** and kept, so each pass runs only the decoder: minutes per run, small GPU memory. The spelling habit lives in the decoder. |
| **MMS adapter (Marathi)** | train only the 2.5M-parameter language adapter | The method made for very small data; a run takes 10–20 minutes. |
| **Omnilingual CTC 300M** | short fine-tune (`fairseq2` recipe) | 300M parameters fits 4 GB; under an hour (only if it installs). |
| Phoneme judge | no training | — |
| Whisper | no training: 3–5 example sentences written as spoken in its prompt | — |

**Training data:**
- the owner's as-spoken transcripts (13 minutes now, more over time);
- a 1–2 hour slice of IndicVoices Marathi Level 1, used for training only, never for testing;
- phone-band simulation (8 kHz, compression) on the IndicVoices clips, so they sound like our calls (checked
  2026-10-24: the Conversation clips already are phone band; only the wide-band Extempore clips need it).

---

## 5. How models are checked quickly

**Screening set (~6 minutes of audio, decoded in 1–3 minutes per model on the GPU).** Lines from the call's 12
scored minutes and the Pappa call, later the owner's new transcripts. It holds four fixed **challenge sets**:
- **S — spelling:** the 54 words the owner wrote as spoken, and 58 written standard.
  - Measures: spoken-form recall, wrong switches.
- **R — short replies:** the 18 short lines that read as invented, and all lines under 2 s.
  - Measures: short-line content CER, invented words.
- **N — meaning:** the lines with negations and numbers (about 10 known flips).
  - Measures: negation and number errors.
- **G — general:** the rest.
  - Measures: content CER (the words) and strict CER (the spelling).

**Every run also reports:** speed (real-time factor), peak GPU memory, load time.

**Three stages:**
1. **Zero-shot.** Every candidate in 4.1 runs once on the screening set; a shortlist follows.
2. **Small-data training** (4.2), with a **fold by recording**:
   - Train on some of the owner's recordings, test on the others, never on the same recording, and report the
     spread across folds.
   - The IndicVoices slice is only ever training data.
3. **Finalists:**
   - a full `trx eval` on all fixtures;
   - a **blind listening check by the owner**: 20 lines, two versions each, unlabelled — which one is what you hear?
     Transcripts have habits and gaps; the ear decides "as spoken".

**Adopted only if:**
- no fixture's content CER gets worse by more than 0.01;
- spoken-form recall improves;
- wrong switches stay ≤ 5 %;
- the owner's listening check prefers it.

**Time budget:**
- screening: ≤ 15 minutes per model;
- a training run: ≤ 1 hour;
- anything longer (overnight): only after asking.

---

## 6. How the winner becomes the transcript
- **Spelling transfer (preferred, low risk).**
  - Indic-Transcribe keeps the words. The verbatim model (trained in 4.2), or the phoneme judge, decides the
    spelling.
  - Both outputs are aligned. Where a word is the **same word** in both (same content once spelling is folded away)
    but spelled differently, the verbatim spelling is used. Only spellings change.
  - Where the two disagree on the word itself, Indic-Transcribe's word stays. A disagreement on a negation or a
    number is flagged (B3).
- **Engine swap (if clearly better):** if a fine-tuned Indic-Transcribe writes spoken forms with no loss of words,
  it simply becomes the engine.
- **Phoneme judge:** the tie-breaker for spellings, and the checker for the owner's hotword names (B4).

---

## 7. Fixing the failures of the call (unchanged in substance, now GPU first)

**B1 — short replies.**
- Score a closed list of common replies with the phoneme judge.
- Otherwise keep the line, and show the best second reading from the screening winners in the review list.

**B2 — lost replies.** A second diarization pass:
- re-scan long turns with 0.6 s windows every 0.1 s;
- give regions under 0.5 s a padded voice sample of their own;
- phone recipe first; a change that makes AMI worse stays out of the meeting recipe.

**B3 — meaning check.** Where two engines differ on a line's negations or numbers, the soft flag `meaning_check`
shows the other reading.

**B4 — names.**
- "written N ways" in `review_list.md`;
- with `--hotwords`, the phoneme judge picks the listed spelling when the sounds match.

**B5 — sound check of words in long lines.** Kept only if at least half of what it marks is really wrong.

**B6 — GPU for the helpers.**
- The models that still run on the CPU each get a GPU path:
  - IndicConformer: `onnxruntime-gpu`;
  - the Silero VAD: torch on CUDA;
  - the speaker embeddings: `onnxruntime-gpu`, or a torch export.
- A helper is moved only if the result is identical (or measured equal) and faster.

---

## 8. Downloads approved by the owner (2026-10-24, as a list)

| # | What | Source | Size | Licence | Where |
|---|---|---|---|---|---|
| 1 | `facebook/wav2vec2-xlsr-53-espeak-cv-ft`, revision 2c733782 (the interrupted download) | Hugging Face (Meta) | 1.26 GB | Apache-2.0 | C: (`toolchain/models/phoneme-xlsr53-espeak`) |
| 2 | IndicWhisper Marathi (`marathi_models.zip`) | AI4Bharat Vistaar (their object store) | 1.41 GB | MIT | USB |
| 3 | `facebook/mms-1b-all` revision 3d33597e: `model.safetensors` + `adapter.mar.safetensors` | Hugging Face (Meta) | 3.87 GB | CC-BY-NC-4.0 (tests only) | USB |
| 4 | Omnilingual ASR CTC 300M + 1B, with the `omnilingual-asr` 0.2.0 and `fairseq2` 0.8.1 packages | Meta / PyPI | ~1.3 + ~3.9 GB (worked out from the parameter counts; checked before download, and asked again if more than 20 % bigger) | Apache-2.0 / MIT | USB (packages in the venv) |
| 5 | `onnxruntime-gpu` 1.31.0 | PyPI | 168 MB | MIT | venv: replaces the CPU `onnxruntime` package, which it contains; lock file updated |
| 6 | `peft` 0.21.2 (LoRA training) | PyPI | 0.8 MB | Apache-2.0 | venv |
| 7 | IndicVoices Marathi, a 1–2 hour slice with Level 1 ("as spoken") text | Hugging Face `ai4bharat/IndicVoices` (gated) | ≤ 10 GB (asked again if a slice needs more) | CC-BY-4.0 (credit AI4Bharat) | USB |

- **Fetched:** item 7 on 2026-10-24: `toolchain/data/indicvoices-mr-slice1` (USB), 1,469 clips, 2.0 h (1.3 h
  Conversation, 0.7 h Extempore), 263 speakers, 28 districts; 67 % of the clips hold at least one spoken form. It
  read 809 MB of the 37.5 GB Marathi part (about 580 h) and stores 107 MB.
- **Totals:** about 12 GB plus the IndicVoices slice. The USB has 42 GB free. Only items 1, 5 and 6 go to C:
  (12 GB free).
- **Pinned:** every file gets a pinned revision or sha256 in `setup/sources.toml` / the lock files, as before.
- **Asked separately:** anything not on this list, such as Allosaurus or ZIPA, WSL if `fairseq2` needs it, or
  `bitsandbytes`. Approved so far: `pyarrow` 25.0.1 (PyPI, 28 MB, Apache-2.0; reads the IndicVoices files),
  2026-10-24.
- **Item 7 needs the owner first:** open the dataset page, accept its terms, and log in with `hf auth login`
  yourself (Claude never types tokens).

---

## 9. Phases

| Phase | Version | What | Needs |
|---|---|---|---|
| 1 | 0.9.0 | the screening harness and challenge sets; the spoken-form metric and gate; as-spoken and standard copies (A1); spoken-variant list v1 (A2); "written N ways" names list (B4a) | nothing |
| 2 | 0.9.1 | downloads 1–6; stage 1, every candidate zero-shot on the GPU; IndicConformer on the GPU | the approved list |
| 3 | 0.9.2 | IndicVoices slice; stage 2, small-data training (Indic-Transcribe LoRA, MMS adapter, Omnilingual 300M) with folds | done 2026-10-24: terms accepted, slice fetched |
| 4 | 0.9.3 | stage 3: full eval of the finalists; owner's listening check; adopt (spelling transfer or engine swap) | owner: 20-line listening check |
| 5 | 0.9.4 | B-track: meaning check, second reading, second diarization pass, sound check, GPU for the helpers | — |
| ongoing | — | each new as-spoken transcript becomes a test file the day it arrives, and training data in later folds | owner |

---

## 10. Risks
- **The phoneme judge may hear phone-call Marathi badly** (zero-shot). The challenge sets show it in phase 2.
- ~~**IndicVoices' released files may not hold Level 1.**~~ Resolved 2026-10-24: they do (column `verbatim`).
- **`fairseq2` may not install on Windows.** Then the Omnilingual tests are skipped (WSL is asked separately).
- **MMS is non-commercial.** Tests only; it is never adopted for paid work.
- **Small data is noisy.** Folds, fixed challenge sets, and the listening check apply; nothing is adopted on one
  recording.
- **Training can damage accuracy.** The content-CER guard stops it.
- **Disk.** Everything optional lives on the USB; C: keeps ≥ 3 GB free.
- **Privacy.** Variant lists and challenge-set definitions are generic or number-only in tracked files. Names and
  phrases of a case never leave `evals/compare/`, the fixtures and the runs; `trx test` checks this.

---

## 11. Decisions

**Answered (2026-10-24):**
- training data: own recordings plus an IndicVoices slice;
- downloads: approved as the list in section 8;
- same-sound spellings: one standard spelling;
- copies: keep both;
- more as-spoken transcripts over time.

**Open:**
- **D3** — review the spoken-variant list for your region (when phase 1 has it ready);
- **D6** — IndicConformer stays on the USB (plugged in during runs) or moves to C: later (2.4 GB);
- the 20-line listening check in phase 4.
