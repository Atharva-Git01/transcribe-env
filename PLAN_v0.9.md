# Plan v0.9: transcripts as spoken, and the failures of the owner's 20-minute call (2026-10-23)

> **Status: draft for the owner.** Nothing in it is built yet. Section 8 lists the decisions it needs. Every step
> goes through `trx eval` (all fixtures, failure monitor) before it is adopted.

Written after the full failure check of the owner's 20-minute call (`knowledge/ERRORS.md` E39, E40) and the owner's
new rule (2026-10-23): **write every word the way the speaker pronounced it, in Devanagari.**

---

## 1. The goal: "as spoken"

| The speaker says | Today's transcript | As spoken |
|---|---|---|
| nay | नाही (nahi) | नाय |
| paishe | paise | paishe |
| mahitiye | mahiti aahe | mahitiye |
| kutha | kuthe | kutha |
| an | aani | an |

- **What counts:** differences you can hear — a dropped h (nahi / nay), contractions with "aahe" (mahiti aahe /
  mahitiye), s or sh (paise / paishe), dental or retroflex n, changed vowels (bhetu / bhitu), short forms
  (aani / an), fillers as heard (hm, haan, hmm, a-a).
- **What does not count:** spelling-only differences with the same sound — short or long i and u, the final anusvara.
  These get one consistent spelling (decision D1).
- **Unchanged:** numbers as spoken words (Indic-Transcribe already writes them as said: its "native" mode has no
  number conversion), English words in Devanagari, speakers, timings, review flags.
- **Forensic rule:** never write a form the speaker did not say. Every changed word is logged in `decisions.json`, and
  the model's standard spelling stays next to it in `transcript.json` (`text_standard`), so nothing is lost.

---

## 2. What was measured today

| Question | Result |
|---|---|
| Can Indic-Transcribe write the spoken form? Each of 54 words where the owner wrote a spoken form, scored against its own audio with the model's own decoder | **No.** It preferred its standard spelling in **54 of 54** (and in 44 of 44 spelling-only cases). It never switched a word either way. |
| Does IndicConformer (the second local model) write spoken forms? | **Hardly:** 2 of 47 with its CTC decoder, 0 with RNNT; the standard form 29–30 times. It is also less accurate on this call (CER 0.265 vs 0.232). |
| Conclusion | Both models were trained on standard-spelled transcripts. Spoken spelling has to come from a model that hears **sounds** (phonemes) rather than spellings, or from teaching a model the owner's spelling. A spelling rule cannot do it: "nahi" in the output does not say whether the speaker said nahi or nay. |
| IndicConformer as a second opinion | Worse on every kind of line (short lines 0.344 vs 0.243; lines flagged `uncertain` 0.523 vs 0.428), but better on 8 of 20 uncertain lines. Where the two disagree on a negation: 8 lines in 12 minutes, 5 of them real errors (of the 14 lines with a negation error). |
| Run time of IndicConformer on the CPU | 232 s for the 12 scored minutes (≈ 0.3 × real time) |

The failure check of the call (E39, E40) found, before today's fixes: speakers right for 98.7 % of the text; 9 of ~30
acknowledgements lost; short lines (< 2 s) 28 % wrong against 18 % for longer ones, often with invented words; ~10 of 60
negations lost or added; one surname in four spellings; the worst lines unmarked (now 10 of 18 marked `uncertain`).

---

## 3. The problems, and why they happen

| # | Problem | Root cause (measured) |
|---|---|---|
| P1 | **Standard spelling instead of what was said** (the new goal) | Both local models learned standard written Marathi; they cannot be steered to spoken forms (section 2). |
| P2 | **Short replies come back with invented words** — an English word, a name, a word that does not fit (now flagged `uncertain` in 16 of 18 cases, but still wrong) | A turn under 2 s is decoded on its own, without context; the model guesses fluently. |
| P3 | **Replies lost, or under the other speaker** (after 0.8.2: 9 lost acknowledgements, 4 wrong-speaker turns, 6 merged lines) | A reply inside or right next to the other person's speech is shorter than the 1 s voice window; regions under 0.5 s get no voice sample of their own. |
| P4 | **Meaning flips:** a negation lost or added ("will" → "will not"), an amount garbled | One misheard word costs a few characters; nothing compares the meaning. |
| P5 | **Names in several spellings** in one call; occasionally a surname nobody said | Names are outside the model's vocabulary, and each turn is decoded alone. |
| P6 | **A single wrong word inside a long, fluent line** is not flagged | The model is confident in those words too (lowest token log-prob −0.6 to −1.1, like right words). |
| P7 | **Too little as-spoken ground truth:** 13 minutes (the call's first 12 minutes + the Pappa call) | The 15 recordings' transcripts and 12:00–20:35 of the call are still to come. |
| P8 | **Every extra check costs run time** (today 0.6–0.8 × real time) | A second model pass per run. |

---

## 4. Fix plan

### A. Write it as spoken (P1)

**A1 — Convention and scoring.**
- The output convention becomes "as spoken" (section 1). README, OPERATOR and the `trx explain` text say so.
- New monitor number: **spoken-form accuracy**. Of the words where the truth and the output are spellings of the
  same word, it is the share where the output has the truth's form. Today on the call: 0 of the 54 words where
  the owner wrote a spoken form.
- Gate for `as_spoken` fixtures: the content CER (the words) must not get worse by more than 0.01, and the strict
  CER and spoken-form accuracy (the spelling) must improve. This replaces today's "judged by the content CER only".

**A2 — A spoken-variant list: `library/spoken/mr.toml`.**
- Generic Marathi only, never words of a case. It is built from rules and a word list:
  - "aahe" contractions after a word;
  - the spoken forms of nahi;
  - adverbs ending in -e that are spoken with -a (kuthe / kutha, ithe / itha);
  - vowel raising in some verbs (bhetu / bhitu);
  - s / sh and n / retroflex n in listed words;
  - common short forms (aani / an).
- You review it (D3): pronunciations differ by region.
- Each corrected transcript you send makes the environment propose new pairs for that review. Only generic forms
  are proposed.

**A3 — An acoustic judge that hears sounds: a phoneme recognizer.**
- **Candidate:** `facebook/wav2vec2-xlsr-53-espeak-cv-ft`.
  - Meta AI on Hugging Face, Apache-2.0, 1.26 GB.
  - It recognizes phonetic symbols in many languages, including languages it was not trained on, so it hears
    sounds rather than spellings.
  - It needs a download (D4).
- **How it works, per line:**
  - Turn the model's text into expected sounds. Devanagari is close to phonetic, so a rule table does this; a
    Marathi word-final schwa rule is part of it.
  - Align those sounds to the line's audio. This gives word timings that Indic-Transcribe does not provide.
  - For each word in the variant list, compare how well each spelling's sounds match the audio of that word. A
    contraction is scored over both words.
  - Take the spoken spelling only by a clear, measured margin.
- **Feasibility test first, before anything is built in.** On the call's variant words (54 spoken + 58 standard,
  as the owner wrote them) and on Pappa:
  - **Pass:** the spoken form chosen in ≥ 70 % of the words the owner wrote spoken, and ≤ 5 % wrong switches where
    the owner wrote the standard form.
  - **Fail:** A4 is the only way.
- **If it passes:** a `spoken` stage after recognition. The phoneme model runs on the GPU once Indic-Transcribe has
  unloaded. Every change is logged, and the standard form is kept.

**A4 — Teach the model the owner's spelling (later; D5).**
- LoRA fine-tuning of Indic-Transcribe's decoder (where the spelling habit lives) on the owner's as-spoken
  transcripts.
- The licence allows fine-tuning; a fine-tuned copy keeps the licence if it is ever given to anyone.
- It needs about 3 hours or more of as-spoken transcripts. There are 13 minutes now.
- The 4 GB GPU can do it only with the encoder frozen, 4-bit weights and one clip at a time: slow, but possible.
- Always measured on recordings it was not trained on.

**A5 — Rejected: rewriting standard forms to spoken ones by rule** (every nahi → nay). That would write words the
speaker did not say.

### B. Fix the failures of the call

**B1 — Short replies (P2).**
- **(a) Reply check.** For a short line flagged `uncertain`, score a closed list of common replies (हं, हां, हो, the
  forms of nahi, बरं, ठीक आहे, the forms of "speak", hello…) with the phoneme judge. Replace the line only by a
  clear margin, logged.
- **(b) Second reading.** Otherwise keep the text and show IndicConformer's reading next to it in the review list.
  That reading was better on 8 of 20 uncertain lines.
- **Measured by:** content CER of lines < 2 s (today 0.243 on the call), and the 18 invented short lines of the
  failure report.

**B2 — Replies lost or under the wrong speaker (P3): a second diarization pass.**
- Re-scan long turns with 0.6 s windows every 0.1 s for short stretches that sound clearly like the other person.
- Give speech regions under 0.5 s a padded voice sample of their own, instead of the neighbour's label.
- **Measured by:** turn outcomes on the call, Pappa and AMI (lost acknowledgements 9, merged 6, wrong speaker 4
  today). A change that makes AMI worse stays out of the meeting recipe (as in 0.8.2).

**B3 — Meaning check (P4).**
- IndicConformer decodes the file a second time. Where the two models differ in a line's negations or numbers, the
  soft flag `meaning_check` shows the other reading.
- Today on the call: 8 lines flagged, 5 of them real.
- **Measured by:** negation and number errors caught, and false alarms per hour.

**B4 — Names (P5).**
- **(a) No new model.** `review_list.md` gets "written N ways": the rare words of one file grouped by sound-alike
  spelling, with their times.
- **(b) With `--hotwords` (your names list).** The phoneme judge picks the listed spelling when the sounds match.
  The change is logged.

**B5 — Confident wrong words in long lines (P6).**
- Try a word-level sound check: words whose expected sounds do not match the audio of their span (A3 gives this
  for free). It becomes a flag only if at least half of what it marks is really wrong (checked on the call's 12
  minutes).
- No working signal exists yet: token confidence fails, and IndicConformer disagrees too often.

**B6 — Run time (P8).**
- Expected total about 1.0–1.3 × real time, against 0.6–0.8 today:
  - the phoneme pass on the GPU, about 0.1 × real time;
  - the IndicConformer pass on the CPU, about 0.3 × real time.
- Each extra pass is switchable per recipe and reported in `decisions.json`.

### C. Ground truth from the owner (P7)
- **C1** — the 15 recordings, written as spoken (each becomes a fixture: `setup/make_fixture.py`).
- **C2** — 12:00–20:35 of the call, written as spoken.
- **C3** — the names in the call (people, places) for `--hotwords`.
- **C4** — your review of `library/spoken/mr.toml` (A2).
- **C5** — later, more as-spoken transcripts for A4 (about 3 hours).

---

## 5. How every step is tested
- **Fixtures.** The call's first 12 minutes and Pappa (both as spoken), then the 15 recordings as they arrive.
  FLEURS, AMI and the music fixture guard against regressions.
- **Numbers read for every change.**
  - Content CER (the words), strict CER and spoken-form accuracy (the spelling).
  - Turn outcomes.
  - Negation and number errors.
  - Short-line CER.
  - Bad lines flagged.
  - Run time.
- **Procedure.**
  - One change at a time.
  - Adopted only if its target improves and no fixture's content CER gets worse by more than 0.01.
  - An ERRORS / CHANGELOG entry for each adopted change; a commit only after `trx test`.

---

## 6. Phases

| Phase | Version | What | Needs |
|---|---|---|---|
| 1 | 0.9.0 | A1 convention and spoken-form accuracy; A2 variant list v1 (rules); B4(a) name list | nothing |
| 2 | 0.9.1 | A3 feasibility test of the phoneme judge; if it passes, the `spoken` stage, B1(a) reply check, B5 sound check | D4 download |
| 3 | 0.9.2 | B3 meaning check + B1(b) second reading (one IndicConformer pass) | D6 |
| 4 | 0.9.3 | B2 second diarization pass | — |
| 5 | later | A4 fine-tuning | C5, D5 |
| ongoing | — | C1–C4: each transcript becomes a fixture the day it arrives | owner |

---

## 7. Costs and risks
- **The phoneme model may hear phone-call Marathi badly.** It is zero-shot and the audio is phone band. The A3
  feasibility test decides before anything is built.
- **A transcript is not always purely phonetic** (the writer's habits). D1 and your review of the variant list
  handle it; the 15 recordings are the real test.
- **Tuning on one call can mislead.** No threshold is adopted on the call alone; Pappa and the 15 recordings must
  agree.
- **Disk.** On C: (12 GB free): the phoneme model takes 1.26 GB; IndicConformer back on C: takes 2.4 GB (D6).
- **Privacy.** The variant list is generic language data in `library/`. Names and phrases of a case never go into
  tracked files; `trx test` checks this.

---

## 8. Decisions for the owner
- **D1** — Spelling-only differences (same sound: short or long i and u, the final anusvara): one standard spelling
  (recommended), or copy the habits of your transcripts?
- **D2** — Keep a standard-spelling copy of every transcript (`--style readable`), for searching and quoting?
  (Recommended: yes.)
- **D3** — Review the spoken-variant list for your region, when Phase 1 has it ready.
- **D4** — Download the phoneme model for the Phase 2 test: `facebook/wav2vec2-xlsr-53-espeak-cv-ft` (Hugging Face,
  Meta AI, 1.26 GB, Apache-2.0).
- **D5** — Fine-tuning later, once about 3 hours of your as-spoken transcripts exist.
- **D6** — Put IndicConformer back on C: (2.4 GB) so the meaning check runs without the USB stick, or keep the stick
  plugged in during runs?
