# Marathi & Hindi in Devanagari — what ASR gets wrong and what the transcript should look like

## Script rules
- Output Marathi and Hindi in **Devanagari**. Never Urdu script, never romanised (except English words, see
  code_switching.md).
- Unicode **NFC** for every output (metrics normalise to NFC too).
- Marathi-specific letters must survive: **ळ** (never ल), **ऱ** (eyelash ra, e.g. ऱ्या), **ॲ/ऑ** for English
  borrowings (ऑफिस, हॅलो).
- Sentence end: `.` or `?` in output; the danda `।` is acceptable but stay consistent within a transcript.
- Numerals: `config/defaults.toml [output].numerals` (default `as_recognised`; conversions are logged).

## Marathi vs Hindi confusions (Whisper trained far more on Hindi)
| Spoken Marathi | Wrong (Hindi-ised) output | Note |
|---|---|---|
| नाही | नहीं | most common confusion |
| आहे / आहेत | है / हैं | |
| काय | क्या | |
| मी | मैं | |
| तू / तुम्ही | तुम / आप | |
| कुठे | कहाँ | |
| हो | हाँ | "yes" |
| झालं | हुआ | |
A Marathi file transcribed with these Hindi forms is a language-ID or prompt failure, not a spelling problem:
check `lid` and the initial prompt before touching vocab.

## Colloquial speech (keep it as spoken in verbatim style)
- Spoken forms are not errors: नाय (नाही), कुटे (कुठे), मंग (मग), आता/अता. Verbatim keeps them; `readable` style may
  normalise them, and logs each change.
- Do not "correct" grammar or dialect in verbatim style.

## Reduplication — real speech, never a loop
हा हा, हो हो, हळूहळू, जा जा, चल चल, हाँ हाँ, अरे अरे. Listed in `library/doubled_words/`. Loop checks ignore them.

## Valid words that look like misspellings (never "correct" one into the other)
आहे/आहेत/आहात · येतो/येते/येता · करतो/करते/करता · होता/होती/होते — gender, number and person, all valid.
This is why the FMS vocab map is disabled (knowledge/ERRORS.md E07).
