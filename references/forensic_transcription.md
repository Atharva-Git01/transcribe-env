# Forensic transcription conventions (default style: verbatim)

## Verbatim (default)
- Write what is said, as said: fillers (अं, हं, umm), false starts, repetitions, dialect, grammar as spoken.
- No paraphrase, no summary, no translation inside the transcript (a translation is a separate file).
- Every segment carries start–end timestamps and a speaker label.

## Tags (exact spellings from config/defaults.toml [output])
| Situation | Tag |
|---|---|
| Speech present but not intelligible | `[inaudible 00:01:23-00:01:25]` |
| Two or more people speaking at once | `[overlap]` |
| Text the system is not confident about | `[review]` (also listed in review_list.md with the reason) |
| Non-speech sound that matters (optional) | `[laughs]`, `[noise]`, `[phone rings]` |

## Speakers
- Labels are `SPK1`, `SPK2`, … in order of first appearance. Never real names unless the owner supplies them for
  the case. Channel = speaker only for true stereo (knowledge/ERRORS.md E01).

## Integrity
- The original file is never modified; work happens on a hash-verified read-only copy.
- `custody.json` holds the sha256 of the original and of every file the run produced.
- `decisions.json` records every automated change (including every accepted/rejected cleanup edit and why).
- Nothing is silently removed. Anything a filter doubts is kept and tagged `[review]`.
- The report states the environment version, recipe, engines and model hashes, so the run can be reproduced.

## Readable style (`--style readable`)
Removes fillers and false starts, normalises colloquial spellings — every change logged in decisions.json, and the
verbatim transcript is still kept beside it.
