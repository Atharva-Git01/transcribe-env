# Code-switching (Marathi / Hindi / English in the same sentence)

Typical: "मी office मध्ये आहे, meeting संपली की call करतो." This is normal speech, not an error.

## Rules
1. English words stay in **Latin script** (`english_words_script = "latin"`): office, meeting, call, okay.
   Never transliterate them to Devanagari (ऑफिस) and never translate them.
2. Marathi/Hindi words stay in Devanagari.
3. Language is decided **per VAD segment** (`lid.per_segment = true`), never forced for the whole file.
4. A segment whose language ID confidence is low is decoded twice (best two languages) in phase 2 and the
   higher-agreement result wins; disagreement goes to review_list.md.
5. Short English fillers (okay, yes, sorry, please) inside a Marathi segment do not make the segment English.

## Known failure
Whisper with a forced language translates the other language's words instead of transcribing them
(forced `mr` turns "meeting" into "बैठक"). Hence no forced language in mixed audio.
