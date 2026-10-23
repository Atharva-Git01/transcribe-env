# library/

| Folder | What | Status |
|---|---|---|
| `vocab/mr_fms_v1.json` | Marathi spelling-variant map copied from FMS (`headword: [variants]`) | **disabled** — some variants are real, different words (e.g. आहेत listed as a variant of आहे, येते of येतो), so applying it would rewrite correct speech. See knowledge/ERRORS.md E07. Needs a reviewed v2 before `config/defaults.toml [vocabulary].enabled = true`. |
| `doubled_words/<lang>.txt` | words genuinely repeated in speech (हा हा, हो हो) | active from phase 1: the loop check never flags these |
| `hotwords/<case>.txt` | names/places for one case, supplied by the owner | optional, per run |
| `scoring/<lang>.toml` | acknowledgements, negations, number endings for the eval's failure monitor (E39) | active in `trx eval`; the acknowledgements also exempt a line from the `uncertain` flag (E40); generic words, never words of a case |

`trx doctor` reports vocab entries whose variants are themselves headwords.
