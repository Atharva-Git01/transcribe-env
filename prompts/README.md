# prompts/

- `initial_prompts/<lang>/<domain>.txt` — passed to Whisper as `initial_prompt`. Purpose: keep the output in the right
  script and register (Devanagari Marathi, not Hindi spellings; English words in Latin script for mixed speech).
  Keep them **short, neutral and generic**. Whisper sometimes copies prompt words into silent stretches, so a prompt must
  never contain names, numbers or facts that could be mistaken for evidence. Every change goes through `trx eval`.
- `gemma3_cleanup/<lang>.toml` — phase 3 (built; cleanup is off by default, E17): system prompt, few-shot pairs and the forbidden-edit list for
  the local Gemma 3 cleanup pass. The 5 rejection gates (config/thresholds.toml [llm_cleanup]) decide; the model
  only suggests.

Recipes reference `initial_prompts/{lang}/<domain>.txt`; `trx doctor` checks that every referenced file exists for
every candidate language of every recipe.
