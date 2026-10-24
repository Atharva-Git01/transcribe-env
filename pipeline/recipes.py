"""Load and check recipes, and build plan.json (the only source of parameters for a run)."""
from __future__ import annotations

from .common import RECIPES_DIR, TrxError, env_version, load_config, load_toml, utc_now
from .stages import CANONICAL_ORDER, is_available, phase_for

META_KEYS = ["id", "version", "description", "when_to_use", "when_not_to_use", "stages"]

# Every key a stage actually reads (E05: a key nobody reads is a lie). Adding a recipe key = adding code that reads it
# AND listing it here; `trx doctor` / `trx test` reject anything else as dead config.
STAGE_KEYS = {
    "convert": {"channels", "highpass_hz", "loudnorm_target_lufs", "sample_rate"},
    "separate": {"required", "engine", "model", "shifts", "overlap", "apply_if_background_db_above",
                 "background_keep_db"},
    "denoise": {"engine", "fallback", "skip_below_dbfs", "attenuation_limit_db", "speech_guard", "speech_guard_threshold",
                "speech_guard_db"},
    "vad": {"engine", "threshold", "min_speech_ms", "min_silence_ms", "speech_pad_ms", "max_segment_s", "gap_audit",
            "gap_threshold", "gap_min_s", "gap_energy_db"},
    "lid": {"mode", "per_segment"},
    "diarize": {"required", "mode", "num_speakers", "max_speakers", "split_cosine", "min_speaker_s", "window_s",
                "hop_s", "switch_penalty", "switch_penalty_pause", "pause_min_s", "min_turn_s", "cluster_threshold",
                "crosstalk_suppression_db", "short_regions", "rescan_long_turns_s", "rescan_window_s", "rescan_hop_s",
                "rescan_margin", "rescan_min_run_s"},
    "recognise": {"audio", "turn_pad_s", "language", "initial_prompt_file", "word_timestamps", "segmentation", "split_max_s", "chunk_minutes", "recover_missed_turns", "recover_min_s", "recover_min_logprob", "group_max_s",
                  "group_max_gap_s"},
    "recognise.faster_whisper": {"compute_type_gpu", "compute_type_cpu", "beam_size", "best_of", "patience",
                                 "temperature", "compression_ratio_threshold", "log_prob_threshold",
                                 "no_speech_threshold", "condition_on_previous_text", "repetition_penalty",
                                 "no_repeat_ngram_size", "hallucination_silence_threshold"},
    "recognise.indic_conformer": {"decoder", "piece_max_s", "threads", "segmentation"},
    "recognise.indic_transcribe": {"piece_max_s", "threads", "segmentation", "dtype", "device", "num_beams"},
    "verify": {"required", "mode", "engine", "decoder"},
    "merge": {"required", "mode", "lora"},
    "polish": {"doubled_words_guard"},
    "gate": {"thresholds"},
    "export": {"formats"},
}


def _dead_keys(data: dict, stages: list[str]) -> list[str]:
    out = []
    for s in stages:
        for k, v in (data.get(s) or {}).items():
            name = f"{s}.{k}" if isinstance(v, dict) else s
            allowed = STAGE_KEYS.get(name, set())
            if isinstance(v, dict):
                out += [f"[{name}] {kk}" for kk in v if kk not in allowed]
                if name not in STAGE_KEYS:
                    out.append(f"[{name}] (no engine reads this table)")
            elif k not in allowed:
                out.append(f"[{s}] {k}")
    return out


def check_recipe(data: dict, file_id: str) -> list[str]:
    p = []
    meta = data.get("recipe")
    if not isinstance(meta, dict):
        return [f"{file_id}: missing [recipe] table"]
    for k in META_KEYS:
        if k not in meta:
            p.append(f"{file_id}: [recipe] missing '{k}'")
    if meta.get("id") != file_id:
        p.append(f"{file_id}: [recipe].id is '{meta.get('id')}', must equal the file name")
    stages = meta.get("stages", [])
    unknown = [s for s in stages if s not in CANONICAL_ORDER]
    if unknown:
        p.append(f"{file_id}: unknown stages {unknown}")
    known = [s for s in stages if s in CANONICAL_ORDER]
    if known != sorted(known, key=CANONICAL_ORDER.index):
        p.append(f"{file_id}: stages must follow the canonical order {CANONICAL_ORDER}")
    for s in known:
        if not isinstance(data.get(s), dict):
            p.append(f"{file_id}: stage '{s}' listed but has no [{s}] table")
    for table in data:
        if table != "recipe" and table not in stages:
            p.append(f"{file_id}: table [{table}] is not a listed stage (dead config)")
    for s in ("recognise", "gate", "export"):
        if s not in stages:
            p.append(f"{file_id}: every recipe must include '{s}'")
    dead = _dead_keys(data, known)
    if dead:
        p.append(f"{file_id}: keys no stage reads (dead config, E05): {', '.join(dead)}")
    return p


def all_recipe_ids() -> list[str]:
    return sorted(f.stem for f in RECIPES_DIR.glob("*.toml"))


def load_recipe(recipe_id: str) -> dict:
    path = RECIPES_DIR / f"{recipe_id}.toml"
    if not path.is_file():
        raise TrxError("CONFIG_INVALID", f"recipes/{recipe_id}.toml not found")
    try:
        data = load_toml(path)
    except Exception as e:  # noqa: BLE001
        raise TrxError("CONFIG_INVALID", f"recipes/{recipe_id}.toml: {e}") from e
    problems = check_recipe(data, recipe_id)
    if problems:
        raise TrxError("CONFIG_INVALID", "; ".join(problems))
    return data


def build_plan(run_id: str, selection: dict, overrides: dict) -> dict:
    recipe = load_recipe(selection["recipe"])
    defaults = load_config("defaults")
    output = dict(defaults["output"])
    if overrides.get("style"):
        output["style"] = overrides["style"]
    stages = []
    for name in recipe["recipe"]["stages"]:
        params = dict(recipe[name])
        if name == "recognise":
            # Copy the engine tables into the plan so the plan alone fully describes the run.
            params["engine_table_primary"] = dict(defaults["engines"]["primary"])
            params["engine_table_cross_check"] = dict(defaults["engines"]["cross_check"])
            if overrides.get("engine"):
                params["engine_table_primary"] = {k: overrides["engine"] for k in params["engine_table_primary"]}
            if overrides.get("lang"):
                params["language"] = overrides["lang"]
            if overrides.get("hotwords"):
                params["hotwords"] = list(overrides["hotwords"])
        if name == "polish":
            params["vocabulary_enabled"] = defaults["vocabulary"]["enabled"]
            params["llm_cleanup_enabled"] = defaults["llm_cleanup"]["enabled"]
            params["llm_cleanup_engines"] = list(defaults["llm_cleanup"].get("engines", []))
        stages.append({
            "name": name,
            "status": "pending" if is_available(name, params) else "not_installed",
            "available_from_phase": phase_for(name, params),
            "params": params,
        })
    return {
        "schema": "trx.plan/1",
        "env_version": env_version(),
        "run_id": run_id,
        "created_utc": utc_now(),
        "recipe": {"id": recipe["recipe"]["id"], "version": int(recipe["recipe"]["version"]),
                   "file": f"recipes/{selection['recipe']}.toml"},
        "selected_by": {"rule_id": selection["rule_id"], "reason": selection["reason"],
                        "forced": selection["forced"]},
        "overrides": overrides,
        "output": output,
        "stages": stages,
    }
