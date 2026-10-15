"""Load and check recipes, and build plan.json (the only source of parameters for a run)."""
from __future__ import annotations

from .common import RECIPES_DIR, TrxError, env_version, load_config, load_toml, utc_now
from .stages import CANONICAL_ORDER, IMPLEMENTED, phase_for

META_KEYS = ["id", "version", "description", "when_to_use", "when_not_to_use", "stages"]


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
            if overrides.get("lang"):
                params["language"] = overrides["lang"]
        if name == "polish":
            params["vocabulary_enabled"] = defaults["vocabulary"]["enabled"]
            params["llm_cleanup_enabled"] = defaults["llm_cleanup"]["enabled"]
        stages.append({
            "name": name,
            "status": "pending" if name in IMPLEMENTED else "not_installed",
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
