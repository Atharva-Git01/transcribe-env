"""Stage: select. Passport -> recipe by the ordered rule table. No AI judgement, fully explainable."""
from __future__ import annotations

from ..common import RECIPES_DIR, TrxError, load_config

OPS = {
    ">": lambda a, b: a > b,
    ">=": lambda a, b: a >= b,
    "<": lambda a, b: a < b,
    "<=": lambda a, b: a <= b,
    "==": lambda a, b: a == b,
    "!=": lambda a, b: a != b,
    "in": lambda a, b: a in b,
}


def get_field(data: dict, dotted: str):
    cur = data
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def condition_true(passport: dict, cond: list) -> bool:
    field, op, ref = cond
    val = get_field(passport, field)
    if val is None:
        return False
    try:
        return bool(OPS[op](val, ref))
    except TypeError:
        return False


def check_rules(rules: list[dict]) -> list[str]:
    """Static checks used by trx doctor."""
    problems = []
    if not rules:
        return ["selection_rules.toml has no [[rule]] entries"]
    seen = set()
    for r in rules:
        rid = r.get("id", "?")
        for key in ("id", "recipe", "when", "reason"):
            if key not in r:
                problems.append(f"rule {rid}: missing '{key}'")
        if rid in seen:
            problems.append(f"rule {rid}: duplicate id")
        seen.add(rid)
        if not (RECIPES_DIR / f"{r.get('recipe')}.toml").is_file():
            problems.append(f"rule {rid}: recipe file recipes/{r.get('recipe')}.toml not found")
        for c in r.get("when", []):
            if not (isinstance(c, list) and len(c) == 3 and c[1] in OPS):
                problems.append(f"rule {rid}: bad condition {c!r}")
    if rules[-1].get("when"):
        problems.append("the last rule must have when = [] (a fallback that always matches)")
    return problems


def select(passport: dict, forced_recipe: str | None = None) -> dict:
    if forced_recipe:
        if not (RECIPES_DIR / f"{forced_recipe}.toml").is_file():
            raise TrxError("USAGE", f"unknown recipe '{forced_recipe}' (see recipes/)")
        return {"rule_id": "forced", "recipe": forced_recipe,
                "reason": "Recipe forced on the command line (--recipe).", "forced": True}
    rules = load_config("selection_rules").get("rule", [])
    problems = check_rules(rules)
    if problems:
        raise TrxError("CONFIG_INVALID", "; ".join(problems))
    for r in rules:
        if all(condition_true(passport, c) for c in r["when"]):
            return {"rule_id": r["id"], "recipe": r["recipe"], "reason": r["reason"], "forced": False}
    raise TrxError("CONFIG_INVALID", "no rule matched, including the fallback")
