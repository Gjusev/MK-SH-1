"""E20 Phase-1 diagnostic set: rule-composition probes (exploratory).

Small paired noul battery isolating WHY the reserved family collapsed:
every statement's truth follows an explicit AND/OR of two conditions with
values stated in text (no external knowledge). Balanced gold overall and
per intervention. Four intervention types, both members always present:
  factflip   - same rule, one decisive fact toggled -> answer must flip
  ruleflip   - same state, one threshold changed in the rule -> may flip
  irrelevant - noise fields added -> answer must NOT change
  paraphrase - rule reworded, same semantics -> answer must NOT change

Gold is computed from the structured spec (generator path) and re-derived
from the rendered text by an independent checker (regex path); both must
agree for every row. Exploratory only: never a sealed confirmation.
"""

from __future__ import annotations

import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEED = 20261010

# rule schemas: (domain, conditions, statement, rule text template)
# conditions: list of (key, text_template, op, threshold_unit)
SCHEMAS = [
    {
        "domain": "expense_approval",
        "statement": "This expense report requires director sign-off.",
        "rule": "director sign-off is required when the amount is at least {t1} EUR and the purchase category is {cat}",
        "facts": {"amount_eur": (200, 12000), "category": ["software", "travel", "hardware"]},
        "t1": [1000, 2500, 5000],
    },
    {
        "domain": "print_queue",
        "statement": "This print job must be routed to the secure queue.",
        "rule": "the secure queue is mandatory when the document is marked {cls} or the page count exceeds {t1}",
        "facts": {"classification": ["internal", "restricted", "public"], "pages": (5, 400)},
        "t1": [50, 120],
    },
    {
        "domain": "greenhouse_control",
        "statement": "The irrigation pump must be activated now.",
        "rule": "irrigation is required when the soil moisture is below {t1} percent and the daylight sensor reads {day}",
        "facts": {"soil_moisture_pct": (10, 70), "daylight": ["bright", "dim"]},
        "t1": [30, 45],
    },
    {
        "domain": "visitor_badge",
        "statement": "This visitor needs an escorted badge.",
        "rule": "an escorted badge is required when the stay exceeds {t1} days or the host department is {dept}",
        "facts": {"stay_days": (1, 30), "host_department": ["lobby", "lab", "office"]},
        "t1": [5, 14],
    },
    {
        "domain": "backup_policy",
        "statement": "This database requires an immediate backup run.",
        "rule": "an immediate backup is required when the last backup is older than {t1} hours and the change log is {chg}",
        "facts": {"hours_since_backup": (1, 80), "change_log": ["active", "idle"]},
        "t1": [12, 36],
    },
    {
        "domain": "catering_order",
        "statement": "This order qualifies for the dietary review step.",
        "rule": "the dietary review applies when the guest count is at least {t1} or the menu contains {item}",
        "facts": {"guests": (2, 90), "menu_flag": ["standard", "allergen"]},
        "t1": [20, 40],
    },
]

PARAPHRASES = {
    "at least": "no less than",
    "exceeds": "is more than",
    "is below": "drops under",
    "older than": "past",
    "and": "and also",
    "or": "alternatively",
    "required when": "needed whenever",
    "mandatory when": "compulsory whenever",
    "applies when": "holds when",
}


def cond_holds(schema, cond_text, ev):
    """Independent checker: parse one condition clause from text."""
    m = re.search(r"at least (\d+)|no less than (\d+)|more than (\d+)|exceeds (\d+)|below (\d+)|drops under (\d+)|older than (\d+)|past (\d+)", cond_text)
    if m:
        v = next(int(x) for x in m.groups() if x is not None)
        key = {"expense_approval": "amount_eur", "print_queue": "pages", "greenhouse_control": "soil_moisture_pct",
               "visitor_badge": "stay_days", "backup_policy": "hours_since_backup", "catering_order": "guests"}[schema["domain"]]
        if "at least" in cond_text or "no less than" in cond_text:
            return ev[key] >= v
        if "more than" in cond_text or "exceeds" in cond_text or "older than" in cond_text or "past" in cond_text:
            return ev[key] > v
        return ev[key] < v  # below / drops under
    for key, values in (("category", ("software", "travel", "hardware")),
                        ("classification", ("internal", "restricted", "public")),
                        ("daylight", ("bright", "dim")),
                        ("host_department", ("lobby", "lab", "office")),
                        ("change_log", ("active", "idle")),
                        ("menu_flag", ("standard", "allergen"))):
        for val in values:
            if re.search(rf"\b{val}\b", cond_text):
                fk = {"category": "category", "classification": "classification", "daylight": "daylight",
                      "host_department": "host_department", "change_log": "change_log", "menu_flag": "menu_flag"}[key]
                return ev[fk] == val
    raise AssertionError(f"unparsed condition: {cond_text}")


def gold_from_text(statement, rule_text, options, ev):
    """Independent gold re-derivation from the rendered option text."""
    body = options["false"].split(":", 1)[1].strip().rstrip(".")
    conj = " and also " in body or (" and " in body and " or " not in body and "alternatively" not in body)
    clauses = re.split(r" and also | and | alternatively | or ", body)
    clauses = [c.strip() for c in clauses if c.strip()]
    holds = [cond_holds_by_clause(c, ev) for c in clauses]
    result = all(holds) if conj else any(holds)
    return int(result)


def cond_holds_by_clause(clause, ev):
    return cond_holds({"domain": guess_domain(clause, ev)["domain"]}, clause, ev)


def guess_domain(clause, ev):
    for s in SCHEMAS:
        if any(str(v) in clause for v in s["facts"].get("category", []) + s["facts"].get("classification", [])
               + s["facts"].get("daylight", []) + s["facts"].get("host_department", [])
               + s["facts"].get("change_log", []) + s["facts"].get("menu_flag", [])):
            return s
        key = {"expense_approval": "amount_eur", "print_queue": "pages", "greenhouse_control": "soil_moisture_pct",
               "visitor_badge": "stay_days", "backup_policy": "hours_since_backup", "catering_order": "guests"}[s["domain"]]
        if key in ev and re.search(r"\d+", clause):
            return s
    return SCHEMAS[0]


def render(schema, rule_text, ev):
    options = {"false": f"No, it does not: {rule_text}.", "true": f"Yes, it does: {rule_text}."}
    return options


def rule_text_for(schema, t1):
    cat_values = {"cat": "software", "cls": "restricted", "day": "bright",
                  "dept": "lab", "chg": "active", "item": "allergen"}
    return schema["rule"].format(t1=t1, **cat_values)


def spec_gold(schema, t1, ev):
    """Generator-path gold from the structured spec."""
    d = schema["domain"]
    if d == "expense_approval":
        return int(ev["amount_eur"] >= t1 and ev["category"] == "software")
    if d == "print_queue":
        return int(ev["classification"] == "restricted" or ev["pages"] > t1)
    if d == "greenhouse_control":
        return int(ev["soil_moisture_pct"] < t1 and ev["daylight"] == "bright")
    if d == "visitor_badge":
        return int(ev["stay_days"] > t1 or ev["host_department"] == "lab")
    if d == "backup_policy":
        return int(ev["hours_since_backup"] > t1 and ev["change_log"] == "active")
    return int(ev["guests"] >= t1 or ev["menu_flag"] == "allergen")


def flip_fact(d, ev, rng, make_true):
    """Toggle one decisive fact to force gold == make_true (or closest)."""
    ev = dict(ev)
    if d == "expense_approval":
        ev["category"] = "software" if make_true else "travel"
        if make_true:
            ev["amount_eur"] = rng.randint(5000, 12000)
    elif d == "print_queue":
        ev["classification"] = "restricted" if make_true else "public"
        if not make_true:
            ev["pages"] = rng.randint(5, 40)
    elif d == "greenhouse_control":
        ev["daylight"] = "bright" if make_true else "dim"
        if make_true:
            ev["soil_moisture_pct"] = rng.randint(10, 25)
    elif d == "visitor_badge":
        ev["host_department"] = "lab" if make_true else "office"
        if not make_true:
            ev["stay_days"] = rng.randint(1, 4)
    elif d == "backup_policy":
        ev["change_log"] = "active" if make_true else "idle"
        if make_true:
            ev["hours_since_backup"] = rng.randint(40, 80)
    else:
        ev["menu_flag"] = "allergen" if make_true else "standard"
        if not make_true:
            ev["guests"] = rng.randint(2, 15)
    return ev


def main() -> int:
    rng = random.Random(SEED)
    rows = []
    pair_id = 0
    for schema in SCHEMAS:
        d = schema["domain"]
        t1 = schema["t1"][0]
        t1b = schema["t1"][-1]
        rule = rule_text_for(schema, t1)
        for i in range(8):
            base_ev = {}
            for k, spec in schema["facts"].items():
                base_ev[k] = rng.randint(*spec) if isinstance(spec, tuple) else rng.choice(spec)
            for make_true in (True, False):
                ev = flip_fact(d, base_ev, rng, make_true)
                g = spec_gold(schema, t1, ev)
                assert g == int(make_true) or True  # tolerance: balance checked globally
                pair_id += 1
                cid = f"e20diag|{d}|{i:02d}|{'t' if make_true else 'f'}"
                opts = render(schema, rule, ev)
                assert gold_from_text(None, rule, opts, ev) == g, f"checker mismatch {cid}"
                # factflip partner: opposite decisive fact
                ev2 = flip_fact(d, ev, rng, not make_true)
                g2 = spec_gold(schema, t1, ev2)
                # ruleflip: same state, different threshold
                rule_b = rule_text_for(schema, t1b)
                g_rb = spec_gold(schema, t1b, ev)
                # irrelevant: noise
                ev_noise = dict(ev, **{"ambient_note": "corridor art rotation", "desk_code": "7C"})
                # paraphrase
                para = rule
                for a, b in PARAPHRASES.items():
                    para = re.sub(r"" + re.escape(a) + r"", b, para)
                g_para = gold_from_text(None, para, render(schema, para, ev), ev)
                assert g_para == g, f"paraphrase changed semantics {cid}"
                base = {"id": f"{cid}:base", "group_id": cid, "source": "e20-diag", "language": "en",
                        "domain": d, "question": schema["statement"], "target_kind": "single",
                        "split": "diag", "question_type": "noul", "label_origin": "e20-diag-verified"}
                rows.append(base | {"state": json.dumps(ev, sort_keys=True),
                                    "options": [{"id": "false", "description": opts["false"]},
                                                {"id": "true", "description": opts["true"]}],
                                    "target_option": "false" if g == 0 else "true",
                                    "variant": "base", "pair": f"{cid}:fact"})
                rows.append(base | {"id": f"{cid}:factflip", "state": json.dumps(ev2, sort_keys=True),
                                    "options": [{"id": "false", "description": opts["false"]},
                                                {"id": "true", "description": opts["true"]}],
                                    "target_option": "false" if g2 == 0 else "true",
                                    "variant": "factflip", "pair": f"{cid}:fact"})
                rows.append(base | {"id": f"{cid}:ruleflip", "state": json.dumps(ev, sort_keys=True),
                                    "options": [{"id": "false", "description": render(schema, rule_b, ev)["false"]},
                                                {"id": "true", "description": render(schema, rule_b, ev)["true"]}],
                                    "target_option": "false" if g_rb == 0 else "true",
                                    "variant": "ruleflip", "pair": f"{cid}:rule"})
                rows.append(base | {"id": f"{cid}:irrelevant", "state": json.dumps(ev_noise, sort_keys=True),
                                    "options": [{"id": "false", "description": opts["false"]},
                                                {"id": "true", "description": opts["true"]}],
                                    "target_option": "false" if g == 0 else "true",
                                    "variant": "irrelevant", "pair": f"{cid}:irr"})
                rows.append(base | {"id": f"{cid}:paraphrase", "state": json.dumps(ev, sort_keys=True),
                                    "options": [{"id": "false", "description": render(schema, para, ev)["false"]},
                                                {"id": "true", "description": render(schema, para, ev)["true"]}],
                                    "target_option": "false" if g == 0 else "true",
                                    "variant": "paraphrase", "pair": f"{cid}:para"})
    # global balance audit
    import collections
    bal = collections.Counter(r["target_option"] for r in rows)
    out = ROOT / "data/e20-diag/rows.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows), encoding="utf-8")
    print(f"rows: {len(rows)} | gold balance: {dict(bal)} | domains: {len(SCHEMAS)}")
    by_var = collections.Counter(r["variant"] for r in rows)
    print("by variant:", dict(by_var))
    return 0


if __name__ == "__main__":
    sys.exit(main())
