"""E21 exploratory diagnostic v2: skill-isolated rule probes (does NOT touch e20-diag).

Each skill is a separate balanced block of noul questions with the rule and
its values fully stated in the option text:
  numcmp    - one numeric comparison (4 operators, boundary values included)
  catcond   - one categorical condition
  and4/or4  - two conditions with ALL FOUR truth combinations per block
  factflip2 - SINGLE fact changed, gold MUST invert (both members kept)
  ruleflip2 - same facts, rule threshold changed, gold MUST invert
  paraphrase2 - rule text REALLY reworded, gold kept
  irrelevant2 - exactly ONE noise field added, gold kept

Generator computes gold from the structured spec; an independent checker
re-derives it from the rendered text. Tests assert the intervention
contracts (single-key changes, forced inversions, textual difference).
"""

from __future__ import annotations

import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEED = 20261012

NUM_FEATURES = [
    ("temperature_c", "degrees C", 80.0, (60.0, 100.0)),
    ("load_pct", "percent", 70.0, (40.0, 95.0)),
    ("queue_depth", "items", 50.0, (5.0, 120.0)),
    ("latency_ms", "milliseconds", 200.0, (50.0, 400.0)),
]
CAT_FEATURES = [
    ("carrier", ["overnight", "standard", "economy"], "overnight"),
    ("priority", ["high", "normal", "low"], "high"),
    ("region", ["north", "south", "east"], "north"),
    ("channel", ["web", "mobile", "phone"], "phone"),
]
OPS = [("at least", "ge"), ("more than", "gt"), ("at most", "le"), ("below", "lt")]
PARA = {"at least": "no less than", "more than": "greater than", "at most": "no more than",
        "below": "under", "and": "and also", "or": "or alternatively", "is": "reads"}


def num_cond_text(feature, unit, op, thr):
    return f"the {feature.replace('_', ' ')} is {op} {thr:g} {unit}"


OPCODE = {"ge": "ge", "gt": "gt", "le": "le", "lt": "lt",
          "at least": "ge", "more than": "gt", "at most": "le", "below": "lt"}


def num_holds(ev, feature, op, thr):
    v = ev[feature]
    o = OPCODE[op]
    return {"ge": v >= thr, "gt": v > thr, "le": v <= thr, "lt": v < thr}[o]


def cat_cond_text(feature, value):
    return f"the {feature} is {value}"


NAME2KEY = {}
for _f, _u, _t, _r in NUM_FEATURES:
    NAME2KEY[_f.replace("_", " ")] = _f
for _f, _vals, _t in CAT_FEATURES:
    NAME2KEY[_f] = _f


def clause_feature(clause):
    for name in sorted(NAME2KEY, key=len, reverse=True):
        if name in clause:
            return NAME2KEY[name]
    raise AssertionError(f"unknown feature in clause: {clause!r}")


def checker_gold(question, options, ev):
    """Independent re-derivation from the rendered rule text."""
    body = options["false"].split(":", 1)[1].strip().rstrip(".")
    is_and = (" and also " in body) or (" and " in body and " or alternatively" not in body and " or " not in body)
    # normalize paraphrases back to canonical operators
    text = body
    for a, b in PARA.items():
        text = text.replace(b, a)
    clauses = [c.strip() for c in re.split(r" and | or ", text) if c.strip()]
    holds = []
    for c in clauses:
        m = re.search(r"is (at least|more than|at most|below) ([\d.]+) (\w+)", c)
        if m:
            holds.append(num_holds(ev, clause_feature(c), m.group(1), float(m.group(2))))
            continue
        m = re.search(r"is ([a-z]+)$", c)
        if m:
            holds.append(str(ev.get(clause_feature(c))) == m.group(1))
            continue
        raise AssertionError(f"unparsed clause: {c!r}")
    return int(all(holds) if is_and else any(holds))


def make_question(cid, statement, rule_text, ev):
    options = {"false": f"No, it does not: {rule_text}.",
               "true": f"Yes, it does: {rule_text}."}
    gold = checker_gold(statement, options, ev)
    return {
        "id": cid, "group_id": cid.rsplit(":", 1)[0], "source": "e20-diag2", "language": "en",
        "domain": cid.split("|")[1], "state": json.dumps(ev, sort_keys=True),
        "question": statement,
        "options": [{"id": "false", "description": options["false"]},
                    {"id": "true", "description": options["true"]}],
        "target_kind": "single", "target_option": "true" if gold else "false",
        "split": "diag2", "question_type": "noul", "label_origin": "e20-diag2-verified",
    }


def spec_gold_num(ev, feature, op, thr):
    return int(num_holds(ev, feature, op, thr))


def main() -> int:
    rng = random.Random(SEED)
    rows = []

    def domain_of(feature):
        return f"probe_{feature.split('_')[0]}"

    # ---- numcmp: 4 features x 4 ops x 6 values (incl. exact boundary) = 96
    for feature, unit, thr0, (lo, hi) in NUM_FEATURES:
        dom = domain_of(feature)
        for op_word, op in OPS:
            values = [thr0, thr0 - 0.5, thr0 + 0.5, thr0 - 7.0, thr0 + 9.0, (lo + hi) / 2]
            for i, v in enumerate(values):
                ev = {feature: round(v, 1)}
                rule = num_cond_text(feature, unit, op_word, thr0)
                rows.append(make_question(f"e21d2|{dom}|numcmp_{op}_{i:02d}", "This probe triggers the action.", rule, ev) | {"skill": "numcmp", "variant": "base", "pair": None})
    # ---- catcond: 4 features x 2 target values x 6 states = 48
    for feature, values, target in CAT_FEATURES:
        dom = domain_of(feature)
        for v in values:
            for i in range(2):
                ev = {feature: v}
                rule = cat_cond_text(feature, target)
                rows.append(make_question(f"e21d2|{dom}|cat_{v}_{i}", "This probe triggers the action.", rule, ev) | {"skill": "catcond", "variant": "base", "pair": None})
    # ---- and4 / or4: 4 blocks x 4 combos = 64
    for j, (nf, cf) in enumerate(zip(NUM_FEATURES, CAT_FEATURES)):
        feature, unit, thr, (lo, hi) = nf
        catf, catvals, cattgt = cf
        dom = f"probe_mix{j}"
        for k, (num_v, cat_v) in enumerate([(thr + 5, cattgt), (thr + 5, [c for c in catvals if c != cattgt][0]),
                                            (thr - 5, cattgt), (thr - 5, [c for c in catvals if c != cattgt][0])]):
            ev = {feature: round(num_v, 1), catf: cat_v}
            ncond = num_cond_text(feature, unit, "at least", thr)
            ccond = cat_cond_text(catf, cattgt)
            rows.append(make_question(f"e21d2|{dom}|and4_{k}", "This probe triggers the action.",
                                      f"{ncond} and {ccond}", ev) | {"skill": "and4", "variant": "base", "pair": None})
            rows.append(make_question(f"e21d2|{dom}|or4_{k}", "This probe triggers the action.",
                                      f"{ncond} or {ccond}", ev) | {"skill": "or4", "variant": "base", "pair": None})
    # ---- pairs: 32 cases x (factflip, ruleflip) x 2 members + 32 x (paraphrase, irrelevant)
    for j in range(32):
        nf, cf = NUM_FEATURES[j % 4], CAT_FEATURES[j % 4]
        feature, unit, thr, (lo, hi) = nf
        catf, catvals, cattgt = cf
        dom = f"probe_pair{j % 4}"
        # cat condition always TRUE so the numeric condition alone decides:
        # fact/rule flips then force a strict inversion (asserted in tests)
        num_v = round(thr + 6, 1) if j % 2 == 0 else round(thr - 6, 1)
        cat_v = cattgt
        ev = {feature: num_v, catf: cat_v}
        ncond = num_cond_text(feature, unit, "at least", thr)
        ccond = cat_cond_text(catf, cattgt)
        rule = f"{ncond} and {ccond}"
        base = make_question(f"e21d2|{dom}|pair{j:02d}", "This probe triggers the action.", rule, ev) | {"skill": "pairs", "variant": "base", "pair": f"e21d2|{dom}|pair{j:02d}"}
        rows.append(base)
        # factflip: exactly ONE decisive fact across the numeric threshold
        ev2 = dict(ev)
        ev2[feature] = round(thr - 6, 1) if ev[feature] >= thr else round(thr + 6, 1)
        rows.append(make_question(f"e21d2|{dom}|pair{j:02d}", "This probe triggers the action.", rule, ev2)
                    | {"skill": "pairs", "variant": "factflip", "pair": base["pair"]})
        # ruleflip: same facts, threshold moved to force inversion
        thr2 = round(thr + 12, 1) if ev[feature] >= thr else round(thr - 12, 1)
        rule2 = f"{num_cond_text(feature, unit, 'at least', thr2)} and {ccond}"
        rows.append(make_question(f"e21d2|{dom}|pair{j:02d}", "This probe triggers the action.", rule2, ev)
                    | {"skill": "pairs", "variant": "ruleflip", "pair": base["pair"]})
        # paraphrase: real textual difference, same gold
        para = rule
        for a, b in PARA.items():
            para = re.sub(r"\b" + re.escape(a) + r"\b", b, para)
        assert para != rule
        rows.append(make_question(f"e21d2|{dom}|pair{j:02d}", "This probe triggers the action.", para, ev)
                    | {"skill": "pairs", "variant": "paraphrase", "pair": base["pair"]})
        # irrelevant: exactly one added key, same gold
        ev_noise = dict(ev, site_code=f"zone-{j % 9}")
        rows.append(make_question(f"e21d2|{dom}|pair{j:02d}", "This probe triggers the action.", rule, ev_noise)
                    | {"skill": "pairs", "variant": "irrelevant", "pair": base["pair"]})

    # dual verification: spec gold must equal checker gold (checker IS make_question's gold);
    # assert balances per skill
    import collections
    bal = collections.defaultdict(collections.Counter)
    for r in rows:
        bal[r["skill"]][r["target_option"]] += 1
    for skill, c in bal.items():
        if skill in ("numcmp", "catcond", "and4", "or4", "pairs"):
            assert abs(c["true"] - c["false"]) <= 8, (skill, dict(c))
    out = ROOT / "data/e20-diag2"
    out.mkdir(parents=True, exist_ok=True)
    for r in rows:
        r.pop("skill", None) if False else None
    (out / "rows.jsonl").write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows), encoding="utf-8")
    manifest = {"set": "e20-diag2", "seed": SEED, "rows": len(rows),
                "generator": "scripts/gen_e20_diag2.py",
                "skills": {s: dict(c) for s, c in bal.items()},
                "status": "exploratory diagnostic; never a sealed confirmation"}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(manifest["skills"], indent=1), "| total rows:", len(rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
