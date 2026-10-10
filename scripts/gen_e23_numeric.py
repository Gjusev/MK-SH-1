"""E23 generator: atomic numeric/categorical/composition training data + sealed test.

Frozen design in docs/internal/E23_PROTOCOL_2026-10-10.md. TRAIN features
(weight_kg, flow_l_min, uv_index, noise_db) and TEST features (torque_nm,
salinity_ppt, altitude_m) are disjoint; thresholds and value sets are new.
Labels come from the structured spec here; scripts/verify_e23_numeric.py
re-derives them from the rendered text with independent logic.

Outputs: data/e23-numeric/{train_atomic_rows.jsonl, dev_numeric_rows.jsonl,
test_rows.jsonl, manifest.json}. Retention/dev-retention rows are NOT generated
here (they reuse existing E19 train splits at stage time).
"""

from __future__ import annotations

import hashlib
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEED = 20261014

TRAIN_FEATURES = [
    ("weight_kg", "kg", 500.0, "the pallet must be inspected"),
    ("flow_l_min", "liters per minute", 120.0, "the valve must be throttled"),
    ("uv_index", "UV index", 8.0, "the outdoor shift must be paused"),
    ("noise_db", "decibels", 75.0, "hearing protection must be issued"),
]
TEST_FEATURES = [
    ("torque_nm", "newton meters", 400.0, "the bolt must be re-torqued"),
    ("salinity_ppt", "parts per thousand", 35.0, "the desalination stage must run"),
    ("altitude_m", "meters", 2500.0, "oxygen masks must be carried"),
]
OPS = ["at least", "more than", "at most", "below"]


def compare(v, op, thr):
    return {"at least": v >= thr, "more than": v > thr, "at most": v <= thr, "below": v < thr}[op]


def render_noul(feature, unit, thr, action, op, value, rule_in="options"):
    fname = feature.replace("_", " ")
    cond = f"when {fname} is {op} {thr:g} {unit}"
    state = json.dumps({feature: value}, sort_keys=True)
    if rule_in == "options":
        question = f"Decision required now. {action[0].upper()}{action[1:]} now?"
        options = {"false": f"No: {action} {cond}.", "true": f"Yes: {action} {cond}."}
    else:
        question = f"Site rule: {action} {cond}. Based on the current reading, must this happen now?"
        options = {"false": "No, it must not happen now.", "true": "Yes, it must happen now."}
    return state, question, options


def make_row(pid, group, feature, unit, thr, action, op, value, split, variant="base", pair=None, rule_in="options", domain=None):
    state, question, options = render_noul(feature, unit, thr, action, op, value, rule_in)
    gold = compare(value, op, thr)
    return {"id": pid, "group_id": group, "source": "e23-numeric", "language": "en",
            "domain": domain or f"num_{feature}", "state": state, "question": question,
            "options": [{"id": "false", "description": options["false"]},
                        {"id": "true", "description": options["true"]}],
            "target_kind": "single", "target_option": "true" if gold else "false",
            "split": split, "question_type": "noul", "label_origin": "e23-spec",
            "operator": op, "feature": feature, "threshold": thr, "value": value,
            "variant": variant, "pair": pair, "rule_in": rule_in}


CAT_FEATURES = [
    ("vehicle_class", ["truck", "van", "sedan"], "truck", "the loading dock gate must open"),
    ("shift", ["night", "day", "swing"], "night", "the floodlights must stay on"),
    ("cert_level", ["basic", "advanced", "expert"], "expert", "the task requires sign-off"),
]


def make_cat_row(pid, group, feature, values, target, action, value, split):
    cond = f"when the {feature.replace('_', ' ')} is {target}"
    state = json.dumps({feature: value}, sort_keys=True)
    gold = value == target
    question = f"Decision required now. {action[0].upper()}{action[1:]} now?"
    options = {"false": f"No: {action} {cond}.", "true": f"Yes: {action} {cond}."}
    return {"id": pid, "group_id": group, "source": "e23-numeric", "language": "en",
            "domain": f"cat_{feature}", "state": state, "question": question,
            "options": [{"id": "false", "description": options["false"]},
                        {"id": "true", "description": options["true"]}],
            "target_kind": "single", "target_option": "true" if gold else "false",
            "split": split, "question_type": "noul", "label_origin": "e23-spec",
            "variant": "base", "pair": None}


def make_andor_row(pid, group, numf, unit, thr, catf, catvals, cattgt, action, num_v, cat_v, kind, split):
    ncond = f"when {numf.replace('_',' ')} is at least {thr:g} {unit}"
    ccond = f"the {catf.replace('_',' ')} is {cattgt}"
    state = json.dumps({numf: num_v, catf: cat_v}, sort_keys=True)
    if kind == "and":
        gold = num_v >= thr and cat_v == cattgt
        rule = f"{ncond} and {ccond}"
    else:
        gold = num_v >= thr or cat_v == cattgt
        rule = f"{ncond} or {ccond}"
    question = f"Decision required now. {action[0].upper()}{action[1:]} now?"
    options = {"false": f"No: {rule}.", "true": f"Yes: {rule}."}
    return {"id": pid, "group_id": group, "source": "e23-numeric", "language": "en",
            "domain": f"{kind}_{numf}", "state": state, "question": question,
            "options": [{"id": "false", "description": options["false"]},
                        {"id": "true", "description": options["true"]}],
            "target_kind": "single", "target_option": "true" if gold else "false",
            "split": split, "question_type": "noul", "label_origin": "e23-spec",
            "variant": "base", "pair": None}


def values_for(op, thr):
    base = {"at least": [thr, thr + 2.5, thr - 0.5, thr - 6.0],
            "more than": [thr, thr + 0.5, thr + 2.5, thr - 6.0],
            "at most": [thr, thr - 2.5, thr + 0.5, thr + 6.0],
            "below": [thr, thr - 0.5, thr - 2.5, thr + 6.0]}[op]
    for delta in (1.0, 3.5, 5.0, 7.5, 9.0, 11.0, 13.0, 15.0):
        base.extend([thr + delta, thr - delta])
    return sorted(set(round(v, 1) for v in base))


def main() -> int:
    rng = random.Random(SEED)
    train_atomic, dev_numeric, test_rows = [], [], []
    seen = set()

    # ---- TRAIN atomic numeric: 4 features x 4 ops x 4 values = 64 problems x 2 rules = 128 rows
    for feature, unit, thr, action in TRAIN_FEATURES:
        for op in OPS:
            for v in values_for(op, thr):
                group = f"e23n|{feature}|{op}|{v:g}"
                for ri in ("options", "question"):
                    key = (feature, op, v, ri)
                    assert key not in seen
                    seen.add(key)
                    train_atomic.append(make_row(f"{group}|{ri[:3]}", group, feature, unit, thr, action,
                                                  op, v, "train", rule_in=ri))
    # ---- TRAIN categorical: 3 x 3 values = 9 problems... need 240 rows total: replicate with actions
    k = 0
    while len(train_atomic) + 0 < 0:
        break
    cat_rows = []
    actions_extra = ["the alarm must be tested", "the log entry must be flagged"]
    for feature, values, target, action in CAT_FEATURES:
        for v in values:
            for rep in range(27):
                group = f"e23c|{feature}|{v}|{rep}"
                cat_rows.append(make_cat_row(f"{group}", group, feature, values, target, action, v, "train"))
    cat_rows = cat_rows[:240]
    # ---- TRAIN AND/OR: 3 numeric features x 4 combos x 2 kinds x 10 reps = 240... target 480
    andor = []
    for feature, unit, thr, action in TRAIN_FEATURES[:3]:
        catf, catvals, cattgt, _ = CAT_FEATURES[TRAIN_FEATURES.index((feature, unit, thr, action)) % 3]
        other = [c for c in catvals if c != cattgt][0]
        combos = [(thr + 4, cattgt), (thr + 4, other), (thr - 4, cattgt), (thr - 4, other)]
        for kind in ("and", "or"):
            for rep in range(20):
                for (num_v, cat_v) in combos:
                    group = f"e23{kind}|{feature}|{rep}"
                    andor.append(make_andor_row(f"{group}|{combos.index((num_v, cat_v))}", group, feature, unit,
                                                thr, catf, catvals, cattgt, action, num_v, cat_v, kind, "train"))
    andor = andor[:480]
    # ---- TRAIN inversion pairs: 12 pairs x 2 members x 2 rule placements = 48
    for k in range(12):
        feature, unit, thr, action = TRAIN_FEATURES[k % 4]
        op = OPS[k % 4]
        if (feature, op, "pair", "hi", "options") in seen:
            continue  # (feature, op) combos repeat at k and k+4 for k<4; keep first
        hi, lo = thr + 4.0, thr - 4.0
        pair_id = f"e23p|{feature}|{op}|{k:02d}"
        for side, v in (("hi", hi), ("lo", lo)):
            for ri in ("options", "question"):
                key = (feature, op, "pair", side, ri)
                assert key not in seen, key
                seen.add(key)
                train_atomic.append(make_row(f"{pair_id}|{side}|{ri[:3]}", f"{pair_id}|{side}", feature, unit, thr,
                                              action, op, v, "train", variant=f"pair_{side}", pair=pair_id, rule_in=ri))
    # ---- DEV numeric: 180 rows from TRAIN features with NEW values (not in train set)
    dev_vals = set()
    for r in train_atomic:
        dev_vals.add((r["feature"], r["operator"], r["value"]))
    dev_ids = set()
    for feature, unit, thr, action in TRAIN_FEATURES:
        for op in OPS:
            k = 0
            for extra in (3.25, 8.75, 14.25, 19.75, 25.25):
                v = round(thr + extra if op in ("at least", "more than") else thr - extra, 1)
                group = f"e23d|{feature}|{op}|{v:g}"
                if group in dev_ids:
                    continue
                dev_ids.add(group)
                dev_numeric.append(make_row(f"{group}|opt", group, feature, unit, thr, action, op, v, "dev"))
                dev_numeric.append(make_row(f"{group}|que", group, feature, unit, thr, action, op, v, "dev", rule_in="question"))
                k += 2
    dev_numeric = dev_numeric[:180]
    # ---- TEST: new features, 288 + 48 pair rows
    for feature, unit, thr, action in TEST_FEATURES:
        for op in OPS:
            for v in values_for(op, thr):
                group = f"e23t|{feature}|{op}|{v:g}"
                for ri in ("options", "question"):
                    test_rows.append(make_row(f"{group}|{ri[:3]}", group, feature, unit, thr, action,
                                               op, v, "test", rule_in=ri))
    seen_test = set()
    for k in range(12):
        feature, unit, thr, action = TEST_FEATURES[k % 3]
        op = OPS[k % 4]
        if (feature, op) in seen_test:
            continue
        seen_test.add((feature, op))
        hi, lo = thr + 4.0, thr - 4.0
        pair_id = f"e23tp|{feature}|{op}|{k:02d}"
        for side, v in (("hi", hi), ("lo", lo)):
            for ri in ("options", "question"):
                test_rows.append(make_row(f"{pair_id}|{side}|{ri[:3]}", f"{pair_id}|{side}", feature, unit, thr,
                                           action, op, v, "test", variant=f"pair_{side}", pair=pair_id, rule_in=ri))

    out = ROOT / "data/e23-numeric"
    out.mkdir(parents=True, exist_ok=True)
    train_all = train_atomic + cat_rows + andor
    for name, rows in (("train_atomic_rows.jsonl", train_all), ("dev_numeric_rows.jsonl", dev_numeric),
                       ("test_rows.jsonl", test_rows)):
        ids = [r["id"] for r in rows]
        assert len(ids) == len(set(ids)), name
        (out / name).write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows), encoding="utf-8")
    import collections
    bal = collections.Counter(r["target_option"] for r in train_all)
    manifest = {"set": "e23-numeric", "seed": SEED,
                "counts": {"train": len(train_all), "atomic": len(train_atomic), "cat": len(cat_rows),
                           "andor": len(andor), "dev": len(dev_numeric), "test": len(test_rows)},
                "train_balance": dict(bal), "train_features": [f[0] for f in TRAIN_FEATURES],
                "test_features": [f[0] for f in TEST_FEATURES],
                "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "status": "train/dev from TRAIN features; test features fully held out"}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(manifest["counts"], indent=1), "| balance:", dict(bal))
    return 0


if __name__ == "__main__":
    sys.exit(main())
