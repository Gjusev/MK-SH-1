"""E22 format-contrast battery: same numeric problems in 4 formats (exploratory).

2x2 design over IDENTICAL facts, operators, thresholds and answers:
  state  : JSON object | prose sentence
  rule   : embedded in the OPTIONS | embedded in the QUESTION
Four operators with boundary values, classes balanced per operator per
format, and single-fact inversion pairs whose pair_id spans all four formats.

- Unique question IDs (one id per row); group_id = problem; pair_id shared
  across the 4 variants AND both members of a pair.
- Content-duplicate detection: normalized (facts, op, threshold, format,
  side) must be unique.
- Labels come from the structured spec here; scripts/verify_e22_format.py
  re-derives them from the rendered text with independent logic (no imports
  from this file). tests/test_e22_format.py contains hand-computed boundary
  examples.
"""

from __future__ import annotations

import hashlib
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEED = 20261013

FEATURES = [
    ("pressure_bar", "bar", 5.0, "the pump must be stopped"),
    ("speed_kmh", "km/h", 120.0, "the vehicle must be flagged"),
    ("humidity_pct", "percent", 60.0, "the archive must be ventilated"),
    ("battery_pct", "percent", 20.0, "the sensor must send a low-battery alert"),
]
OPS = ["at least", "more than", "at most", "below"]
FORMATS = ("json_opt", "json_q", "prose_opt", "prose_q")


def compare(value, op, thr):
    return {"at least": value >= thr, "more than": value > thr,
            "at most": value <= thr, "below": value < thr}[op]


def render(feature, unit, thr, action, op, value, fmt):
    """Return (state_str, question, options_dict) for one probe."""
    fname = feature.replace("_", " ")
    cond = f"when {fname} is {op} {thr:g} {unit}"
    if fmt.startswith("json"):
        state = json.dumps({feature: value}, sort_keys=True)
    else:
        state = f"The current reading shows {fname} at {value:g} {unit}. The gauge was verified today."
    if fmt.endswith("_opt"):
        question = f"This requires attention: {action.replace('the ', 'the ')} now."
        question = f"Decision required now. {action[0].upper()}{action[1:]} now?"
        options = {"false": f"No: {action} {cond}.",
                   "true": f"Yes: {action} {cond}."}
    else:
        question = (f"Site rule: {action} {cond}. Based on the current reading, "
                    f"must {action.replace('the ', 'the ')} happen now?")
        options = {"false": "No, it must not happen now.", "true": "Yes, it must happen now."}
    return state, question, options


def make_row(pid, group, fmt, feature, unit, thr, action, op, value, variant, pair_id):
    state, question, options = render(feature, unit, thr, action, op, value, fmt)
    gold = compare(value, op, thr)
    return {
        "id": pid, "group_id": group, "source": "e22-format", "language": "en",
        "domain": f"fmt_{feature}", "state": state, "question": question,
        "options": [{"id": "false", "description": options["false"]},
                    {"id": "true", "description": options["true"]}],
        "target_kind": "single", "target_option": "true" if gold else "false",
        "split": "diag", "question_type": "noul", "label_origin": "e22-spec",
        "format": fmt, "operator": op, "feature": feature, "threshold": thr,
        "value": value, "variant": variant, "pair": pair_id,
    }


def main() -> int:
    rng = random.Random(SEED)
    rows = []
    seen = set()
    # base problems: 4 features x 4 ops x 4 values (boundary, +-eps, +-large)
    for feature, unit, thr, action in FEATURES:
        for op in OPS:
            # choose values so the op's class is balanced 2 true / 2 false
            values = {
                "at least": [thr, thr + 3.5, thr - 0.5, thr - 8.0],
                "more than": [thr, thr + 0.5, thr + 3.5, thr - 8.0],
                "at most": [thr, thr - 3.5, thr + 0.5, thr + 8.0],
                "below": [thr, thr - 0.5, thr - 3.5, thr + 8.0],
            }[op]
            for v in values:
                group = f"e22f|{feature}|{op}|{v:g}"
                for fmt in FORMATS:
                    key = (feature, op, v, fmt)
                    assert key not in seen, f"duplicate content {key}"
                    seen.add(key)
                    rows.append(make_row(f"{group}|{fmt}", group, fmt, feature, unit, thr, action,
                                         op, v, "base", None))
    # inversion pairs: single fact crosses the threshold; all 4 formats, both members
    combos = [(f, op) for f in FEATURES for op in OPS]
    for k, (feature_unit_thr_action, op) in enumerate(combos[:12]):
        feature, unit, thr, action = feature_unit_thr_action
        hi, lo = thr + 4.0, thr - 4.0
        pair_id = f"e22f|pair{k:02d}|{feature}|{op}"
        for side, v in (("hi", hi), ("lo", lo)):
            group = f"{pair_id}|{side}"
            for fmt in FORMATS:
                key = (feature, op, "pair", side, fmt)
                assert key not in seen, f"duplicate content {key}"
                seen.add(key)
                rows.append(make_row(f"{group}|{fmt}", group, fmt, feature, unit, thr, action,
                                     op, v, f"pair_{side}", pair_id))
    # balances
    import collections
    bal = collections.Counter((r["format"], r["operator"], r["target_option"]) for r in rows)
    ids = [r["id"] for r in rows]
    assert len(ids) == len(set(ids))
    for fmt in FORMATS:
        for op in OPS:
            t, f = bal[(fmt, op, "true")], bal[(fmt, op, "false")]
            assert t == f, (fmt, op, t, f)
    out = ROOT / "data/e22-format"
    out.mkdir(parents=True, exist_ok=True)
    (out / "rows.jsonl").write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows), encoding="utf-8")
    manifest = {"set": "e22-format", "seed": SEED, "rows": len(rows),
                "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "formats": FORMATS, "features": [f[0] for f in FEATURES],
                "distinct_contents": len(seen),
                "status": "exploratory diagnostic; never a sealed confirmation"}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"rows": len(rows), "distinct_contents": len(seen)}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
