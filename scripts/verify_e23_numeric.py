"""Independent verifier for E23 numeric data (no generator imports).

Re-derives every gold from the rendered text: state is JSON (one or two keys),
the rule is in the option text or the question. Operator words map to
comparisons; categorical conditions match exact values.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

OPS = {"at least": lambda v, t: v >= t, "more than": lambda v, t: v > t,
       "at most": lambda v, t: v <= t, "below": lambda v, t: v < t}
NUM_KEYS = ("weight_kg", "flow_l_min", "uv_index", "noise_db", "torque_nm", "salinity_ppt", "altitude_m")


def _num_key(ev):
    return next(k for k in ev if k in NUM_KEYS)


def _cat_key(ev):
    return next(k for k in ev if k not in NUM_KEYS)


def gold(row: dict) -> int:
    ev = json.loads(row["state"])
    text = (row["options"][0]["description"] + " " + row["options"][1]["description"] + " " + row["question"]).lower()
    if " and " in text or " or " in text:
        m = re.search(r"is (at least|more than|at most|below) ([\d.]+)", text)
        if not m:
            raise AssertionError("no numeric clause")
        num_ok = OPS[m.group(1)](ev[_num_key(ev)], float(m.group(2)))
        vals = re.findall(r"\b(truck|van|sedan|night|day|swing|basic|advanced|expert)\b", text)
        if not vals:
            raise AssertionError("no categorical clause")
        cat_ok = ev[_cat_key(ev)] == vals[0]
        is_and = " and " in text
        return int((num_ok and cat_ok) if is_and else (num_ok or cat_ok))
    m = re.search(r"is (at least|more than|at most|below) ([\d.]+)", text)
    if m:
        return int(OPS[m.group(1)](ev[_num_key(ev)], float(m.group(2))))
    m = re.search(r"\b(truck|van|sedan|night|day|swing|basic|advanced|expert)\b", text)
    if m:
        return int(ev[_cat_key(ev)] == m.group(1))
    raise AssertionError(f"no rule found: {row['id']}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/e23-numeric")
    args = parser.parse_args()
    data = Path(args.data)
    bad = 0
    total = 0
    for fname in ("train_atomic_rows.jsonl", "dev_numeric_rows.jsonl", "test_rows.jsonl"):
        for line in (data / fname).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            total += 1
            try:
                if gold(row) != int(row["target_option"] == "true"):
                    bad += 1
                    print("MISMATCH", row["id"])
            except AssertionError as e:
                bad += 1
                print("ERROR", row["id"], e)
    print(f"verified {total} rows: {total - bad} agree, {bad} disagree")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
