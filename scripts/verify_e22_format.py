"""Independent verifier for the E22 format battery (no generator imports).

Re-derives every gold from the rendered text alone:
- state: JSON object -> json.loads; prose -> regex "at <value> <unit>"
- rule: option-embedded ("when <feature> is <op> <thr> <unit>") or
  question-embedded ("Site rule: ... when ...")
Written from scratch; shares no code with gen_e22_format.py.

Usage: python scripts/verify_e22_format.py [--data data/e22-format]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

OPS = {"at least": lambda v, t: v >= t, "more than": lambda v, t: v > t,
       "at most": lambda v, t: v <= t, "below": lambda v, t: v < t}


def value_from_state(state: str) -> float:
    s = state.strip()
    if s.startswith("{"):
        obj = json.loads(s)
        if len(obj) != 1:
            raise AssertionError(f"json state with {len(obj)} keys")
        return float(next(iter(obj.values())))
    m = re.search(r"at (-?[\d.]+) (?:bar|km/h|percent)", s)
    if not m:
        raise AssertionError(f"no numeric reading in prose state: {s!r}")
    return float(m.group(1))


def rule_from_row(row: dict) -> tuple[str, float]:
    text = (row["options"][0]["description"] + " " + row["options"][1]["description"] + " " + row["question"]).lower()
    m = re.search(r"is (at least|more than|at most|below) (-?[\d.]+) (?:bar|km/h|percent)", text)
    if not m:
        raise AssertionError("no rule clause found")
    return m.group(1), float(m.group(2))


def gold(row: dict) -> int:
    v = value_from_state(row["state"])
    op, thr = rule_from_row(row)
    return int(OPS[op](v, thr))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/e22-format")
    args = parser.parse_args()
    rows = [json.loads(l) for l in (Path(args.data) / "rows.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    bad = 0
    for r in rows:
        try:
            if gold(r) != int(r["target_option"] == "true"):
                bad += 1
                print("MISMATCH", r["id"])
        except AssertionError as e:
            bad += 1
            print("ERROR", r["id"], e)
    print(f"verified {len(rows)} rows: {len(rows) - bad} agree, {bad} disagree")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
