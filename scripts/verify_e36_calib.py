"""E36 independent verifier: re-derives the calibration split from the
RENDERED text + frozen rule, without importing the generator.

Checks (all hard):
1. Labels re-derived from state JSON + question regex (value >= threshold).
2. Balance 4 Yes / 4 No per value; global 240/240; 480 rows; 60 values.
3. Disjointness vs the prior numeric universe (re-collected independently).
4. 2-decimal freshness of every value/threshold; thresholds within [10,130]
   and not reserved; substitutions match the frozen ladder rule.
5. Manifest hashes match the files on disk.

Exit 0 = verified.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/e36-calib"
RESERVED = {25.0, 44.5, 56.5, 66.5, 78.5, 88.5, 115.0, 30.0, 100.0, 45.0, 55.0, 65.0, 75.0, 85.0}
UNIVERSE = [
    ("e24-fit", "train_fit_rows.jsonl"), ("e24-fit", "dev_fit_rows.jsonl"),
    ("e24-fit", "eval_newfield_rows.jsonl"), ("e24-fit", "eval_newthreshold_rows.jsonl"),
    ("e24-fit", "eval_newvalues_rows.jsonl"),
    ("e25-thrvar", "train_thr_rows.jsonl"), ("e25-thrvar", "dev_thr_rows.jsonl"),
    ("e25-thrvar", "test_known_thr_rows.jsonl"), ("e25-thrvar", "test_new_thr_rows.jsonl"),
    ("e26-order", "train_ctrl_rows.jsonl"), ("e26-order", "train_treat_rows.jsonl"),
    ("e26-order", "dev_rows.jsonl"), ("e26-order", "test_c_rows.jsonl"),
    ("e26-order", "test_d_rows.jsonl"), ("e26-order", "mono_rows.jsonl"),
    ("e27-grid", "train_rows.jsonl"), ("e27-grid", "test_e_rows.jsonl"),
    ("e28-ext", "train_rows.jsonl"), ("e35-extremes", "train_rows.jsonl"),
]
QRE = re.compile(r"^Is the motor temperature reading at least (\d+(?:\.\d+)?) degrees Celsius\?$")


def main() -> int:
    rows = [json.loads(l) for l in (DATA / "rows.jsonl").read_text(encoding="utf-8").splitlines()
            if l.strip()]
    manifest = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))

    # 1. labels re-derived from rendered text
    for r in rows:
        st = json.loads(r["state"])
        m = QRE.match(r["question"])
        assert m, f"question not parseable: {r['question']!r}"
        t_txt, v_txt = float(m.group(1)), float(st["motor_temp_c"])
        assert t_txt == round(r["threshold"], 2) and v_txt == round(r["value"], 2), r["id"]
        assert r["target_option"] == ("true" if v_txt >= t_txt else "false"), r["id"]
        assert st["equipment_id"].startswith("cu-"), r["id"]  # fresh equipment prefix

    # 2. structure and balance
    assert len(rows) == 480
    by_val = defaultdict(Counter)
    for r in rows:
        by_val[r["value"]][r["target_option"]] += 1
    assert len(by_val) == 60
    assert all(c["true"] == 4 and c["false"] == 4 for c in by_val.values())
    assert Counter(r["target_option"] for r in rows) == {"true": 240, "false": 240}

    # 3. disjointness (independent collection)
    prior = set(RESERVED)
    for d, f in UNIVERSE:
        for l in (ROOT / "data" / d / f).read_text(encoding="utf-8").splitlines():
            if l.strip():
                rr = json.loads(l)
                prior.add(round(rr["value"], 2))
                prior.add(round(rr["threshold"], 2))
    used = {round(r["value"], 2) for r in rows} | {round(r["threshold"], 2) for r in rows}
    assert not (used & prior), f"overlap with prior universe: {sorted(used & prior)}"

    # 4. freshness + threshold rules + ladder consistency
    sub_map = {(s["value"], s["original_offset"]): s for s in manifest["substitutions"]}
    for r in rows:
        for x in (r["value"], r["threshold"]):
            s = f"{x:.2f}"
            assert s.endswith(("1", "2", "3", "4", "6", "7", "8", "9")), (r["id"], x)
        assert 10.0 <= r["threshold"] <= 130.0 and r["threshold"] not in RESERVED, r["id"]
    for v, c in by_val.items():
        thrs = sorted({round(r["threshold"], 2) for r in rows if r["value"] == v})
        assert len(thrs) == 8, (v, thrs)
        for t in thrs:
            off = round(t - v, 2)
            if off.is_integer() and int(off) in (-25, -7, -1, 0, 1, 7, 15, 25):
                continue
            hit = [s for s in manifest["substitutions"] if s["value"] == v and s["threshold"] == t]
            assert hit, f"threshold {t} for v={v} is neither an offset nor a logged substitution"
            sign = 1 if hit[0]["original_offset"] > 0 else -1
            assert round(t - v, 2) == sign * hit[0]["substituted_offset"], (v, t, hit)

    # 5. manifest hashes
    assert manifest["rows_sha256"] == hashlib.sha256((DATA / "rows.jsonl").read_bytes()).hexdigest()
    assert manifest["generator_sha256"] == hashlib.sha256(
        (ROOT / "scripts/gen_e36_calib.py").read_bytes()).hexdigest()

    print(f"e36-calib VERIFIED: 480 rows, 60 values, 240/240, "
          f"disjoint from {len(prior)} prior numbers, {len(sub_map)} substitutions consistent")
    return 0


if __name__ == "__main__":
    sys.exit(main())
