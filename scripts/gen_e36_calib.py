"""E36 generator: dedicated calibration split for MK-SH-1 v1.2 (frozen criteria
docs/E36_CRITERIA_2026-10-12.md).

60 hand-frozen fresh two-decimal values (disjoint from the ENTIRE prior
numeric universe: e24/e25/e26/e27/e28/e35 values AND thresholds — asserted at
build), each against the 8-offset rule from E27/E35 with the reserved set and
same-sign ladder {3,15,10,5,20}. 4 Yes / 4 No per value EXACTLY (the E27
invariant). Split is for temperature fitting and policy choice ONLY — never
for training. scripts/verify_e36_calib.py re-derives everything independently
from the rendered text and the frozen rule.

Outputs: data/e36-calib/{rows.jsonl, manifest.json}.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIELD, FIELD_WORD, OP = "motor_temp_c", "motor temperature", "at least"
RESERVED = {25.0, 44.5, 56.5, 66.5, 78.5, 88.5, 115.0, 30.0, 100.0, 45.0, 55.0, 65.0, 75.0, 85.0}
OFFSETS = [-25, -7, -1, 0, 1, 7, 15, 25]
YES_OFFSETS = {-25, -7, -1, 0}
LADDER = [3, 15, 10, 5, 20]
THR_MIN, THR_MAX = 10.0, 130.0
SITES = ["site-alpha", "site-bravo", "site-charlie", "site-delta", "site-echo", "site-foxtrot"]

# prior numeric universe: (dir, file) pairs whose value/threshold sets the split must avoid
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

# hand-frozen fresh values (2 decimals, no RNG) — disjointness asserted at build
INTERIOR = [30.43, 32.61, 34.87, 36.29, 38.07, 40.29, 43.43, 45.61, 47.43, 49.29,
            51.87, 53.73, 55.87, 57.07, 59.19, 62.61, 64.73, 66.43, 68.87, 70.29,
            72.51, 74.61, 76.73, 78.37, 81.73, 83.29, 85.43, 87.29, 89.43, 91.61,
            93.07, 95.51, 97.87, 100.29, 102.43, 104.29]
LOWS = [17.43, 18.29, 18.61, 19.07, 20.87, 21.73, 22.43, 23.29, 24.07, 25.51, 26.73, 26.19]
HIGHS = [105.87, 106.43, 107.29, 109.61, 110.87, 112.29, 113.07, 114.61, 116.73, 117.43,
         118.51, 120.19]
VALUES = INTERIOR + LOWS + HIGHS


def thr_ok(t: float) -> bool:
    return THR_MIN <= t <= THR_MAX and t not in RESERVED


def substitute(v: float, off: int, used_for_value: set[float]):
    """Frozen same-sign ladder from E27/E28/E35: {3, 15, 10, 5, 20}; skips
    candidates already used for this value (no duplicate questions per value)."""
    for sub in LADDER:
        t = round(v + (sub if off > 0 else -sub), 2)
        if thr_ok(t) and t not in used_for_value:
            return t, sub
    return None, None


def prior_numbers() -> set[float]:
    prior: set[float] = set(RESERVED)
    for d, f in UNIVERSE:
        p = ROOT / "data" / d / f
        if not p.exists():
            raise SystemExit(f"universe file missing: {p}")
        for l in p.read_text(encoding="utf-8").splitlines():
            if l.strip():
                r = json.loads(l)
                prior.add(round(r["value"], 2))
                prior.add(round(r["threshold"], 2))
    return prior


def main() -> int:
    prior = prior_numbers()
    overlap = {v for v in VALUES if v in prior}
    assert not overlap, f"value overlap with prior numeric universe: {sorted(overlap)}"

    rows, subs = [], []
    seq = 1
    for v in VALUES:
        used_for_value: set[float] = set()
        for off in OFFSETS:
            t = round(v + off, 2)
            used_sub = None
            if not thr_ok(t):
                t, used_sub = substitute(v, off, used_for_value)
                if t is None:
                    raise AssertionError(f"no ladder candidate for v={v} off={off}")
                subs.append({"value": v, "original_offset": off,
                             "substituted_offset": used_sub, "threshold": t})
            used_for_value.add(t)
            assert t not in prior, f"threshold {t} (v={v} off={off}) hits prior universe"
            gold = v >= t
            eq = f"cu-{seq:04d}"
            state = json.dumps({"equipment_id": eq, FIELD: v, "site": SITES[seq % len(SITES)]},
                               sort_keys=True)
            question = f"Is the {FIELD_WORD} reading at least {t:g} degrees Celsius?"
            rows.append({"id": f"e36cal|{seq:04d}", "group_id": f"e36cal|{seq:04d}",
                         "source": "e36-calib", "language": "en", "domain": f"num_{FIELD}",
                         "state": state, "question": question,
                         "options": [{"id": "false", "description": "No"},
                                     {"id": "true", "description": "Yes"}],
                         "target_kind": "single", "target_option": "true" if gold else "false",
                         "split": "cal_temperature", "question_type": "noul",
                         "label_origin": "e36-spec",
                         "operator": OP, "feature": FIELD, "threshold": t, "value": v,
                         "variant": "base", "pair": None, "rule_in": "question",
                         "thr_range": "calib"})
            seq += 1

    # invariants (E27 rule): 8 rows per value, 4 Yes / 4 No EXACT per value
    from collections import Counter, defaultdict

    by_val = defaultdict(Counter)
    for r in rows:
        by_val[r["value"]][r["target_option"]] += 1
    assert len(by_val) == 60 and all(c["true"] == 4 and c["false"] == 4 for c in by_val.values())
    golds = [r["target_option"] == "true" for r in rows]
    assert abs(sum(golds) / len(golds) - 0.5) < 1e-12  # question-ignoring rule == 0.500
    assert len({r["state"] for r in rows}) == len(rows)
    assert len({(r["value"], r["threshold"]) for r in rows}) == len(rows)

    out = ROOT / "data/e36-calib"
    out.mkdir(parents=True, exist_ok=True)
    (out / "rows.jsonl").write_text(
        "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows), encoding="utf-8")
    yes = sum(golds)
    manifest = {
        "set": "e36-calib", "protocol": "docs/E36_CRITERIA_2026-10-12.md",
        "purpose": "temperature fit + abstention policy choice ONLY (MK-SH-1 v1.2); never training",
        "values": {"interior": INTERIOR, "lows": LOWS, "highs": HIGHS},
        "n_values": len(VALUES), "offsets": OFFSETS, "ladder": LADDER,
        "reserved": sorted(RESERVED), "thr_range": [THR_MIN, THR_MAX],
        "universe_files": [f"{d}/{f}" for d, f in UNIVERSE],
        "counts": {"rows": len(rows)},
        "balance": {"true": yes, "false": len(rows) - yes},
        "substitutions": subs,
        "prior_numeric_universe_disjoint": True,
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "rows_sha256": hashlib.sha256((out / "rows.jsonl").read_bytes()).hexdigest(),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True),
                                       encoding="utf-8")
    print(f"rows={len(rows)} balance={yes}/{len(rows) - yes} values={len(VALUES)} subs={len(subs)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
