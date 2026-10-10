"""E24 fitting microset: hand-computed labels + generator/verifier agreement.

The HAND_CHECKS literals are computed by hand from the protocol text
("at least" = >=), independently of both the generator and the verifier.
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
DATA = ROOT / "data/e24-fit"

spec = importlib.util.spec_from_file_location("gen_e24_fit", ROOT / "scripts/gen_e24_fit.py")
gen = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gen)
spec_v = importlib.util.spec_from_file_location("verify_e24_fit", ROOT / "scripts/verify_e24_fit.py")
ver = importlib.util.module_from_spec(spec_v)
spec_v.loader.exec_module(ver)

# (value, threshold, expected answer) — hand-derived from "at least" = >=
HAND_CHECKS = [
    (59.5, 60.0, False), (59.9, 60.0, False), (59.1, 60.0, False),
    (60.0, 60.0, True),   # exact boundary satisfies "at least"
    (60.1, 60.0, True), (60.5, 60.0, True),
    (19.3, 60.0, False), (120.6, 60.0, True), (7.5, 60.0, False), (122.4, 60.0, True),
    (44.9, 45.0, False), (45.0, 45.0, True), (45.4, 45.0, True),
    (74.9, 75.0, False), (75.0, 75.0, True),
    (89.9, 90.0, False), (90.0, 90.0, True), (90.5, 90.0, True),
]


@pytest.mark.parametrize("value,threshold,expected", HAND_CHECKS)
def test_hand_computed_labels(value, threshold, expected):
    assert gen.compare(value, threshold) is expected
    row = gen.make_row(999, "handcheck", value, threshold)
    assert row["target_option"] == ("true" if expected else "false")
    assert ver.gold_from_text(row)[0] is expected


def test_hand_check_values_exist_in_generated_data():
    rows = [json.loads(l) for l in (DATA / "train_fit_rows.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    by_value = {r["value"]: r for r in rows}
    for value, threshold, expected in HAND_CHECKS:
        if threshold == 60.0 and value in by_value:
            assert by_value[value]["target_option"] == ("true" if expected else "false")


def test_generator_builds_frozen_counts_and_balance(tmp_path):
    monkey = tmp_path / "root"
    monkey.mkdir()
    saved = gen.ROOT
    gen.ROOT = monkey
    try:
        assert gen.main() == 0
        out = monkey / "data/e24-fit"
        counts = {"train_fit_rows.jsonl": 48, "dev_fit_rows.jsonl": 32,
                  "eval_newvalues_rows.jsonl": 64, "eval_newthreshold_rows.jsonl": 48,
                  "eval_newfield_rows.jsonl": 64}
        for fname, n in counts.items():
            rows = [json.loads(l) for l in (out / fname).read_text(encoding="utf-8").splitlines() if l.strip()]
            assert len(rows) == n
            yes = sum(r["target_option"] == "true" for r in rows)
            assert yes == n // 2, fname
    finally:
        gen.ROOT = saved


def test_pair_members_stay_in_the_same_split_and_invert():
    rows = [json.loads(l) for l in (DATA / "train_fit_rows.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    pairs = {}
    for r in rows:
        if r["pair"]:
            pairs.setdefault(r["pair"], []).append(r)
    assert len(pairs) == 16
    for members in pairs.values():
        lo = next(m for m in members if m["variant"] == "pair_lo")
        hi = next(m for m in members if m["variant"] == "pair_hi")
        assert lo["target_option"] == "false" and hi["target_option"] == "true"
        assert lo["value"] < 60.0 < hi["value"]
        # ONLY the value (and its derived answer) differs between members
        def strip(r):
            state = json.loads(r["state"])
            state[r["feature"]] = "<V>"
            return {k: v for k, v in r.items()
                    if k not in ("id", "group_id", "value", "variant", "state", "target_option")} | {"state": json.dumps(state, sort_keys=True)}
        assert strip(lo) == strip(hi)
