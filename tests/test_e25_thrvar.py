"""E25 threshold-variable data: hand-computed labels + pair integrity.

HAND_CHECKS are computed by hand from the protocol text ("at least" = >=),
independently of the generator and the verifier. Boundary semantics: an exact
reading equal to the threshold is Yes.
"""

import importlib.util
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
DATA = ROOT / "data/e25-thrvar"

spec = importlib.util.spec_from_file_location("gen_e25_thrvar", ROOT / "scripts/gen_e25_thrvar.py")
gen = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gen)
spec_v = importlib.util.spec_from_file_location("verify_e25_thrvar", ROOT / "scripts/verify_e25_thrvar.py")
ver = importlib.util.module_from_spec(spec_v)
spec_v.loader.exec_module(ver)

# (value, threshold, expected answer) — hand-derived
HAND_CHECKS = [
    (60.0, 60.0, True), (59.6, 60.0, False), (60.4, 60.0, True),   # dev near-boundary
    (40.0, 40.0, True), (39.6, 40.0, False),
    (65.0, 65.0, True), (64.4, 65.0, False), (65.6, 65.0, True),   # reserved threshold
    (100.0, 100.0, True), (99.4, 100.0, False), (100.6, 100.0, True),
    (30.0, 30.0, True), (29.4, 30.0, False),
    (42.5, 40.0, True), (42.5, 50.0, False),                       # the threshold-only pair idea
    (86.9, 90.0, False), (92.8, 90.0, True),
    (21.4, 40.0, False), (103.7, 90.0, True),
]


@pytest.mark.parametrize("value,threshold,expected", HAND_CHECKS)
def test_hand_computed_labels(value, threshold, expected):
    question = f"Is the motor temperature reading at least {threshold:g} degrees Celsius?"
    state = json.dumps({"equipment_id": "unit-aa", "motor_temp_c": value, "site": "site-alpha"},
                       sort_keys=True)
    assert (value >= threshold) is expected
    row = {"question": question, "state": state,
           "options": [{"id": "false", "description": "No"}, {"id": "true", "description": "Yes"}]}
    gold, v, t = ver.gold_from_text({**row, "target_option": "true"})
    assert gold is expected and v == value and t == threshold


def test_generated_files_carry_the_hand_checked_answers():
    rows = {r["value"]: r for r in
            [json.loads(l) for l in (DATA / "train_thr_rows.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]}
    for value, threshold, expected in HAND_CHECKS:
        if value in rows and rows[value]["threshold"] == threshold:
            assert rows[value]["target_option"] == ("true" if expected else "false"), (value, threshold)


def test_threshold_only_pairs_change_exactly_one_thing():
    for fname, n_pairs in (("train_thr_rows.jsonl", 15), ("dev_thr_rows.jsonl", 10),
                           ("test_known_thr_rows.jsonl", 10), ("test_new_thr_rows.jsonl", 6)):
        rows = [json.loads(l) for l in (DATA / fname).read_text(encoding="utf-8").splitlines() if l.strip()]
        pairs = defaultdict(list)
        for r in rows:
            if r["pair"]:
                pairs[r["pair"]].append(r)
        assert len(pairs) == n_pairs, fname
        for pid, members in pairs.items():
            lo = next(m for m in members if m["variant"] == "pair_thrlo")
            hi = next(m for m in members if m["variant"] == "pair_thrhi")
            assert lo["state"] == hi["state"], (fname, pid)          # identical state
            assert lo["threshold"] < lo["value"] < hi["threshold"]   # straddles, inverts
            assert lo["target_option"] == "true" and hi["target_option"] == "false"
            assert lo["group_id"] == hi["group_id"] == pid           # grouped by base case


def test_per_threshold_balance_is_exact_where_declared():
    for fname, exact in (("train_thr_rows.jsonl", True), ("dev_thr_rows.jsonl", True),
                         ("test_known_thr_rows.jsonl", True)):
        rows = [json.loads(l) for l in (DATA / fname).read_text(encoding="utf-8").splitlines() if l.strip()]
        by_thr = defaultdict(Counter)
        for r in rows:
            by_thr[r["threshold"]][r["target_option"]] += 1
        for t, c in by_thr.items():
            assert c["true"] == c["false"], (fname, t, dict(c))


def test_question_ignoring_rules_cannot_solve_the_pairs():
    for fname in ("train_thr_rows.jsonl", "dev_thr_rows.jsonl",
                  "test_known_thr_rows.jsonl", "test_new_thr_rows.jsonl"):
        rows = [json.loads(l) for l in (DATA / fname).read_text(encoding="utf-8").splitlines() if l.strip()]
        pair_rows = [r for r in rows if r["pair"]]
        gold = [r["target_option"] == "true" for r in pair_rows]
        yes_acc = sum(gold) / len(gold)
        ge60 = [json.loads(r["state"])["motor_temp_c"] >= 60 for r in pair_rows]
        ge60_acc = sum(a == g for a, g in zip(ge60, gold)) / len(gold)
        assert yes_acc <= 0.5 + 1e-9 and ge60_acc <= 0.5 + 1e-9, fname


def test_reserved_thresholds_absent_from_train_and_dev():
    reserved = {45.0, 55.0, 65.0, 75.0, 85.0, 30.0, 100.0}
    for fname in ("train_thr_rows.jsonl", "dev_thr_rows.jsonl"):
        rows = [json.loads(l) for l in (DATA / fname).read_text(encoding="utf-8").splitlines() if l.strip()]
        assert not reserved & {r["threshold"] for r in rows}, fname


def test_generator_rebuild_is_deterministic(tmp_path):
    saved = gen.ROOT
    gen.ROOT = tmp_path
    try:
        assert gen.main() == 0
        for fname in ("train_thr_rows.jsonl", "dev_thr_rows.jsonl",
                      "test_known_thr_rows.jsonl", "test_new_thr_rows.jsonl", "manifest.json"):
            assert (tmp_path / "data/e25-thrvar" / fname).read_bytes() == \
                (DATA / fname).read_bytes(), fname
    finally:
        gen.ROOT = saved
