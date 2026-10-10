"""Meaningful E25 reporting regressions, independent of local datasets/weights."""
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


report = load("e25_report")
verifier = load("verify_e25_thrvar")


def fixtures():
    meta, pred = {}, []
    for pid, ranges, scores in [("pure", ("interp", "interp"), (1, 1)),
                                ("mixed_ok", ("extrap", "interp"), (1, 1)),
                                ("mixed_fail", ("interp", "extrap"), (0, 1))]:
        for side, (domain, correct) in enumerate(zip(ranges, scores)):
            identity = f"{pid}-{side}"
            meta[identity] = {"pair": pid, "variant": "pair_thrlo" if side == 0 else "pair_thrhi",
                              "thr_range": domain}
            pred.append({"id": identity, "correct": correct})
    return meta, pred


def test_mixed_pairs_are_actual_successes_not_incomplete_pair_count():
    meta, pred = fixtures()
    stats = report.pairs_by_range(pred, meta)
    assert stats["mixed"] == {"n": 2, "both_correct": 1, "rate": 0.5}
    assert stats["interp"]["rate"] == 1
    assert stats["extrap"] == {"n": 0, "both_correct": 0, "rate": None}


def test_incomplete_or_duplicate_pairs_fail():
    meta, pred = fixtures()
    for invalid in (pred[:-1], pred + [pred[0]]):
        with pytest.raises(ValueError):
            report.pairs_by_range(invalid, meta)


def test_perfect_accuracy_does_not_replace_missing_registered_pairs():
    stats = {"accuracy": 1.0, "pairs_both": None, "pairs_n": 0}
    assert report.registered_verdict(stats) == "not_assessable_missing_pairs"
    assert report.registered_verdict({"accuracy": 1.0, "pairs_both": 0.5, "pairs_n": 2}) == "failed"


def test_same_decade_is_not_greater_than_comparison():
    assert report.same_decade_reference({"value": 54.4, "threshold": 55}) is True
    assert report.same_decade_reference({"value": 62.2, "threshold": 55}) is False


def test_fixed_rule_has_zero_both_correct_on_opposite_pairs():
    state = json.dumps({"motor_temp_c": 65})
    rows = [{"pair": "p", "variant": variant, "state": state, "target_option": gold}
            for variant, gold in [("pair_thrlo", "true"), ("pair_thrhi", "false")]]
    caps = verifier.pair_accuracy_cap(rows)
    assert caps["pairs"]["fixed_rule_both_correct"] == 0
    assert caps["pairs"]["gold_opposite_pairs"] == 1


def test_validate_eval_rejects_missing_duplicate_and_wrong_gold():
    meta = {"a": {"group_id": "g", "options": [{"id": "false"}, {"id": "true"}],
                   "target_option": "true"}}
    row = {"id": "a", "group_id": "g", "gold_idx": 1, "pred_idx": 1, "correct": 1}
    report.validate_eval([row], meta, {"a"})
    for rows in ([], [row, row], [row | {"gold_idx": 0}], [row | {"correct": 0}]):
        with pytest.raises(ValueError):
            report.validate_eval(rows, meta, {"a"})


def test_all_correct_bootstrap_is_markedly_degenerate():
    rows = [{"group_id": str(i // 2), "correct": 1} for i in range(8)]
    assert report.grouped_bootstrap_ci(rows, draws=100) == [1.0, 1.0]
