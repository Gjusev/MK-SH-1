"""E19 statistical regressions; no model weights or generated corpus required."""
import copy

import pytest

from scripts.e19_rulevar_report import (
    choice_noul_delta,
    pooled_case_deltas,
    validate_aligned_rows,
)


def row(identifier, group, correct):
    return dict(id=identifier, group_id=group, family="f", type="choice",
                gold_idx=0, variant="base", pair=None, correct=correct)


def test_both_seed_pairs_contribute_and_seed_order_does_not_matter():
    control = [row("a", "g1", 0), row("b", "g2", 1)]
    candidate_seed1 = [row("a", "g1", 1), row("b", "g2", 1)]
    candidate_seed2 = [row("a", "g1", 0), row("b", "g2", 0)]
    pairs = [(control, candidate_seed1), (control, candidate_seed2)]
    expected = {"g1": 0.5, "g2": -0.5}
    assert pooled_case_deltas(pairs) == expected
    assert pooled_case_deltas(pairs[::-1]) == expected


def test_regression_guard_uses_actual_subset_denominator():
    # A one percentage point improvement over 450 rows remains one point.
    a = {"by_type": {"choice": {"n": 300, "accuracy": 0.4},
                      "noul": {"n": 150, "accuracy": 0.5}}}
    b = {"by_type": {"choice": {"n": 300, "accuracy": 0.41},
                      "noul": {"n": 150, "accuracy": 0.51}}}
    assert choice_noul_delta(a, b) == pytest.approx(0.01)


@pytest.mark.parametrize("corruption", ["duplicate", "missing", "group", "gold", "type"])
def test_pairing_rejects_corrupted_inputs(corruption):
    a = [row("a", "g1", 1), row("b", "g2", 0)]
    b = copy.deepcopy(a)
    if corruption == "duplicate":
        b[1] = copy.deepcopy(b[0])
    elif corruption == "missing":
        b.pop()
    else:
        field = {"group": "group_id", "gold": "gold_idx", "type": "type"}[corruption]
        b[0][field] = "different"
    with pytest.raises(ValueError):
        validate_aligned_rows({"a": a, "b": b})
