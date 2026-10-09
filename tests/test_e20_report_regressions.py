"""E20 decision guards: synthetic aggregates, no models or downloads."""
import copy

import pytest

from scripts.e20_report import frozen_decision, macro_acc, validate_aligned_rows


def decision(types=None, known=None, deltas=None):
    return frozen_decision(
        deltas if deltas is not None else {42: 0.06, 43: 0.07, 44: 0.05},
        types if types is not None else {"choice": 0.03, "noul": 0.1, "score": 0.02},
        known if known is not None else {"dev_new": 0.0, "e19_diagnostic": 0.0})


def test_one_improving_type_cannot_hide_another_type_regression():
    result = decision(types={"choice": -0.1, "noul": 0.2, "score": 0.1})
    assert result["verdict"] == "inconclusive"
    assert result["checks"]["type_regressions_within_2pp"]["choice"] is False


def test_success_requires_all_frozen_checks():
    assert decision()["verdict"] == "improvement"
    assert decision(known={"dev_new": -0.03, "e19_diagnostic": 0.0})["verdict"] == "inconclusive"
    result = decision(known={"dev_new": 0.0, "e19_diagnostic": None})
    assert result["verdict"] == "inconclusive"
    assert result["missing_checks"] == ["e19_diagnostic"]


def test_exact_regression_boundary_is_permitted():
    assert decision(types={"choice": -0.02, "noul": 0.2, "score": 0.1})["verdict"] == "improvement"


def test_negative_rule_includes_control_ties():
    assert decision(deltas={42: 0.0, 43: -0.01, 44: 0.04})["verdict"] == "negative"


def test_missing_family_cannot_silently_change_macro_target():
    with pytest.raises(ValueError, match="coverage"):
        macro_acc([{"family": "loan_underwriting", "correct": 1}])


@pytest.mark.parametrize("corruption", ["duplicate", "missing", "gold", "group"])
def test_pairing_rejects_corrupted_inputs(corruption):
    a = [dict(id=str(i), group_id="g", gold_idx=0, type="noul") for i in range(2)]
    b = copy.deepcopy(a)
    if corruption == "duplicate":
        b[1] = copy.deepcopy(b[0])
    elif corruption == "missing":
        b.pop()
    else:
        b[0]["gold_idx" if corruption == "gold" else "group_id"] = "changed"
    with pytest.raises(ValueError):
        validate_aligned_rows({"ctrl": a, "cand": b})
