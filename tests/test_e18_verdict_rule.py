"""Verdict-rule regressions for the E18 analyzer (review errata #2).

The frozen rule counts ties as candidate-not-worse (>=) and the negative
branch fires when control wins >=2/3 OR KL worsens in >=2/3 pairs.
"""

import importlib.util
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    "e18_report", Path(__file__).resolve().parents[1] / "scripts" / "e18_supervision_report.py")
e18 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(e18)


def test_improvement_requires_kl_consistency():
    assert e18.frozen_verdict(2, 1, 2, -0.01, 1, 1, 1) == "improvement"
    assert e18.frozen_verdict(3, 0, 3, -0.02, 0, 0, 0) == "improvement"


def test_tie_counts_as_not_worse_not_as_two_wins():
    # 1 win + 1 tie + 1 loss -> wins=2 by >=, but KL worsens 2/3 -> negative
    assert e18.frozen_verdict(2, 1, 1, 0.005, 1, 1, 0) == "negative"


def test_negative_branch_matches_protocol_wording():
    assert e18.frozen_verdict(2, 1, 1, -0.01, 0, 0, 0) == "negative"      # KL worsens 2/3
    assert e18.frozen_verdict(1, 2, 3, -0.05, 0, 0, 0) == "negative"      # control wins 2/3
    assert e18.frozen_verdict(0, 3, 0, 0.02, 2, 2, 2) == "negative"       # both


def test_inconclusive_middle_ground():
    # wins>=2 and kl_wins>=2 but mean_kl positive, no uniform worsening
    assert e18.frozen_verdict(2, 1, 2, 0.001, 1, 1, 1) == "inconclusive"
    # 1 win + 2 ties: wins=2, kl improves 3/3 but a prob metric worsens 2/3
    assert e18.frozen_verdict(2, 0, 3, -0.01, 2, 1, 0) == "inconclusive"
