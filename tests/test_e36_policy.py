"""E36 tests: calibration/policy semantics, generator invariants, export package.

No model load here: the heavy fresh-process package verification lives in
scripts/export_mksh_v12.py --verify (one-time, isolated interpreter).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/e36-calib"
LOGITS = ROOT / "runs/e36-calib/logits"
EXPORT = ROOT / "runs/hf-export/mksh-1-v1.2"


def probs_at(logits, t):
    scaled = [l / t for l in logits]
    m = max(scaled)
    exps = [math.exp(s - m) for s in scaled]
    z = sum(exps)
    return [e / z for e in exps]


def test_temperature_preserves_argmax_moves_confidence():
    """Monotone transform: same choice, different confidence (the v1.2 note)."""
    logits = [2.0, -1.5]
    p1, pt = probs_at(logits, 1.0), probs_at(logits, 7.653133)
    assert p1.index(max(p1)) == pt.index(max(pt))
    assert abs(max(p1) - max(pt)) > 0.1  # saturation softened
    assert sum(pt) == 1.0


def test_policy_abstention_semantics():
    """Real code path: abstain -> choice null + reason_code; accept otherwise."""
    from mokka_decisions.contracts import DecisionCase, Option
    from mokka_decisions.policy import apply_policy

    case = DecisionCase(
        id="t1", group_id="t1", source="test", language="en", domain="test",
        state="{}", question="q?",
        options=(Option(id="false", description="No"), Option(id="true", description="Yes")),
        target_option="true",
    )
    low = apply_policy(case, {"false": 0.25, "true": 0.75}, threshold=0.79)
    assert low.abstain and low.choice is None and low.reason_code == "below_validated_threshold"
    high = apply_policy(case, {"false": 0.05, "true": 0.95}, threshold=0.79)
    assert not high.abstain and high.choice == "true" and high.reason_code == "accepted"


def test_generator_invariants():
    """480 rows, 60 values, exact 4/4 balance, disjoint from the prior universe."""
    rows = [json.loads(l) for l in (DATA / "rows.jsonl").read_text(encoding="utf-8")
            .splitlines() if l.strip()]
    assert len(rows) == 480
    from collections import Counter, defaultdict

    by_val = defaultdict(Counter)
    for r in rows:
        by_val[r["value"]][r["target_option"]] += 1
    assert len(by_val) == 60
    assert all(c["true"] == 4 and c["false"] == 4 for c in by_val.values())
    assert Counter(r["target_option"] for r in rows) == {"true": 240, "false": 240}

    prior = set()
    universe = [
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
    for d, f in universe:
        for l in (ROOT / "data" / d / f).read_text(encoding="utf-8").splitlines():
            if l.strip():
                rr = json.loads(l)
                prior.add(round(rr["value"], 2))
                prior.add(round(rr["threshold"], 2))
    used = {round(r["value"], 2) for r in rows} | {round(r["threshold"], 2) for r in rows}
    assert not (used & prior)


def test_inference_policy_json_structure():
    if not (EXPORT / "inference-policy.json").exists():
        import pytest

        pytest.skip("v1.2 export not assembled locally")
    p = json.loads((EXPORT / "inference-policy.json").read_text(encoding="utf-8"))
    assert p["format_version"] == 1
    assert p["model"]["id"] == "Gjusev/MK-SH-1"
    assert p["model"]["weights_sha256"].startswith("ea66cdde")
    assert "no retraining" in p["model"]["weights_note"]
    assert p["temperature"]["value"] > 1.0  # fitted, not defaulted
    assert p["abstention"]["enabled"] and p["abstention"]["threshold"] is not None
    assert p["abstention"]["reason_code"] == "below_validated_threshold"
    assert p["provenance"]["experiment"] == "E36"


def test_export_policy_consistent_with_frozen_logits():
    """Choice invariance + abstention consistency, replayed on frozen logits."""
    if not (EXPORT / "inference-policy.json").exists() or not (LOGITS / "split.jsonl").exists():
        import pytest

        pytest.skip("v1.2 export / frozen logits not present locally")
    p = json.loads((EXPORT / "inference-policy.json").read_text(encoding="utf-8"))
    t, tau = p["temperature"]["value"], p["abstention"]["threshold"]
    rows = [json.loads(l) for l in (LOGITS / "split.jsonl").read_text(encoding="utf-8")
            .splitlines() if l.strip()]
    n_abst = 0
    for r in rows:
        raw, cal = probs_at(r["logits"], 1.0), probs_at(r["logits"], t)
        assert max(range(len(raw)), key=lambda i: raw[i]) == max(range(len(cal)), key=lambda i: cal[i])
        if max(cal) < tau:
            n_abst += 1
    assert n_abst == 48  # 480 * (1 - 0.900) coverage at tau (frozen report: 432 accepted)


def test_clopper_pearson_matches_reference():
    """Exact CP bounds (reference values from scipy.stats.beta, 95 % two-sided)."""
    import sys
    sys.path.insert(0, str(ROOT / "scripts"))
    from e36_heldout_policy import clopper_pearson

    for k, n, lo, hi in [(21, 432, 0.030339, 0.073350), (16, 194, 0.047876, 0.130478),
                         (1, 47, 0.000539, 0.112938), (0, 10, 0.0, 0.308497)]:
        got_lo, got_hi = clopper_pearson(k, n)
        assert abs(got_lo - lo) < 1e-5 and abs(got_hi - hi) < 1e-5
