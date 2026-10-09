"""E19 generator tests: boundary golds, verifier agreement, structural invariants."""

import importlib.util
import json
import random
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


gen = load("gen_e19_rulevar")
ver = load("verify_e19_labels")


def facts(family, **over):
    rng = random.Random(7)
    f = gen.FAMILIES[family]["sample"](rng)["facts"]
    f.update(over)
    return f


# ------------------------------------------------------------- boundaries
def test_f1_boundaries():
    f = facts("insurance_claims_triage", incident="collision", documentation_complete=True,
              claim_amount_eur=3000, coverage_limit_eur=12000)
    assert gen.f1_action_gold(f, 3000) == "investigate"      # amount == thr -> at least
    f2 = dict(f, claim_amount_eur=2999.99)
    assert gen.f1_action_gold(f2, 3000) == "standard"
    assert gen.f1_action_gold(dict(f, claim_amount_eur=12000.01), 3000) == "underwriting"  # just over limit
    assert gen.f1_risk_gold(dict(f, claim_amount_eur=2000)) == 1       # >= r1
    assert gen.f1_risk_gold(dict(f, claim_amount_eur=1999.99)) == 0    # < r1 (doc complete)
    assert gen.f1_risk_gold(dict(f, police_flag=True, claim_amount_eur=500)) == 3
    assert gen.f1_urgency_gold(dict(f, report_delay_hours=48)) == 1    # 25..48 without police
    assert gen.f1_urgency_gold(dict(f, report_delay_hours=49)) == 2
    assert gen.f1_urgency_gold(dict(f, report_delay_hours=48, police_flag=True)) == 2


def test_f2_boundaries():
    f = facts("facility_maintenance", bearing_temp_c=90.0, alarm_active=True, safety_class="A")
    assert gen.f2_risk_gold(f) == 3                            # hot at threshold + alarm + A
    assert gen.f2_risk_gold(dict(f, bearing_temp_c=89.9)) == 1
    f2 = dict(f, alarm_active=False, safety_class="B", runtime_hours_since_service=2001,
              redundancy_available=False)
    assert gen.f2_action_gold(f2) == "schedule"                # runtime > thr, no alarm
    assert gen.f2_action_gold(dict(f2, runtime_hours_since_service=1999)) == "monitor"
    assert gen.f2_urgency_gold(dict(f, equipment_age_years=10)) == 2  # age 5..10 with alarm


def test_f3_f4_boundaries():
    f = facts("it_access_review")
    assert gen.f3_urgency_gold(dict(f, temporary_grant_days=30, mfa_used=True)) == 0
    assert gen.f3_urgency_gold(dict(f, temporary_grant_days=31, mfa_used=True)) == 1
    assert gen.f3_urgency_gold(dict(f, temporary_grant_days=90, mfa_used=False)) == 2
    assert gen.f3_urgency_gold(dict(f, temporary_grant_days=91, mfa_used=False)) == 3
    g = facts("records_retention", legal_hold_active=False, audit_flagged=False)
    g4 = dict(g, contains_pii=True, record_age_years=8, retention_limit_years=7)
    assert gen.f4_risk_gold(g4) == 3                           # pii + over
    assert gen.f4_risk_gold(dict(g4, record_age_years=7)) == 1  # pii, within, no hold/flag


# ------------------------------------------------------- verifier agreement
@pytest.mark.parametrize("family", list(gen.SPLIT_PLAN))
def test_verifier_agrees_on_generated_cases(family):
    rng = random.Random(hash(family) % 10_000)
    for idx in range(3):
        case = gen.build_case(family, rng, idx)
        for slot in case["questions"]:
            q = case["questions"][slot]
            m = case["meta"][slot]
            options = [{"id": k, "description": v} for k, v in q["criteria"].items()]
            target = options[case["gold"][slot]]["id"]
            row = {"id": f"{case['id']}:{slot}", "domain": family, "state": m["state"],
                   "question": q["instructions"], "options": options,
                   "question_type": q["type"], "target_option": target}
            assert ver.verify_row(row), f"{row['id']} disagrees"


def test_flip_and_irrelevant_invariants():
    rng = random.Random(11)
    case = gen.build_case("insurance_claims_triage", rng, 0)
    assert case["gold"]["action_flip"] != case["gold"]["action"] or \
        [c for c in case["questions"]["action_flip"]["criteria"].values()] != \
        [c for c in case["questions"]["action"]["criteria"].values()]
    assert case["gold"]["risk_flip"] != case["gold"]["risk"]
    assert case["meta"]["action_flip"]["state"] == case["meta"]["action"]["state"]  # rule flip: same state
    assert case["meta"]["risk_flip"]["state"] != case["meta"]["risk"]["state"]      # evidence flip: state changed
    assert case["gold"]["needs_review_irr"] == case["gold"]["needs_review"]


# ------------------------------------------------------------ structural
def test_corpus_files_structure():
    data = ROOT / "data/e19-rulevar"
    manifest = json.loads((data / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["seed"] == gen.SEED
    assert manifest["counts"] == {"train_ctrl": 1650, "train_cand": 1650, "dev": 330,
                                  "test": 750, "test_cases": 150}
    splits = {}
    for name, split in (("train_ctrl", "train"), ("train_cand", "train"), ("dev", "dev"), ("test", "test")):
        rows = [json.loads(l) for l in (data / f"{name}_rows.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
        for r in rows:
            assert r["question_type"] in ("choice", "noul", "score")
            assert len(r["options"]) == (2 if r["question_type"] == "noul" else 4)
            s = splits.setdefault(r["group_id"], set())
            assert s <= {split}, "case crosses splits"
            s.add(split)
    # same underlying train cases in both arms; action base shared
    ctrl = {json.loads(l)["id"] for l in (data / "train_ctrl_rows.jsonl").read_text(encoding="utf-8").splitlines()}
    cand = {json.loads(l)["id"] for l in (data / "train_cand_rows.jsonl").read_text(encoding="utf-8").splitlines()}
    shared = ctrl & cand
    assert all(i.endswith(":action") for i in shared) and len(shared) == 330
    # reserved family only in test
    for line in (data / "train_ctrl_rows.jsonl").read_text(encoding="utf-8").splitlines():
        assert "records_retention" not in line
    for line in (data / "dev_rows.jsonl").read_text(encoding="utf-8").splitlines():
        assert "records_retention" not in line
