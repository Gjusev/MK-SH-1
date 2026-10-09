"""E20 FASE 2/3 generator: composition-paired vs coverage data (frozen design).

Implements docs/internal/E20_PROTOCOL_2026-10-09.md. Reuses the E19 family
machinery (scripts/gen_e19_rulevar.py) for the three TRAIN families with NEW
cases and balanced golds, adds flip renderings for every slot (the treatment),
and defines two NEW sealed families (loan_underwriting, chemical_batch_release)
with explicit values in every rule text.

Outputs (typed-row schema + engine cases for the sealed test):
  data/e20-contrast/train_ctrl_rows.jsonl   330 cases x 5 base questions
  data/e20-contrast/train_cand_rows.jsonl   165 cases x 10 (base+flip, all slots)
  data/e20-contrast/dev_rows.jsonl          66 cases x 5 (train families, paired structure)
  data/e20-contrast/test_rows.jsonl         150 cases x 5 (2 sealed families, paired)
  data/e20-contrast/manifest.json

Determinism: pinned seed; splits by family/case BEFORE variant generation.
An independent verifier (scripts/verify_e20_labels.py) must accept 100%.
"""

from __future__ import annotations

import hashlib
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import gen_e19_rulevar as e19  # noqa: E402

SEED = 20261011
TRAIN_FAMILIES = ("insurance_claims_triage", "facility_maintenance", "it_access_review")
SEALED_FAMILIES = ("loan_underwriting", "chemical_batch_release")
FORBIDDEN = ("correct", "answer", "gold", "label", "solution")


def stable_shuffle(seq, key):
    order = sorted(range(len(seq)), key=lambda i: hashlib.sha256(f"{key}|{i}".encode()).hexdigest())
    return [seq[i] for i in order]


# ---------------------------------------------------------------- slot flips for the E19 train families
def e20_action_flip(family, f, base):
    """e19.action_flip with a local fallback for f3 deny cases whose facts the
    polarity adjustment moved off the sampler manifold (deny without high
    sensitivity): loosen GRANT instead, by sensitivity range or by role."""
    try:
        return e19.action_flip(family, f, base)
    except AssertionError:
        assert family == "it_access_review" and base == "deny"
        role = f["requester_role"]
        if f["resource_sensitivity"] == "high":
            raise
        if f["security_training_current"]:

            def rules_loosen_sens(x):
                base_rules = e19.f3_action_rules(x)
                for o in base_rules:
                    if o["key"] == "grant":
                        o["desc"] = o["desc"].replace("sensitivity is low", "sensitivity is low or medium")
                return base_rules
            return dict(f), rules_loosen_sens, "grant"

        def rules_loosen_role(x):
            base_rules = e19.f3_action_rules(x)
            for o in base_rules:
                if o["key"] == "grant":
                    o["desc"] = o["desc"].replace("and security training is current",
                                                    f"and the requester role is {role}")
            return base_rules
        return dict(f), rules_loosen_role, "grant"


def noul_flip(family, f):
    """Fact flip on the review-deciding condition; returns (new_facts, new_gold_bool, kind)."""
    import copy
    nf = copy.deepcopy(f)
    if family == "insurance_claims_triage":
        # decisive trio: delay>48 | amount>=10000 | !doc
        if nf["documentation_complete"]:
            nf["report_delay_hours"] = 60 if nf["report_delay_hours"] <= 48 else 10
        else:
            nf["documentation_complete"] = True
    elif family == "facility_maintenance":
        if nf["safety_class"] == "A" and not nf["redundancy_available"]:
            nf["redundancy_available"] = True
        else:
            nf["bearing_temp_c"] = round(95.0 if nf["bearing_temp_c"] < 90.0 else 50.0, 1)
            nf["alarm_active"] = True if nf["bearing_temp_c"] >= 90.0 else False
            if not nf["alarm_active"]:
                nf["safety_class"] = "A"
                nf["redundancy_available"] = False
    else:
        if nf["data_class"] in ("restricted", "confidential"):
            nf["data_class"] = "internal"
        elif nf["privilege_breadth"] > 4:
            nf["privilege_breadth"] = 2
        else:
            nf["privilege_breadth"] = 5  # internal + breadth>4 -> review true; keeps other slots intact
    return nf


def outcome_flip(family, f):
    import copy
    nf = copy.deepcopy(f)
    if family == "insurance_claims_triage":
        if nf["claim_amount_eur"] > nf["coverage_limit_eur"]:
            nf["claim_amount_eur"] = round(nf["coverage_limit_eur"] * 0.5, 2)
        elif nf["incident"] == "glass":
            nf["incident"] = "collision"
        else:
            nf["documentation_complete"] = not nf["documentation_complete"]
    elif family == "facility_maintenance":
        if nf["bearing_temp_c"] >= nf["temp_threshold_c"]:
            nf["bearing_temp_c"] = round(nf["temp_threshold_c"] - 5.0, 1)
        else:
            nf["alarm_active"] = not nf["alarm_active"]
    else:
        nf["justification_documented"] = not nf["justification_documented"]
    return nf


def urgency_flip(family, f):
    import copy
    nf = copy.deepcopy(f)
    if family == "insurance_claims_triage":
        nf["report_delay_hours"] = 70 if nf["report_delay_hours"] <= 48 else 20
    elif family == "facility_maintenance":
        nf["equipment_age_years"] = 15 if nf["equipment_age_years"] <= 10 else 4
    else:
        nf["temporary_grant_days"] = 120 if nf["temporary_grant_days"] <= 90 else 30
    return nf


# ---------------------------------------------------------------- sealed family: loan_underwriting
def s1_loan(rng):
    limits = {"personal": 15000, "auto": 25000, "mortgage": 200000}
    product = rng.choice(list(limits))
    amount = round(rng.uniform(2000, 250000), 2)
    dti = round(rng.uniform(5, 60), 1)  # debt-to-income %
    collateral = rng.random() < 0.5
    insured = rng.random() < 0.5
    target = rng.choice(["approve", "memo", "conditional", "decline"])
    if target == "approve":
        amount = round(rng.uniform(2000, min(limits[product], 100000) * 0.9), 2)
        dti = round(rng.uniform(5, 35), 1)
        collateral, insured = True, True
    elif target == "memo":
        amount = round(rng.uniform(2000, min(limits[product], 100000) * 0.9), 2)
        dti = round(rng.uniform(5, 35), 1)
        collateral, insured = False, True
    elif target == "conditional":
        amount = round(rng.uniform(2000, min(limits[product], 100000) * 0.9), 2)
        dti = round(rng.uniform(36, 50), 1)
    else:
        dti = round(rng.uniform(36, 60), 1)
        if dti <= 50:
            amount = round(min(limits[product], 100000) * 1.4, 2)  # over limit
    return {"family": "loan_underwriting", "facts": {
        "product": product, "requested_amount_eur": amount, "product_limit_eur": limits[product],
        "debt_to_income_pct": dti, "collateral_pledged": collateral, "borrower_insured": insured,
        "dti_threshold_pct": 35.0, "dti_high_pct": 50.0}}


def s1_action_rules(f):
    lim, dti, dtih = f["product_limit_eur"], f["dti_threshold_pct"], f["dti_high_pct"]
    return [
        {"key": "approve", "desc": f"Approve the loan as requested: applies when the requested amount is within the {f['product']} limit of {lim:,.0f} EUR, the debt-to-income ratio is below {dti:.0f} percent, collateral is pledged, and the borrower carries life insurance."},
        {"key": "memo", "desc": f"Approve after a credit memo: applies when the requested amount is within the {f['product']} limit of {lim:,.0f} EUR, the debt-to-income ratio is below {dti:.0f} percent, collateral is not pledged, and the borrower carries life insurance."},
        {"key": "conditional", "desc": f"Approve with conditions: applies when the requested amount is within the {f['product']} limit of {lim:,.0f} EUR and the debt-to-income ratio is from {dti:.0f} to {dtih:.0f} percent."},
        {"key": "decline", "desc": f"Decline the application: applies in every other case, including amounts above the product limit of {lim:,.0f} EUR or a debt-to-income ratio above {dtih:.0f} percent."},
    ]


def s1_action_gold(f):
    within = f["requested_amount_eur"] <= f["product_limit_eur"]
    if not within or f["debt_to_income_pct"] > f["dti_high_pct"]:
        return "decline"
    if f["debt_to_income_pct"] > f["dti_threshold_pct"]:
        return "conditional"
    if f["collateral_pledged"] and f["borrower_insured"]:
        return "approve"
    if not f["collateral_pledged"] and f["borrower_insured"]:
        return "memo"
    return "decline"


def s1_review(f):
    return (f["requested_amount_eur"] > f["product_limit_eur"] * 0.8
            and f["debt_to_income_pct"] > f["dti_threshold_pct"])


def s1_review_rule():
    return ("manual underwriter review is required when the requested amount exceeds 80 percent of the "
            "product limit and the debt-to-income ratio is above 35.0 percent")


def s1_outcome_rules(f):
    return [
        {"key": "full_disbursement", "desc": "The loan will pay out in full at signing: applies to approved loans with collateral pledged."},
        {"key": "staged_disbursement", "desc": "The loan will pay out in stages: applies to conditional approvals whatever the collateral."},
        {"key": "insurance_first", "desc": "Payout waits for an insurance policy: applies to memo approvals, whatever the collateral."},
        {"key": "no_payout", "desc": "No payout will occur: applies to declined applications."},
    ]


def s1_outcome_gold(f):
    a = s1_action_gold(f)
    if a == "approve":
        return "full_disbursement"
    if a == "conditional":
        return "staged_disbursement"
    if a == "memo":
        return "insurance_first"
    return "no_payout"


def s1_risk_rubric():
    return [
        {"key": 0, "desc": "low: debt-to-income below 20.0 percent and collateral pledged"},
        {"key": 1, "desc": "moderate: debt-to-income below 20.0 percent without collateral, or from 20.0 to 35.0 percent with collateral"},
        {"key": 2, "desc": "high: debt-to-income from 20.0 to 35.0 percent without collateral, or above 35.0 percent with collateral"},
        {"key": 3, "desc": "critical: debt-to-income above 35.0 percent without collateral, or an amount above the product limit"},
    ]


def s1_risk_gold(f):
    if f["requested_amount_eur"] > f["product_limit_eur"]:
        return 3
    dti, col = f["debt_to_income_pct"], f["collateral_pledged"]
    if dti > 35.0:
        return 2 if col else 3
    if dti > 20.0:
        return 1 if not col else 1
    return 0 if col else 1


def s1_urgency_rubric():
    return [
        {"key": 0, "desc": "standard: the borrower is insured and the debt-to-income ratio is at most 25.0 percent"},
        {"key": 1, "desc": "prompt: the borrower is not insured with debt-to-income at most 35.0 percent, or insured with debt-to-income above 25.0 percent"},
        {"key": 2, "desc": "priority: the borrower is not insured with debt-to-income above 35.0 percent and collateral is pledged"},
        {"key": 3, "desc": "immediate: the borrower is not insured with debt-to-income above 35.0 percent without collateral"},
    ]


def s1_urgency_gold(f):
    dti, ins, col = f["debt_to_income_pct"], f["borrower_insured"], f["collateral_pledged"]
    if not ins and dti > 35.0:
        return 3 if not col else 2
    if (not ins and dti <= 35.0) or (ins and dti > 25.0):
        return 1
    return 0


# ---------------------------------------------------------------- sealed family: chemical_batch_release
def s2_chem(rng):
    temp = round(rng.uniform(15, 45), 1)
    ph = round(rng.uniform(4, 10), 2)
    impurity = round(rng.uniform(0.01, 1.2), 3)
    certified = rng.random() < 0.7
    vessel_ok = rng.random() < 0.8
    target = rng.choice(["release", "quarantine", "reprocess", "hold_lab"])
    if target == "release":
        temp, ph, impurity = round(rng.uniform(18, 28), 1), round(rng.uniform(6.5, 8), 2), round(rng.uniform(0.01, 0.2), 3)
    elif target == "quarantine":
        impurity = round(rng.uniform(0.6, 1.2), 3)
    elif target == "reprocess":
        ph = round(rng.uniform(4.2, 6.2), 2)
    return {"family": "chemical_batch_release", "facts": {
        "batch_temp_c": temp, "ph": ph, "impurity_pct": impurity,
        "operator_certified": certified, "vessel_integrity_ok": vessel_ok,
        "temp_max_c": 30.0, "ph_min": 6.5, "impurity_max_pct": 0.5}}


def s2_action_rules(f):
    tm, pm, im = f["temp_max_c"], f["ph_min"], f["impurity_max_pct"]
    return [
        {"key": "release", "desc": f"Release the batch to packaging: applies when the temperature is at or below {tm:.1f} degrees C, pH is at least {pm:.1f}, and impurities are at or below {im:.1f} percent."},
        {"key": "reprocess", "desc": f"Send the batch to reprocessing: applies when the temperature is at or below {tm:.1f} degrees C and pH is below {pm:.1f}, whatever the impurities."},
        {"key": "quarantine", "desc": f"Quarantine the batch: applies when impurities exceed {im:.1f} percent, whatever the temperature or pH."},
        {"key": "hold_lab", "desc": "Hold the batch for lab analysis: applies in every other case, including temperatures above the maximum."},
    ]


def s2_action_gold(f):
    if f["impurity_pct"] > f["impurity_max_pct"]:
        return "quarantine"
    if f["batch_temp_c"] > f["temp_max_c"]:
        return "hold_lab"
    if f["ph"] < f["ph_min"]:
        return "reprocess"
    return "release"


def s2_review(f):
    return (not f["operator_certified"] and f["impurity_pct"] > f["impurity_max_pct"]) or (
        not f["vessel_integrity_ok"] and f["batch_temp_c"] > f["temp_max_c"])


def s2_review_rule():
    return ("a quality supervisor must sign the disposition when the operator is not certified with "
            "impurities above 0.5 percent, or the vessel integrity check failed with a temperature above 30.0 degrees C")


def s2_outcome_rules(f):
    return [
        {"key": "ship_ambient", "desc": "The batch ships in ambient containers: applies to released batches with a certified operator."},
        {"key": "ship_cooled", "desc": "The batch ships in cooled containers: applies to released batches with an uncertified operator."},
        {"key": "blend_back", "desc": "The batch will be blended back: applies to reprocessed batches and to batches held for lab analysis."},
        {"key": "dispose", "desc": "The batch will be disposed: applies to quarantined batches with impurities above 0.9 percent."},
    ]


def s2_outcome_gold(f):
    a = s2_action_gold(f)
    if a == "release":
        return "ship_ambient" if f["operator_certified"] else "ship_cooled"
    if a == "reprocess":
        return "blend_back"
    if a == "quarantine":
        return "dispose" if f["impurity_pct"] > 0.9 else "blend_back"
    return "blend_back"


def s2_risk_rubric():
    return [
        {"key": 0, "desc": "low: pH at least 6.5 and impurities at or below 0.3 percent"},
        {"key": 1, "desc": "moderate: pH at least 6.5 with impurities above 0.3 percent, or pH below 6.5 with impurities at or below 0.3 percent"},
        {"key": 2, "desc": "high: pH below 6.5 with impurities above 0.3 percent and at or below 0.5 percent"},
        {"key": 3, "desc": "critical: impurities above 0.5 percent, whatever the pH"},
    ]


def s2_risk_gold(f):
    if f["impurity_pct"] > f["impurity_max_pct"]:
        return 3
    if f["ph"] < f["ph_min"]:
        return 2 if f["impurity_pct"] > 0.3 else 1
    return 1 if f["impurity_pct"] > 0.3 else 0


def s2_urgency_rubric():
    return [
        {"key": 0, "desc": "routine: vessel integrity ok and an operator is certified"},
        {"key": 1, "desc": "same-shift: vessel integrity ok with an uncertified operator"},
        {"key": 2, "desc": "same-hour: vessel integrity failed with temperature at or below 30.0 degrees C"},
        {"key": 3, "desc": "now: vessel integrity failed with temperature above 30.0 degrees C"},
    ]


def s2_urgency_gold(f):
    if not f["vessel_integrity_ok"]:
        return 3 if f["batch_temp_c"] > f["temp_max_c"] else 2
    return 1 if not f["operator_certified"] else 0


SEALED = {
    "loan_underwriting": {
        "sample": s1_loan, "action_rules": s1_action_rules, "action_gold": s1_action_gold,
        "review": s1_review, "review_rule": s1_review_rule,
        "outcome_rules": s1_outcome_rules, "outcome_gold": s1_outcome_gold,
        "risk_rubric": lambda f: s1_risk_rubric(), "risk_gold": s1_risk_gold,
        "urgency_rubric": lambda f: s1_urgency_rubric(), "urgency_gold": lambda f: s1_urgency_gold(f),
        "question": {"action": "What should the underwriting system do with this application?",
                     "needs_review": "This application requires manual underwriter review.",
                     "outcome": "How will this loan most likely be disbursed?",
                     "risk": "How risky is this loan under the risk rubric?",
                     "urgency": "How urgent is processing this application under the urgency rubric?"}},
    "chemical_batch_release": {
        "sample": s2_chem, "action_rules": s2_action_rules, "action_gold": s2_action_gold,
        "review": s2_review, "review_rule": s2_review_rule,
        "outcome_rules": s2_outcome_rules, "outcome_gold": s2_outcome_gold,
        "risk_rubric": lambda f: s2_risk_rubric(), "risk_gold": s2_risk_gold,
        "urgency_rubric": lambda f: s2_urgency_rubric(), "urgency_gold": s2_urgency_gold,
        "question": {"action": "What should the quality system do with this batch?",
                     "needs_review": "This disposition requires a quality supervisor signature.",
                     "outcome": "What will happen to this batch?",
                     "risk": "How risky is releasing this batch under the risk rubric?",
                     "urgency": "How urgent is the disposition under the urgency rubric?"}},
}


def build_train_case(family, rng, idx, noul_polarity=None):
    """E19-family case with base + flip renderings for every slot (10 questions).

    noul_polarity: force the base noul gold to True/False (protocol: 50/50 per
    family for ctrl/dev; cand pairs are self-balancing).
    """
    spec = e19.FAMILIES[family]
    for _ in range(160):
        f = spec["sample"](rng)["facts"]
        if noul_polarity is not None and bool(spec["review_gold"](f)) != noul_polarity:
            f = noul_flip(family, f)
            if bool(spec["review_gold"](f)) != noul_polarity:
                continue
        base_action = spec["action_gold"](f)
        ff, rules_fn, flip_gold = e20_action_flip(family, f, base_action)
        rf = e19.risk_flip(family, f)
        nf = noul_flip(family, f)
        of = outcome_flip(family, f)
        uf = urgency_flip(family, f)
        if flip_gold == base_action or flip_gold is None:
            continue
        if spec["risk_gold"](rf) == spec["risk_gold"](f):
            continue
        if spec["review_gold"](nf) == spec["review_gold"](f):
            continue
        if spec["outcome_gold"](of) == spec["outcome_gold"](f):
            continue
        if spec["urgency_gold"](uf) == spec["urgency_gold"](f):
            continue
        case_id = f"e20|{family}|{idx:04d}"
        out = {"id": case_id, "family": family, "questions": {}, "gold": {}, "meta": {}}

        def add(slot, qtype, instructions, criteria, gold, facts, variant, pair=None):
            state = e19.state_json(family, facts)
            out["questions"][slot] = {"instructions": instructions, "type": qtype, "criteria": criteria}
            out["gold"][slot] = gold
            out["meta"][slot] = {"variant": variant, "pair": pair, "state": state}

        q = spec["question"]
        ac, ag = e19.render_choice(spec["action_rules"](f), base_action, f"{case_id}:action")
        add("action", "choice", q["action"], ac, ag, f, "base", pair=f"{case_id}:action")
        fc, fg = e19.render_choice(rules_fn(ff), flip_gold, f"{case_id}:action_flip")
        add("action_flip", "choice", q["action"], fc, fg, f, "ruleflip", pair=f"{case_id}:action")
        oc, og = e19.render_choice(spec["outcome_rules"](f), spec["outcome_gold"](f), f"{case_id}:outcome")
        add("outcome", "choice", q["outcome"], oc, og, f, "base", pair=f"{case_id}:outcome")
        ofc, ofg = e19.render_choice(spec["outcome_rules"](of), spec["outcome_gold"](of), f"{case_id}:outcome_flip")
        add("outcome_flip", "choice", q["outcome"], ofc, ofg, of, "factflip", pair=f"{case_id}:outcome")
        noul = {"false": f"No, it does not: {spec['review_rule']()}.", "true": f"Yes, it does: {spec['review_rule']()}."}
        add("needs_review", "noul", q["needs_review"], noul, int(spec["review_gold"](f)), f, "base", pair=f"{case_id}:noul")
        add("needs_review_flip", "noul", q["needs_review"], noul, int(spec["review_gold"](nf)), nf, "factflip", pair=f"{case_id}:noul")
        rk, rg = e19.render_score(spec["risk_rubric"](f), spec["risk_gold"](f))
        add("risk", "score", q["risk"], rk, rg, f, "base", pair=f"{case_id}:risk")
        rk2, rg2 = e19.render_score(spec["risk_rubric"](rf), spec["risk_gold"](rf))
        add("risk_flip", "score", q["risk"], rk2, rg2, rf, "evidenceflip", pair=f"{case_id}:risk")
        ub, ug = e19.render_score(spec["urgency_rubric"](f), spec["urgency_gold"](f))
        add("urgency", "score", q["urgency"], ub, ug, f, "base", pair=f"{case_id}:urgency")
        ub2, ug2 = e19.render_score(spec["urgency_rubric"](uf), spec["urgency_gold"](uf))
        add("urgency_flip", "score", q["urgency"], ub2, ug2, uf, "factflip", pair=f"{case_id}:urgency")
        return out
    raise RuntimeError(f"{family}: no fully flip-ready case")


def build_sealed_case(family, rng, idx):
    spec = SEALED[family]
    for _ in range(80):
        f = spec["sample"](rng)["facts"]
        # flip machinery for the sealed families: fact flips per slot
        import copy
        nf = copy.deepcopy(f)
        rf = copy.deepcopy(f)
        if family == "loan_underwriting":
            # noul flip: toggle amount near 80% of limit while dti>35, or dti across 35
            if s1_review(f):
                nf["requested_amount_eur"] = round(f["product_limit_eur"] * 0.5, 2)
            else:
                nf["requested_amount_eur"] = round(f["product_limit_eur"] * 0.9, 2)
                nf["debt_to_income_pct"] = 40.0
            # risk flip: dti across 20 or collateral toggle at dti>35
            if rf["debt_to_income_pct"] > 20.0:
                rf["debt_to_income_pct"] = 15.0
            else:
                rf["debt_to_income_pct"] = 22.0
        else:
            if s2_review(f):
                nf["operator_certified"] = True
            else:
                nf["operator_certified"] = False
                nf["impurity_pct"] = 0.8
            if rf["ph"] >= 6.5:
                rf["ph"] = round(5.5, 2)
            else:
                rf["ph"] = round(7.0, 2)
        if spec["review"](nf) == spec["review"](f):
            continue
        if spec["risk_gold"](rf) == spec["risk_gold"](f):
            continue
        case_id = f"e20|{family}|{idx:04d}"
        out = {"id": case_id, "family": family, "questions": {}, "gold": {}, "meta": {}}

        def add(slot, qtype, instructions, criteria, gold, facts, variant, pair=None):
            ev = {k: v for k, v in facts.items() if k not in ("dti_threshold_pct", "dti_high_pct")}
            state = json.dumps(ev, sort_keys=True)
            out["questions"][slot] = {"instructions": instructions, "type": qtype, "criteria": criteria}
            out["gold"][slot] = gold
            out["meta"][slot] = {"variant": variant, "pair": pair, "state": state}

        q = spec["question"]
        ac, ag = e19.render_choice(spec["action_rules"](f), spec["action_gold"](f), f"{case_id}:action")
        add("action", "choice", q["action"], ac, ag, f, "base", pair=f"{case_id}:action")
        noul = {"false": f"No, it does not: {spec['review_rule']()}.", "true": f"Yes, it does: {spec['review_rule']()}."}
        add("needs_review", "noul", q["needs_review"], noul, int(spec["review"](f)), f, "base", pair=f"{case_id}:noul")
        add("needs_review_flip", "noul", q["needs_review"], noul, int(spec["review"](nf)), nf, "factflip", pair=f"{case_id}:noul")
        rk, rg = e19.render_score(spec["risk_rubric"](f), spec["risk_gold"](f))
        add("risk", "score", q["risk"], rk, rg, f, "base", pair=f"{case_id}:risk")
        rk2, rg2 = e19.render_score(spec["risk_rubric"](rf), spec["risk_gold"](rf))
        add("risk_flip", "score", q["risk"], rk2, rg2, rf, "evidenceflip", pair=f"{case_id}:risk")
        oc, og = e19.render_choice(spec["outcome_rules"](f), spec["outcome_gold"](f), f"{case_id}:outcome")
        add("outcome", "choice", q["outcome"], oc, og, f, "base")
        return out
    raise RuntimeError(f"{family}: no flip-ready sealed case")


def rows_for(case, slots, split):
    out = []
    for slot in slots:
        q = case["questions"][slot]
        m = case["meta"][slot]
        options = [{"id": k, "description": v} for k, v in q["criteria"].items()]
        out.append({
            "id": f"{case['id']}:{slot}", "group_id": case["id"], "source": "e20-contrast",
            "language": "en", "domain": case["family"], "state": m["state"],
            "question": q["instructions"], "options": options, "target_kind": "single",
            "target_option": options[case["gold"][slot]]["id"], "split": split,
            "question_type": q["type"], "label_origin": "e20-rule-verified",
            "variant": m["variant"], "pair": m["pair"],
        })
    return out


def engine_case(case, slots):
    questions, gold, meta, states = {}, {}, {}, {}
    for slot in slots:
        q = case["questions"][slot]
        questions[slot] = {"instructions": q["instructions"], "type": q["type"], "criteria": q["criteria"]}
        gold[slot] = case["gold"][slot]
        meta[slot] = {"variant": case["meta"][slot]["variant"], "pair": case["meta"][slot]["pair"]}
        states[slot] = case["meta"][slot]["state"]
    return {"id": case["id"], "family": case["family"], "states": states,
            "questions": questions, "gold": gold, "meta": meta}


CTRL_SLOTS = ["action", "needs_review", "outcome", "risk", "urgency"]
CAND_SLOTS = ["action", "action_flip", "needs_review", "needs_review_flip", "outcome",
              "outcome_flip", "risk", "risk_flip", "urgency", "urgency_flip"]
DEV_SLOTS = ["action", "needs_review", "outcome", "risk", "urgency"]
TEST_SLOTS = ["action", "needs_review", "needs_review_flip", "risk", "risk_flip"]


def main() -> int:
    rng = random.Random(SEED)
    train_ctrl, train_cand, dev_rows, test_rows, test_cases = [], [], [], [], []
    # train families: 110 ctrl cases + 55 cand cases + 22 dev cases each
    for fam_i, family in enumerate(TRAIN_FAMILIES):
        idx_pool = list(range(187))
        order = stable_shuffle(idx_pool, f"e20-split|{family}")
        ctrl_ids, cand_ids, dev_ids = order[:110], order[110:165], order[165:187]
        noul_balance = {"ctrl": [0, 0], "dev": [0, 0]}
        for k, idx in enumerate(sorted(ctrl_ids)):
            case = build_train_case(family, rng, idx, noul_polarity=(k % 2 == 0))
            rows = rows_for(case, CTRL_SLOTS, "train")
            b = noul_balance["ctrl"]
            b[int(rows[1]["target_option"] == "true")] += 1
            train_ctrl.extend(rows)
        for idx in sorted(cand_ids):
            case = build_train_case(family, rng, idx)
            train_cand.extend(rows_for(case, CAND_SLOTS, "train"))
        for k, idx in enumerate(sorted(dev_ids)):
            case = build_train_case(family, rng, idx, noul_polarity=(k % 2 == 0))
            dev_rows.extend(rows_for(case, DEV_SLOTS, "dev"))
    # sealed families: 75 cases each, test only
    for family in SEALED_FAMILIES:
        for idx in range(75):
            case = build_sealed_case(family, rng, idx)
            test_rows.extend(rows_for(case, TEST_SLOTS, "test"))
            test_cases.append(engine_case(case, TEST_SLOTS))

    for name, rows in (("train_ctrl", train_ctrl), ("train_cand", train_cand),
                       ("dev", dev_rows), ("test", test_rows)):
        ids = [r["id"] for r in rows]
        assert len(ids) == len(set(ids)), f"duplicate ids in {name}"
    by_case = {}
    for r in train_ctrl + train_cand + dev_rows + test_rows:
        by_case.setdefault(r["group_id"], set()).add(r["split"])
    assert all(len(v) == 1 for v in by_case.values()), "case crosses splits"

    import collections
    hist = collections.Counter()
    for r in train_ctrl + train_cand + dev_rows + test_rows:
        text = " ".join([r["question"], r["state"]] + [o["description"] for o in r["options"]]).lower()
        for tok in FORBIDDEN:
            assert tok not in text, f"leak {tok!r} in {r['id']}"
        if r["question_type"] == "choice":
            hist[f"choice|{r['target_option']}"] += 1

    out = ROOT / "data/e20-contrast"
    out.mkdir(parents=True, exist_ok=True)
    (out / "test_sealed").mkdir(exist_ok=True)
    for name, rows in (("train_ctrl", train_ctrl), ("train_cand", train_cand),
                       ("dev", dev_rows), ("test", test_rows)):
        (out / f"{name}_rows.jsonl").write_text(
            "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows), encoding="utf-8")
    (out / "test_sealed" / "cases.jsonl").write_text(
        "".join(json.dumps(c, sort_keys=True) + "\n" for c in test_cases), encoding="utf-8")

    manifest = {
        "set": "e20-contrast", "protocol": "docs/internal/E20_PROTOCOL_2026-10-09.md",
        "generator": "scripts/gen_e20_contrast.py",
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "seed": SEED,
        "counts": {"train_ctrl": len(train_ctrl), "train_cand": len(train_cand),
                   "dev": len(dev_rows), "test": len(test_rows), "test_cases": len(test_cases)},
        "noul_gold_balance": {fam: dict(collections.Counter(
            r["target_option"] for r in train_ctrl + dev_rows
            if r["domain"] == fam and r["question_type"] == "noul")) for fam in TRAIN_FAMILIES},
        "test_noul_balance": dict(collections.Counter(
            r["target_option"] for r in test_rows if r["question_type"] == "noul")),
        "gold_position_histogram": dict(hist),
        "checks": {"unique_ids": True, "no_split_crossing": True, "forbidden_absent": True},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in ("counts", "noul_gold_balance", "test_noul_balance",
                                               "gold_position_histogram")}, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
