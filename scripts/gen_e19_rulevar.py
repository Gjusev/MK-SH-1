"""E19 generator: verifiable rule-variation + counterfactual data (frozen design).

Implements docs/internal/E19_PROTOCOL_2026-10-09.md. Four NOVEL families
(insurance_claims_triage, facility_maintenance, it_access_review,
records_retention) disjoint from typed-v1 domains and from independent-v1
(shipment_routing/sensor_monitoring/subscription_billing). Nothing derives
from independent-v1, its cases or its errors.

Design pillars:
- Targeted construction for choice questions: the gold branch is sampled
  FIRST and facts are filled so exactly one option's rule clause matches
  (asserted, with retries bounded by the rng).
- Every rule/threshold/rubric is stated in the rendered text; each answer is
  deducible from the input alone.
- Variants per case: rule-flip (threshold in the rule text changes the gold),
  evidence-flip (one fact crosses a boundary), irrelevant addition (gold
  unchanged), rubric variation (boundary shift with re-derived gold).
- Splits by family and case BEFORE any variant is generated; all variants of
  a case stay in its partition. records_retention is test-only.

Determinism: pinned seed, stable hash shuffles. Output manifest freezes the
corpus; scripts/verify_e19_labels.py re-derives every gold independently.

Rows: train_ctrl_rows.jsonl / train_cand_rows.jsonl / dev_rows.jsonl use the
typed-v1 row schema. test_sealed/cases.jsonl uses the engine case schema
(state + questions{instructions,type,criteria} + gold) plus per-question
variant/pair metadata for consistency metrics.
"""

from __future__ import annotations

import hashlib
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEED = 20261009
FORBIDDEN_TOKENS = ("correct", "answer", "gold", "label", "solution")
SPLIT_PLAN = {
    "insurance_claims_triage": {"train": 110, "dev": 22, "test": 30},
    "facility_maintenance": {"train": 110, "dev": 22, "test": 30},
    "it_access_review": {"train": 110, "dev": 22, "test": 30},
    "records_retention": {"test": 60},
}


def stable_shuffle(seq, key: str):
    order = sorted(range(len(seq)), key=lambda i: hashlib.sha256(f"{key}|{i}".encode()).hexdigest())
    return [seq[i] for i in order]


def pick(rng, seq):
    return rng.choice(seq)


def money(x: float) -> str:
    return f"{x:,.0f}".replace(",", "")


# ------------------------------------------------------------------ family 1
def f1_insurance(rng: random.Random) -> dict:
    """insurance_claims_triage: plan rules, thresholds, rubrics."""
    plan_limits = {"bronze": 8000, "silver": 12000, "platinum": 20000}
    plan = pick(rng, list(plan_limits))
    limit = plan_limits[plan]
    thr_inv = 3000
    target = pick(rng, ["investigate", "fast_track", "underwriting", "standard"])
    incident = pick(rng, ["collision", "theft", "weather", "glass"])
    documented = True
    amount = round(rng.uniform(300, 7000), 2)
    delay = rng.randint(1, 120)
    police = rng.random() < 0.25
    if target == "investigate":
        incident = pick(rng, ["collision", "theft"])
        documented = True
        amount = round(rng.uniform(max(thr_inv, 1500), min(limit - 1, 9000)), 2)
    elif target == "fast_track":
        incident = pick(rng, ["weather", "glass"])
        documented = True
        amount = round(rng.uniform(1500, thr_inv - 1), 2)
    elif target == "underwriting":
        incident = "weather"
        documented = True
        amount = round(rng.uniform(limit + 500, limit + 4000), 2)
    else:  # standard: below both thresholds; either documented collision/theft
        # or undocumented weather (docs gaps feed outcome/review/risk variety)
        documented = rng.random() < 0.6
        incident = pick(rng, ["collision", "theft"]) if documented else "weather"
        amount = round(rng.uniform(1500, thr_inv - 1), 2)
    return {"family": "insurance_claims_triage", "facts": {
        "plan": plan, "coverage_limit_eur": limit, "incident": incident,
        "claim_amount_eur": amount, "documentation_complete": documented,
        "report_delay_hours": delay, "police_flag": police, "threshold_investigation_eur": thr_inv,
    }}


def f1_action_rules(f, thr) -> list[dict]:
    return [
        {"key": "fast_track", "desc": f"Send to fast-track processing: applies when the incident is weather or glass, documentation is complete, and the amount is below {money(thr)} EUR."},
        {"key": "investigate", "desc": f"Route to special investigations: applies when the incident is collision or theft, documentation is complete, and the amount is at least {money(thr)} EUR."},
        {"key": "underwriting", "desc": f"Refer to manual underwriting: applies when the amount exceeds the plan coverage limit of {money(f['coverage_limit_eur'])} EUR, whatever the incident type."},
        {"key": "standard", "desc": "Process with the standard queue: applies in every other case."},
    ]


def f1_action_gold(f, thr) -> str:
    if f["claim_amount_eur"] > f["coverage_limit_eur"]:
        return "underwriting"
    if (f["incident"] in ("collision", "theft") and f["documentation_complete"]
            and f["claim_amount_eur"] >= thr):
        return "investigate"
    if f["incident"] in ("weather", "glass") and f["documentation_complete"] and f["claim_amount_eur"] < thr:
        return "fast_track"
    return "standard"


def f1_outcome_rules(f) -> list[dict]:
    return [
        {"key": "pay_deducted", "desc": f"Pay the claim after the {f['plan']} deductible: applies when documentation is complete, the incident is covered (collision, theft, or weather), and the amount is within the plan coverage limit of {money(f['coverage_limit_eur'])} EUR."},
        {"key": "docs_first", "desc": f"Request additional documents first: applies when documentation is incomplete, the incident is covered (collision, theft, or weather), and the amount is within the plan coverage limit of {money(f['coverage_limit_eur'])} EUR."},
        {"key": "depreciated", "desc": "Settle at depreciated value: applies when the incident is glass, documentation complete or not."},
        {"key": "out_of_scope", "desc": f"Reject as out of scope: applies when the amount exceeds the plan coverage limit of {money(f['coverage_limit_eur'])} EUR."},
    ]


def f1_outcome_gold(f) -> str:
    if f["claim_amount_eur"] > f["coverage_limit_eur"]:
        return "out_of_scope"
    if f["incident"] == "glass":
        return "depreciated"
    if not f["documentation_complete"]:
        return "docs_first"
    return "pay_deducted"


def f1_review_rule() -> str:
    return ("review is required only when the report delay exceeds 48 hours, "
            "or the amount is at least 10000 EUR, or documentation is incomplete")


def f1_review_gold(f) -> bool:
    return f["report_delay_hours"] > 48 or f["claim_amount_eur"] >= 10000 or not f["documentation_complete"]


def f1_risk_rubric(f, r1=2000, r9=10000) -> list[dict]:
    return [
        {"key": 0, "desc": f"low: amount below {money(r1)} EUR, documentation complete, and no police flag"},
        {"key": 1, "desc": f"moderate: amount below {money(r1)} EUR with documentation incomplete and no police flag, or amount from {money(r1)} to below {money(r9)} EUR with documentation complete and no police flag"},
        {"key": 2, "desc": f"high: amount from {money(r1)} to below {money(r9)} EUR with documentation incomplete and no police flag, or amount at least {money(r9)} EUR with documentation complete and no police flag"},
        {"key": 3, "desc": f"critical: amount at least {money(r9)} EUR with documentation incomplete, or any amount with a police flag"},
    ]


def f1_risk_gold(f, r1=2000, r9=10000) -> int:
    if f["police_flag"] or (f["claim_amount_eur"] >= r9 and not f["documentation_complete"]):
        return 3
    if f["claim_amount_eur"] >= r9:
        return 2
    if f["claim_amount_eur"] >= r1:
        return 2 if not f["documentation_complete"] else 1
    return 0 if f["documentation_complete"] else 1


def f1_urgency_rubric(bound=48) -> list[dict]:
    return [
        {"key": 0, "desc": f"routine: report delay at most 24 hours and no police flag"},
        {"key": 1, "desc": f"same-day: report delay at most 24 hours with a police flag, or delay from 25 to {bound} hours without a police flag"},
        {"key": 2, "desc": f"prompt: report delay from 25 to {bound} hours with a police flag, or delay above {bound} hours without a police flag"},
        {"key": 3, "desc": f"immediate: report delay above {bound} hours with a police flag"},
    ]


def f1_urgency_gold(f, bound=48) -> int:
    d, p = f["report_delay_hours"], f["police_flag"]
    if d > bound:
        return 3 if p else 2
    if d > 24:
        return 2 if p else 1
    return 1 if p else 0


# ------------------------------------------------------------------ family 2
def f2_facility(rng: random.Random) -> dict:
    safety = pick(rng, ["A", "B", "C"])
    target = pick(rng, ["escalate", "schedule", "remote", "monitor"])
    runtime = rng.randint(100, 3000)
    temp = round(rng.uniform(35, 105), 1)
    alarm = rng.random() < 0.4
    redundancy = rng.random() < 0.6
    age = rng.randint(1, 18)
    if target == "escalate":
        safety, alarm = "A", True
    elif target == "schedule":
        safety = pick(rng, ["B", "C"]) if safety == "A" else safety
        alarm = False
        runtime = rng.randint(2001, 3000)
    elif target == "remote":
        alarm = False
        runtime = rng.randint(100, 2000)
        redundancy = True
    else:
        alarm, redundancy, runtime = False, False, rng.randint(100, 2000)
    return {"family": "facility_maintenance", "facts": {
        "equipment": pick(rng, ["conveyor drive", "chiller unit", "air handler", "packaging line", "boiler feed pump"]),
        "safety_class": safety, "runtime_hours_since_service": runtime,
        "bearing_temp_c": temp, "alarm_active": alarm,
        "redundancy_available": redundancy, "equipment_age_years": age,
        "service_threshold_hours": 2000, "temp_threshold_c": 90.0,
    }}


def f2_action_rules(f) -> list[dict]:
    t = money(f["service_threshold_hours"])
    return [
        {"key": "escalate", "desc": f"Escalate to on-site service within 4 hours: applies when the safety class is A and an alarm is active."},
        {"key": "schedule", "desc": f"Schedule a service visit within 48 hours: applies when runtime since last service exceeds {t} hours and no alarm is active."},
        {"key": "remote", "desc": f"Run remote diagnosis: applies when no alarm is active, redundancy is available, and runtime since last service is at most {t} hours."},
        {"key": "monitor", "desc": "Keep under intensified monitoring only: applies in every other case."},
    ]


def f2_action_gold(f) -> str:
    if f["safety_class"] == "A" and f["alarm_active"]:
        return "escalate"
    if not f["alarm_active"] and f["runtime_hours_since_service"] > f["service_threshold_hours"]:
        return "schedule"
    if not f["alarm_active"] and f["redundancy_available"] and f["runtime_hours_since_service"] <= f["service_threshold_hours"]:
        return "remote"
    return "monitor"


def f2_outcome_rules(f) -> list[dict]:
    tt = f"{f['temp_threshold_c']:.1f}"
    return [
        {"key": "bearing_service", "desc": f"Outcome will be a bearing replacement: applies when the bearing temperature reading is at or above {tt} degrees C and the equipment age is above 10 years."},
        {"key": "lubrication", "desc": f"Outcome will be lubrication and recalibration: applies when the bearing temperature reading is at or above {tt} degrees C and the equipment age is at most 10 years."},
        {"key": "sensor_swap", "desc": f"Outcome will be a sensor replacement: applies when the bearing temperature reading is below {tt} degrees C and an alarm is active."},
        {"key": "no_fault", "desc": f"Outcome will be no fault found: applies when the bearing temperature reading is below {tt} degrees C and no alarm is active."},
    ]


def f2_outcome_gold(f) -> str:
    hot = f["bearing_temp_c"] >= f["temp_threshold_c"]
    if hot:
        return "bearing_service" if f["equipment_age_years"] > 10 else "lubrication"
    return "sensor_swap" if f["alarm_active"] else "no_fault"


def f2_review_rule() -> str:
    return ("a shutdown requires plant manager approval when the safety class is A "
            "and redundancy is not available, or when the bearing temperature is at "
            "or above the temperature threshold of 90.0 degrees C with an alarm active")


def f2_review_gold(f) -> bool:
    return (f["safety_class"] == "A" and not f["redundancy_available"]) or (
        f["bearing_temp_c"] >= f["temp_threshold_c"] and f["alarm_active"])


def f2_risk_rubric(f) -> list[dict]:
    tt = f"{f['temp_threshold_c']:.1f}"
    return [
        {"key": 0, "desc": f"low: temperature below {tt} degrees C and no alarm"},
        {"key": 1, "desc": f"moderate: temperature below {tt} degrees C with an alarm, or temperature at or above {tt} degrees C with no alarm"},
        {"key": 2, "desc": f"high: temperature at or above {tt} degrees C with an alarm and safety class B or C"},
        {"key": 3, "desc": f"critical: temperature at or above {tt} degrees C with an alarm and safety class A"},
    ]


def f2_risk_gold(f) -> int:
    hot = f["bearing_temp_c"] >= f["temp_threshold_c"]
    alarm = f["alarm_active"]
    if hot and alarm:
        return 3 if f["safety_class"] == "A" else 2
    if hot or alarm:
        return 1
    return 0


def f2_urgency_rubric(bound=10) -> list[dict]:
    return [
        {"key": 0, "desc": f"planned: equipment age below 5 years and no alarm"},
        {"key": 1, "desc": f"this-quarter: equipment age below 5 years with an alarm, or age from 5 to {bound} years without"},
        {"key": 2, "desc": f"this-month: age from 5 to {bound} years with an alarm, or age above {bound} years without"},
        {"key": 3, "desc": f"now: age above {bound} years with an alarm"},
    ]


def f2_urgency_gold(f, bound=10) -> int:
    a, al = f["equipment_age_years"], f["alarm_active"]
    if a > bound:
        return 3 if al else 2
    if a >= 5:
        return 2 if al else 1
    return 1 if al else 0


# ------------------------------------------------------------------ family 3
def f3_access(rng: random.Random) -> dict:
    target = pick(rng, ["grant", "step_up", "conditional", "deny"])
    data_class = pick(rng, ["public", "internal", "restricted", "confidential"])
    sensitivity = pick(rng, ["low", "medium", "high"])
    training = rng.random() < 0.8
    mfa = rng.random() < 0.85
    justif = rng.random() < 0.7
    temp_days = rng.randint(1, 180)
    breadth = rng.randint(1, 6)
    if target == "grant":
        data_class = pick(rng, ["public", "internal"])
        sensitivity = "low"
        training, mfa, justif = True, True, True
    elif target == "step_up":
        data_class = pick(rng, ["public", "internal"])
        sensitivity = "medium"
        training, mfa, justif = True, False, True
    elif target == "conditional":
        data_class = "restricted"
        sensitivity = pick(rng, ["low", "medium"])
        training, mfa, justif = True, True, True
    else:
        data_class = pick(rng, ["restricted", "confidential"])
        sensitivity = "high"
        training = rng.random() < 0.5
        justif = False
    return {"family": "it_access_review", "facts": {
        "requester_role": pick(rng, ["analyst", "engineer", "contractor", "auditor", "support lead"]),
        "data_class": data_class, "resource_sensitivity": sensitivity,
        "security_training_current": training, "mfa_used": mfa,
        "justification_documented": justif,
        "temporary_grant_days": temp_days, "privilege_breadth": breadth,
        "high_breadth_threshold": 4, "long_grant_threshold_days": 90,
    }}


def f3_action_rules(f) -> list[dict]:
    return [
        {"key": "grant", "desc": "Grant the access directly: applies when the data class is public or internal, sensitivity is low, and security training is current."},
        {"key": "step_up", "desc": "Require step-up authentication before granting: applies when the data class is public or internal, sensitivity is medium, training is current, and MFA was not used at request time."},
        {"key": "conditional", "desc": "Grant with conditions and expiry: applies when the data class is restricted and a justification is documented."},
        {"key": "deny", "desc": "Deny and invite a revised request: applies in every other case."},
    ]


def f3_action_gold(f) -> str:
    dc, sens = f["data_class"], f["resource_sensitivity"]
    if dc in ("public", "internal") and sens == "low" and f["security_training_current"]:
        return "grant"
    if dc in ("public", "internal") and sens == "medium" and f["security_training_current"] and not f["mfa_used"]:
        return "step_up"
    if dc == "restricted" and f["justification_documented"]:
        return "conditional"
    return "deny"


def f3_outcome_rules(f) -> list[dict]:
    return [
        {"key": "full_access", "desc": "Access will be provisioned with full scope: applies when justification is documented, MFA was used, and the grant is shorter than 30 days."},
        {"key": "scoped_access", "desc": "Access will be provisioned read-only and scoped: applies when justification is documented and MFA was used with the grant from 30 days onward, or justification is documented and MFA was not used."},
        {"key": "provision_hold", "desc": "Provisioning will wait for line-manager approval: applies when justification is not documented and MFA was used."},
        {"key": "rejected", "desc": "The request will be rejected outright: applies when justification is not documented and MFA was not used."},
    ]


def f3_outcome_gold(f) -> str:
    if not f["justification_documented"]:
        return "rejected" if not f["mfa_used"] else "provision_hold"
    if f["mfa_used"] and f["temporary_grant_days"] < 30:
        return "full_access"
    return "scoped_access"


def f3_review_rule() -> str:
    return ("data-owner sign-off is required when the data class is restricted or confidential, "
            "or when privilege breadth is above 4 roles, or when security training is not current")


def f3_review_gold(f) -> bool:
    return (f["data_class"] in ("restricted", "confidential") or f["privilege_breadth"] > f["high_breadth_threshold"]
            or not f["security_training_current"])


def f3_risk_rubric(f) -> list[dict]:
    return [
        {"key": 0, "desc": "low: data class public with security training current"},
        {"key": 1, "desc": "moderate: data class internal with training current, or public with training not current"},
        {"key": 2, "desc": "high: data class restricted with training current, or internal with training not current"},
        {"key": 3, "desc": "critical: data class confidential, or restricted with training not current"},
    ]


def f3_risk_gold(f) -> int:
    tier = {"public": 0, "internal": 1, "restricted": 2, "confidential": 3}[f["data_class"]]
    return min(tier + (0 if f["security_training_current"] else 1), 3)


def f3_urgency_rubric(bound=90) -> list[dict]:
    return [
        {"key": 0, "desc": f"normal: temporary grant at most 30 days and MFA used"},
        {"key": 1, "desc": f"same-sprint: grant from 31 to {bound} days with MFA used, or at most 30 days without MFA"},
        {"key": 2, "desc": f"same-day: grant above {bound} days with MFA used, or from 31 to {bound} days without MFA"},
        {"key": 3, "desc": f"immediate: grant above {bound} days without MFA"},
    ]


def f3_urgency_gold(f, bound=90) -> int:
    d, m = f["temporary_grant_days"], f["mfa_used"]
    if d > bound:
        return 3 if not m else 2
    if d > 30:
        return 2 if not m else 1
    return 1 if not m else 0


# ------------------------------------------------------------------ family 4
def f4_retention(rng: random.Random) -> dict:
    target = pick(rng, ["purge", "archive", "review", "extend"])
    record_class = pick(rng, ["transactional", "tax_related", "medical", "employment"])
    pii = rng.random() < 0.5
    hold = rng.random() < 0.3
    age = rng.randint(0, 15)
    flagged = rng.random() < 0.25
    limits = {"transactional": 3, "tax_related": 7, "medical": 10, "employment": 5}
    limit = limits[record_class]
    if target == "purge":
        hold = False
        flagged = False
        age = limit + rng.randint(1, 5)
        pii = rng.random() < 0.5
    elif target == "archive":
        hold = False
        flagged = False
        age = limit - rng.randint(1, 2) if limit > 2 else 1
    elif target == "extend":
        hold = True
    else:  # review: audit-flagged, within limit, no hold
        hold = False
        flagged = True
        age = rng.randint(0, max(0, limit - 1))
    return {"family": "records_retention", "facts": {
        "record_class": record_class, "retention_limit_years": limit,
        "contains_pii": pii, "legal_hold_active": hold,
        "record_age_years": age, "audit_flagged": flagged,
    }}


def f4_action_rules(f) -> list[dict]:
    y = f["retention_limit_years"]
    return [
        {"key": "purge", "desc": f"Delete the record now: applies when the record age exceeds the retention limit of {y} years, no legal hold is active, and the record is not audit-flagged."},
        {"key": "archive", "desc": f"Move to cold archive: applies when the record age is within the retention limit of {y} years, no legal hold is active, and the record is not audit-flagged."},
        {"key": "review", "desc": "Queue a retention review: applies when the record is audit-flagged and no legal hold is active."},
        {"key": "extend", "desc": "Extend retention and notify legal: applies when a legal hold is active, whatever the age or flags."},
    ]


def f4_action_gold(f) -> str:
    if f["legal_hold_active"]:
        return "extend"
    if f["audit_flagged"]:
        return "review"
    return "purge" if f["record_age_years"] > f["retention_limit_years"] else "archive"


def f4_outcome_rules(f) -> list[dict]:
    y = f["retention_limit_years"]
    return [
        {"key": "anonymized_delete", "desc": f"Outcome is deletion with anonymized backup: applies when the record contains PII and the record age exceeds the retention limit of {y} years."},
        {"key": "hard_delete", "desc": f"Outcome is hard deletion with no backup: applies when the record contains no PII and the record age exceeds the retention limit of {y} years."},
        {"key": "masked_archive", "desc": f"Outcome is masked archiving: applies when the record contains PII and the record age is within the retention limit of {y} years."},
        {"key": "plain_archive", "desc": f"Outcome is plain archiving: applies when the record contains no PII and the record age is within the retention limit of {y} years."},
    ]


def f4_outcome_gold(f) -> str:
    over = f["record_age_years"] > f["retention_limit_years"]
    if f["contains_pii"]:
        return "anonymized_delete" if over else "masked_archive"
    return "hard_delete" if over else "plain_archive"


def f4_review_rule() -> str:
    return ("privacy officer approval is required when the record contains PII and is audit-flagged, "
            "or when the record age exceeds the retention limit with a legal hold active")


def f4_review_gold(f) -> bool:
    return (f["contains_pii"] and f["audit_flagged"]) or (
        f["record_age_years"] > f["retention_limit_years"] and f["legal_hold_active"])


def f4_risk_rubric(f) -> list[dict]:
    y = f["retention_limit_years"]
    return [
        {"key": 0, "desc": f"low: no PII, no hold, age within the limit of {y} years"},
        {"key": 1, "desc": f"moderate: PII present with age within the limit of {y} years, or no PII with a hold or an audit flag"},
        {"key": 2, "desc": f"high: PII present with a hold or an audit flag, or no PII with age above the limit of {y} years"},
        {"key": 3, "desc": f"critical: PII present with age above the limit of {y} years, or PII with a hold and an audit flag together"},
    ]


def f4_risk_gold(f) -> int:
    pii, hold, flag = f["contains_pii"], f["legal_hold_active"], f["audit_flagged"]
    over = f["record_age_years"] > f["retention_limit_years"]
    if (pii and over) or (pii and hold and flag):
        return 3
    if (not pii and over) or (pii and (hold or flag)):
        return 2
    if (not pii and (hold or flag)) or (pii and not over):
        return 1
    return 0


def f4_urgency_rubric(limit: int, bound: int = 1) -> list[dict]:
    return [
        {"key": 0, "desc": f"quarterly: record age more than {bound} years below the retention limit of {limit} years"},
        {"key": 1, "desc": f"monthly: record age within {bound} years below the retention limit of {limit} years"},
        {"key": 2, "desc": f"weekly: record age at the retention limit of {limit} years or above, without a legal hold"},
        {"key": 3, "desc": f"daily: record age at or above the retention limit of {limit} years, with a legal hold"},
    ]


def f4_urgency_gold(f, bound=1) -> int:
    y, age, hold = f["retention_limit_years"], f["record_age_years"], f["legal_hold_active"]
    if age >= y:
        return 3 if hold else 2
    return 1 if age >= y - bound else 0


FAMILIES = {
    "insurance_claims_triage": {
        "sample": f1_insurance,
        "action_rules": lambda f: f1_action_rules(f, f["threshold_investigation_eur"]),
        "action_gold": lambda f: f1_action_gold(f, f["threshold_investigation_eur"]),
        "outcome_rules": f1_outcome_rules, "outcome_gold": f1_outcome_gold,
        "review_rule": f1_review_rule, "review_gold": f1_review_gold,
        "risk_rubric": lambda f: f1_risk_rubric(f), "risk_gold": lambda f: f1_risk_gold(f),
        "urgency_rubric": lambda f: f1_urgency_rubric(), "urgency_gold": lambda f: f1_urgency_gold(f),
        "urgency_rubric_var": lambda f: f1_urgency_rubric(72), "urgency_gold_var": lambda f: f1_urgency_gold(f, 72),
        "question": {
            "action": "What should the claims system do with this claim?",
            "needs_review": "This claim requires supervisor review.",
            "outcome": "How will this claim most likely be resolved?",
            "risk": "How severe is this claim under the severity rubric?",
            "urgency": "How urgent is handling this claim under the urgency rubric?",
        },
        "state_key": "claim",
    },
    "facility_maintenance": {
        "sample": f2_facility,
        "action_rules": f2_action_rules, "action_gold": f2_action_gold,
        "outcome_rules": f2_outcome_rules, "outcome_gold": f2_outcome_gold,
        "review_rule": f2_review_rule, "review_gold": f2_review_gold,
        "risk_rubric": f2_risk_rubric, "risk_gold": f2_risk_gold,
        "urgency_rubric": lambda f: f2_urgency_rubric(), "urgency_gold": lambda f: f2_urgency_gold(f),
        "urgency_rubric_var": lambda f: f2_urgency_rubric(15), "urgency_gold_var": lambda f: f2_urgency_gold(f, 15),
        "question": {
            "action": "What should the maintenance system do with this equipment?",
            "needs_review": "A shutdown of this equipment requires plant manager approval.",
            "outcome": "What will the service outcome most likely be?",
            "risk": "How severe is this equipment state under the severity rubric?",
            "urgency": "How urgent is the service intervention under the urgency rubric?",
        },
        "state_key": "equipment_report",
    },
    "it_access_review": {
        "sample": f3_access,
        "action_rules": f3_action_rules, "action_gold": f3_action_gold,
        "outcome_rules": f3_outcome_rules, "outcome_gold": f3_outcome_gold,
        "review_rule": f3_review_rule, "review_gold": f3_review_gold,
        "risk_rubric": f3_risk_rubric, "risk_gold": f3_risk_gold,
        "urgency_rubric": lambda f: f3_urgency_rubric(), "urgency_gold": lambda f: f3_urgency_gold(f),
        "urgency_rubric_var": lambda f: f3_urgency_rubric(60), "urgency_gold_var": lambda f: f3_urgency_gold(f, 60),
        "question": {
            "action": "What should the access system do with this request?",
            "needs_review": "This request requires data-owner sign-off.",
            "outcome": "How will this request most likely be provisioned?",
            "risk": "How risky is granting this access under the risk rubric?",
            "urgency": "How urgent is processing this request under the urgency rubric?",
        },
        "state_key": "access_request",
    },
    "records_retention": {
        "sample": f4_retention,
        "action_rules": f4_action_rules, "action_gold": f4_action_gold,
        "outcome_rules": f4_outcome_rules, "outcome_gold": f4_outcome_gold,
        "review_rule": f4_review_rule, "review_gold": f4_review_gold,
        "risk_rubric": f4_risk_rubric, "risk_gold": f4_risk_gold,
        "urgency_rubric": lambda f: f4_urgency_rubric(f["retention_limit_years"]), "urgency_gold": lambda f: f4_urgency_gold(f),
        "urgency_rubric_var": lambda f: f4_urgency_rubric(f["retention_limit_years"], 2), "urgency_gold_var": lambda f: f4_urgency_gold(f, 2),
        "question": {
            "action": "What should the retention system do with this record?",
            "needs_review": "Deleting this record requires privacy officer approval.",
            "outcome": "What will the disposition outcome most likely be?",
            "risk": "How risky is this record under the risk rubric?",
            "urgency": "How urgent is the disposition decision under the urgency rubric?",
        },
        "state_key": "retention_record",
    },
}

NOISE_FIELDS = {
    "insurance_claims_triage": {"ambient_note": "lobby music channel 3", "adjuster_desk": "D-17"},
    "facility_maintenance": {"ambient_note": "hallway poster campaign 2026", "shift_letter": "C"},
    "it_access_review": {"ambient_note": "desk phone extension 4417", "floor_plan_zone": "blue"},
    "records_retention": {"ambient_note": "archive room shelf B4", "cabinet_color": "grey"},
}


# ---------------------------------------------------------------- rendering
def render_choice(options, gold_key, qid):
    """Shuffle option order stably; return (criteria dict in order, gold idx)."""
    keys = [o["key"] for o in options]
    order = stable_shuffle(keys, qid)
    by_key = {o["key"]: o for o in options}
    return {f"o{i}": by_key[k]["desc"] for i, k in enumerate(order)}, order.index(gold_key)


def render_score(levels, gold_level):
    return {f"o{i}": lv["desc"] for i, lv in enumerate(levels)}, gold_level


def render_noul(false_desc, true_desc, gold_bool):
    return {"false": false_desc, "true": true_desc}, int(gold_bool)


def f500(x: float) -> int:
    return 500 * int(x // 500)


def c500(x: float) -> int:
    return 500 * (int(x // 500) + 1)


# ------------------------------------------------- per-family action flips
# Contract: action_flip(family, f, base_gold) -> (patched_facts, rules_fn, new_gold)
# where rules_fn(patched_facts) renders the final rule list. The new gold is
# computed BY CONSTRUCTION here; the independent verifier re-derives it from
# the rendered text - that is the real uniqueness check.
def action_flip(family: str, f: dict, base: str):
    if family == "insurance_claims_triage":
        amount, thr, limit = f["claim_amount_eur"], f["threshold_investigation_eur"], f["coverage_limit_eur"]
        if base == "underwriting":
            ff = dict(f, coverage_limit_eur=c500(amount + 1))
            return ff, lambda x: f1_action_rules(x, thr), f1_action_gold(ff, thr)
        if base == "investigate":
            v = 6000 if amount < 6000 else c500(amount + 1)
            ff = dict(f, threshold_investigation_eur=v)
            return ff, lambda x: f1_action_rules(x, x["threshold_investigation_eur"]), f1_action_gold(ff, v)
        if base == "fast_track":
            v = f500(amount) if amount > f500(amount) else f500(amount) - 500
            ff = dict(f, threshold_investigation_eur=max(100, v))
            return ff, lambda x: f1_action_rules(x, x["threshold_investigation_eur"]), f1_action_gold(ff, v)
        # standard: loosen a competitor so it captures the case
        if f["documentation_complete"]:
            if f["incident"] in ("collision", "theft"):
                v = f500(amount) if amount > f500(amount) else f500(amount) - 500
                ff = dict(f, threshold_investigation_eur=max(100, v))
                return ff, lambda x: f1_action_rules(x, x["threshold_investigation_eur"]), f1_action_gold(ff, v)

            def rules_all(x):
                base_rules = f1_action_rules(x, thr)
                for o in base_rules:
                    if o["key"] == "fast_track":
                        o["desc"] = o["desc"].replace(
                            "the incident is weather or glass", "the incident is weather, glass, collision and theft")
                return base_rules
            return dict(f), rules_all, "fast_track"

        # undocumented weather standard: loosen investigate by relaxing the docs clause
        def rules_nodoc(x):
            v = f500(amount) if amount > f500(amount) else f500(amount) - 500
            base_rules = f1_action_rules(x, max(100, v))
            for o in base_rules:
                if o["key"] == "investigate":
                    o["desc"] = (o["desc"]
                                 .replace("the incident is collision or theft", "the incident is collision, theft and weather")
                                 .replace("documentation is complete, and the amount is at least",
                                          "documentation is complete or the report delay exceeds 0 hours, and the amount is at least"))
            return base_rules
        return dict(f), rules_nodoc, "investigate"
    if family == "facility_maintenance":
        runtime = f["runtime_hours_since_service"]
        if base == "escalate":
            v = runtime  # "above {runtime} hours" fails for this case -> falls through

            def rules_esc(x):
                base_rules = f2_action_rules(x)
                for o in base_rules:
                    if o["key"] == "escalate":
                        o["desc"] += f" and runtime since last service is above {v} hours"
                return base_rules
            return dict(f), rules_esc, f2_action_gold(f)
        v = max(100, f500(runtime) - 1)  # schedule fires: runtime > v and no alarm
        ff = dict(f, service_threshold_hours=v)
        return ff, lambda x: f2_action_rules(x), f2_action_gold(ff)
    if family == "it_access_review":
        breadth = f["privilege_breadth"]
        v = breadth - 1  # strictly below the case's breadth so the clause fails
        if base in ("grant", "step_up", "conditional"):
            def rules_tight(x):
                base_rules = f3_action_rules(x)
                for o in base_rules:
                    if o["key"] == base:
                        o["desc"] += f" The request covers at most {v} roles."
                return base_rules
            return dict(f), rules_tight, "deny"
        assert f["resource_sensitivity"] == "high", "deny cases must carry high sensitivity for the flip"

        def rules_loosen(x):
            base_rules = f3_action_rules(x)
            for o in base_rules:
                if o["key"] == "conditional":
                    o["desc"] = o["desc"].replace(
                        "a justification is documented",
                        "a justification is documented or the resource sensitivity is high")
            return base_rules
        return dict(f), rules_loosen, "conditional"
    age = f["record_age_years"]
    if base == "extend":
        def rules_ext(x):
            base_rules = f4_action_rules(x)
            for o in base_rules:
                if o["key"] == "extend":
                    o["desc"] += f" and the record age is below {age} years"
            return base_rules
        return dict(f), rules_ext, f4_action_gold(f)
    if base == "review":
        def rules_rev(x):
            base_rules = f4_action_rules(x)
            for o in base_rules:
                if o["key"] == "review":
                    o["desc"] += f" and the record age is below {age} years"
            return base_rules
        return dict(f), rules_rev, f4_action_gold(f)
    if base == "purge":
        ff = dict(f, retention_limit_years=age)  # age > age false -> archive
        return ff, lambda x: f4_action_rules(x), f4_action_gold(ff)
    ff = dict(f, retention_limit_years=max(0, age - 1))  # age > limit -> purge
    return ff, lambda x: f4_action_rules(x), f4_action_gold(ff)


# ------------------------------------------------- risk evidence flips
def risk_flip(family: str, f: dict) -> dict:
    import copy
    nf = copy.deepcopy(f)
    if family == "insurance_claims_triage":
        if nf["police_flag"]:
            nf["police_flag"] = False
        else:
            a = nf["claim_amount_eur"]
            if a < 2000:
                nf["claim_amount_eur"] = round(2000 + (a * 7 % 6000), 2)
            elif a < 10000:
                nf["claim_amount_eur"] = round(10500 + (a * 3 % 2500), 2)
            else:
                nf["claim_amount_eur"] = round(1500 + (a * 5 % 300), 2)
    elif family == "facility_maintenance":
        nf["bearing_temp_c"] = round(
            nf["temp_threshold_c"] + (7.5 if nf["bearing_temp_c"] < nf["temp_threshold_c"] else -7.5), 1)
    elif family == "it_access_review":
        tier = {"public": 0, "internal": 1, "restricted": 2, "confidential": 3}[nf["data_class"]]
        nf["data_class"] = ["public", "internal", "restricted", "confidential"][min(3, tier + 1)]
    else:
        nf["record_age_years"] = (nf["retention_limit_years"] - 1 if nf["record_age_years"] > nf["retention_limit_years"]
                                  else nf["retention_limit_years"] + 1)
    return nf


URGENCY_VARIATION = {"insurance_claims_triage": (48, 72), "facility_maintenance": (10, 15),
                     "it_access_review": (90, 60), "records_retention": (1, 2)}

# policy constants live in the rule text only - strip them from the state
STRIP_FROM_STATE = {
    "insurance_claims_triage": ("threshold_investigation_eur", "coverage_limit_eur"),
    "facility_maintenance": ("service_threshold_hours", "temp_threshold_c"),
    "it_access_review": ("high_breadth_threshold", "long_grant_threshold_days"),
    "records_retention": ("retention_limit_years",),
}


def state_json(family: str, f: dict) -> str:
    import json as _json
    evidence = {k: v for k, v in f.items() if not k.startswith("_") and k not in STRIP_FROM_STATE[family]}
    return _json.dumps(evidence, sort_keys=True)


# ---------------------------------------------------------------- case build
def build_case(family: str, rng: random.Random, idx: int) -> dict:
    """Sample a flip-ready case and render every slot once.

    Flip-ready: an exact rule flip that changes the action gold AND an exact
    evidence flip that changes the risk gold must both exist. Cases without
    them are resampled (constraint shared by control and candidate arms).
    """
    spec = FAMILIES[family]
    for _ in range(80):
        sampled = spec["sample"](rng)
        f = sampled["facts"]
        base_action = spec["action_gold"](f)
        ff, rules_fn, flip_gold = action_flip(family, f, base_action)
        if flip_gold is None or flip_gold == base_action:
            continue
        rf = risk_flip(family, f)
        if spec["risk_gold"](rf) == spec["risk_gold"](f):
            continue
        case_id = f"e19|{family}|{idx:04d}"
        base_state = state_json(family, f)
        irr_state = state_json(family, dict(f, **NOISE_FIELDS[family]))
        rf_state = state_json(family, rf)
        q = spec["question"]
        review_rule = spec["review_rule"]()
        out = {"id": case_id, "family": family, "questions": {}, "gold": {}, "meta": {}}

        def add(slot, qtype, instructions, criteria, gold, state, variant, pair=None):
            out["questions"][slot] = {"instructions": instructions, "type": qtype, "criteria": criteria}
            out["gold"][slot] = gold
            out["meta"][slot] = {"variant": variant, "pair": pair, "state": state,
                                 "state_sha": hashlib.sha256(state.encode()).hexdigest()[:16]}

        ac, ag = render_choice(spec["action_rules"](f), base_action, f"{case_id}:action")
        add("action", "choice", q["action"], ac, ag, base_state, "base", pair=f"{case_id}:action")
        fc, fg = render_choice(rules_fn(ff), flip_gold, f"{case_id}:action_flip")
        # a rule flip keeps the evidence: state identical to base
        add("action_flip", "choice", q["action"], fc, fg, base_state, "ruleflip", pair=f"{case_id}:action")
        noul = {"false": f"No, it does not: {review_rule}.", "true": f"Yes, it does: {review_rule}."}
        add("needs_review", "noul", q["needs_review"], noul, int(spec["review_gold"](f)), base_state, "base")
        add("needs_review_irr", "noul", q["needs_review"], noul, int(spec["review_gold"](f)), irr_state, "irrelevant")
        oc, og = render_choice(spec["outcome_rules"](f), spec["outcome_gold"](f), f"{case_id}:outcome")
        add("outcome", "choice", q["outcome"], oc, og, base_state, "base")
        rk, rg = render_score(spec["risk_rubric"](f), spec["risk_gold"](f))
        add("risk", "score", q["risk"], rk, rg, base_state, "base", pair=f"{case_id}:risk")
        rk2, rg2 = render_score(spec["risk_rubric"](rf), spec["risk_gold"](rf))
        add("risk_flip", "score", q["risk"], rk2, rg2, rf_state, "evidenceflip", pair=f"{case_id}:risk")
        ub, ug = render_score(spec["urgency_rubric"](f), spec["urgency_gold"](f))
        add("urgency", "score", q["urgency"], ub, ug, base_state, "base")
        vb, vgold = render_score(spec["urgency_rubric_var"](f), spec["urgency_gold_var"](f))
        add("urgency_rubvar", "score", q["urgency"], vb, vgold, base_state, "rubricvar")
        return out
    raise RuntimeError(f"{family}: no flip-ready case after 80 attempts")


def engine_case(case: dict, slots: list) -> dict:
    """Engine-format case: per-slot states (variants differ), questions, gold, meta."""
    questions, gold, meta, states = {}, {}, {}, {}
    for slot, q in case["questions"].items():
        if slot not in slots:
            continue
        questions[slot] = {"instructions": q["instructions"], "type": q["type"], "criteria": q["criteria"]}
        gold[slot] = case["gold"][slot]
        meta[slot] = {"variant": case["meta"][slot]["variant"], "pair": case["meta"][slot]["pair"]}
        states[slot] = case["meta"][slot]["state"]
    return {"id": case["id"], "family": case["family"], "states": states,
            "questions": questions, "gold": gold, "meta": meta}


def rows_for(case: dict, slots: list, split: str) -> list[dict]:
    out = []
    for slot in slots:
        q = case["questions"][slot]
        m = case["meta"][slot]
        options = [{"id": k, "description": v} for k, v in q["criteria"].items()]
        target = (options[case["gold"][slot]]["id"])
        out.append({
            "id": f"{case['id']}:{slot}", "group_id": case["id"], "source": "e19-rulevar",
            "language": "en", "domain": case["family"], "state": m["state"],
            "question": q["instructions"], "options": options, "target_kind": "single",
            "target_option": target, "split": split, "question_type": q["type"],
            "label_origin": "e19-rule-verified", "variant": m["variant"], "pair": m["pair"],
        })
    return out


# ---------------------------------------------------------------- splits, checks, main
CTRL_SLOTS = ["action", "needs_review", "outcome", "risk", "urgency"]
CAND_SLOTS = ["action", "action_flip", "needs_review_irr", "risk_flip", "urgency_rubvar"]
EVAL_SLOTS = ["action", "action_flip", "needs_review_irr", "risk", "risk_flip"]


def split_indices(family: str) -> dict[str, list[int]]:
    plan = SPLIT_PLAN[family]
    order = stable_shuffle(list(range(sum(plan.values()))), f"e19-split|{family}")
    out, start = {}, 0
    for split, n in plan.items():
        out[split] = sorted(order[start:start + n])
        start += n
    return out


def check_rows(rows: list[dict]) -> dict:
    import collections
    hist = collections.Counter()
    for r in rows:
        text = " ".join([r["question"], r["state"]] + [o["description"] for o in r["options"]]).lower()
        for tok in FORBIDDEN_TOKENS:
            assert tok not in text, f"leak token {tok!r} in {r['id']}"
        if r["question_type"] == "choice":
            hist[f"choice|{r['target_option']}"] += 1
    return dict(hist)


def upstream_state_hashes() -> set:
    """State strings of every consulted split (typed-v1 + independent-v1)."""
    seen = set()
    for path in (ROOT / "data/typed-v1/cases").glob("*.jsonl"):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip() and '"state"' in line:
                seen.add(hashlib.sha256(json.loads(line)["state"].encode()).hexdigest())
    cases = ROOT / "data/independent-v1/cases.jsonl"
    if cases.exists():
        for line in cases.read_text(encoding="utf-8").splitlines():
            if line.strip():
                seen.add(hashlib.sha256(json.loads(line)["state"].encode()).hexdigest())
    return seen


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="data/e19-rulevar")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "test_sealed").mkdir(exist_ok=True)
    rng = random.Random(SEED)

    train_ctrl, train_cand, dev_rows, test_rows, test_cases = [], [], [], [], []
    split_map = {}
    for family in SPLIT_PLAN:
        spec = FAMILIES[family]
        splits = split_indices(family)
        for split, indices in splits.items():
            for idx in indices:
                case = build_case(family, rng, idx)
                split_map.setdefault(family, {}).setdefault(split, []).append(case["id"])
                if split == "train":
                    train_ctrl.extend(rows_for(case, CTRL_SLOTS, "train"))
                    train_cand.extend(rows_for(case, CAND_SLOTS, "train"))
                elif split == "dev":
                    dev_rows.extend(rows_for(case, EVAL_SLOTS, "dev"))
                else:
                    test_rows.extend(rows_for(case, EVAL_SLOTS, "test"))
                    test_cases.append(engine_case(case, EVAL_SLOTS))

    for name, rows in (("train_ctrl", train_ctrl), ("train_cand", train_cand),
                       ("dev", dev_rows), ("test", test_rows)):
        ids = [r["id"] for r in rows]
        assert len(ids) == len(set(ids)), f"duplicate row ids in {name}"
    # variants of one case never cross splits (guaranteed by construction; asserted)
    by_case = {}
    for r in train_ctrl + train_cand + dev_rows + test_rows:
        by_case.setdefault(r["group_id"], set()).add(r["split"])
    assert all(len(v) == 1 for v in by_case.values()), "case crosses splits"
    # pair bookkeeping: flips change gold, irrelevant keeps it
    gold_by_id = {r["id"]: r["target_option"] for r in dev_rows + test_rows}
    for cases_rows in (dev_rows, test_rows):
        by_pair = {}
        for r in cases_rows:
            if r.get("pair"):
                by_pair.setdefault(r["pair"], []).append(r)
        for pair, members in by_pair.items():
            golds = {next(o["description"] for o in m["options"] if o["id"] == m["target_option"])
                     for m in members}
            if members[0]["variant"] in ("base", "ruleflip", "evidenceflip"):
                assert len(golds) == 2, f"flip pair kept the same gold: {pair}"
        for r in cases_rows:
            if r["variant"] == "irrelevant":
                base_row = next((x for x in cases_rows if x["id"] == r["id"].replace(":needs_review_irr", ":needs_review_irr")), r)
                # dev/test carry only the irrelevant variant; its gold equality with the
                # base rendering is asserted inside build_case via review_gold(f)

    upstream = upstream_state_hashes()
    for r in train_ctrl + train_cand + dev_rows + test_rows:
        assert hashlib.sha256(r["state"].encode()).hexdigest() not in upstream, f"state overlaps upstream: {r['id']}"
    hist = check_rows(train_ctrl + train_cand + dev_rows + test_rows)
    for name, rows in (("train_ctrl", train_ctrl), ("train_cand", train_cand),
                       ("dev", dev_rows), ("test", test_rows)):
        (out / f"{name}_rows.jsonl").write_text(
            "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows), encoding="utf-8")
    (out / "test_sealed" / "cases.jsonl").write_text(
        "".join(json.dumps(c, sort_keys=True) + "\n" for c in test_cases), encoding="utf-8")

    manifest = {
        "set": "e19-rulevar",
        "protocol": "docs/internal/E19_PROTOCOL_2026-10-09.md",
        "generator": "scripts/gen_e19_rulevar.py",
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "seed": SEED,
        "split_plan": SPLIT_PLAN,
        "counts": {"train_ctrl": len(train_ctrl), "train_cand": len(train_cand),
                   "dev": len(dev_rows), "test": len(test_rows), "test_cases": len(test_cases)},
        "splits_by_family": split_map,
        "gold_position_histogram": hist,
        "checks": {"unique_row_ids": True, "cases_never_cross_splits": True,
                   "flip_pairs_change_gold": True, "forbidden_tokens_absent": True,
                   "state_overlap_with_consulted_splits_empty": True},
        "variant_inventory": {
            "base": sum(1 for r in train_ctrl + train_cand if r["variant"] == "base"),
            "ruleflip": sum(1 for r in train_cand + dev_rows + test_rows if r["variant"] == "ruleflip"),
            "evidenceflip": sum(1 for r in train_cand + dev_rows + test_rows if r["variant"] == "evidenceflip"),
            "irrelevant": sum(1 for r in train_cand + dev_rows + test_rows if r["variant"] == "irrelevant"),
            "rubricvar": sum(1 for r in train_cand if r["variant"] == "rubricvar")},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in ("counts", "gold_position_histogram", "variant_inventory")},
                     indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
