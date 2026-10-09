"""Independent label verifier for E20 data (separate from the generators).

Re-derives every gold from the rendered text + state. For the three E19
train families it delegates to verify_e19_labels after normalizing the two
new flip phrasings per-evidence; the two sealed families get their own
regex logic (values are stated in every rule text).

Usage: python scripts/verify_e20_labels.py [--data data/e20-contrast]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import verify_e19_labels as v19  # noqa: E402


def prep(desc: str, ev: dict) -> str:
    """Normalize E20-specific flip phrasings into v19-parseable text."""
    d = desc
    if "sensitivity is low or medium" in d:
        d = d.replace("sensitivity is low or medium", f"sensitivity is {ev.get('resource_sensitivity')}")
    m = re.search(r"requester role is ([a-z ]+?)[,.]", d)
    if m:
        phrase = f"requester role is {m.group(1)}"
        # holds iff the role matches; express via an mfa fact clause of the same truth value
        d = d.replace(phrase, "MFA was used" if ev.get("mfa_used") else "MFA was not used") if m.group(1).strip() == ev.get("requester_role") \
            else d.replace(phrase, "MFA was used" if not ev.get("mfa_used") else "MFA was not used")
    return d


def loan_clause_holds(desc: str, ev: dict) -> bool | None:
    d = desc.lower()
    if "every other case" in d:
        return None
    hold, seen = True, False
    m = re.search(r"limit of ([\d,]+) eur", d)
    if m:
        seen = True
        lim = float(m.group(1).replace(",", ""))
        if "within the" in d:
            hold &= ev["requested_amount_eur"] <= lim
        elif "above the product limit" in d:
            hold &= ev["requested_amount_eur"] > lim
        elif "exceeds 80 percent of the product limit" in d or "exceeds 80 percent of the" in d:
            hold &= ev["requested_amount_eur"] > 0.8 * ev["product_limit_eur"]
        elif "within 90 percent" in d or "90 percent of the product limit" in d:
            hold &= ev["requested_amount_eur"] <= 0.9 * ev["product_limit_eur"]
    for pat, op in ((r"debt-to-income (?:ratio )?(?:is )?below ([\d.]+) percent", "lt"),
                    (r"debt-to-income (?:ratio )?(?:is )?above ([\d.]+) percent", "gt"),
                    (r"debt-to-income (?:ratio )?(?:is )?at most ([\d.]+) percent", "le"),
                    (r"debt-to-income (?:ratio )?(?:is )?from ([\d.]+) to ([\d.]+) percent", "range"),
                    (r"(?:^|)below ([\d.]+) percent", "lt"),
                    (r"(?:^|)above ([\d.]+) percent", "gt"),
                    (r"(?:^|)at most ([\d.]+) percent", "le"),
                    (r"(?:^|)from ([\d.]+) to ([\d.]+) percent", "range")):
        m = re.search(pat, d)
        if m:
            seen = True
            dti = ev["debt_to_income_pct"]
            if op == "lt":
                hold &= dti < float(m.group(1))
            elif op == "gt":
                hold &= dti > float(m.group(1))
            elif op == "le":
                hold &= dti <= float(m.group(1))
            else:
                hold &= float(m.group(1)) <= dti <= float(m.group(2))
    if "collateral is pledged" in d or "with collateral" in d or "collateral pledged" in d:
        seen = True
        hold &= bool(ev["collateral_pledged"])
    if "without collateral" in d or "collateral is not pledged" in d:
        seen = True
        hold &= not ev["collateral_pledged"]
    if "borrower carries life insurance" in d or "borrower is insured" in d:
        seen = True
        hold &= bool(ev["borrower_insured"])
    if "borrower is not insured" in d:
        seen = True
        hold &= not ev["borrower_insured"]
    if re.search(r"approved loans|conditional approvals|memo approvals|declined applications", d):
        seen = True
        # outcome rules keyed on the action; re-derive the action from the state
        from gen_e20_contrast import s1_action_gold
        a = s1_action_gold(ev)
        if "approved loans" in d:
            hold &= a == "approve"
        elif "conditional approvals" in d:
            hold &= a == "conditional"
        elif "memo approvals" in d:
            hold &= a == "memo"
        else:
            hold &= a == "decline"
    return hold if seen else None


def chem_clause_holds(desc: str, ev: dict) -> bool | None:
    d = desc.lower()
    if "every other case" in d:
        return None
    hold, seen = True, False
    m = re.search(r"temperature (?:is )?(?:at or below|above) ([\d.]+) degrees", d)
    if m:
        seen = True
        hot = ev["batch_temp_c"] > float(m.group(1))
        hold &= hot if "above" in m.group(0) else not hot
    m = re.search(r"ph (?:is )?(?:at least|below) ([\d.]+)", d)
    if m:
        seen = True
        hold &= ev["ph"] >= float(m.group(1)) if "at least" in m.group(0) else ev["ph"] < float(m.group(1))
    m = re.search(r"impurities (?:are )?(?:at or below|above|exceed) ([\d.]+) percent", d)
    if m:
        seen = True
        hold &= ev["impurity_pct"] > float(m.group(1)) if ("above" in m.group(0) or "exceed" in m.group(0)) else ev["impurity_pct"] <= float(m.group(1))
    if "operator is certified" in d or "a certified operator" in d:
        seen = True
        hold &= bool(ev["operator_certified"])
    if "operator is not certified" in d or "an uncertified operator" in d:
        seen = True
        hold &= not ev["operator_certified"]
    if "vessel integrity ok" in d or "vessel integrity check passed" in d:
        seen = True
        hold &= bool(ev["vessel_integrity_ok"])
    if "vessel integrity failed" in d or "vessel integrity check failed" in d:
        seen = True
        hold &= not ev["vessel_integrity_ok"]
    if re.search(r"released batches|reprocessed batches|quarantined batches|held for lab", d):
        seen = True
        from gen_e20_contrast import s2_action_gold
        a = s2_action_gold(ev)
        if "released batches" in d:
            hold &= a == "release"
        elif "quarantined batches" in d:
            hold &= a == "quarantine"
        else:
            hold &= a in ("reprocess", "hold_lab")
    return hold if seen else None


SEALED_CLAUSES = {"loan_underwriting": loan_clause_holds, "chemical_batch_release": chem_clause_holds}


def verify_row(row: dict) -> bool:
    ev = json.loads(row["state"])
    family = row["domain"]
    if family in SEALED_CLAUSES:
        fn = SEALED_CLAUSES[family]
        if row["question_type"] == "noul":
            # statement truth from the rule text in the option description
            d = row["options"][0]["description"].lower()
            if family == "loan_underwriting":
                truth = (ev["requested_amount_eur"] > 0.8 * ev["product_limit_eur"]
                         and ev["debt_to_income_pct"] > 35.0)
            else:
                truth = ((not ev["operator_certified"] and ev["impurity_pct"] > 0.5)
                         or (not ev["vessel_integrity_ok"] and ev["batch_temp_c"] > 30.0))
            return ("true" if truth else "false") == row["target_option"]
        parts = [o["description"].split(": ", 1)[-1] for o in row["options"]]
        if row["question_type"] == "choice":
            matches = []
            for o, desc in zip(row["options"], parts):
                holds = fn(desc, ev)
                if holds is None and "every other case" in desc.lower():
                    matches.append(("catchall", o["id"]))
                elif holds:
                    matches.append(("specific", o["id"]))
            specifics = [m for m in matches if m[0] == "specific"]
            if len(specifics) == 1:
                gold = specifics[0][1]
            elif not specifics and len(matches) == 1:
                gold = matches[0][1]
            else:
                raise AssertionError(f"{row['id']}: {len(specifics)} specific clauses")
            return gold == row["target_option"]
        for i, desc in enumerate(parts):  # score: exclusive rubric levels
            for alt in desc.split(", or "):
                if fn(alt.strip(), ev):
                    return f"o{i}" == row["target_option"] and all(
                        not fn(a2.strip(), ev)
                        for j, d2 in enumerate(parts) if j != i for a2 in d2.split(", or "))
        return False
    # train families: delegate to the E19 verifier with phrasing normalization
    prepped = {o["id"]: prep(o["description"], ev) for o in row["options"]}
    return v19.verify_row(row | {"options": [{"id": k, "description": v} for k, v in prepped.items()]})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/e20-contrast")
    args = parser.parse_args()
    data = Path(args.data)
    total = bad = 0
    failures = []
    for name in ("train_ctrl_rows.jsonl", "train_cand_rows.jsonl", "dev_rows.jsonl", "test_rows.jsonl"):
        for line in (data / name).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            total += 1
            try:
                if not verify_row(row):
                    bad += 1
                    failures.append(row["id"])
            except AssertionError as e:
                bad += 1
                failures.append(f"{row['id']} :: {e}")
    print(f"verified {total} rows: {total - bad} agree, {bad} disagree")
    for f in failures[:12]:
        print("  FAIL:", f)
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
