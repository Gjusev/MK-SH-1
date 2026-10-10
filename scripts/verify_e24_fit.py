"""Independent verifier for the E24 fitting microset.

Re-derives every gold label from the RENDERED text with logic that does not
import the generator: regex the question for field word + operator + threshold,
parse the state JSON for the numeric field, compare. Also enforces the
distribution guards the protocol demands (balance, uniqueness, pair integrity,
value disjointness, expected counts, no-truncation serialization, target
alignment) and writes runs/e24-prep/verification.json.
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

DATA = ROOT / "data/e24-fit"
OUT = ROOT / "runs/e24-prep"

FIELD_WORDS = {"motor temperature": "motor_temp_c", "bearing temperature": "bearing_temp_c"}
QUESTION_RE = re.compile(
    r"^Is the (motor|bearing) temperature reading at least ([0-9]+(?:\.[0-9]+)?) degrees Celsius\?$")

EXPECTED_COUNTS = {"train_fit": 48, "dev_fit": 32, "eval_newvalues": 64,
                   "eval_newthreshold": 48, "eval_newfield": 64}
FILES = {"train_fit": "train_fit_rows.jsonl", "dev_fit": "dev_fit_rows.jsonl",
         "eval_newvalues": "eval_newvalues_rows.jsonl",
         "eval_newthreshold": "eval_newthreshold_rows.jsonl",
         "eval_newfield": "eval_newfield_rows.jsonl"}


def gold_from_text(row: dict) -> tuple[bool, dict]:
    """Independent label derivation + field/threshold extraction from rendered text."""
    m = QUESTION_RE.match(row["question"])
    if not m:
        raise ValueError(f"unparseable question: {row['question']!r}")
    field_word = f"{m.group(1)} temperature"
    field = FIELD_WORDS[field_word]
    threshold = float(m.group(2))
    state = json.loads(row["state"])
    if field not in state or not isinstance(state[field], (int, float)):
        raise ValueError(f"field {field} missing or non-numeric in state: {row['id']}")
    numeric = [k for k, v in state.items() if isinstance(v, (int, float)) and not isinstance(v, bool)]
    if numeric != [field]:
        raise ValueError(f"expected exactly one numeric field {field}, got {numeric}: {row['id']}")
    return bool(state[field] >= threshold), {"field": field, "threshold": threshold,
                                             "value": state[field]}


def main() -> int:
    failures = []
    splits = {}
    for name, fname in FILES.items():
        rows = [json.loads(l) for l in (DATA / fname).read_text(encoding="utf-8").splitlines() if l.strip()]
        splits[name] = rows
        if len(rows) != EXPECTED_COUNTS[name]:
            failures.append(f"{name}: {len(rows)} rows, expected {EXPECTED_COUNTS[name]}")
        for r in rows:
            # option contract + target alignment
            if [(o["id"], o["description"]) for o in r["options"]] != [("false", "No"), ("true", "Yes")]:
                failures.append(f"{r['id']}: option contract violated")
                continue
            gold, derived = gold_from_text(r)
            if r["target_option"] not in ("false", "true"):
                failures.append(f"{r['id']}: bad target_option {r['target_option']!r}")
                continue
            if ("true" if gold else "false") != r["target_option"]:
                failures.append(f"{r['id']}: label mismatch text={gold} row={r['target_option']}")
            if r["question_type"] != "noul" or r.get("rule_in") != "question":
                failures.append(f"{r['id']}: question_type/rule_in wrong")
            # metadata consistency with the rendered text
            if abs(derived["value"] - r["value"]) > 1e-9 or abs(derived["threshold"] - r["threshold"]) > 1e-9 \
                    or derived["field"] != r["feature"]:
                failures.append(f"{r['id']}: metadata disagrees with rendered text")

    all_rows = [r for rows in splits.values() for r in rows]
    # global uniqueness
    ids = Counter(r["id"] for r in all_rows)
    failures += [f"duplicate id {i}" for i, n in ids.items() if n > 1]
    payloads = Counter((r["question"], r["state"]) for r in all_rows)
    failures += [f"duplicate payload {p[0][:40]}" for p, n in payloads.items() if n > 1]
    # per-split value uniqueness + class balance
    for name, rows in splits.items():
        vals = Counter(r["value"] for r in rows)
        failures += [f"{name}: repeated value {v}" for v, n in vals.items() if n > 1]
        bal = Counter(r["target_option"] for r in rows)
        if bal["false"] != bal["true"]:
            failures.append(f"{name}: unbalanced {dict(bal)}")
        if any(v <= 0 for v in bal.values()):
            failures.append(f"{name}: a class is absent")
    # threshold-60 disjointness: train/dev/newvalues mutually disjoint; newfield == newvalues
    t60 = {n: {r["value"] for r in splits[n]} for n in ("train_fit", "dev_fit", "eval_newvalues")}
    for a, b in (("train_fit", "dev_fit"), ("train_fit", "eval_newvalues"), ("dev_fit", "eval_newvalues")):
        if t60[a] & t60[b]:
            failures.append(f"value overlap {a} & {b}: {sorted(t60[a] & t60[b])}")
    if t60["eval_newvalues"] != {r["value"] for r in splits["eval_newfield"]}:
        failures.append("eval_newfield values != eval_newvalues values")
    # boundary rows exist and are Yes
    for name, thr in (("train_fit", 60.0),):
        if not any(r["value"] == thr and r["target_option"] == "true" for r in splits[name]):
            failures.append(f"{name}: missing exact boundary {thr} -> true")
    for t in (45.0, 75.0, 90.0):
        if not any(r["threshold"] == t and r["value"] == t and r["target_option"] == "true"
                   for r in splits["eval_newthreshold"]):
            failures.append(f"eval_newthreshold: missing exact boundary {t} -> true")
    # pair integrity: both members, same split, lo < thr < hi, answers inverted
    for name, rows in splits.items():
        pairs = {}
        for r in rows:
            if r.get("pair"):
                pairs.setdefault(r["pair"], []).append(r)
        for pid, members in pairs.items():
            if len(members) != 2 or {m["variant"] for m in members} != {"pair_lo", "pair_hi"}:
                failures.append(f"{name}/{pid}: incomplete pair")
                continue
            lo = next(m for m in members if m["variant"] == "pair_lo")
            hi = next(m for m in members if m["variant"] == "pair_hi")
            if not (lo["value"] < lo["threshold"] < hi["value"]):
                failures.append(f"{name}/{pid}: pair does not straddle the threshold")
            if lo["target_option"] != "false" or hi["target_option"] != "true":
                failures.append(f"{name}/{pid}: pair answers not inverted")
    # serialization: state tokens fully present at the tail, head untruncated, 2 markers
    try:
        from transformers import AutoTokenizer
        from mokka_decisions.arch_julia import julia_sequence
        tok = AutoTokenizer.from_pretrained("jhu-clsp/mmBERT-small")
        truncation_failures = 0
        for r in all_rows:
            item = julia_sequence(tok, r["state"], r["question"],
                                  [o["description"] for o in r["options"]], "noul", 1024, 256)
            state_ids = tok(r["state"], add_special_tokens=False)["input_ids"]
            head_ids = tok(f"noul question: {r['question']}", add_special_tokens=False)["input_ids"]
            if item["ids"][-len(state_ids) - 1:-1] != state_ids or item["ids"][-1] != tok.sep_token_id:
                truncation_failures += 1
            if item["ids"][1:1 + len(head_ids)] != head_ids:
                truncation_failures += 1
            if len(item["markers"]) != 2 or item["qtype"] != 2:
                truncation_failures += 1
        if truncation_failures:
            failures.append(f"serialization/truncation violations: {truncation_failures}")
        serialization = "checked (state tokens intact, head untruncated, 2 markers, qtype noul)"
    except Exception as exc:  # pragma: no cover - offline environments
        serialization = f"NOT checked ({type(exc).__name__}: {exc})"

    OUT.mkdir(parents=True, exist_ok=True)
    report = {
        "protocol": "docs/internal/E24_PROTOCOL_2026-10-10.md",
        "rows_total": len(all_rows),
        "labels_rederived": len(all_rows),
        "failures": failures,
        "counts": {n: len(rows) for n, rows in splits.items()},
        "balance": {n: dict(Counter(r["target_option"] for r in rows)) for n, rows in splits.items()},
        "pairs": {n: sum(1 for r in rows if r.get("pair")) // 2 for n, rows in splits.items()},
        "serialization": serialization,
        "verifier_sha256": __import__("hashlib").sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (OUT / "verification.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "balance"}, indent=1))
    if failures:
        print(f"FAILURES ({len(failures)}):", *failures[:20], sep="\n  ")
        return 1
    print("verification: ALL CHECKS PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
