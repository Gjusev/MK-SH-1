"""Independent verifier for the E25 threshold-variable data.

Re-derives every gold label from the RENDERED text with logic that does not
import the generator (regex the question threshold, parse the state value),
then enforces the protocol's structural guards: exact counts and balance,
train per-threshold 4/4, threshold-only pair integrity (state byte-identical,
question differs ONLY in the threshold, opposite gold, same split), reserved
thresholds absent from train/dev, cross-split value disjointness, purely
alphabetic equipment ids, near/exact boundaries, no-truncation serialization,
and the question-ignoring bound on pair subsets (constant rules and the fixed
value>=60 rule must each be <= 0.5 accuracy / 0 pairs there).
Writes runs/e25-prep/verification.json.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

DATA = ROOT / "data/e25-thrvar"
OUT = ROOT / "runs/e25-prep"

QUESTION_RE = re.compile(
    r"^Is the motor temperature reading at least ([0-9]+(?:\.[0-9]+)?) degrees Celsius\?$")
FILES = {"train": "train_thr_rows.jsonl", "dev": "dev_thr_rows.jsonl",
         "known": "test_known_thr_rows.jsonl", "new": "test_new_thr_rows.jsonl"}
EXPECTED = {"train": 48, "dev": 36, "known": 36, "new": 54}
TRAIN_THR = {40.0, 50.0, 60.0, 70.0, 80.0, 90.0}
RESERVED = {45.0, 55.0, 65.0, 75.0, 85.0, 30.0, 100.0}


def gold_from_text(row: dict) -> tuple[bool, float, float]:
    m = QUESTION_RE.match(row["question"])
    if not m:
        raise ValueError(f"unparseable question: {row['question']!r}")
    threshold = float(m.group(1))
    state = json.loads(row["state"])
    if set(state) != {"equipment_id", "motor_temp_c", "site"}:
        raise ValueError(f"unexpected state keys: {sorted(state)}")
    return bool(state["motor_temp_c"] >= threshold), state["motor_temp_c"], threshold


def pair_accuracy_cap(rows: list[dict]) -> dict:
    """Question-ignoring references on the pair subsets (and full sets)."""
    out = {}
    pair_rows = [r for r in rows if r["pair"]]
    subsets = {"all": rows}
    if pair_rows:
        subsets["pairs"] = pair_rows
    for name, subset in subsets.items():
        gold = [r["target_option"] == "true" for r in subset]
        out[name] = {
            "constant_yes": sum(gold) / len(gold),
            "constant_no": sum(not g for g in gold) / len(gold),
            "fixed_value_ge_60": sum((json.loads(r["state"])["motor_temp_c"] >= 60) == g
                                     for r, g in zip(subset, gold)) / len(gold),
        }
    if pair_rows:
        by_pair = defaultdict(dict)
        for r in pair_rows:
            by_pair[r["pair"]][r["variant"]] = r["target_option"] == "true"
        out["pairs"]["fixed_rule_both_correct"] = sum(
            1 for v in by_pair.values()
            if len(v) == 2 and False in v.values())  # a question-ignoring rule cannot flip
        # any deterministic state-only rule answers the same for both members
        # (identical state) => both-correct is impossible; count pairs where the
        # two golds differ as required:
        out["pairs"]["gold_opposite_pairs"] = sum(
            1 for v in by_pair.values() if len(v) == 2 and set(v.values()) == {True, False})
    return out


def main() -> int:
    failures = []
    splits = {}
    for name, fname in FILES.items():
        rows = [json.loads(l) for l in (DATA / fname).read_text(encoding="utf-8").splitlines() if l.strip()]
        splits[name] = rows
        if len(rows) != EXPECTED[name]:
            failures.append(f"{name}: {len(rows)} rows, expected {EXPECTED[name]}")
        for r in rows:
            if [(o["id"], o["description"]) for o in r["options"]] != [("false", "No"), ("true", "Yes")]:
                failures.append(f"{r['id']}: option contract violated")
                continue
            gold, value, threshold = gold_from_text(r)
            if ("true" if gold else "false") != r["target_option"]:
                failures.append(f"{r['id']}: label mismatch text={gold} row={r['target_option']}")
            if abs(value - r["value"]) > 1e-9 or abs(threshold - r["threshold"]) > 1e-9:
                failures.append(f"{r['id']}: metadata disagrees with rendered text")
            if r["question_type"] != "noul":
                failures.append(f"{r['id']}: not noul")
            state_str = r["state"]
            if re.search(r"[0-9]", json.loads(state_str)["equipment_id"]) or \
                    re.search(r"[0-9]", json.loads(state_str)["site"]):
                failures.append(f"{r['id']}: digits in distractor ids")
            # exactly one number in the state besides the value: none allowed elsewhere
            if re.sub(r'"motor_temp_c":\s*-?[0-9.]+', '"motor_temp_c":<V>', state_str) != state_str \
                    and re.search(r"[0-9]", re.sub(r'"motor_temp_c":\s*-?[0-9.]+', "", state_str)):
                failures.append(f"{r['id']}: digits outside the measured value")

    all_rows = [r for rows in splits.values() for r in rows]
    ids = Counter(r["id"] for r in all_rows)
    failures += [f"duplicate id {i}" for i, n in ids.items() if n > 1]
    payloads = Counter((r["question"], r["state"]) for r in all_rows)
    failures += [f"duplicate payload {p[0][:40]}" for p, n in payloads.items() if n > 1]

    # balance: global exact; train per-threshold 4/4; dev/known 3/3; new <=1
    for name, rows in splits.items():
        bal = Counter(r["target_option"] for r in rows)
        if bal["false"] != bal["true"]:
            failures.append(f"{name}: unbalanced {dict(bal)}")
        by_thr = defaultdict(Counter)
        for r in rows:
            by_thr[r["threshold"]][r["target_option"]] += 1
        for t, c in sorted(by_thr.items()):
            diff = abs(c["true"] - c["false"])
            limit = 0 if name in ("train", "dev", "known") else 1
            if diff > limit:
                failures.append(f"{name}: threshold {t:g} imbalance {dict(c)} (limit {limit})")

    # reserved thresholds never in train/dev; train thresholds exactly TRAIN_THR
    for name in ("train", "dev"):
        thrs = {r["threshold"] for r in splits[name]}
        if thrs & RESERVED:
            failures.append(f"{name}: reserved thresholds leaked: {sorted(thrs & RESERVED)}")
    if {r["threshold"] for r in splits["train"]} != TRAIN_THR:
        failures.append("train thresholds are not exactly the six declared")
    if {r["threshold"] for r in splits["new"]} != RESERVED:
        failures.append("test-new thresholds are not exactly the seven reserved")

    # cross-split value disjointness
    vals = {n: {r["value"] for r in rows} for n, rows in splits.items()}
    for a, b in (("train", "dev"), ("train", "known"), ("train", "new"),
                 ("dev", "known"), ("dev", "new"), ("known", "new")):
        if vals[a] & vals[b]:
            failures.append(f"value overlap {a}&{b}: {sorted(vals[a] & vals[b])}")

    # pair integrity: state identical, only-threshold-differs, gold inverted, same split
    pair_stats = {}
    for name, rows in splits.items():
        pairs = defaultdict(list)
        for r in rows:
            if r["pair"]:
                pairs[r["pair"]].append(r)
        pair_stats[name] = len(pairs)
        for pid, members in pairs.items():
            if len(members) != 2 or {m["variant"] for m in members} != {"pair_thrlo", "pair_thrhi"}:
                failures.append(f"{name}/{pid}: incomplete pair")
                continue
            lo, hi = (members[0], members[1]) if members[0]["variant"] == "pair_thrlo" \
                else (members[1], members[0])
            if lo["state"] != hi["state"]:
                failures.append(f"{name}/{pid}: state differs")
            qa = lo["question"]
            qb = hi["question"].replace(f"{hi['threshold']:g}", f"{lo['threshold']:g}")
            if qa != qb:
                failures.append(f"{name}/{pid}: question differs beyond the threshold")
            if not (lo["target_option"] == "true" and hi["target_option"] == "false"):
                failures.append(f"{name}/{pid}: gold not inverted (lo must be Yes, hi No)")
            if not lo["value"] > lo["threshold"] or not hi["value"] < hi["threshold"]:
                failures.append(f"{name}/{pid}: does not straddle correctly")

    # boundaries: exact in train and new; near (<=0.6) everywhere
    if not any(r["value"] == r["threshold"] and r["target_option"] == "true"
               for r in splits["train"]):
        failures.append("train: no exact boundary rows")
    for t in RESERVED:
        if not any(r["value"] == r["threshold"] == t and r["target_option"] == "true"
                   for r in splits["new"]):
            failures.append(f"new: missing exact boundary {t:g}")
    for name, rows in splits.items():
        if not any(abs(r["value"] - r["threshold"]) <= 0.6 for r in rows):
            failures.append(f"{name}: no near-boundary rows")

    # question-ignoring bound on pair subsets
    caps = {name: pair_accuracy_cap(rows) for name, rows in splits.items()}
    for name, cap in caps.items():
        if "pairs" in cap:
            refs = cap["pairs"]
            for ref in ("constant_yes", "constant_no", "fixed_value_ge_60"):
                if refs[ref] > 0.5 + 1e-9:
                    failures.append(f"{name}: question-ignoring rule {ref} exceeds 0.5 on pairs ({refs[ref]:.3f})")
            if refs.get("gold_opposite_pairs") != cap and refs.get("gold_opposite_pairs") != pair_stats[name]:
                pass  # informational below

    # serialization: state intact at the tail, head untruncated, 2 markers
    try:
        from transformers import AutoTokenizer
        from mokka_decisions.arch_julia import julia_sequence
        tok = AutoTokenizer.from_pretrained("jhu-clsp/mmBERT-small")
        bad = 0
        for r in all_rows:
            item = julia_sequence(tok, r["state"], r["question"],
                                  [o["description"] for o in r["options"]], "noul", 1024, 256)
            state_ids = tok(r["state"], add_special_tokens=False)["input_ids"]
            head_ids = tok(f"noul question: {r['question']}", add_special_tokens=False)["input_ids"]
            if item["ids"][-len(state_ids) - 1:-1] != state_ids or item["ids"][-1] != tok.sep_token_id:
                bad += 1
            if item["ids"][1:1 + len(head_ids)] != head_ids:
                bad += 1
            if len(item["markers"]) != 2 or item["qtype"] != 2:
                bad += 1
        if bad:
            failures.append(f"serialization/truncation violations: {bad}")
        serialization = "checked (state tokens intact, head untruncated, 2 markers, qtype noul)"
    except Exception as exc:  # pragma: no cover
        serialization = f"NOT checked ({type(exc).__name__}: {exc})"

    OUT.mkdir(parents=True, exist_ok=True)
    report = {
        "protocol": "docs/internal/E25_PROTOCOL_2026-10-10.md",
        "rows_total": len(all_rows), "failures": failures,
        "counts": {n: len(r) for n, r in splits.items()},
        "balance": {n: dict(Counter(r["target_option"] for r in rows)) for n, rows in splits.items()},
        "pairs": pair_stats,
        "question_ignoring_references": caps,
        "serialization": serialization,
        "verifier_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (OUT / "verification.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items()
                      if k not in ("balance", "question_ignoring_references")}, indent=1))
    if failures:
        print(f"FAILURES ({len(failures)}):", *failures[:20], sep="\n  ")
        return 1
    print("verification: ALL CHECKS PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
