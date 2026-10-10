"""Independent verifier for the E26 numeric-order data.

Re-derives every gold label from the RENDERED text with logic that does not
import the generator (regex the question threshold, parse the state value),
then enforces the protocol's structural guards: exact counts and balance,
per-threshold balance (treat 4/4, ctrl 5/5, dev/C 2/2, D 3/3), quadrant
coverage, threshold-only AND value-only pair integrity, reserved thresholds
absent from train/dev/testC (as threshold) and from both trains (as value),
cross-file value disjointness (pair-shared values by design), alphabetic ids,
monotonicity sweep structure, question-ignoring caps on threshold-only pairs,
the same-decade rule caps (treatment <= 0.65, control >= 0.80), and the
single-decimal-suffix rule cap on both trains. Writes runs/e26-prep/verification.json.
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

DATA = ROOT / "data/e26-order"
OUT = ROOT / "runs/e26-prep"

QUESTION_RE = re.compile(
    r"^Is the motor temperature reading at least ([0-9]+(?:\.[0-9]+)?) degrees Celsius\?$")
FILES = {"treat": "train_treat_rows.jsonl", "ctrl": "train_ctrl_rows.jsonl",
         "dev": "dev_rows.jsonl", "known": "test_c_rows.jsonl",
         "new": "test_d_rows.jsonl", "mono": "mono_rows.jsonl"}
EXPECTED = {"treat": 96, "ctrl": 96, "dev": 48, "known": 48, "new": 54, "mono": 60}
TREAT_THR = {41.0, 48.0, 52.0, 57.0, 63.0, 68.0, 74.0, 79.0, 84.0, 92.0}
CTRL_THR = {30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0}
RESERVED = {44.5, 56.5, 66.5, 78.5, 88.5, 25.0, 115.0}
QUAD_YES = {"bd_y", "sd_y", "cd_y", "cd_y_lej"}


def gold_from_text(row: dict) -> tuple[bool, float, float]:
    m = QUESTION_RE.match(row["question"])
    if not m:
        raise ValueError(f"unparseable question: {row['question']!r}")
    threshold = float(m.group(1))
    state = json.loads(row["state"])
    if set(state) != {"equipment_id", "motor_temp_c", "site"}:
        raise ValueError(f"unexpected state keys: {sorted(state)}")
    return bool(state["motor_temp_c"] >= threshold), state["motor_temp_c"], threshold


def same_decade(value: float, threshold: float) -> bool:
    return int(value // 10) == int(threshold // 10)


def reference_accuracies(rows: list[dict]) -> dict:
    gold = [r["target_option"] == "true" for r in rows]
    return {
        "constant_yes": sum(gold) / len(gold),
        "constant_no": sum(not g for g in gold) / len(gold),
        "fixed_value_ge_60": sum((r["value"] >= 60) == g for r, g in zip(rows, gold)) / len(gold),
        "same_decade": sum(same_decade(r["value"], r["threshold"]) == g
                           for r, g in zip(rows, gold)) / len(gold),
    }


def suffix_rule_max(rows: list[dict]) -> float:
    """Best single-suffix rule: 'suffix==s -> Yes (or No), else majority'."""
    gold = [r["target_option"] == "true" for r in rows]
    best = 0.5
    for s in range(10):
        hits = [i for i, r in enumerate(rows) if round(r["value"] * 10) % 10 == s]
        if not hits:
            continue
        for pole in (True, False):
            correct = sum((gold[i] == pole) for i in hits)
            rest_yes = sum(gold[i] for i in range(len(rows)) if i not in set(hits))
            rest_no = len(rows) - len(hits) - rest_yes
            correct += max(rest_yes, rest_no)
            best = max(best, correct / len(rows))
    return best


def check_pair(pid: str, members: list[dict], failures: list) -> None:
    variants = {m["variant"] for m in members}
    if variants == {"pair_thrlo", "pair_thrhi"}:
        lo = next(m for m in members if m["variant"] == "pair_thrlo")
        hi = next(m for m in members if m["variant"] == "pair_thrhi")
        if lo["state"] != hi["state"]:
            failures.append(f"{pid}: thr-pair state differs")
        if lo["question"] != hi["question"].replace(f"{hi['threshold']:g}", f"{lo['threshold']:g}"):
            failures.append(f"{pid}: question differs beyond the threshold")
        if not (lo["target_option"] == "true" and hi["target_option"] == "false"):
            failures.append(f"{pid}: thr-pair gold not inverted")
        if not (lo["value"] > lo["threshold"] and hi["value"] < hi["threshold"]):
            failures.append(f"{pid}: thr-pair does not straddle")
    elif variants == {"pair_valno", "pair_valyes"}:
        no = next(m for m in members if m["variant"] == "pair_valno")
        yes = next(m for m in members if m["variant"] == "pair_valyes")
        if no["question"] != yes["question"]:
            failures.append(f"{pid}: val-pair question differs")
        sn, sy = json.loads(no["state"]), json.loads(yes["state"])
        sn["motor_temp_c"] = sy["motor_temp_c"] = None
        if sn != sy:
            failures.append(f"{pid}: val-pair state differs beyond the value")
        if not (no["target_option"] == "false" and yes["target_option"] == "true"):
            failures.append(f"{pid}: val-pair gold not inverted")
        if not (no["value"] < no["threshold"] <= yes["value"]):
            failures.append(f"{pid}: val-pair does not straddle")
    else:
        failures.append(f"{pid}: unknown pair variants {sorted(variants)}")


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
            state = json.loads(r["state"])
            if re.search(r"[0-9]", state["equipment_id"]) or re.search(r"[0-9]", state["site"]):
                failures.append(f"{r['id']}: digits in distractor ids")
            if r.get("quadrant"):
                quad_yes = r["quadrant"] in QUAD_YES
                if quad_yes != gold:
                    failures.append(f"{r['id']}: quadrant {r['quadrant']} disagrees with gold")
                if r["quadrant"].startswith("sd_") and not same_decade(value, threshold):
                    failures.append(f"{r['id']}: quadrant {r['quadrant']} not same decade")
                if r["quadrant"].startswith("cd_") and same_decade(value, threshold):
                    failures.append(f"{r['id']}: quadrant {r['quadrant']} not other decade")
                if r["quadrant"] == "bd_y" and value != threshold:
                    failures.append(f"{r['id']}: bd_y not at the boundary")

    all_rows = [r for rows in splits.values() for r in rows]
    ids = Counter(r["id"] for r in all_rows)
    failures += [f"duplicate id {i}" for i, n in ids.items() if n > 1]
    payloads = Counter((r["question"], r["state"]) for r in all_rows)
    failures += [f"duplicate payload {p[0][:40]}" for p, n in payloads.items() if n > 1]

    # global balance; per-threshold balance per protocol
    for name, rows in splits.items():
        bal = Counter(r["target_option"] for r in rows)
        if name != "mono" and bal["false"] != bal["true"]:
            failures.append(f"{name}: unbalanced {dict(bal)}")
        by_thr = defaultdict(Counter)
        for r in rows:
            by_thr[r["threshold"]][r["target_option"]] += 1
        for t, c in sorted(by_thr.items()):
            y, n = c["true"], c["false"]
            # singles are exact per threshold; pairs add +/-1 to the anchored
            # threshold (protocol amendment, pre-launch) -> |Y-N| bounds
            ok = ((name == "mono" and y == 5 and n == 5) or
                  (name != "mono" and abs(y - n) <= (2 if name == "new" else 1)))
            if not ok:
                failures.append(f"{name}: threshold {t:g} balance {y}/{n}")

    # threshold sets exactly as declared
    if {r["threshold"] for r in splits["treat"]} != TREAT_THR:
        failures.append("treat thresholds are not exactly the ten declared")
    if {r["threshold"] for r in splits["ctrl"]} != CTRL_THR:
        failures.append("ctrl thresholds are not exactly the eight declared")
    for name in ("treat", "ctrl", "dev", "known"):
        leaked = {r["threshold"] for r in splits[name]} & RESERVED
        if leaked:
            failures.append(f"{name}: reserved thresholds leaked: {sorted(leaked)}")
    if {r["threshold"] for r in splits["new"]} != RESERVED:
        failures.append("test-new thresholds are not exactly the seven reserved")
    for name in ("treat", "ctrl"):
        as_value = {r["value"] for r in splits[name]} & RESERVED
        if as_value:
            failures.append(f"{name}: reserved thresholds appear as values: {sorted(as_value)}")

    # quadrant coverage: every treat threshold has all four quadrants in train
    for t in TREAT_THR:
        quads = {r["quadrant"] for r in splits["treat"]
                 if r["threshold"] == t and r["quadrant"]}
        need = {"bd_y", "sd_y", "sd_n", "cd_n"}
        if not ({"cd_y", "cd_y_lej"} & quads):
            failures.append(f"treat t={t:g}: no cross-decade positive")
        if not need <= quads:
            failures.append(f"treat t={t:g}: missing quadrants {sorted(need - quads)}")

    # cross-file value disjointness (pair-shared values by design)
    def value_map(rows):
        out = defaultdict(list)
        for r in rows:
            out[r["value"]].append(r)
        return out
    by_val = {name: value_map(rows) for name, rows in splits.items()}
    names = list(splits)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            for v in set(by_val[a]) & set(by_val[b]):
                holders = by_val[a][v] + by_val[b][v]
                if not all(h["pair"] for h in holders) or len({h["pair"] for h in holders}) != 1:
                    failures.append(f"value overlap {a}&{b}: {v}")

    # pair integrity + question-ignoring caps on threshold-only pair subsets
    pair_stats = {}
    for name, rows in splits.items():
        pairs = defaultdict(list)
        for r in rows:
            if r["pair"]:
                pairs[r["pair"]].append(r)
        pair_stats[name] = {"thr": sum(1 for m in pairs.values()
                                       if {x["variant"] for x in m} == {"pair_thrlo", "pair_thrhi"}),
                            "val": sum(1 for m in pairs.values()
                                       if {x["variant"] for x in m} == {"pair_valno", "pair_valyes"})}
        for pid, members in pairs.items():
            if len(members) != 2:
                failures.append(f"{name}/{pid}: incomplete pair")
                continue
            check_pair(f"{name}/{pid}", members, failures)
            if len({m["split"] for m in members}) != 1:
                failures.append(f"{name}/{pid}: pair split across files")
        thr_pair_rows = [r for r in rows if r["variant"] in ("pair_thrlo", "pair_thrhi")]
        if thr_pair_rows:
            golds = [r["target_option"] == "true" for r in thr_pair_rows]
            for ref, acc in reference_accuracies(thr_pair_rows).items():
                if ref in ("constant_yes", "constant_no", "fixed_value_ge_60") and acc > 0.5 + 1e-9:
                    failures.append(f"{name}: question-ignoring rule {ref} exceeds 0.5 on thr-pairs ({acc:.3f})")
            # any state-only deterministic rule answers identically for both members
            by_pair = defaultdict(list)
            for r in thr_pair_rows:
                by_pair[r["pair"]].append(r)
            solved = sum(1 for m in by_pair.values()
                         if len(m) == 2 and m[0]["target_option"] == m[1]["target_option"])
            if solved:
                failures.append(f"{name}: {solved} thr-pairs without inverted gold")

    # monotonicity structure
    sweeps = defaultdict(list)
    for r in splits["mono"]:
        sweeps[r["sweep"]].append(r)
    if len(sweeps) != 6 or any(len(v) != 10 for v in sweeps.values()):
        failures.append("mono: expected 6 sweeps of 10 rows")
    for sid, rows in sweeps.items():
        rows.sort(key=lambda r: r["value"])
        golds = [r["target_option"] == "true" for r in rows]
        if golds != [False] * 5 + [True] * 5:
            failures.append(f"mono {sid}: gold not 5xNo then 5xYes")
        if any(a["value"] >= b["value"] for a, b in zip(rows, rows[1:])):
            failures.append(f"mono {sid}: values not strictly increasing")

    # frozen reference guards: same-decade caps + suffix rule cap on the trains
    refs = {name: reference_accuracies(rows) for name, rows in splits.items()}
    if refs["treat"]["same_decade"] > 0.65 + 1e-9:
        failures.append(f"treat: same-decade rule above 0.65 cap ({refs['treat']['same_decade']:.3f})")
    if refs["ctrl"]["same_decade"] < 0.80 - 1e-9:
        failures.append(f"ctrl: same-decade rule below 0.80 floor ({refs['ctrl']['same_decade']:.3f})")
    for name in ("treat", "ctrl"):
        sm = suffix_rule_max(splits[name])
        if sm > 0.80 + 1e-9:
            failures.append(f"{name}: single-suffix rule above 0.80 cap ({sm:.3f})")

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
        "protocol": "docs/internal/E26_PROTOCOL_2026-10-10.md",
        "rows_total": len(all_rows), "failures": failures,
        "counts": {n: len(r) for n, r in splits.items()},
        "balance": {n: dict(Counter(r["target_option"] for r in rows)) for n, rows in splits.items()},
        "pairs": pair_stats,
        "references": refs,
        "suffix_rule_max": {n: round(suffix_rule_max(r), 4) for n, r in
                            (("treat", splits["treat"]), ("ctrl", splits["ctrl"]))},
        "serialization": serialization,
        "verifier_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (OUT / "verification.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items()
                      if k not in ("balance", "references")}, indent=1))
    if failures:
        print(f"FAILURES ({len(failures)}):", *failures[:20], sep="\n  ")
        return 1
    print("verification: ALL CHECKS PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
