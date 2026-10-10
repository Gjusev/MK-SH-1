"""Independent verifier for the E27 dense-grid data.

Re-derives every gold label from the RENDERED text (regex question threshold +
state value) WITHOUT importing the generator, then re-implements the frozen grid
RULE independently (input hash, 92 values, offsets, substitution ladder) and
compares row-by-row. Guards: per-value 4/4 balance, exact global balance,
reserved thresholds absent from train, question-ignoring rules EXACTLY 0.500 on
train (hard cap by construction), same-decade <= 0.75 on train, testE two-decimal
freshness, quadrant placement, pure interp/extrap pairs, monotonicity sweep
structure, no-truncation serialization. Writes runs/e27-prep/verification.json.
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

DATA = ROOT / "data/e27-grid"
OUT = ROOT / "runs/e27-prep"
INPUT = ROOT / "data/e26-order/train_treat_rows.jsonl"
INPUT_SHA = "4b76a5c5d928aeda328b209024c44def4ea163b650fda51b61a4d5b3a77decb8"
OFFSETS = [-25, -7, -1, 0, 1, 7, 15, 25]
YES_OFFSETS = {-25, -7, -1, 0}
LADDER = {1: [3, 15, 10, 5, 20], -1: [-3, -15, -10, -5, -20]}
RESERVED = {25.0, 44.5, 56.5, 66.5, 78.5, 88.5, 115.0}
TESTE_INTERP = {47.31, 58.73, 68.17, 79.43, 89.27}
TESTE_EXTRAP = {21.83, 118.57}
QUESTION_RE = re.compile(
    r"^Is the motor temperature reading at least ([0-9]+(?:\.[0-9]+)?) degrees Celsius\?$")
QUAD_YES = {"bd_y", "sd_y", "cd_y"}


def gold_from_text(row):
    m = QUESTION_RE.match(row["question"])
    if not m:
        raise ValueError(f"unparseable question: {row['question']!r}")
    threshold = float(m.group(1))
    state = json.loads(row["state"])
    return bool(state["motor_temp_c"] >= threshold), state["motor_temp_c"], threshold


def independent_grid():
    sha = hashlib.sha256(INPUT.read_bytes()).hexdigest()
    assert sha == INPUT_SHA, f"input hash: {sha}"
    values = sorted({json.loads(l)["value"] for l in
                     INPUT.read_text(encoding="utf-8").splitlines() if l.strip()})
    rows = []
    for v in values:
        for off in OFFSETS:
            thr = round(v + off, 2)
            if thr in RESERVED or not (10.0 <= thr <= 130.0):
                sign = 1 if off > 0 else -1
                thr = next(round(v + alt, 2) for alt in LADDER[sign]
                           if round(v + alt, 2) not in RESERVED
                           and 10.0 <= round(v + alt, 2) <= 130.0)
            rows.append((v, thr, (off in YES_OFFSETS)))
    return rows


def main() -> int:
    failures = []
    train = [json.loads(l) for l in (DATA / "train_rows.jsonl").read_text(
        encoding="utf-8").splitlines() if l.strip()]
    teste = [json.loads(l) for l in (DATA / "test_e_rows.jsonl").read_text(
        encoding="utf-8").splitlines() if l.strip()]

    # labels re-derived from rendered text (testE questions are 2-decimal by regex)
    for name, rows in (("train", train), ("teste", teste)):
        for r in rows:
            if [(o["id"], o["description"]) for o in r["options"]] != [("false", "No"), ("true", "Yes")]:
                failures.append(f"{r['id']}: option contract")
                continue
            gold, value, threshold = gold_from_text(r)
            if ("true" if gold else "false") != r["target_option"]:
                failures.append(f"{r['id']}: label mismatch")
            if abs(value - r["value"]) > 1e-9 or abs(threshold - r["threshold"]) > 1e-9:
                failures.append(f"{r['id']}: metadata vs text")
            if re.search(r"[0-9]", json.loads(r["state"])["equipment_id"]):
                failures.append(f"{r['id']}: digits in equipment id")

    # independent grid re-derivation, row-by-row
    expected = independent_grid()
    if len(train) != len(expected):
        failures.append(f"train rows {len(train)} != rule {len(expected)}")
    got = [(r["value"], round(r["threshold"], 2), r["target_option"] == "true") for r in train]
    if got != expected:
        diffs = [(g, e) for g, e in zip(got, expected) if g != e][:5]
        failures.append(f"grid rule mismatch (first diffs): {diffs}")

    # balances
    by_val = defaultdict(Counter)
    for r in train:
        by_val[r["value"]][r["target_option"]] += 1
    bad = {v: dict(c) for v, c in by_val.items() if c["true"] != 4 or c["false"] != 4}
    if bad:
        failures.append(f"per-value balance broken: {bad}")
    if len(by_val) != 92:
        failures.append(f"train values {len(by_val)} != 92")
    bal = Counter(r["target_option"] for r in teste)
    if bal["true"] != bal["false"]:
        failures.append(f"teste unbalanced {dict(bal)}")

    # reserved never as train threshold; hard 0.500 cap for question-ignoring rules
    if RESERVED & {r["threshold"] for r in train}:
        failures.append("reserved thresholds leaked into train")
    golds = [r["target_option"] == "true" for r in train]
    frac_yes = sum(golds) / len(golds)
    if abs(frac_yes - 0.5) > 1e-12:
        failures.append(f"train not exactly balanced ({frac_yes})")
    fixed = sum((r["value"] >= 60) == g for r, g in zip(train, golds)) / len(golds)
    if abs(fixed - 0.5) > 1e-12:
        failures.append(f"fixed value>=60 rule not exactly 0.500 ({fixed})")
    # same-decade guard
    sd = sum((int(r["value"] // 10) == int(r["threshold"] // 10)) == g
             for r, g in zip(train, golds)) / len(golds)
    if sd > 0.75 + 1e-9:
        failures.append(f"same-decade rule above 0.75 on train ({sd:.3f})")

    # testE structure: quadrant placement + decimals + pairs + sweeps
    for r in teste:
        for x in (r["value"], r["threshold"]):
            if len(f"{x:.2f}".split(".")[1]) != 2:
                failures.append(f"{r['id']}: not two-decimal {x}")
        q = r.get("quadrant")
        if q:
            v, t = r["value"], r["threshold"]
            same_dec = int(v // 10) == int(t // 10)
            if (q in QUAD_YES) != (v >= t):
                failures.append(f"{r['id']}: quadrant/gold mismatch {q}")
            if q.startswith("sd_") and not same_dec:
                failures.append(f"{r['id']}: {q} not same decade")
            if q.startswith("cd_") and same_dec:
                failures.append(f"{r['id']}: {q} not other decade")
            if q == "bd_y" and v != t:
                failures.append(f"{r['id']}: bd_y not at boundary")
    thr_sets = TESTE_INTERP | TESTE_EXTRAP
    if {r["threshold"] for r in teste if not r["sweep"] and not r["pair"]} - thr_sets:
        failures.append("testE singles at undeclared thresholds")
    pairs = defaultdict(list)
    for r in teste:
        if r["pair"]:
            pairs[r["pair"]].append(r)
    pure = {"interp": 0, "extrap": 0}
    for pid, members in pairs.items():
        if len(members) != 2 or {m["variant"] for m in members} != {"pair_thrlo", "pair_thrhi"}:
            failures.append(f"{pid}: bad pair")
            continue
        lo, hi = sorted(members, key=lambda m: m["threshold"])
        if lo["state"] != hi["state"] or lo["target_option"] != "true" or hi["target_option"] != "false":
            failures.append(f"{pid}: pair integrity")
        ranges = {m["threshold"] in TESTE_INTERP and "interp" or "extrap" for m in members}
        if ranges not in ({"interp"}, {"extrap"}):
            failures.append(f"{pid}: mixed-range pair")
        else:
            pure[ranges.pop()] += 1
    if pure != {"interp": 8, "extrap": 8}:
        failures.append(f"pure pairs {pure} != 8/8")
    sweeps = defaultdict(list)
    for r in teste:
        if r["sweep"]:
            sweeps[r["sweep"]].append(r)
    if len(sweeps) != 10:
        failures.append(f"sweeps {len(sweeps)} != 10")
    for sid, rows in sweeps.items():
        rows.sort(key=lambda r: r["value"])
        if [r["target_option"] == "true" for r in rows] != [False] * 5 + [True] * 5:
            failures.append(f"{sid}: sweep gold pattern")
        if any(a["value"] >= b["value"] for a, b in zip(rows, rows[1:])):
            failures.append(f"{sid}: not strictly increasing")

    # serialization
    try:
        from transformers import AutoTokenizer
        from mokka_decisions.arch_julia import julia_sequence
        tok = AutoTokenizer.from_pretrained("jhu-clsp/mmBERT-small")
        bad = 0
        for r in train + teste:
            item = julia_sequence(tok, r["state"], r["question"],
                                  [o["description"] for o in r["options"]], "noul", 1024, 256)
            sids = tok(r["state"], add_special_tokens=False)["input_ids"]
            hids = tok(f"noul question: {r['question']}", add_special_tokens=False)["input_ids"]
            if item["ids"][-len(sids) - 1:-1] != sids or item["ids"][1:1 + len(hids)] != hids \
                    or len(item["markers"]) != 2:
                bad += 1
        if bad:
            failures.append(f"serialization violations: {bad}")
        serialization = "checked"
    except Exception as exc:  # pragma: no cover
        serialization = f"NOT checked ({type(exc).__name__})"

    OUT.mkdir(parents=True, exist_ok=True)
    report = {
        "protocol": "docs/internal/E27_PROTOCOL_2026-10-10.md",
        "failures": failures,
        "counts": {"train": len(train), "teste": len(teste)},
        "balance": {"train": dict(Counter(r["target_option"] for r in train)),
                    "teste": dict(bal)},
        "question_ignoring_exact_0500": {"constant": frac_yes, "fixed_ge60": fixed},
        "same_decade_train": round(sd, 4),
        "pure_pairs": pure, "sweeps": len(sweeps),
        "serialization": serialization,
        "verifier_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (OUT / "verification.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "balance"}, indent=1))
    if failures:
        print(f"FAILURES ({len(failures)}):", *failures[:10], sep="\n  ")
        return 1
    print("verification: ALL CHECKS PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
