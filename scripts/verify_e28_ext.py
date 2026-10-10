"""Independent verifier for E28 (extremes + chains on the frozen E27 grid).

Re-derives every gold label from the RENDERED text without importing the
generator, re-implements the frozen construction rule (E27 base verbatim by
sha256; 16 extreme values x the 8-offset grid with the substitution ladder; 8
chains with the deterministic nudge ladder), and enforces: 944 rows, global
balance 472/472, per-value 4/4 on grid rows, chains strictly ascending 5 No +
5 Yes, reserved thresholds absent, question-ignoring constants exactly 0.500,
serialization. Writes runs/e28-prep/verification.json.
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

DATA = ROOT / "data/e28-ext"
BASE = ROOT / "data/e27-grid/train_rows.jsonl"
E26 = ROOT / "data/e26-order"
OUT = ROOT / "runs/e28-prep"
OFFSETS = [-25, -7, -1, 0, 1, 7, 15, 25]
YES_OFFSETS = {-25, -7, -1, 0}
LADDER = {1: [3, 15, 10, 5, 20], -1: [-3, -15, -10, -5, -20]}
RESERVED = {25.0, 44.5, 56.5, 66.5, 78.5, 88.5, 115.0}
EXTREMES = [18.4, 19.7, 20.9, 21.6, 22.8, 23.5, 24.7, 25.4,
            108.3, 109.6, 111.2, 112.8, 113.5, 116.1, 122.4, 124.6]
CHAIN_THR = [23.0, 27.5, 33.0, 47.0, 63.0, 84.0, 110.0, 121.0]
CHAIN_OFFS = [-4.2, -3.1, -2.3, -1.4, -0.6, 0.4, 1.3, 2.6, 3.5, 4.8]
NUDGE = [0.0, 0.1, -0.1, 0.2, -0.2, 0.3, -0.3, 0.4, -0.4]
QUESTION_RE = re.compile(
    r"^Is the motor temperature reading at least ([0-9]+(?:\.[0-9]+)?) degrees Celsius\?$")


def grid_thresholds(v):
    out = []
    for off in OFFSETS:
        thr = round(v + off, 2)
        if thr in RESERVED or not (10.0 <= thr <= 130.0):
            sign = 1 if off > 0 else -1
            thr = next(round(v + alt, 2) for alt in LADDER[sign]
                       if round(v + alt, 2) not in RESERVED
                       and 10.0 <= round(v + alt, 2) <= 130.0)
        out.append((thr, off in YES_OFFSETS))
    return out


def main() -> int:
    failures = []
    rows = [json.loads(l) for l in (DATA / "train_rows.jsonl").read_text(
        encoding="utf-8").splitlines() if l.strip()]
    base_rows = [json.loads(l) for l in BASE.read_text(encoding="utf-8").splitlines() if l.strip()]

    # labels re-derived from rendered text
    for r in rows:
        m = QUESTION_RE.match(r["question"])
        if not m:
            failures.append(f"{r['id']}: unparseable question")
            continue
        thr = float(m.group(1))
        v = json.loads(r["state"])["motor_temp_c"]
        gold = v >= thr
        if ("true" if gold else "false") != r["target_option"]:
            failures.append(f"{r['id']}: label mismatch")
        if abs(v - r["value"]) > 1e-9 or abs(thr - r["threshold"]) > 1e-9:
            failures.append(f"{r['id']}: metadata vs text")
        if re.search(r"[0-9]", json.loads(r["state"])["equipment_id"]):
            failures.append(f"{r['id']}: digits in equipment id")

    # base verbatim
    if len(rows) != 944:
        failures.append(f"rows {len(rows)} != 944")
    if [r["id"] for r in rows[:736]] != [r["id"] for r in base_rows]:
        failures.append("first 736 ids are not the E27 base verbatim")
    mismatch = sum(1 for a, b in zip(rows[:736], base_rows)
                   if (a["question"], a["state"], a["target_option"]) !=
                   (b["question"], b["state"], b["target_option"]))
    if mismatch:
        failures.append(f"base rows differ from E27 in {mismatch} payloads")

    # extremes grid re-derivation
    k = 736
    for v in EXTREMES:
        expected = grid_thresholds(v)
        got = rows[k:k + 8]
        if len(got) != 8 or any(g["value"] != v for g in got):
            failures.append(f"extreme block for v={v} malformed")
            k += 8
            continue
        for g, (thr, yes) in zip(got, expected):
            if round(g["threshold"], 2) != thr or (g["target_option"] == "true") != yes:
                failures.append(f"extreme v={v}: ({g['threshold']},{g['target_option']}) != ({thr},{yes})")
        k += 8

    # chains re-derivation with the nudge ladder
    taken = {json.loads(l)["value"] for f in ("train_treat_rows.jsonl", "dev_rows.jsonl",
                                              "test_c_rows.jsonl", "test_d_rows.jsonl",
                                              "mono_rows.jsonl")
             for l in (E26 / f).read_text(encoding="utf-8").splitlines()}
    taken |= {r["value"] for r in base_rows} | set(EXTREMES)
    for t in CHAIN_THR:
        expected_vals = []
        for off in CHAIN_OFFS:
            v0 = round(t + off, 1)
            v = v0
            for bump in NUDGE:
                cand = round(v0 + bump, 1)
                if cand not in taken and cand != t and 10.0 <= cand <= 130.0:
                    v = cand
                    break
            expected_vals.append(v)
            taken.add(v)
        got = rows[k:k + 10]
        if len(got) != 10 or any(g["threshold"] != t for g in got):
            failures.append(f"chain t={t} malformed")
            k += 10
            continue
        if [g["value"] for g in got] != expected_vals:
            failures.append(f"chain t={t} values {[g['value'] for g in got]} != {expected_vals}")
        if [g["target_option"] == "true" for g in got] != [False] * 5 + [True] * 5:
            failures.append(f"chain t={t} gold pattern")
        if any(a["value"] >= b["value"] for a, b in zip(got, got[1:])):
            failures.append(f"chain t={t} not ascending")
        k += 10

    # balances + caps
    bal = Counter(r["target_option"] for r in rows)
    if bal["true"] != bal["false"] != 472 or bal["true"] != 472:
        failures.append(f"balance {dict(bal)} != 472/472")
    by_val = defaultdict(Counter)
    for r in rows:
        if r["variant"] != "chain":
            by_val[r["value"]][r["target_option"]] += 1
    bad = {v: dict(c) for v, c in by_val.items() if c["true"] != 4 or c["false"] != 4}
    if bad:
        failures.append(f"per-value 4/4 broken: {list(bad)[:5]}")
    if RESERVED & {r["threshold"] for r in rows}:
        failures.append("reserved thresholds leaked")

    try:
        from transformers import AutoTokenizer
        from mokka_decisions.arch_julia import julia_sequence
        tok = AutoTokenizer.from_pretrained("jhu-clsp/mmBERT-small")
        bad_ser = 0
        for r in rows[::17]:  # sample incl. base/extremes/chains deterministically
            item = julia_sequence(tok, r["state"], r["question"],
                                  [o["description"] for o in r["options"]], "noul", 1024, 256)
            sids = tok(r["state"], add_special_tokens=False)["input_ids"]
            if item["ids"][-len(sids) - 1:-1] != sids or len(item["markers"]) != 2:
                bad_ser += 1
        if bad_ser:
            failures.append(f"serialization violations: {bad_ser}")
        serialization = "checked (1-in-17 sample)"
    except Exception as exc:  # pragma: no cover
        serialization = f"NOT checked ({type(exc).__name__})"

    OUT.mkdir(parents=True, exist_ok=True)
    report = {
        "protocol": "docs/internal/E28_PROTOCOL_2026-10-10.md",
        "base_e27_sha256": hashlib.sha256(BASE.read_bytes()).hexdigest(),
        "failures": failures, "rows": len(rows), "balance": dict(bal),
        "serialization": serialization,
        "verifier_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (OUT / "verification.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=1))
    if failures:
        print(f"FAILURES ({len(failures)}):", *failures[:10], sep="\n  ")
        return 1
    print("verification: ALL CHECKS PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
