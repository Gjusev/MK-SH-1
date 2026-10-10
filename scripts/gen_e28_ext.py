"""E28 generator: extremes + monotone chains on top of the frozen E27 grid.

docs/internal/E28_PROTOCOL_2026-10-10.md. Base = the E27 train rows VERBATIM
(file pinned by sha256); +16 extreme values (one decimal, frozen literals,
outside the E26 value range) each with the same 8-offset grid and substitution
ladder; +8 monotone chains (frozen thresholds, 10 ascending fresh values each,
5 No / 5 Yes) teaching the value-dimension at fixed threshold that the mono
sweeps measure. Guards: global balance exact, reserved absent, chain values
nudged deterministically away from any existing value, question-ignoring
constants exactly 0.500. scripts/verify_e28_ext.py re-derives independently.

Outputs: data/e28-ext/{train_rows.jsonl, manifest.json}.
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

FIELD = "motor_temp_c"
FIELD_WORD = "motor temperature"
OP = "at least"
BASE = ROOT / "data/e27-grid/train_rows.jsonl"
OFFSET_SETS = None  # set below after BASE load
OFFSETS = [-25, -7, -1, 0, 1, 7, 15, 25]
YES_OFFSETS = {-25, -7, -1, 0}
LADDER = {1: [3, 15, 10, 5, 20], -1: [-3, -15, -10, -5, -20]}
RESERVED = {25.0, 44.5, 56.5, 66.5, 78.5, 88.5, 115.0}
THR_MIN, THR_MAX = 10.0, 130.0
SITES = ["site-alpha", "site-bravo", "site-charlie", "site-delta", "site-echo", "site-foxtrot"]
_LETTERS = "abcdefghjkmnpqrstuvwxyz"

EXTREME_VALUES = [18.4, 19.7, 20.9, 21.6, 22.8, 23.5, 24.7, 25.4,
                  108.3, 109.6, 111.2, 112.8, 113.5, 116.1, 122.4, 124.6]
CHAIN_THRESHOLDS = [23.0, 27.5, 33.0, 47.0, 63.0, 84.0, 110.0, 121.0]
CHAIN_OFFSETS = [-4.2, -3.1, -2.3, -1.4, -0.6, 0.4, 1.3, 2.6, 3.5, 4.8]
E26_ORDER = ROOT / "data/e26-order"


def eq_codes():
    i = 0
    n = len(_LETTERS)
    while True:
        j, chars = i, []
        while True:
            chars.append(_LETTERS[j % n])
            j //= n
            if j == 0:
                break
        yield "unit-" + "".join(reversed(chars))
        i += 1


def fmt(x: float) -> str:
    return f"{x:g}"


def existing_values():
    """Every value that already exists anywhere (train values + eval values)."""
    vals = set()
    for fname in ("train_treat_rows.jsonl", "dev_rows.jsonl", "test_c_rows.jsonl",
                  "test_d_rows.jsonl", "mono_rows.jsonl"):
        for l in (E26_ORDER / fname).read_text(encoding="utf-8").splitlines():
            vals.add(json.loads(l)["value"])
    return vals


def main() -> int:
    base_rows = [json.loads(l) for l in BASE.read_text(encoding="utf-8").splitlines() if l.strip()]
    base_sha = hashlib.sha256(BASE.read_bytes()).hexdigest()
    assert len(base_rows) == 736, len(base_rows)
    taken = existing_values()
    assert not (set(EXTREME_VALUES) & taken), set(EXTREME_VALUES) & taken
    assert not (set(CHAIN_THRESHOLDS) & RESERVED)

    codes = eq_codes()
    rows = list(base_rows)
    seq = len(rows) + 1
    substitutions = []
    nudge_log = []

    def add(value, threshold, *, source):
        nonlocal seq
        eq = f"unit-{'zz'}"  # placeholder replaced below
        eq = next(codes)
        site = SITES[len(rows) % len(SITES)]
        state = json.dumps({"equipment_id": eq, FIELD: value, "site": site}, sort_keys=True)
        question = f"Is the {FIELD_WORD} reading at least {fmt(threshold)} degrees Celsius?"
        gold = value >= threshold
        rows.append({"id": f"e28|{source}|{seq:04d}", "group_id": f"e28|{source}|{seq:04d}",
                     "source": "e28-ext", "language": "en", "domain": f"num_{FIELD}",
                     "state": state, "question": question,
                     "options": [{"id": "false", "description": "No"},
                                 {"id": "true", "description": "Yes"}],
                     "target_kind": "single", "target_option": "true" if gold else "false",
                     "split": "train", "question_type": "noul", "label_origin": "e28-spec",
                     "operator": OP, "feature": FIELD, "threshold": threshold, "value": value,
                     "variant": "grid_ext" if source == "ext" else "chain",
                     "pair": None, "quadrant": None,
                     "sweep": f"e28c|{fmt(threshold)}" if source == "chain" else None,
                     "rule_in": "question", "thr_range": "grid"})
        seq += 1

    # 16 extreme values with the frozen E27 grid rule
    for v in EXTREME_VALUES:
        for off in OFFSETS:
            thr = round(v + off, 2)
            if thr in RESERVED or not (THR_MIN <= thr <= THR_MAX):
                sign = 1 if off > 0 else -1
                for alt in LADDER[sign]:
                    cand = round(v + alt, 2)
                    if cand not in RESERVED and THR_MIN <= cand <= THR_MAX:
                        substitutions.append({"value": v, "original_offset": off,
                                              "substituted_offset": alt, "threshold": cand})
                        thr = cand
                        break
                else:
                    raise AssertionError(f"no ladder candidate for v={v} off={off}")
            add(v, thr, source="ext")

    # 8 monotone chains: 10 ascending fresh values per frozen threshold (5 No, 5 Yes);
    # deterministic nudge away from any taken value (incl. the new extremes)
    taken |= set(EXTREME_VALUES)
    for t in CHAIN_THRESHOLDS:
        for off in CHAIN_OFFSETS:
            v = round(t + off, 1)
            for bump in (0.0, 0.1, -0.1, 0.2, -0.2, 0.3, -0.3, 0.4, -0.4):
                cand = round(v + bump, 1)
                if cand not in taken and cand != t and 10.0 <= cand <= 130.0:
                    v = cand
                    break
            else:
                raise AssertionError(f"no free chain value for t={t} off={off}")
            if v != round(t + off, 1):
                nudge_log.append({"threshold": t, "formula_value": round(t + off, 1), "used": v})
            taken.add(v)
            add(v, t, source="chain")

    # guards
    ids = [r["id"] for r in rows]
    payloads = [(r["question"], r["state"]) for r in rows]
    assert len(ids) == len(set(ids)) and len(payloads) == len(set(payloads))
    assert len(rows) == 736 + 128 + 80 == 944
    bal = Counter(r["target_option"] for r in rows)
    assert bal["true"] == bal["false"] == 472, dict(bal)
    grid_vals = {r["value"] for r in rows if r["variant"] != "chain"}
    from collections import defaultdict
    by_val = defaultdict(Counter)
    for r in rows:
        if r["variant"] != "chain":
            by_val[r["value"]][r["target_option"]] += 1
    assert all(c["true"] == 4 and c["false"] == 4 for c in by_val.values())
    assert not RESERVED & {r["threshold"] for r in rows}
    # chains: strictly ascending, 5 No then 5 Yes
    sweeps = defaultdict(list)
    for r in rows:
        if r["variant"] == "chain":
            sweeps[r["sweep"]].append(r)
    assert len(sweeps) == 8
    for sid, members in sweeps.items():
        members.sort(key=lambda r: r["value"])
        assert all(a["value"] < b["value"] for a, b in zip(members, members[1:])), sid
        assert [m["target_option"] == "true" for m in members] == [False] * 5 + [True] * 5, sid

    out = ROOT / "data/e28-ext"
    out.mkdir(parents=True, exist_ok=True)
    (out / "train_rows.jsonl").write_text(
        "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows), encoding="utf-8")
    manifest = {
        "set": "e28-ext", "protocol": "docs/internal/E28_PROTOCOL_2026-10-10.md",
        "base_e27_train_sha256": base_sha,
        "extreme_values": EXTREME_VALUES, "chain_thresholds": CHAIN_THRESHOLDS,
        "chain_value_formula": "t + [-4.2,-3.1,-2.3,-1.4,-0.6,0.4,1.3,2.6,3.5,4.8], deterministic nudge ladder +0.1/-0.1..+0.4/-0.4 on collision",
        "substitutions": substitutions, "nudges": nudge_log,
        "counts": {"train": len(rows)},
        "balance": dict(bal),
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True),
                                        encoding="utf-8")
    print(json.dumps({"rows": len(rows), "balance": dict(bal),
                      "substitutions": substitutions, "nudges": nudge_log[:6]}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
