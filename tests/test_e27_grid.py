"""E27 grid-data tests: hand literals + frozen-rule re-derivation + hard caps."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/e27-grid"
INPUT = ROOT / "data/e26-order/train_treat_rows.jsonl"
OFFSETS = [-25, -7, -1, 0, 1, 7, 15, 25]
YES = {-25, -7, -1, 0}
LADDER = {1: [3, 15, 10, 5, 20], -1: [-3, -15, -10, -5, -20]}
RESERVED = {25.0, 44.5, 56.5, 66.5, 78.5, 88.5, 115.0}


def load(name):
    return [json.loads(l) for l in (DATA / name).read_text(encoding="utf-8").splitlines() if l.strip()]


def test_hand_computed_grid_rows():
    """Frozen literal spot-checks incl. the five substitutions."""
    rows = {(r["value"], round(r["threshold"], 2)): r["target_option"] for r in load("train_rows.jsonl")}
    hand = [
        (92.0, 67.0, "true"),    # 92 - 25
        (92.0, 117.0, "false"),  # 92 + 25
        (92.0, 92.0, "true"),    # boundary offset 0
        (92.0, 91.0, "true"),    # 92 - 1
        (92.0, 93.0, "false"),   # 92 + 1
        (41.0, 41.0, "true"),    # value 41.0 boundary
        (27.3, 24.3, "true"),    # SUBSTITUTION: 27.3-25 out of range -> -3
        (31.2, 28.2, "true"),    # SUBSTITUTION: 31.2-25 out of range -> -3
        (87.5, 90.5, "false"),   # SUBSTITUTION: 87.5+1 = 88.5 reserved -> +3
        (91.5, 76.5, "true"),    # SUBSTITUTION: 91.5-25 = 66.5 reserved -> -15
        (106.9, 109.9, "false"), # SUBSTITUTION: 106.9+25 out of range -> +3
    ]
    for v, t, gold in hand:
        assert rows[(v, t)] == gold, (v, t, gold, rows.get((v, t)))


def test_rule_rederivation_matches_file():
    values = sorted({json.loads(l)["value"] for l in
                     INPUT.read_text(encoding="utf-8").splitlines() if l.strip()})
    assert len(values) == 92
    expected = []
    for v in values:
        for off in OFFSETS:
            thr = round(v + off, 2)
            if thr in RESERVED or not (10.0 <= thr <= 130.0):
                sign = 1 if off > 0 else -1
                thr = next(round(v + alt, 2) for alt in LADDER[sign]
                           if round(v + alt, 2) not in RESERVED
                           and 10.0 <= round(v + alt, 2) <= 130.0)
            expected.append((v, thr, off in YES))
    got = [(r["value"], round(r["threshold"], 2), r["target_option"] == "true")
           for r in load("train_rows.jsonl")]
    assert got == expected


def test_question_ignoring_hard_cap():
    """Identical value with 4 Yes / 4 No: any question-ignoring rule == 0.500 exact."""
    train = load("train_rows.jsonl")
    by_val = defaultdict(Counter)
    for r in train:
        by_val[r["value"]][r["target_option"]] += 1
    assert len(by_val) == 92
    assert all(c["true"] == 4 and c["false"] == 4 for c in by_val.values())
    golds = [r["target_option"] == "true" for r in train]
    assert sum(golds) == len(golds) // 2
    fixed = sum((r["value"] >= 60) == g for r, g in zip(train, golds))
    assert fixed == len(golds) // 2  # the fixed rule also lands exactly at half
    if not {r["threshold"] for r in train} & RESERVED:
        pass  # covered by test_reserved_absent; kept here for locality


def test_reserved_absent_and_same_decade_guard():
    train = load("train_rows.jsonl")
    assert not RESERVED & {r["threshold"] for r in train}
    sd = sum((int(r["value"] // 10) == int(r["threshold"] // 10))
             == (r["target_option"] == "true") for r in train) / len(train)
    assert sd <= 0.75, sd


def test_teste_structure():
    teste = load("test_e_rows.jsonl")
    assert len(teste) == 202
    bal = Counter(r["target_option"] for r in teste)
    assert bal["true"] == bal["false"] == 101
    for r in teste:  # two-decimal freshness everywhere
        for x in (r["value"], r["threshold"]):
            assert len(f"{x:.2f}".split(".")[1]) == 2, (r["id"], x)
    pairs = defaultdict(list)
    for r in teste:
        if r["pair"]:
            pairs[r["pair"]].append(r)
    interp = extrap = 0
    for members in pairs.values():
        assert len(members) == 2
        lo, hi = sorted(members, key=lambda m: m["threshold"])
        assert lo["state"] == hi["state"]
        assert lo["target_option"] == "true" and hi["target_option"] == "false"
        both_interp = {m["threshold"] for m in members} <= {47.31, 58.73, 68.17, 79.43, 89.27}
        both_extrap = {m["threshold"] for m in members} <= {21.83, 118.57}
        assert both_interp or both_extrap
        interp += both_interp
        extrap += both_extrap
    assert interp == 8 and extrap == 8
    sweeps = defaultdict(list)
    for r in teste:
        if r["sweep"]:
            sweeps[r["sweep"]].append(r)
    assert len(sweeps) == 10
    for rows in sweeps.values():
        rows.sort(key=lambda r: r["value"])
        assert [r["target_option"] == "true" for r in rows] == [False] * 5 + [True] * 5
    quads = Counter(r["quadrant"] for r in teste if r["quadrant"])
    assert quads == Counter({"cd_n": 21, "sd_n": 14, "sd_y": 14, "cd_y": 14, "bd_y": 7})


def main():  # pragma: no cover
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all e27 grid tests pass")


if __name__ == "__main__":
    raise SystemExit(main())
