"""E28 data tests: base verbatim, extremes grid, chains, balances, caps."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/e28-ext/train_rows.jsonl"
BASE = ROOT / "data/e27-grid/train_rows.jsonl"
EXTREMES = [18.4, 19.7, 20.9, 21.6, 22.8, 23.5, 24.7, 25.4,
            108.3, 109.6, 111.2, 112.8, 113.5, 116.1, 122.4, 124.6]
RESERVED = {25.0, 44.5, 56.5, 66.5, 78.5, 88.5, 115.0}


def load(p):
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def test_structure_and_base_verbatim():
    rows, base = load(DATA), load(BASE)
    assert len(rows) == 944 and len(base) == 736
    assert [(r["question"], r["state"], r["target_option"]) for r in rows[:736]] == \
           [(r["question"], r["state"], r["target_option"]) for r in base]
    bal = Counter(r["target_option"] for r in rows)
    assert bal["true"] == bal["false"] == 472
    ext = rows[736:864]
    assert {r["value"] for r in ext} == set(EXTREMES)
    by_val = defaultdict(Counter)
    for r in ext:
        by_val[r["value"]][r["target_option"]] += 1
    assert all(c["true"] == 4 and c["false"] == 4 for c in by_val.values())
    assert not RESERVED & {r["threshold"] for r in rows}


def test_chain_rows():
    rows = load(DATA)[864:]
    assert len(rows) == 80
    sweeps = defaultdict(list)
    for r in rows:
        assert r["variant"] == "chain"
        sweeps[r["sweep"]].append(r)
    assert len(sweeps) == 8
    for sid, members in sweeps.items():
        members.sort(key=lambda r: r["value"])
        assert len(members) == 10
        assert all(a["value"] < b["value"] for a, b in zip(members, members[1:]))
        assert [m["target_option"] == "true" for m in members] == [False] * 5 + [True] * 5
        assert len({m["threshold"] for m in members}) == 1


def test_hand_literals():
    """(value, threshold) -> gold, hand-derived (incl. ladder substitutions)."""
    rows = {(r["value"], round(r["threshold"], 2)): r["target_option"] for r in load(DATA)}
    hand = [
        (18.4, 15.4, "true"),   # 18.4-25 out of range -> -3 (substitution)
        (18.4, 11.4, "true"),   # -7
        (18.4, 18.4, "true"),   # boundary
        (18.4, 43.4, "false"),  # +25
        (124.6, 127.6, "false"),  # +25 out -> +3 (substitution)
        (124.6, 124.6, "true"),
        # chains: t=23.0 first and last members
        (18.8, 23.0, "false"),
        (27.8, 23.0, "true"),
    ]
    for v, t, gold in hand:
        assert rows[(v, t)] == gold, (v, t)


def test_constants_cap():
    rows = load(DATA)
    golds = [r["target_option"] == "true" for r in rows]
    assert sum(golds) == len(golds) // 2  # constant rules exactly 0.500


def main():  # pragma: no cover
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")


if __name__ == "__main__":
    raise SystemExit(main())
