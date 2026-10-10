"""E26 order-data tests: hand-computed literals + structural guards.

Hand-checked boundary/quadrant examples from the frozen protocol
(docs/internal/E26_PROTOCOL_2026-10-10.md), plus the structural properties the
independent verifier also enforces (kept here as fast regression tests).
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/e26-order"


def load(name):
    return [json.loads(l) for l in (DATA / name).read_text(encoding="utf-8").splitlines() if l.strip()]


def test_hand_computed_labels():
    """(value, threshold) -> gold, hand-derived from the protocol tables."""
    hand = [
        # treatment train: quadrants incl. the ones aligned thresholds cannot have
        (40.6, 41.0, False, "same-dec negative"),
        (40.1, 41.0, False, "same-dec negative"),
        (43.8, 41.0, True, "same-dec positive"),
        (71.4, 41.0, True, "far positive, other decade"),
        (38.9, 41.0, False, "other-dec negative"),
        (49.3, 48.0, True, "same-dec positive"),
        (46.4, 48.0, False, "same-dec negative"),
        (79.8, 79.0, True, "same-dec positive at mod-9 threshold"),
        (77.3, 79.0, False, "same-dec negative at mod-9 threshold"),
        (106.9, 84.0, True, "far positive"),
        (78.2, 84.0, False, "other-dec negative"),
        (92.0, 92.0, True, "exact boundary"),
        (90.2, 92.0, False, "same-dec negative"),
        # control train (aligned): same-dec positive is v in (t, t+10)
        (38.2, 30.0, True, "aligned same-dec positive"),
        (18.9, 30.0, False, "aligned other-dec negative"),
        (53.2, 40.0, True, "aligned far positive (the one rule-breaking row type)"),
        (88.4, 90.0, False, "aligned other-dec negative"),
        (92.6, 90.0, True, "aligned same-dec positive"),
        (77.5, 90.0, False, "aligned other-dec negative"),
        # reserved thresholds (test D)
        (115.0, 115.0, True, "exact boundary"),
        (113.8, 115.0, False, "same-dec negative"),
        (25.0, 25.0, True, "exact boundary"),
        (23.1, 25.0, False, "same-dec negative"),
        (14.2, 25.0, False, "far other-dec negative"),
        (121.6, 115.0, True, "far positive"),
    ]
    pool = {}
    for fname in ("train_treat_rows.jsonl", "train_ctrl_rows.jsonl", "dev_rows.jsonl",
                  "test_c_rows.jsonl", "test_d_rows.jsonl", "mono_rows.jsonl"):
        for r in load(fname):
            pool[(round(r["value"], 6), round(r["threshold"], 6))] = r
    for value, threshold, gold, why in hand:
        r = pool[(round(value, 6), round(threshold, 6))]
        assert (r["target_option"] == "true") == gold, f"({value},{threshold}) {why}"


def test_counts_and_global_balance():
    counts = {"train_treat_rows.jsonl": 96, "train_ctrl_rows.jsonl": 96,
              "dev_rows.jsonl": 48, "test_c_rows.jsonl": 48,
              "test_d_rows.jsonl": 54, "mono_rows.jsonl": 60}
    for fname, n in counts.items():
        rows = load(fname)
        assert len(rows) == n, (fname, len(rows))
        if fname != "mono_rows.jsonl":
            bal = Counter(r["target_option"] for r in rows)
            assert bal["true"] == bal["false"], (fname, dict(bal))


def test_per_threshold_balance_bounds():
    limits = {"treat": 1, "ctrl": 1, "dev": 1, "testc": 1, "testd": 2}
    fname_of = {"treat": "train_treat_rows.jsonl", "ctrl": "train_ctrl_rows.jsonl",
                "dev": "dev_rows.jsonl", "testc": "test_c_rows.jsonl",
                "testd": "test_d_rows.jsonl"}
    for name, limit in limits.items():
        by_thr = defaultdict(Counter)
        for r in load(fname_of[name]):
            by_thr[r["threshold"]][r["target_option"]] += 1
        for t, c in by_thr.items():
            assert abs(c["true"] - c["false"]) <= limit, (name, t, dict(c))


def test_pair_integrity_both_kinds():
    for fname in ("train_treat_rows.jsonl", "train_ctrl_rows.jsonl", "dev_rows.jsonl",
                  "test_c_rows.jsonl", "test_d_rows.jsonl"):
        rows = load(fname)
        pairs = defaultdict(list)
        for r in rows:
            if r["pair"]:
                pairs[r["pair"]].append(r)
        assert pairs, fname
        for pid, members in pairs.items():
            assert len(members) == 2, (pid, len(members))
            variants = {m["variant"] for m in members}
            if variants == {"pair_thrlo", "pair_thrhi"}:
                lo = next(m for m in members if m["variant"] == "pair_thrlo")
                hi = next(m for m in members if m["variant"] == "pair_thrhi")
                assert lo["state"] == hi["state"], pid           # state byte-identical
                assert lo["question"] == hi["question"].replace(
                    f"{hi['threshold']:g}", f"{lo['threshold']:g}"), pid
                assert lo["target_option"] == "true" and hi["target_option"] == "false", pid
            else:
                assert variants == {"pair_valno", "pair_valyes"}, (pid, variants)
                no = next(m for m in members if m["variant"] == "pair_valno")
                yes = next(m for m in members if m["variant"] == "pair_valyes")
                assert no["question"] == yes["question"], pid    # question byte-identical
                sn, sy = json.loads(no["state"]), json.loads(yes["state"])
                sn["motor_temp_c"] = sy["motor_temp_c"] = None
                assert sn == sy, pid                              # only the value differs
                assert no["target_option"] == "false" and yes["target_option"] == "true", pid


def test_threshold_pairs_cap_question_ignoring_rules():
    """On threshold-only pairs any question-ignoring rule is <= 0.5 / 0 both-correct."""
    for fname in ("train_treat_rows.jsonl", "train_ctrl_rows.jsonl", "dev_rows.jsonl",
                  "test_c_rows.jsonl", "test_d_rows.jsonl"):
        rows = [r for r in load(fname) if r["variant"] in ("pair_thrlo", "pair_thrhi")]
        assert rows
        golds = [r["target_option"] == "true" for r in rows]
        assert abs(sum(golds) / len(golds) - 0.5) < 1e-9, fname
        for rule in (lambda r: True, lambda r: False, lambda r: r["value"] >= 60):
            acc = sum(rule(r) == g for r, g in zip(rows, golds)) / len(rows)
            assert acc <= 0.5 + 1e-9, (fname, acc)
        by_pair = defaultdict(list)
        for r in rows:
            by_pair[r["pair"]].append(r)
        for pid, members in by_pair.items():
            # identical states => any deterministic state-only rule answers the same
            assert len({m["target_option"] for m in members}) == 2, pid


def test_same_decade_shortcut_separation():
    """The frozen design contract: shortcut broken in treatment, alive in control."""
    def acc(fname):
        rows = load(fname)
        hit = sum((int(r["value"] // 10) == int(r["threshold"] // 10))
                  == (r["target_option"] == "true") for r in rows)
        return hit / len(rows)
    treat, ctrl = acc("train_treat_rows.jsonl"), acc("train_ctrl_rows.jsonl")
    assert treat <= 0.65, treat          # 0.6042 at freeze
    assert ctrl >= 0.80, ctrl            # 0.9167 at freeze


def test_reserved_thresholds_never_in_training():
    reserved = {44.5, 56.5, 66.5, 78.5, 88.5, 25.0, 115.0}
    for fname in ("train_treat_rows.jsonl", "train_ctrl_rows.jsonl",
                  "dev_rows.jsonl", "test_c_rows.jsonl"):
        for r in load(fname):
            assert r["threshold"] not in reserved, (fname, r["id"])
    for fname in ("train_treat_rows.jsonl", "train_ctrl_rows.jsonl"):
        for r in load(fname):
            assert r["value"] not in reserved, (fname, r["id"], r["value"])


def test_quadrant_coverage_in_treatment_train():
    treat = load("train_treat_rows.jsonl")
    for t in {r["threshold"] for r in treat}:
        quads = {r["quadrant"] for r in treat if r["threshold"] == t}
        assert {"bd_y", "sd_y", "sd_n", "cd_n"} <= quads, (t, quads)
        assert quads & {"cd_y", "cd_y_lej"}, (t, quads)


def test_monotonicity_sweeps_structure():
    rows = load("mono_rows.jsonl")
    sweeps = defaultdict(list)
    for r in rows:
        sweeps[r["sweep"]].append(r)
    assert len(sweeps) == 6
    for sid, members in sweeps.items():
        members.sort(key=lambda r: r["value"])
        assert len(members) == 10
        assert all(a["value"] < b["value"] for a, b in zip(members, members[1:])), sid
        golds = [r["target_option"] == "true" for r in members]
        assert golds == [False] * 5 + [True] * 5, sid
        t = members[0]["threshold"]
        assert all(r["threshold"] == t for r in members), sid


def test_pure_pairs_exist_for_both_reserved_ranges():
    """Fixes the E25 V3b flaw: pure interp AND pure extrap threshold pairs exist."""
    testd = load("test_d_rows.jsonl")
    ranges = {"interp": {44.5, 56.5, 66.5, 78.5, 88.5}, "extrap": {25.0, 115.0}}
    for rng, reserved in ranges.items():
        pairs = defaultdict(list)
        for r in testd:
            if r["variant"] in ("pair_thrlo", "pair_thrhi") and r["threshold"] in reserved:
                pairs[r["pair"]].append(r)
        pure = [pid for pid, m in pairs.items() if len(m) == 2
                and {x["thr_range"] for x in m} == {rng}]
        assert len(pure) >= 2, (rng, len(pure))


def main():  # pragma: no cover - direct runner
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all e26 order tests pass")


if __name__ == "__main__":
    sys.exit(main())
