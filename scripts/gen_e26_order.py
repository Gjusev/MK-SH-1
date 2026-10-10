"""E26 generator: numeric ORDER data with the same-decade shortcut BROKEN (frozen protocol).

docs/internal/E26_PROTOCOL_2026-10-10.md. Two arms with EQUAL rows/steps/eval:
treatment (non-decade-aligned thresholds, all four quadrants incl. same-decade
negatives and far positives, threshold-only AND value-only pairs) vs control
(decade-aligned thresholds, E25-style shortcut-compatible placement). Shared
dev/C/D/monotonicity evals; reserved thresholds never seen in training and
never inspected in E25 (44.5..88.5 interp, 25/115 extrap). Hand-frozen
literals, no RNG. scripts/verify_e26_order.py re-derives labels independently.

Outputs: data/e26-order/{train_treat_rows.jsonl, train_ctrl_rows.jsonl,
dev_rows.jsonl, test_c_rows.jsonl, test_d_rows.jsonl, mono_rows.jsonl,
manifest.json}.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

FIELD = "motor_temp_c"
FIELD_WORD = "motor temperature"
OP = "at least"
TREAT_THR = [41.0, 48.0, 52.0, 57.0, 63.0, 68.0, 74.0, 79.0, 84.0, 92.0]
CTRL_THR = [30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
RESERVED_INTERP = [44.5, 56.5, 66.5, 78.5, 88.5]
RESERVED_EXTRAP = [25.0, 115.0]
SITES = ["site-alpha", "site-bravo", "site-charlie", "site-delta", "site-echo", "site-foxtrot"]
_LETTERS = "abcdefghjkmnpqrstuvwxyz"


def eq_codes():
    """Deterministic purely-alphabetic 2-letter equipment codes (no digits)."""
    i = 0
    while True:
        yield f"unit-{_LETTERS[i // len(_LETTERS)]}{_LETTERS[i % len(_LETTERS)]}"
        i += 1


# ---- hand-frozen literals (protocol tables) ----
# treatment singles per threshold: (value, threshold, quadrant) — quadrants:
# bd_y (boundary), sd_y (same decade yes), cd_y (other decade yes, far for
# the "lej" ones), sd_n (same decade no — impossible with aligned thresholds),
# cd_n (other decade no).
TREAT_SINGLE = {
    41.0: [(41.0, "bd_y"), (43.8, "sd_y"), (46.2, "sd_y"), (71.4, "cd_y_lej"),
           (40.6, "sd_n"), (40.1, "sd_n"), (38.9, "cd_n"), (27.3, "cd_n")],
    48.0: [(48.0, "bd_y"), (48.7, "sd_y"), (49.3, "sd_y"), (51.8, "cd_y"),
           (46.4, "sd_n"), (42.9, "sd_n"), (37.6, "cd_n"), (31.2, "cd_n")],
    52.0: [(52.0, "bd_y"), (54.3, "sd_y"), (58.6, "sd_y"), (83.9, "cd_y_lej"),
           (51.4, "sd_n"), (50.6, "sd_n"), (47.1, "cd_n"), (39.8, "cd_n")],
    57.0: [(57.0, "bd_y"), (57.8, "sd_y"), (59.2, "sd_y"), (62.4, "cd_y"),
           (55.1, "sd_n"), (53.6, "sd_n"), (48.3, "cd_n"), (44.7, "cd_n")],
    63.0: [(63.0, "bd_y"), (65.7, "sd_y"), (68.1, "sd_y"), (94.2, "cd_y_lej"),
           (62.3, "sd_n"), (61.8, "sd_n"), (58.4, "cd_n"), (52.9, "cd_n")],
    68.0: [(68.0, "bd_y"), (68.9, "sd_y"), (69.4, "sd_y"), (71.2, "cd_y"),
           (66.7, "sd_n"), (64.8, "sd_n"), (59.1, "cd_n"), (56.2, "cd_n")],
    74.0: [(74.0, "bd_y"), (76.5, "sd_y"), (79.1, "sd_y"), (102.8, "cd_y_lej"),
           (73.2, "sd_n"), (71.9, "sd_n"), (68.4, "cd_n"), (63.7, "cd_n")],
    79.0: [(79.0, "bd_y"), (79.6, "sd_y"), (79.8, "sd_y"), (81.4, "cd_y"),
           (77.3, "sd_n"), (74.6, "sd_n"), (64.1, "cd_n"), (66.8, "cd_n")],
    84.0: [(84.0, "bd_y"), (86.3, "sd_y"), (88.7, "sd_y"), (106.9, "cd_y_lej"),
           (83.5, "sd_n"), (81.7, "sd_n"), (78.2, "cd_n"), (72.4, "cd_n")],
    92.0: [(92.0, "bd_y"), (93.8, "sd_y"), (97.6, "sd_y"), (103.4, "cd_y"),
           (91.5, "sd_n"), (90.2, "sd_n"), (87.1, "cd_n"), (84.6, "cd_n")],
}
TREAT_THR_PAIRS = [(44.3, 41.0, 48.0), (54.6, 52.0, 57.0),
                   (65.8, 63.0, 68.0), (87.5, 84.0, 92.0)]
TREAT_VAL_PAIRS = [(48.0, 46.7, 49.6), (57.0, 55.8, 58.1),
                   (68.0, 66.2, 70.3), (84.0, 82.4, 85.2)]

# control singles per aligned threshold: boundary + 3 same-decade yes + 1 far
# yes + 5 below-decade no (E25-style placement).
CTRL_SINGLE = {
    30.0: [(30.0, "bd_y"), (32.4, "sd_y"), (35.7, "sd_y"), (38.2, "sd_y"), (44.1, "cd_y_lej"),
           (28.6, "cd_n"), (26.3, "cd_n"), (23.8, "cd_n"), (21.5, "cd_n"), (18.9, "cd_n")],
    40.0: [(40.0, "bd_y"), (41.6, "sd_y"), (44.8, "sd_y"), (47.3, "sd_y"), (53.2, "cd_y_lej"),
           (38.4, "cd_n"), (35.9, "cd_n"), (33.1, "cd_n"), (29.6, "cd_n"), (25.7, "cd_n")],
    50.0: [(50.0, "bd_y"), (52.7, "sd_y"), (56.4, "sd_y"), (58.9, "sd_y"), (64.3, "cd_y_lej"),
           (48.2, "cd_n"), (45.6, "cd_n"), (42.8, "cd_n"), (39.4, "cd_n"), (36.1, "cd_n")],
    60.0: [(60.0, "bd_y"), (61.9, "sd_y"), (65.3, "sd_y"), (68.7, "sd_y"), (74.5, "cd_y_lej"),
           (58.3, "cd_n"), (55.7, "cd_n"), (52.6, "cd_n"), (49.2, "cd_n"), (45.8, "cd_n")],
    70.0: [(70.0, "bd_y"), (72.8, "sd_y"), (76.1, "sd_y"), (79.4, "sd_y"), (85.6, "cd_y_lej"),
           (68.5, "cd_n"), (65.9, "cd_n"), (62.7, "cd_n"), (59.3, "cd_n"), (56.8, "cd_n")],
    80.0: [(80.0, "bd_y"), (82.8, "sd_y"), (84.9, "sd_y"), (88.3, "sd_y"), (94.7, "cd_y_lej"),
           (78.6, "cd_n"), (75.4, "cd_n"), (72.9, "cd_n"), (69.5, "cd_n"), (66.1, "cd_n")],
    90.0: [(90.0, "bd_y"), (92.6, "sd_y"), (95.8, "sd_y"), (98.4, "sd_y"), (103.9, "cd_y_lej"),
           (88.4, "cd_n"), (86.4, "cd_n"), (83.2, "cd_n"), (80.8, "cd_n"), (77.5, "cd_n")],
    100.0: [(100.0, "bd_y"), (101.8, "sd_y"), (104.6, "sd_y"), (107.3, "sd_y"), (112.4, "cd_y_lej"),
            (98.6, "cd_n"), (96.9, "cd_n"), (94.1, "cd_n"), (91.7, "cd_n"), (88.9, "cd_n")],
}
CTRL_THR_PAIRS = [(34.9, 30.0, 40.0), (47.8, 40.0, 50.0),
                  (67.3, 60.0, 70.0), (86.9, 80.0, 90.0)]
CTRL_VAL_PAIRS = [(40.0, 37.9, 41.3), (60.0, 57.2, 60.8),
                  (80.0, 78.9, 81.2), (100.0, 97.9, 101.2)]

# shared dev/C: per treatment threshold {sd_y, cd_y, sd_n, cd_n} (2Y/2N)
DEV_QUAD = {
    41.0: [(45.1, "sd_y"), (58.7, "cd_y"), (40.8, "sd_n"), (36.4, "cd_n")],
    48.0: [(48.4, "sd_y"), (50.9, "cd_y"), (45.3, "sd_n"), (37.2, "cd_n")],
    52.0: [(56.1, "sd_y"), (66.4, "cd_y"), (50.2, "sd_n"), (46.8, "cd_n")],
    57.0: [(58.2, "sd_y"), (61.6, "cd_y"), (54.8, "sd_n"), (49.5, "cd_n")],
    63.0: [(64.2, "sd_y"), (72.6, "cd_y"), (62.6, "sd_n"), (59.7, "cd_n")],
    68.0: [(69.1, "sd_y"), (73.8, "cd_y"), (67.4, "sd_n"), (57.5, "cd_n")],
    74.0: [(78.4, "sd_y"), (89.2, "cd_y"), (73.6, "sd_n"), (69.2, "cd_n")],
    79.0: [(79.3, "sd_y"), (82.2, "cd_y"), (76.8, "sd_n"), (67.8, "cd_n")],
    84.0: [(87.8, "sd_y"), (99.1, "cd_y"), (82.6, "sd_n"), (76.0, "cd_n")],
    92.0: [(95.4, "sd_y"), (108.2, "cd_y"), (90.6, "sd_n"), (88.1, "cd_n")],
}
DEV_THR_PAIRS = [(43.6, 41.0, 48.0), (77.1, 74.0, 79.0)]
DEV_VAL_PAIRS = [(52.0, 51.7, 52.8), (63.0, 59.9, 63.4)]

TESTC_QUAD = {
    41.0: [(42.6, "sd_y"), (51.0, "cd_y"), (40.4, "sd_n"), (33.9, "cd_n")],
    48.0: [(49.1, "sd_y"), (53.3, "cd_y"), (47.4, "sd_n"), (38.7, "cd_n")],
    52.0: [(55.4, "sd_y"), (60.9, "cd_y"), (51.1, "sd_n"), (48.6, "cd_n")],
    57.0: [(58.8, "sd_y"), (60.7, "cd_y"), (56.6, "sd_n"), (48.9, "cd_n")],
    63.0: [(65.4, "sd_y"), (70.6, "cd_y"), (61.1, "sd_n"), (57.6, "cd_n")],
    68.0: [(69.8, "sd_y"), (75.1, "cd_y"), (65.2, "sd_n"), (55.5, "cd_n")],
    74.0: [(77.7, "sd_y"), (85.9, "cd_y"), (72.2, "sd_n"), (68.6, "cd_n")],
    79.0: [(79.5, "sd_y"), (83.8, "cd_y"), (75.8, "sd_n"), (68.3, "cd_n")],
    84.0: [(89.6, "sd_y"), (104.1, "cd_y"), (83.1, "sd_n"), (77.6, "cd_n")],
    92.0: [(96.7, "sd_y"), (109.3, "cd_y"), (91.8, "sd_n"), (89.1, "cd_n")],
}
TESTC_THR_PAIRS = [(49.4, 48.0, 52.0), (81.9, 79.0, 84.0)]
TESTC_VAL_PAIRS = [(57.0, 56.9, 57.4), (74.0, 73.9, 75.6)]

# test D: per reserved threshold (boundary, sd_y, cd_y | sd_n, cd_n, cd_n)
TESTD_SINGLE = {
    44.5: [(44.5, "bd_y"), (45.9, "sd_y"), (50.8, "cd_y"),
           (43.3, "sd_n"), (39.1, "cd_n"), (36.9, "cd_n")],
    56.5: [(56.5, "bd_y"), (58.0, "sd_y"), (60.1, "cd_y"),
           (54.9, "sd_n"), (47.2, "cd_n"), (47.7, "cd_n")],
    66.5: [(66.5, "bd_y"), (67.9, "sd_y"), (70.9, "cd_y"),
           (65.6, "sd_n"), (59.4, "cd_n"), (57.9, "cd_n")],
    78.5: [(78.5, "bd_y"), (79.7, "sd_y"), (82.7, "cd_y"),
           (77.9, "sd_n"), (69.3, "cd_n"), (66.3, "cd_n")],
    88.5: [(88.5, "bd_y"), (89.8, "sd_y"), (92.3, "cd_y"),
           (87.6, "sd_n"), (78.1, "cd_n"), (78.7, "cd_n")],
    25.0: [(25.0, "bd_y"), (26.7, "sd_y"), (31.8, "cd_y"),
           (23.1, "sd_n"), (17.6, "cd_n"), (14.2, "cd_n")],
    115.0: [(115.0, "bd_y"), (117.4, "sd_y"), (121.6, "cd_y"),
            (113.8, "sd_n"), (108.7, "cd_n"), (104.8, "cd_n")],
}
TESTD_THR_PAIRS_INTERP = [(50.4, 44.5, 56.5), (68.2, 66.5, 78.5)]
TESTD_THR_PAIRS_EXTRAP = [(98.2, 25.0, 115.0), (105.7, 25.0, 115.0)]
TESTD_VAL_PAIRS = [(44.5, 42.2, 46.6), (88.5, 86.6, 90.4)]

# monotonicity sweeps: threshold -> 5 values below (No) then 5 above (Yes)
MONO_SWEEPS = {
    41.0: [32.5, 36.5, 39.6, 40.7, 40.9, 41.7, 42.5, 45.5, 49.7, 59.0],
    63.0: [55.0, 58.5, 61.5, 62.8, 62.9, 63.3, 64.5, 67.1, 71.6, 80.2],
    84.0: [76.3, 79.2, 82.1, 83.3, 83.7, 84.2, 85.8, 88.6, 93.1, 100.7],
    44.5: [36.8, 40.3, 43.5, 44.2, 44.4, 44.6, 45.0, 48.5, 52.3, 62.2],
    25.0: [17.1, 20.4, 23.7, 24.5, 24.8, 25.2, 25.9, 29.4, 33.8, 43.1],
    115.0: [107.1, 110.4, 113.2, 114.3, 114.7, 115.3, 116.8, 119.5, 124.0, 132.5],
}


def fmt(x: float) -> str:
    return f"{x:g}"


def make_row(seq, split_name, value, threshold, *, variant="base", pair=None,
             eq=None, site=None, quadrant=None, sweep=None):
    state = json.dumps({"equipment_id": eq, FIELD: value, "site": site}, sort_keys=True)
    question = f"Is the {FIELD_WORD} reading at least {fmt(threshold)} degrees Celsius?"
    gold = value >= threshold
    return {"id": f"e26|{split_name}|{seq:04d}",
            "group_id": pair or f"e26|{split_name}|{seq:04d}",
            "source": "e26-order", "language": "en", "domain": f"num_{FIELD}",
            "state": state, "question": question,
            "options": [{"id": "false", "description": "No"}, {"id": "true", "description": "Yes"}],
            "target_kind": "single", "target_option": "true" if gold else "false",
            "split": {"train": "train", "dev": "dev"}.get(split_name, "test"),
            "question_type": "noul", "label_origin": "e26-spec", "operator": OP,
            "feature": FIELD, "threshold": threshold, "value": value,
            "variant": variant, "pair": pair, "quadrant": quadrant, "sweep": sweep,
            "rule_in": "question",
            "thr_range": ("extrap" if threshold in RESERVED_EXTRAP
                          else "interp" if threshold in RESERVED_INTERP else "train")}


class Builder:
    def __init__(self, split_name):
        self.split_name = split_name
        self.codes = eq_codes()
        self.rows = []
        self.seq = 1

    def _site(self):
        return SITES[len(self.rows) % len(SITES)]

    def add(self, value, threshold, **kw):
        r = make_row(self.seq, self.split_name, value, threshold, **kw)
        self.rows.append(r)
        self.seq += 1
        return r

    def add_thr_pair(self, value, t_lo, t_hi):
        pair_id = f"e26p|{self.split_name}|thr|{fmt(value)}"
        eq, site = next(self.codes), self._site()
        self.add(value, t_lo, variant="pair_thrlo", pair=pair_id, eq=eq, site=site)
        self.add(value, t_hi, variant="pair_thrhi", pair=pair_id, eq=eq, site=site)

    def add_val_pair(self, threshold, v_no, v_yes):
        pair_id = f"e26p|{self.split_name}|val|{fmt(threshold)}"
        eq, site = next(self.codes), self._site()
        self.add(v_no, threshold, variant="pair_valno", pair=pair_id, eq=eq, site=site)
        self.add(v_yes, threshold, variant="pair_valyes", pair=pair_id, eq=eq, site=site)


def build_arm_train(name, singles, thr_pairs, val_pairs):
    b = Builder(name)
    for t, rows in singles.items():
        for value, quad in rows:
            b.add(value, t, eq=next(b.codes), site=b._site(), quadrant=quad)
    for value, t_lo, t_hi in thr_pairs:
        b.add_thr_pair(value, t_lo, t_hi)
    for threshold, v_no, v_yes in val_pairs:
        b.add_val_pair(threshold, v_no, v_yes)
    return b.rows


def build_eval_quad(name, quad_map, thr_pairs, val_pairs):
    b = Builder(name)
    for t, rows in quad_map.items():
        for value, quad in rows:
            b.add(value, t, eq=next(b.codes), site=b._site(), quadrant=quad)
    for value, t_lo, t_hi in thr_pairs:
        b.add_thr_pair(value, t_lo, t_hi)
    for threshold, v_no, v_yes in val_pairs:
        b.add_val_pair(threshold, v_no, v_yes)
    return b.rows


def build_testd():
    b = Builder("testd")
    for t, rows in TESTD_SINGLE.items():
        for value, quad in rows:
            b.add(value, t, eq=next(b.codes), site=b._site(), quadrant=quad)
    for value, t_lo, t_hi in TESTD_THR_PAIRS_INTERP + TESTD_THR_PAIRS_EXTRAP:
        b.add_thr_pair(value, t_lo, t_hi)
    for threshold, v_no, v_yes in TESTD_VAL_PAIRS:
        b.add_val_pair(threshold, v_no, v_yes)
    return b.rows


def build_mono():
    b = Builder("mono")
    for t, values in MONO_SWEEPS.items():
        sweep_id = f"e26m|{fmt(t)}"
        for value in values:
            b.add(value, t, eq=next(b.codes), site=b._site(),
                  variant="mono", sweep=sweep_id)
    return b.rows


def main() -> int:
    treat = build_arm_train("treat", TREAT_SINGLE, TREAT_THR_PAIRS, TREAT_VAL_PAIRS)
    ctrl = build_arm_train("ctrl", CTRL_SINGLE, CTRL_THR_PAIRS, CTRL_VAL_PAIRS)
    dev = build_eval_quad("dev", DEV_QUAD, DEV_THR_PAIRS, DEV_VAL_PAIRS)
    testc = build_eval_quad("testc", TESTC_QUAD, TESTC_THR_PAIRS, TESTC_VAL_PAIRS)
    testd = build_testd()
    mono = build_mono()

    files = {"treat": (treat, "train_treat_rows.jsonl"),
             "ctrl": (ctrl, "train_ctrl_rows.jsonl"),
             "dev": (dev, "dev_rows.jsonl"),
             "testc": (testc, "test_c_rows.jsonl"),
             "testd": (testd, "test_d_rows.jsonl"),
             "mono": (mono, "mono_rows.jsonl")}
    all_rows = [r for rows, _ in files.values() for r in rows]
    ids = [r["id"] for r in all_rows]
    payloads = [(r["question"], r["state"]) for r in all_rows]
    assert len(ids) == len(set(ids)), "duplicate ids"
    assert len(payloads) == len(set(payloads)), "duplicate payloads"

    # value disjointness across the six files (pair members share by design;
    # mono additionally only guarded against the two trains downstream)
    def values(rows):
        out = {}
        for r in rows:
            out.setdefault(r["value"], []).append(r)
        return out
    by_name = {k: values(rows) for k, (rows, _) in files.items()}
    names = list(files)
    for i, a in enumerate(names):
        for b_ in names[i + 1:]:
            overlap = set(by_name[a]) & set(by_name[b_])
            for v in overlap:
                # allowed only when all sharers are members of ONE pair
                holders = by_name[a][v] + by_name[b_][v]
                if not all(h["pair"] for h in holders) or \
                        len({h["pair"] for h in holders}) != 1:
                    raise AssertionError(f"value overlap {a}&{b_}: {v}")
    # reserved thresholds never as threshold or value in either train
    reserved = set(RESERVED_INTERP + RESERVED_EXTRAP)
    for k in ("treat", "ctrl"):
        for r in files[k][0]:
            assert r["threshold"] not in reserved
            assert r["value"] not in reserved, f"reserved as value in {k}: {r['value']}"
    for k in ("dev", "testc"):
        assert not reserved & {r["threshold"] for r in files[k][0]}

    # pair integrity (both kinds)
    pairs = {}
    for r in all_rows:
        if r["pair"]:
            pairs.setdefault(r["pair"], []).append(r)
    for pid, members in pairs.items():
        variants = {m["variant"] for m in members}
        if variants == {"pair_thrlo", "pair_thrhi"}:
            lo = next(m for m in members if m["variant"] == "pair_thrlo")
            hi = next(m for m in members if m["variant"] == "pair_thrhi")
            assert lo["state"] == hi["state"], f"{pid}: state differs"
            assert lo["question"] == hi["question"].replace(
                fmt(hi["threshold"]), fmt(lo["threshold"])), f"{pid}: question differs beyond threshold"
            assert lo["target_option"] == "true" and hi["target_option"] == "false", pid
            assert lo["value"] > lo["threshold"] and hi["value"] < hi["threshold"], pid
        elif variants == {"pair_valno", "pair_valyes"}:
            no = next(m for m in members if m["variant"] == "pair_valno")
            yes = next(m for m in members if m["variant"] == "pair_valyes")
            assert no["question"] == yes["question"], f"{pid}: question differs"
            sn, sy = json.loads(no["state"]), json.loads(yes["state"])
            sn[FIELD], sy[FIELD] = "<V>", "<V>"
            assert sn == sy, f"{pid}: state differs beyond the value"
            assert no["target_option"] == "false" and yes["target_option"] == "true", pid
            assert no["value"] < no["threshold"] <= yes["value"], pid
        else:
            raise AssertionError(f"{pid}: unknown pair variants {variants}")
        assert len({m["split"] for m in members}) == 1, f"{pid}: pair split across files"

    # monotonicity structure: strictly increasing, first 5 No, last 5 Yes
    for t, vals in MONO_SWEEPS.items():
        assert all(a < b2 for a, b2 in zip(vals, vals[1:])), t
        assert all(v < t for v in vals[:5]) and all(v >= t for v in vals[5:]), t

    out = ROOT / "data/e26-order"
    out.mkdir(parents=True, exist_ok=True)
    for name, (rows, fname) in files.items():
        (out / fname).write_text(
            "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows), encoding="utf-8")
    import collections
    manifest = {
        "set": "e26-order", "protocol": "docs/internal/E26_PROTOCOL_2026-10-10.md",
        "counts": {name: len(rows) for name, (rows, _) in files.items()},
        "treat_thresholds": TREAT_THR, "ctrl_thresholds": CTRL_THR,
        "reserved_thresholds": {"interp": RESERVED_INTERP, "extrap": RESERVED_EXTRAP},
        "balance": {name: dict(collections.Counter(r["target_option"] for r in rows))
                    for name, (rows, _) in files.items()},
        "thr_pairs": {name: sum(1 for r in rows if r["variant"] == "pair_thrlo")
                      for name, (rows, _) in files.items()},
        "val_pairs": {name: sum(1 for r in rows if r["variant"] == "pair_valno")
                      for name, (rows, _) in files.items()},
        "mono_sweeps": {fmt(t): v for t, v in MONO_SWEEPS.items()},
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "status": ("numeric-order data with the same-decade shortcut broken in the "
                   "treatment arm and an equal-budget shortcut-compatible control; "
                   "tests/mono opened only after finalists are fixed (frozen rule)"),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True),
                                        encoding="utf-8")
    print(json.dumps({"counts": manifest["counts"], "balance": manifest["balance"],
                      "thr_pairs": manifest["thr_pairs"], "val_pairs": manifest["val_pairs"]},
                     indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
