"""E25 generator: threshold-VARIABLE numeric comparison data (frozen protocol).

docs/internal/E25_PROTOCOL_2026-10-10.md. One numeric field (motor_temp_c),
units degrees Celsius, operator >=, direct No/Yes question. Multiple TRAIN
thresholds (40..90) with threshold-only inversion pairs (state byte-identical,
ONLY the threshold in the question changes, answers flip) — any question-ignoring
predictor is capped at 0.5 accuracy / 0 pairs there. Held-out tests: new values
at known thresholds (C) and NEVER-USED thresholds (D: interp 45-85, extrap
30/100). All values/offsets are hand-frozen literals; no RNG. Equipment ids are
purely alphabetic (no digits in the state except the measured value).
scripts/verify_e25_thrvar.py re-derives labels from the rendered text with
independent logic.

Outputs: data/e25-thrvar/{train_thr_rows.jsonl, dev_thr_rows.jsonl,
test_known_thr_rows.jsonl, test_new_thr_rows.jsonl, manifest.json}.
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
TRAIN_THRESHOLDS = [40.0, 50.0, 60.0, 70.0, 80.0, 90.0]
NEW_THRESHOLDS_INTERP = [45.0, 55.0, 65.0, 75.0, 85.0]
NEW_THRESHOLDS_EXTRAP = [30.0, 100.0]
SITES = ["site-alpha", "site-bravo", "site-charlie", "site-delta", "site-echo", "site-foxtrot"]
_LETTERS = "abcdefghjkmnpqrstuvwxyz"


def eq_codes():
    """Deterministic purely-alphabetic 2-letter equipment codes (no digits)."""
    i = 0
    while True:
        yield f"unit-{_LETTERS[i // len(_LETTERS)]}{_LETTERS[i % len(_LETTERS)]}"
        i += 1


# ---- hand-frozen value lists (see protocol) ----
TRAIN_STRADDLE = {  # gap -> values inside it, 3 per gap
    (40.0, 50.0): [42.5, 45.2, 47.0],
    (50.0, 60.0): [52.5, 55.3, 57.5],
    (60.0, 70.0): [61.7, 63.9, 68.5],
    (70.0, 80.0): [71.9, 74.0, 76.8],
    (80.0, 90.0): [82.4, 84.0, 88.5],
}
TRAIN_BOUNDARY = [40.0, 50.0, 60.0, 70.0, 80.0, 90.0]  # v = t -> Yes
TRAIN_BALANCE = [  # (value, threshold, is_yes)
    (36.5, 40.0, False), (33.1, 40.0, False), (28.7, 40.0, False), (21.4, 40.0, False),
    (46.2, 50.0, False), (57.1, 60.0, False), (67.3, 70.0, False), (78.6, 80.0, False),
    (86.9, 90.0, False),
    (92.8, 90.0, True), (96.4, 90.0, True), (103.7, 90.0, True),
]
DEV_STRADDLE = {
    (40.0, 50.0): [44.1, 47.9], (50.0, 60.0): [53.7, 58.2], (60.0, 70.0): [62.8, 68.9],
    (70.0, 80.0): [73.5, 78.1], (80.0, 90.0): [83.6, 87.4],
}
DEV_EXTRA = [(35.2, 40.0, False), (29.8, 40.0, False), (91.6, 90.0, True), (98.3, 90.0, True)]
TESTC_STRADDLE = {
    (40.0, 50.0): [43.3, 48.6], (50.0, 60.0): [51.4, 59.0], (60.0, 70.0): [64.2, 66.8],
    (70.0, 80.0): [72.7, 79.3], (80.0, 90.0): [81.2, 89.1],
}
TESTC_EXTRA = [(32.4, 40.0, False), (26.9, 40.0, False), (93.5, 90.0, True), (99.7, 90.0, True)]
TESTD_PAIRS = [  # (value, thr_lo -> Yes, thr_hi -> No) among reserved thresholds
    (33.6, 30.0, 45.0), (47.3, 45.0, 55.0), (57.8, 55.0, 65.0),
    (67.1, 65.0, 75.0), (77.5, 75.0, 85.0), (92.4, 85.0, 100.0),
]


def fmt_thr(t: float) -> str:
    return f"{t:g}"


def make_row(seq, split, value, threshold, *, variant="base", pair=None, eq=None):
    state = json.dumps({"equipment_id": eq, FIELD: value, "site": None}, sort_keys=True)
    # site filled by caller-side wrapper below; kept explicit here for clarity
    question = f"Is the {FIELD_WORD} reading at least {fmt_thr(threshold)} degrees Celsius?"
    gold = value >= threshold
    return {"id": f"e25|{split}|{seq:04d}", "group_id": pair or f"e25|{split}|{seq:04d}",
            "source": "e25-thrvar", "language": "en", "domain": f"num_{FIELD}",
            "state": state, "question": question,
            "options": [{"id": "false", "description": "No"}, {"id": "true", "description": "Yes"}],
            "target_kind": "single", "target_option": "true" if gold else "false",
            "split": {"train": "train", "dev": "dev"}.get(split, "test"), "question_type": "noul",
            "label_origin": "e25-spec", "operator": OP, "feature": FIELD,
            "threshold": threshold, "value": value, "variant": variant, "pair": pair,
            "rule_in": "question",
            "thr_range": ("extrap" if threshold in NEW_THRESHOLDS_EXTRAP
                          else "interp" if threshold in NEW_THRESHOLDS_INTERP else "train")}


def build_split(split, straddle, boundary_like, extra, new_thr_singles=None, testd_pairs=None):
    """Assemble one split. Base cases (pairs) share equipment/site; everything
    that is not a pair gets its own. `boundary_like`: [(value, thr, yes)] singles."""
    codes = eq_codes()
    rows, seq = [], 1

    def add(value, threshold, *, variant="base", pair=None, eq=None, site=None):
        nonlocal seq
        r = make_row(seq, split, value, threshold, variant=variant, pair=pair, eq=eq)
        st = json.loads(r["state"])
        st["site"] = site
        r["state"] = json.dumps(st, sort_keys=True)
        rows.append(r)
        seq += 1

    for (t_lo, t_hi), values in straddle.items():
        for v in values:
            pair_id = f"e25p|{split}|{v:g}"
            eq, site = next(codes), SITES[len(rows) % len(SITES)]
            add(v, t_lo, variant="pair_thrlo", pair=pair_id, eq=eq, site=site)
            add(v, t_hi, variant="pair_thrhi", pair=pair_id, eq=eq, site=site)
    for value, threshold, _is_yes in boundary_like:
        add(value, threshold, eq=next(codes), site=SITES[len(rows) % len(SITES)])
    for value, threshold, _is_yes in extra:
        add(value, threshold, eq=next(codes), site=SITES[len(rows) % len(SITES)])
    if new_thr_singles:
        for threshold, triples in new_thr_singles:
            for sign_off, yes in triples:
                value = round(threshold + sign_off, 1)
                add(value, threshold, eq=next(codes), site=SITES[len(rows) % len(SITES)])
    if testd_pairs:
        for v, t_lo, t_hi in testd_pairs:
            pair_id = f"e25p|{split}|{v:g}"
            eq, site = next(codes), SITES[len(rows) % len(SITES)]
            add(v, t_lo, variant="pair_thrlo", pair=pair_id, eq=eq, site=site)
            add(v, t_hi, variant="pair_thrhi", pair=pair_id, eq=eq, site=site)
    return rows


def main() -> int:
    near_dev = [(t + 0.4, t, True) for t in TRAIN_THRESHOLDS] + \
               [(t - 0.4, t, False) for t in TRAIN_THRESHOLDS]
    near_c = [(t + 0.3, t, True) for t in TRAIN_THRESHOLDS] + \
             [(t - 0.3, t, False) for t in TRAIN_THRESHOLDS]
    testd_singles = [(t, [(0.6, True), (-0.6, False), (7.2, True), (-4.8, False),
                          (0.0, True), (-12.0, False)])
                     for t in NEW_THRESHOLDS_INTERP + NEW_THRESHOLDS_EXTRAP]

    train = build_split("train", TRAIN_STRADDLE,
                        [(t, t, True) for t in TRAIN_BOUNDARY], TRAIN_BALANCE)
    dev = build_split("dev", DEV_STRADDLE, [], near_dev, DEV_EXTRA and None or None) if False else \
        build_split("dev", DEV_STRADDLE, [], near_dev + DEV_EXTRA, [])
    testc = build_split("testc", TESTC_STRADDLE, [], near_c + TESTC_EXTRA, [])
    testd = build_split("testd", {}, [], [], new_thr_singles=testd_singles, testd_pairs=TESTD_PAIRS)

    files = {"train_thr_rows.jsonl": train, "dev_thr_rows.jsonl": dev,
             "test_known_thr_rows.jsonl": testc, "test_new_thr_rows.jsonl": testd}
    all_rows = [r for rs in files.values() for r in rs]
    ids = [r["id"] for r in all_rows]
    payloads = [(r["question"], r["state"]) for r in all_rows]
    assert len(ids) == len(set(ids)), "duplicate ids"
    assert len(payloads) == len(set(payloads)), "duplicate payloads"
    # cross-split value disjointness (pairs reuse their own value by design)
    def split_values(rows, skip_pairs=False):
        return {r["value"] for r in rows if not (skip_pairs and r["pair"])}
    by_name = {"train": train, "dev": dev, "testc": testc, "testd": testd}
    for a, b in (("train", "dev"), ("train", "testc"), ("train", "testd"),
                 ("dev", "testc"), ("dev", "testd"), ("testc", "testd")):
        overlap = split_values(by_name[a]) & split_values(by_name[b])
        assert not overlap, f"value overlap {a}&{b}: {sorted(overlap)}"
    # reserved thresholds never appear in train/dev
    for rows in (train, dev):
        assert not ({*NEW_THRESHOLDS_INTERP, *NEW_THRESHOLDS_EXTRAP} & {r["threshold"] for r in rows})
    # pair property: identical state, question differs ONLY in threshold, gold inverted
    pairs = {}
    for r in all_rows:
        if r["pair"]:
            pairs.setdefault((r["pair"]), []).append(r)
    for pid, members in pairs.items():
        assert len(members) == 2 and {m["variant"] for m in members} == {"pair_thrlo", "pair_thrhi"}, pid
        lo = next(m for m in members if m["variant"] == "pair_thrlo")
        hi = next(m for m in members if m["variant"] == "pair_thrhi")
        assert lo["state"] == hi["state"], f"{pid}: state differs"
        assert lo["question"] == hi["question"].replace(fmt_thr(hi["threshold"]), fmt_thr(lo["threshold"])), \
            f"{pid}: question differs in more than the threshold"
        assert lo["target_option"] == "true" and hi["target_option"] == "false", pid
        assert lo["split"] == hi["split"], f"{pid}: pair split across files"

    out = ROOT / "data/e25-thrvar"
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in files.items():
        (out / name).write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows),
                                encoding="utf-8")
    import collections
    manifest = {
        "set": "e25-thrvar", "protocol": "docs/internal/E25_PROTOCOL_2026-10-10.md",
        "counts": {k.split("_rows")[0]: len(v) for k, v in files.items()},
        "train_thresholds": TRAIN_THRESHOLDS,
        "reserved_thresholds": {"interp": NEW_THRESHOLDS_INTERP, "extrap": NEW_THRESHOLDS_EXTRAP},
        "balance": {k.split("_rows")[0]: dict(collections.Counter(r["target_option"] for r in v))
                    for k, v in files.items()},
        "threshold_only_pairs": {k.split("_rows")[0]: sum(1 for r in v if r["pair"]) // 2
                                 for k, v in files.items()},
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "status": ("threshold-variable fitting; pairs force reading the question threshold; "
                   "tests C/D opened only after finalists are fixed (frozen rule)"),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"counts": manifest["counts"], "balance": manifest["balance"],
                      "pairs": manifest["threshold_only_pairs"]}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
