"""E24 generator: minimal numeric FITTING microset (frozen protocol).

docs/internal/E24_PROTOCOL_2026-10-10.md. One numeric field (motor_temp_c),
one operator (>=, "at least"), one threshold (60), direct question with No/Yes
answers. Train 48 / dev 32 (disjoint values), plus three conditional eval
contrasts: newvalues (64), newthreshold (48, thresholds 45/75/90), newfield
(64, same values as newvalues under a renamed field). All value lists are
hand-frozen literals; no RNG. Labels come from the spec here;
scripts/verify_e24_fit.py re-derives them from the rendered text with
independent logic.

Outputs: data/e24-fit/{train_fit_rows.jsonl, dev_fit_rows.jsonl,
eval_newvalues_rows.jsonl, eval_newthreshold_rows.jsonl, eval_newfield_rows.jsonl,
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
THRESHOLD = 60.0
OP = "at least"
NEW_FIELD = "bearing_temp_c"
NEW_FIELD_WORD = "bearing temperature"
SITES = ["site-alpha", "site-bravo", "site-charlie", "site-delta", "site-echo", "site-foxtrot"]

TRAIN_PAIRS = [(59.5, 60.5), (59.2, 60.8), (58.2, 61.8), (56.0, 65.0), (52.8, 69.8),
               (47.3, 74.5), (41.8, 80.0), (34.6, 86.9), (25.4, 97.2), (18.9, 106.3),
               (12.7, 114.0), (7.5, 122.4), (44.0, 78.2), (38.6, 91.7), (30.1, 99.8), (22.3, 110.5)]
TRAIN_NO = [59.9, 59.1, 54.5, 50.0, 45.6, 39.5, 31.2, 19.3]
TRAIN_YES = [60.0, 60.1, 61.2, 67.2, 72.0, 77.1, 101.5, 120.6]

DEV_PAIRS = [(59.7, 60.3), (58.6, 62.4), (55.5, 66.8), (51.2, 71.9), (44.9, 83.6),
             (38.4, 89.4), (30.7, 95.5), (21.6, 104.2)]
DEV_NO = [59.8, 59.4, 58.9, 53.7, 48.2, 42.5, 35.1, 16.7]
DEV_YES = [60.2, 60.4, 61.7, 68.5, 75.9, 85.0, 99.1, 112.4]

NV_PAIRS = [(59.3, 60.6), (59.0, 60.9), (58.4, 62.1), (57.1, 63.9), (55.2, 67.5),
            (50.9, 70.3), (46.4, 76.4), (43.1, 81.7), (36.9, 88.2), (29.4, 93.6),
            (24.6, 98.4), (17.2, 108.9), (10.8, 118.2), (5.9, 126.7), (14.3, 103.1), (33.3, 96.3)]
NV_NO = [59.6, 58.8, 58.0, 56.9, 54.0, 49.3, 44.5, 41.2, 37.7, 32.5, 27.8, 23.1, 15.4, 11.2, 8.4, 3.6]
NV_YES = [60.7, 61.0, 61.5, 62.8, 64.4, 66.0, 68.9, 73.3, 79.4, 84.7, 90.2, 94.8, 100.5, 107.1, 116.3, 124.8]

NEW_THRESHOLDS = [
    (45.0, [(44.6, 45.4), (43.8, 47.1), (41.5, 50.3), (36.9, 55.6)],
     [44.9, 44.1, 42.7, 39.4], [45.0, 45.2, 48.6, 58.3]),
    (75.0, [(74.6, 75.4), (73.5, 78.0), (70.2, 82.4), (64.8, 88.6)],
     [74.9, 74.1, 71.6, 66.3], [75.0, 75.3, 79.7, 92.5]),
    (90.0, [(89.6, 90.4), (88.7, 92.1), (85.3, 95.8), (79.6, 101.2)],
     [89.9, 89.0, 86.4, 81.5], [90.0, 90.5, 94.2, 108.7]),
]


def compare(value: float, threshold: float) -> bool:
    return value >= threshold


def make_row(seq, split, value, threshold, *, field=FIELD, field_word=FIELD_WORD,
             variant="base", pair=None, eq_seq=None, contract_split=None):
    # pair members share eq_seq so that ONLY the value differs between them;
    # eval files are test-role data -> contract split "test" (id prefix keeps
    # the eval-set identity)
    eq = seq if eq_seq is None else eq_seq
    split_field = contract_split or split
    state = json.dumps({"equipment_id": f"unit-{eq:03d}", field: value,
                        "site": SITES[eq % len(SITES)]}, sort_keys=True)
    question = (f"Is the {field_word} reading at least {threshold:g} degrees Celsius?")
    gold = compare(value, threshold)
    return {"id": f"e24|{split}|{seq:04d}", "group_id": f"e24|{split}|{seq:04d}",
            "source": "e24-fit", "language": "en", "domain": f"num_{field}",
            "state": state, "question": question,
            "options": [{"id": "false", "description": "No"}, {"id": "true", "description": "Yes"}],
            "target_kind": "single", "target_option": "true" if gold else "false",
            "split": split_field, "question_type": "noul", "label_origin": "e24-spec",
            "operator": OP, "feature": field, "threshold": threshold, "value": value,
            "variant": variant, "pair": pair, "rule_in": "question"}


def build_split(seq_start, split, pairs, no_values, yes_values, *, threshold=THRESHOLD,
                field=FIELD, field_word=FIELD_WORD, pair_prefix="p", contract_split=None):
    rows, seq = [], seq_start
    for k, (lo, hi) in enumerate(pairs):
        pair_id = f"e24{pair_prefix}|{split}|{k:02d}"
        eq_seq = seq  # both members share the same equipment/site: only the value differs
        for side, v in (("lo", lo), ("hi", hi)):
            rows.append(make_row(seq, split, v, threshold, field=field, field_word=field_word,
                                 variant=f"pair_{side}", pair=pair_id, eq_seq=eq_seq,
                                 contract_split=contract_split))
            seq += 1
    for v in no_values:
        rows.append(make_row(seq, split, v, threshold, field=field, field_word=field_word,
                             contract_split=contract_split))
        seq += 1
    for v in yes_values:
        rows.append(make_row(seq, split, v, threshold, field=field, field_word=field_word,
                             contract_split=contract_split))
        seq += 1
    return rows, seq


def main() -> int:
    out = ROOT / "data/e24-fit"
    out.mkdir(parents=True, exist_ok=True)
    seq = 1
    train, seq = build_split(seq, "train", TRAIN_PAIRS, TRAIN_NO, TRAIN_YES, pair_prefix="p")
    dev, seq = build_split(seq, "dev", DEV_PAIRS, DEV_NO, DEV_YES, pair_prefix="d")
    newvalues, seq = build_split(seq, "evalnv", NV_PAIRS, NV_NO, NV_YES, pair_prefix="n",
                                 contract_split="test")
    newfield, seq = build_split(seq, "evalnf", NV_PAIRS, NV_NO, NV_YES,
                                field=NEW_FIELD, field_word=NEW_FIELD_WORD, pair_prefix="f",
                                contract_split="test")
    newthreshold = []
    for threshold, pairs, nos, yeses in NEW_THRESHOLDS:
        rows, seq = build_split(seq, "evalnt", pairs, nos, yeses, threshold=threshold,
                                pair_prefix=f"t{int(threshold)}", contract_split="test")
        newthreshold.extend(rows)

    files = {"train_fit_rows.jsonl": train, "dev_fit_rows.jsonl": dev,
             "eval_newvalues_rows.jsonl": newvalues,
             "eval_newthreshold_rows.jsonl": newthreshold,
             "eval_newfield_rows.jsonl": newfield}
    all_rows = [r for rs in files.values() for r in rs]
    ids = [r["id"] for r in all_rows]
    payloads = [(r["question"], r["state"]) for r in all_rows]
    assert len(ids) == len(set(ids)), "duplicate ids"
    assert len(payloads) == len(set(payloads)), "duplicate payloads"
    # newfield reuses the newvalues values BY DESIGN (pure field-name contrast)
    t60_values = [r["value"] for r in train + dev + newvalues]
    assert len(t60_values) == len(set(t60_values)), "value collision across train/dev/newvalues"
    assert sorted(r["value"] for r in newfield) == sorted(r["value"] for r in newvalues)

    for name, rows in files.items():
        (out / name).write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows),
                                encoding="utf-8")
    manifest = {
        "set": "e24-fit", "protocol": "docs/internal/E24_PROTOCOL_2026-10-10.md",
        "counts": {name.split("_rows")[0]: len(rows) for name, rows in files.items()},
        "field": FIELD, "operator": OP, "threshold": THRESHOLD,
        "new_thresholds": [t for t, *_ in NEW_THRESHOLDS],
        "new_field": NEW_FIELD,
        "balance": {name.split("_rows")[0]: {"no": sum(r["target_option"] == "false" for r in rows),
                                             "yes": sum(r["target_option"] == "true" for r in rows)}
                    for name, rows in files.items()},
        "train_pairs": len(TRAIN_PAIRS), "dev_pairs": len(DEV_PAIRS),
        "boundary_values": {"train": 60.0, "eval_newthreshold": [45.0, 75.0, 90.0]},
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "status": ("fitting microset; measures FIT not generalization; eval files used only "
                   "if train fit >= 0.95 (frozen rule)"),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"counts": manifest["counts"], "balance": manifest["balance"]}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
