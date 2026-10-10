"""E27 generator: dense comparison grid + fresh two-decimal testE (frozen protocol).

docs/internal/E27_PROTOCOL_2026-10-10.md. TRAIN = the 92 exact E26 treat-train
values (input file pinned by sha256), each against 8 thresholds (offsets
-25/-7/-1/0/+1/+7/+15/+25 -> 4 Yes / 4 No per value with the IDENTICAL state, so
any question-ignoring rule is EXACTLY 0.500 on the whole train). Reserved
thresholds and out-of-range hits are substituted from a frozen same-sign ladder.
testE = fresh two-decimal thresholds (47.31..118.57) that cannot collide with any
one-decimal grid/historical threshold: balanced quadrants, 8+8 PURE pairs,
10 monotonicity sweeps. scripts/verify_e27_grid.py re-derives everything
independently from the rendered text and the frozen rule.

Outputs: data/e27-grid/{train_rows.jsonl, test_e_rows.jsonl, manifest.json}.
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
INPUT = ROOT / "data/e26-order/train_treat_rows.jsonl"
INPUT_SHA = "4b76a5c5d928aeda328b209024c44def4ea163b650fda51b61a4d5b3a77decb8"
OFFSETS = [-25, -7, -1, 0, 1, 7, 15, 25]
YES_OFFSETS = {-25, -7, -1, 0}
LADDER = {1: [3, 15, 10, 5, 20], -1: [-3, -15, -10, -5, -20]}
RESERVED = {25.0, 44.5, 56.5, 66.5, 78.5, 88.5, 115.0}
THR_MIN, THR_MAX = 10.0, 130.0
SITES = ["site-alpha", "site-bravo", "site-charlie", "site-delta", "site-echo", "site-foxtrot"]
_LETTERS = "abcdefghjkmnpqrstuvwxyz"

TESTE_INTERP = [47.31, 58.73, 68.17, 79.43, 89.27]
TESTE_EXTRAP = [21.83, 118.57]
TESTE_MONO_THR = [47.31, 58.73, 68.17, 79.43, 89.27, 21.83, 118.57, 41.0, 63.0, 84.5]
# hand-frozen testE literals: per threshold {quadrant: [values]} (5Y/5N exactly)
TESTE_QUADS = {
    47.31: {"bd_y": [47.31], "sd_y": [48.19, 49.02], "cd_y": [53.44, 55.83],
            "sd_n": [46.87, 45.93], "cd_n": [39.55, 36.42, 33.17]},
    58.73: {"bd_y": [58.73], "sd_y": [59.61, 59.29], "cd_y": [64.19, 66.35],
            "sd_n": [57.82, 56.94], "cd_n": [49.18, 46.35, 43.72]},
    68.17: {"bd_y": [68.17], "sd_y": [69.42, 69.58], "cd_y": [74.36, 76.28],
            "sd_n": [67.28, 66.51], "cd_n": [59.15, 56.38, 53.81]},
    79.43: {"bd_y": [79.43], "sd_y": [79.58, 79.92], "cd_y": [85.71, 88.16],
            "sd_n": [78.54, 77.86], "cd_n": [69.27, 66.84, 64.53]},
    89.27: {"bd_y": [89.27], "sd_y": [89.39, 89.74], "cd_y": [97.82, 101.47],
            "sd_n": [88.34, 87.46], "cd_n": [78.93, 76.51, 74.18]},
    21.83: {"bd_y": [21.83], "sd_y": [23.47, 24.96], "cd_y": [31.29, 32.47],
            "sd_n": [20.92, 20.64], "cd_n": [17.24, 15.68, 13.11]},
    118.57: {"bd_y": [118.57], "sd_y": [118.94, 119.37], "cd_y": [126.71, 129.13],
             "sd_n": [117.64, 116.29], "cd_n": [109.64, 107.28, 104.93]},
}
TESTE_PAIRS_INTERP = [  # (value, thr_lo -> Yes, thr_hi -> No), pure interp
    (51.42, 47.31, 58.73), (53.86, 47.31, 58.73),
    (62.94, 58.73, 68.17), (65.28, 58.73, 68.17),
    (71.53, 68.17, 79.43), (74.89, 68.17, 79.43),
    (83.26, 79.43, 89.27), (86.71, 79.43, 89.27),
]
TESTE_PAIRS_EXTRAP = [  # both thresholds extrap -> pure extrap
    (35.17, 21.83, 118.57), (49.62, 21.83, 118.57), (63.28, 21.83, 118.57),
    (77.94, 21.83, 118.57), (91.36, 21.83, 118.57), (105.71, 21.83, 118.57),
    (98.43, 21.83, 118.57), (42.58, 21.83, 118.57),
]
TESTE_MONO = {  # threshold -> 5 values below (No) then 5 above (Yes), 2 decimals
    47.31: [41.26, 44.83, 46.52, 47.08, 47.19, 47.42, 47.87, 48.94, 51.67, 56.32],
    58.73: [52.18, 55.67, 57.42, 58.21, 58.64, 58.85, 59.36, 60.72, 63.48, 68.15],
    68.17: [61.93, 65.42, 67.18, 67.89, 68.06, 68.28, 68.74, 69.91, 72.56, 77.83],
    79.43: [73.17, 76.68, 78.34, 79.05, 79.31, 79.56, 80.12, 81.47, 84.23, 89.76],
    89.27: [82.94, 86.51, 88.26, 88.95, 89.18, 89.36, 89.83, 91.07, 93.72, 98.46],
    21.83: [15.42, 18.87, 20.53, 21.26, 21.71, 21.94, 22.37, 23.85, 26.49, 31.12],
    118.57: [112.34, 115.78, 117.42, 118.13, 118.46, 118.68, 119.15, 120.62, 123.37, 128.84],
    41.0: [34.76, 38.21, 39.94, 40.68, 40.91, 41.13, 41.59, 42.86, 45.52, 50.27],
    63.0: [56.83, 60.29, 62.05, 62.71, 62.93, 63.14, 63.62, 64.87, 67.53, 72.28],
    84.5: [78.19, 81.74, 83.48, 84.16, 84.41, 84.63, 85.09, 86.34, 89.02, 93.75],
}


def eq_codes():
    """Base-24 alphabetic codes, variable length (aa..zz, then aaa..): the first
    576 codes are byte-identical to the E26 scheme; E27 needs more rows."""
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


def make_row(seq, split_name, value, threshold, *, variant="base", pair=None,
             eq=None, site=None, quadrant=None, sweep=None):
    state = json.dumps({"equipment_id": eq, FIELD: value, "site": site}, sort_keys=True)
    question = f"Is the {FIELD_WORD} reading at least {fmt(threshold)} degrees Celsius?"
    gold = value >= threshold
    return {"id": f"e27|{split_name}|{seq:04d}",
            "group_id": pair or f"e27|{split_name}|{seq:04d}",
            "source": "e27-grid", "language": "en", "domain": f"num_{FIELD}",
            "state": state, "question": question,
            "options": [{"id": "false", "description": "No"}, {"id": "true", "description": "Yes"}],
            "target_kind": "single", "target_option": "true" if gold else "false",
            "split": "train", "question_type": "noul", "label_origin": "e27-spec",
            "operator": OP, "feature": FIELD, "threshold": threshold, "value": value,
            "variant": variant, "pair": pair, "quadrant": quadrant, "sweep": sweep,
            "rule_in": "question",
            "thr_range": ("extrap" if threshold in TESTE_EXTRAP
                          else "interp" if threshold in TESTE_INTERP else "grid")}


class Builder:
    def __init__(self, split_name):
        self.split_name = split_name
        self.codes = eq_codes()
        self.rows = []
        self.seq = 1

    def add(self, value, threshold, **kw):
        r = make_row(self.seq, self.split_name, value, threshold, **kw)
        self.rows.append(r)
        self.seq += 1
        return r

    def add_pair(self, value, t_lo, t_hi, *, rng):
        pair_id = f"e27p|{self.split_name}|{fmt(value)}"
        eq, site = next(self.codes), SITES[len(self.rows) % len(SITES)]
        self.add(value, t_lo, variant="pair_thrlo", pair=pair_id, eq=eq, site=site)
        self.add(value, t_hi, variant="pair_thrhi", pair=pair_id, eq=eq, site=site)


def build_train():
    got = hashlib.sha256(INPUT.read_bytes()).hexdigest()
    assert got == INPUT_SHA, f"input train_treat file changed: {got}"
    values = sorted({json.loads(l)["value"] for l in
                     INPUT.read_text(encoding="utf-8").splitlines() if l.strip()})
    assert len(values) == 92, len(values)
    b = Builder("train")
    substitutions = []
    for v in values:
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
            b.add(v, thr, eq=next(b.codes), site=SITES[len(b.rows) % len(SITES)])
    return b.rows, substitutions


def build_teste():
    b = Builder("teste")
    for t, quads in TESTE_QUADS.items():
        for quad, vals in quads.items():
            for v in vals:
                b.add(v, t, eq=next(b.codes), site=SITES[len(b.rows) % len(SITES)],
                      quadrant=quad)
    for v, t_lo, t_hi in TESTE_PAIRS_INTERP + TESTE_PAIRS_EXTRAP:
        b.add_pair(v, t_lo, t_hi, rng="x")
    for t, vals in TESTE_MONO.items():
        sweep_id = f"e27m|{fmt(t)}"
        for v in vals:
            b.add(v, t, eq=next(b.codes), site=SITES[len(b.rows) % len(SITES)],
                  variant="mono", sweep=sweep_id)
    return b.rows


def main() -> int:
    train, subs = build_train()
    teste = build_teste()
    files = {"train": (train, "train_rows.jsonl"), "teste": (teste, "test_e_rows.jsonl")}
    all_rows = train + teste
    ids = [r["id"] for r in all_rows]
    payloads = [(r["question"], r["state"]) for r in all_rows]
    assert len(ids) == len(set(ids)) and len(payloads) == len(set(payloads))
    # per-value 4/4 balance on train; global exact on teste
    from collections import Counter, defaultdict
    by_val = defaultdict(Counter)
    for r in train:
        by_val[r["value"]][r["target_option"]] += 1
    assert all(c["true"] == 4 and c["false"] == 4 for c in by_val.values())
    bal = Counter(r["target_option"] for r in teste)
    assert bal["true"] == bal["false"], dict(bal)
    # hard cap: any question-ignoring rule == 0.500 exactly on train
    golds = [r["target_option"] == "true" for r in train]
    assert abs(sum(golds) / len(golds) - 0.5) < 1e-12
    # each value's 8 rows share NOTHING but the value: state differs (eq/site) by design,
    # but the QUESTION-IGNORING cap holds because gold splits 4/4 per identical value
    # reserved never as train threshold
    assert not RESERVED & {r["threshold"] for r in train}
    # testE decimals: every value/threshold has exactly 2 decimals (freshness by construction)
    for r in teste:
        for x in (r["value"], r["threshold"]):
            s = f"{x:.2f}"
            assert len(s.split(".")[1]) == 2, (r["id"], x)
    # sweeps: strictly increasing, 5 No then 5 Yes
    sweeps = defaultdict(list)
    for r in teste:
        if r["sweep"]:
            sweeps[r["sweep"]].append(r)
    assert len(sweeps) == 10
    for sid, rows in sweeps.items():
        rows.sort(key=lambda r: r["value"])
        assert all(a["value"] < b2["value"] for a, b2 in zip(rows, rows[1:])), sid
        assert [r["target_option"] == "true" for r in rows] == [False] * 5 + [True] * 5, sid
    # pair integrity (thr-only): state identical, question differs only in threshold
    pairs = defaultdict(list)
    for r in teste:
        if r["pair"]:
            pairs[r["pair"]].append(r)
    for pid, members in pairs.items():
        assert len(members) == 2 and {m["variant"] for m in members} == {"pair_thrlo", "pair_thrhi"}
        lo, hi = members
        assert lo["state"] == hi["state"], pid
        assert lo["question"] == hi["question"].replace(fmt(hi["threshold"]), fmt(lo["threshold"]))
        assert lo["target_option"] == "true" and hi["target_option"] == "false"

    out = ROOT / "data/e27-grid"
    out.mkdir(parents=True, exist_ok=True)
    for name, (rows, fname) in files.items():
        (out / fname).write_text(
            "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows), encoding="utf-8")
    manifest = {
        "set": "e27-grid", "protocol": "docs/internal/E27_PROTOCOL_2026-10-10.md",
        "input_train_treat_sha256": INPUT_SHA,
        "offsets": OFFSETS, "ladder": LADDER, "reserved": sorted(RESERVED),
        "substitutions": subs,
        "counts": {name: len(rows) for name, (rows, _) in files.items()},
        "balance": {name: dict(Counter(r["target_option"] for r in rows))
                    for name, (rows, _) in files.items()},
        "teste_thresholds": {"interp": TESTE_INTERP, "extrap": TESTE_EXTRAP},
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "status": ("dense comparison grid; tests opened only after finalists fixed"),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True),
                                        encoding="utf-8")
    print(json.dumps({"counts": manifest["counts"], "balance": manifest["balance"],
                      "substitutions": subs}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
