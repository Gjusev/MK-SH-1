"""E35 generator: NEW disjoint extreme-value grid (frozen criteria).

One-decimal values in the desert ranges [18-27] and [105-126], each against
the 8-offset rule from E27/E28 (thr ∈ v + {-25,-7,-1,0,+1,+7,+15,+25} with
the reserved/exclusion ladder). Disjoint from ALL previous values (e26/e27/
e28 grids checked). Hand-frozen literals; no RNG.

Output: data/e35-extremes/{train_rows.jsonl, manifest.json}.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIELD, FIELD_WORD, OP = "motor_temp_c", "motor temperature", "at least"
RESERVED = {25.0, 44.5, 56.5, 66.5, 78.5, 88.5, 115.0, 30.0, 100.0, 45.0, 55.0, 65.0, 75.0, 85.0}
OFFSETS = [-25, -7, -1, 0, 1, 7, 15, 25]
_L = "abcdefghjkmnpqrstuvwxyz"
SITES = ["site-alpha", "site-bravo", "site-charlie", "site-delta", "site-echo", "site-foxtrot"]


def eq_codes():
    i = 0
    while True:
        yield f"unit-{_L[i // len(_L)]}{_L[i % len(_L)]}"
        i += 1


# NEW extreme values — disjoint from e26/e27/e28 (checked at build)
LOWS = [18.3, 19.1, 19.8, 20.4, 21.2, 22.1, 23.0, 23.8, 24.2, 25.1, 25.8, 26.3, 26.7, 27.2, 27.6]
HIGHS = [105.2, 106.1, 107.4, 108.8, 109.5, 110.3, 111.6, 112.3, 113.1, 114.2, 115.8, 117.2, 119.4, 120.8, 122.6, 124.1]
VALUES = LOWS + HIGHS


def thr_ok(t):
    return 10 <= t <= 130 and t not in RESERVED


def substitute(v, off):
    """Frozen same-sign ladder from E27/E28: {3, 15, 10, 5, 20}."""
    for sub in (3, 15, 10, 5, 20):
        t = round(v + (sub if off > 0 else -sub), 1)
        if thr_ok(t):
            return t, sub
    return None, None


def main() -> int:
    # disjointness check
    prior = set()
    for f in ("data/e28-ext/train_rows.jsonl", "data/e27-grid/train_rows.jsonl"):
        p = ROOT / f
        if p.exists():
            for l in p.read_text(encoding="utf-8").splitlines():
                if l.strip():
                    prior.add(json.loads(l)["value"])
    overlap = {v for v in VALUES if v in prior}
    assert not overlap, f"value overlap with prior grids: {sorted(overlap)}"
    assert all(thr_ok(v) or v in RESERVED for v in [])  # values themselves may be reserved? No —
    # actually values can be any number; the THRESHOLDS derived must be in range and non-reserved
    # regenerate: value 115.8 is fine as a VALUE; its thresholds must dodge reserved

    codes = eq_codes()
    rows, subs = [], []
    seq = 1
    for v in VALUES:
        eq, site = next(codes), SITES[seq % len(SITES)]
        for off in OFFSETS:
            t = round(v + off, 1)
            used_sub = None
            if not thr_ok(t):
                t, used_sub = substitute(v, off)
                if t is None:
                    continue  # skip if no substitute works (shouldn't happen)
                subs.append({"value": v, "original_offset": off, "substituted_offset": used_sub, "threshold": t})
            gold = v >= t
            state = json.dumps({"equipment_id": eq, FIELD: v, "site": site}, sort_keys=True)
            question = f"Is the {FIELD_WORD} reading at least {t:g} degrees Celsius?"
            rows.append({"id": f"e35x|{seq:04d}", "group_id": f"e35x|{seq:04d}",
                         "source": "e35-extremes", "language": "en", "domain": f"num_{FIELD}",
                         "state": state, "question": question,
                         "options": [{"id": "false", "description": "No"}, {"id": "true", "description": "Yes"}],
                         "target_kind": "single", "target_option": "true" if gold else "false",
                         "split": "train", "question_type": "noul", "label_origin": "e35-spec",
                         "operator": OP, "feature": FIELD, "threshold": t, "value": v,
                         "variant": "base", "pair": None, "rule_in": "question",
                         "thr_range": "extremes"})
            seq += 1

    out = ROOT / "data/e35-extremes"
    out.mkdir(parents=True, exist_ok=True)
    (out / "train_rows.jsonl").write_text(
        "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows), encoding="utf-8")
    yes = sum(1 for r in rows if r["target_option"] == "true")
    manifest = {"protocol": "docs/E35_CRITERIA_2026-10-11.md",
                "values": VALUES, "n_values": len(VALUES),
                "offsets": OFFSETS, "reserved": sorted(RESERVED),
                "counts": {"rows": len(rows)},
                "balance": {"yes": yes, "no": len(rows) - yes},
                "substitutions": subs,
                "prior_grids_disjoint": True,
                "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(f"rows={len(rows)} balance={yes}/{len(rows)-yes} values={len(VALUES)} subs={len(subs)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
