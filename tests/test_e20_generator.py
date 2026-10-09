"""E20 generator structural regressions."""

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


g = load("gen_e20_contrast")
v = load("verify_e20_labels")


def test_counts_and_balance():
    data = ROOT / "data/e20-contrast"
    m = json.loads((data / "manifest.json").read_text(encoding="utf-8"))
    assert m["counts"] == {"train_ctrl": 1650, "train_cand": 1650, "dev": 330,
                           "test": 750, "test_cases": 150}
    for fam, bal in m["noul_gold_balance"].items():
        assert abs(bal["true"] - bal["false"]) <= 2, (fam, bal)
    assert m["test_noul_balance"] == {"false": 150, "true": 150}


def test_type_mix_equal_across_arms():
    import collections
    data = ROOT / "data/e20-contrast"

    def mix(name):
        c = collections.Counter(json.loads(l)["question_type"]
                                for l in (data / name).read_text(encoding="utf-8").splitlines() if l.strip())
        return dict(c)
    assert mix("train_ctrl_rows.jsonl") == mix("train_cand_rows.jsonl") == {"choice": 660, "noul": 330, "score": 660}


def test_flips_change_gold_and_irrelevant_style_pairs():
    data = ROOT / "data/e20-contrast"
    rows = {}
    for name in ("train_cand_rows.jsonl", "test_rows.jsonl"):
        for l in (data / name).read_text(encoding="utf-8").splitlines():
            if l.strip():
                r = json.loads(l)
                rows[r["id"]] = r
    checked = 0
    for r in rows.values():
        if r["variant"] == "base" and r.get("pair"):
            for variant in ("action_flip", "outcome_flip", "needs_review_flip", "risk_flip", "urgency_flip"):
                other = rows.get(f"{r['id'].rsplit(':', 1)[0]}:{variant}")
                if other and other["variant"] != "base":
                    desc = lambda row: next(o["description"] for o in row["options"] if o["id"] == row["target_option"])
                    assert desc(other) != desc(r), r["id"]
                    checked += 1
    assert checked > 500


def test_verifier_agrees_on_sealed_sample():
    data = ROOT / "data/e20-contrast"
    n = 0
    for l in (data / "test_rows.jsonl").read_text(encoding="utf-8").splitlines():
        if l.strip() and n < 120:
            row = json.loads(l)
            assert v.verify_row(row), row["id"]
            n += 1
    assert n == 120


def test_den_flip_fallback():
    rng = __import__("random").Random(3)
    spec = g.e19.FAMILIES["it_access_review"]
    for _ in range(200):
        f = spec["sample"](rng)["facts"]
        if f["resource_sensitivity"] != "high":
            continue
        base = spec["action_gold"](f)
        ff, rules_fn, flip_gold = g.e20_action_flip("it_access_review", f, base)
        assert flip_gold != base
        return
