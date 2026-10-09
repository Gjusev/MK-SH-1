"""E21 diagnostic-v2 regressions: intervention contracts and verifier agreement."""

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


g = load("gen_e20_diag2")


def rows():
    return [json.loads(l) for l in (ROOT / "data/e20-diag2/rows.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]


def gold_int(r):
    return int(r["target_option"] == "true")


def test_parity_and_and_or_cover_all_four_combos():
    rs = rows()
    for skill in ("and4", "or4"):
        block = [r for r in rs if r.get("variant") == "base" and ("and4" in r["id"] or "or4" in r["id"])]
    base = {r["id"]: r for r in rs}
    for skill in ("and4", "or4"):
        got = set()
        for r in rs:
            if r["id"].endswith((f"{skill}_0", f"{skill}_1", f"{skill}_2", f"{skill}_3")) or f"|{skill}_" in r["id"]:
                pass
        combos = [tuple(json.loads(r["state"]).values()) for r in rs if f"|{skill}_" in r["id"]]
        assert len(combos) == 16  # 4 blocks x 4 combos
    # AND: exactly one combo-true per block; OR: exactly one combo-false
    for dom in ("probe_mix0", "probe_mix1", "probe_mix2", "probe_mix3"):
        ands = [gold_int(r) for r in rs if dom in r["id"] and "|and4_" in r["id"]]
        ors = [gold_int(r) for r in rs if dom in r["id"] and "|or4_" in r["id"]]
        assert sum(ands) == 1 and sum(ors) == 3


def test_factflip_changes_exactly_one_key_and_inverts():
    base = {r["id"]: r for r in rows()}
    checked = 0
    for r in rows():
        if r.get("variant") == "factflip":
            b = base[r["id"].replace("factflip", "").rstrip(":") if False else r["id"]]
        if r.get("variant") == "factflip":
            partner = next(x for x in rows() if x["id"] == r["id"] and x["variant"] == "base")
            ev_a, ev_b = json.loads(partner["state"]), json.loads(r["state"])
            changed = [k for k in set(ev_a) | set(ev_b) if ev_a.get(k) != ev_b.get(k)]
            assert len(changed) == 1, (r["id"], changed)
            assert gold_int(r) != gold_int(partner), r["id"]
            checked += 1
    assert checked == 32


def test_ruleflip_same_state_and_inverts():
    rs = rows()
    base = {r["id"]: r for r in rs if r["variant"] == "base"}
    checked = 0
    for r in rs:
        if r.get("variant") == "ruleflip":
            partner = base[r["id"]]
            assert r["state"] == partner["state"]
            assert r["options"][0]["description"] != partner["options"][0]["description"]
            assert gold_int(r) != gold_int(partner)
            checked += 1
    assert checked == 32


def test_paraphrase_actually_rewords_and_keeps_gold():
    rs = rows()
    base = {r["id"]: r for r in rs if r["variant"] == "base"}
    checked = 0
    for r in rs:
        if r.get("variant") == "paraphrase":
            partner = base[r["id"]]
            assert r["options"][0]["description"] != partner["options"][0]["description"]
            assert r["state"] == partner["state"]
            assert gold_int(r) == gold_int(partner)
            checked += 1
    assert checked == 32


def test_irrelevant_adds_exactly_one_key():
    rs = rows()
    base = {r["id"]: r for r in rs if r["variant"] == "base"}
    checked = 0
    for r in rs:
        if r.get("variant") == "irrelevant":
            partner = base[r["id"]]
            ev_a, ev_b = json.loads(partner["state"]), json.loads(r["state"])
            assert set(ev_a) < set(ev_b) and len(set(ev_b) - set(ev_a)) == 1
            assert all(ev_a[k] == ev_b[k] for k in ev_a)
            assert gold_int(r) == gold_int(partner)
            checked += 1
    assert checked == 32


def test_checker_rederives_every_gold():
    rs = rows()
    for r in rs:
        opts = {o["id"]: o["description"] for o in r["options"]}
        assert g.checker_gold(r["question"], opts, json.loads(r["state"])) == gold_int(r), r["id"]


def test_numcmp_includes_exact_boundaries():
    rs = rows()
    thresholds = {f: t for f, u, t, r in g.NUM_FEATURES}
    seen = set()
    for r in rs:
        if "|numcmp_" in r["id"]:
            ev = json.loads(r["state"])
            for k, v in ev.items():
                if k in thresholds and abs(v - thresholds[k]) < 1e-9:
                    seen.add(k)
    assert seen == set(thresholds)  # every feature has an at-threshold probe
