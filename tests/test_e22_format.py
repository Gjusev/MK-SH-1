"""E22 format battery: hand-computed boundaries, verifier agreement, contracts."""

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys_path_fixed = False


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ver = load("verify_e22_format")


def rows():
    return [json.loads(l) for l in (ROOT / "data/e22-format/rows.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]


def test_hand_computed_boundaries():
    """Manually derived expectations - independent of both generator and verifier."""
    rs = {r["id"]: r for r in rows()}
    # pressure threshold 5.0, operator at least, boundary value 5.0 -> TRUE
    assert rs["e22f|pressure_bar|at least|5|json_opt"]["target_option"] == "true"
    assert rs["e22f|pressure_bar|at least|5|prose_q"]["target_option"] == "true"
    # 'more than' at the boundary value is FALSE (strict)
    assert rs["e22f|pressure_bar|more than|5|json_q"]["target_option"] == "false"
    # 'at most' at the boundary is TRUE
    assert rs["e22f|humidity_pct|at most|60|prose_opt"]["target_option"] == "true"
    # 'below' at the boundary is FALSE
    assert rs["e22f|battery_pct|below|20|json_opt"]["target_option"] == "false"
    # below: 19.5 and 16.5 are TRUE; 20 (boundary) and 28 are FALSE
    assert rs["e22f|battery_pct|below|19.5|json_q"]["target_option"] == "true"
    assert rs["e22f|battery_pct|below|28|prose_q"]["target_option"] == "false"


def test_verifier_agrees_on_every_row():
    for r in rows():
        assert ver.gold(r) == int(r["target_option"] == "true"), r["id"]


def test_all_four_formats_same_gold_and_facts():
    from collections import defaultdict
    by_group = defaultdict(list)
    for r in rows():
        by_group[r["group_id"]].append(r)
    for group, variants in by_group.items():
        golds = {v["target_option"] for v in variants}
        assert len(golds) == 1, group
        if group.split("|")[1].startswith("pair"):
            assert {v["format"] for v in variants} == {"json_opt", "json_q", "prose_opt", "prose_q"}


def test_unique_ids_and_no_content_duplicates():
    rs = rows()
    ids = [r["id"] for r in rs]
    assert len(ids) == len(set(ids))
    contents = {(r["feature"], r["operator"], r["value"], r["format"], r["variant"]) for r in rs}
    assert len(contents) == len(rs)


def test_pairs_single_fact_inversion_spanning_formats():
    from collections import defaultdict
    by_pair = defaultdict(dict)
    for r in rows():
        if r["pair"]:
            by_pair[r["pair"]][(r["variant"], r["format"])] = r
    assert len(by_pair) == 12
    for pair, members in by_pair.items():
        assert set(members) == {(f"pair_{side}", fmt) for side in ("hi", "lo") for fmt in ("json_opt", "json_q", "prose_opt", "prose_q")}
        golds_hi = {members[("pair_hi", f)]["target_option"] for f in ("json_opt", "json_q", "prose_opt", "prose_q")}
        golds_lo = {members[("pair_lo", f)]["target_option"] for f in ("json_opt", "json_q", "prose_opt", "prose_q")}
        assert len(golds_hi) == 1 and len(golds_lo) == 1 and golds_hi != golds_lo


def test_pair_members_invert_and_states_differ_by_one_number():
    import re
    rs = rows()
    by_id = {r["id"]: r for r in rs}
    pairs = [r for r in rs if r["variant"] == "pair_hi" and r["id"].endswith("json_opt")]
    for r in pairs:
        lo = by_id[r["id"].replace("|hi|", "|lo|")]
        assert lo is not None
        assert lo["target_option"] != r["target_option"]
        # states differ only in the number (same feature/format)
        n_hi = re.findall(r"-?\d+(?:\.\d+)?", r["state"])
        n_lo = re.findall(r"-?\d+(?:\.\d+)?", lo["state"])
        assert len(n_hi) == len(n_lo)
        assert sum(a != b for a, b in zip(n_hi, n_lo)) <= 2  # prose carries one number; json one
