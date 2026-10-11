"""E32 harness tests (torch-free): metric joins + frozen-literal guards."""

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("e32_report", ROOT / "scripts/e32_report.py")
e32 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(e32)


def _meta(rows):
    return {r["id"]: r for r in rows}


def test_metric_join_ranges_pairs_and_sweeps():
    meta = _meta([
        {"id": "a", "thr_range": "interp", "pair": None, "variant": "base", "sweep": None},
        {"id": "b", "thr_range": "interp", "pair": None, "variant": "base", "sweep": None},
        {"id": "p1", "thr_range": "interp", "pair": "P", "variant": "pair_thrlo", "sweep": None},
        {"id": "p2", "thr_range": "interp", "pair": "P", "variant": "pair_thrhi", "sweep": None},
        {"id": "q1", "thr_range": "extrap", "pair": "Q", "variant": "pair_thrlo", "sweep": None},
        {"id": "q2", "thr_range": "extrap", "pair": "Q", "variant": "pair_thrhi", "sweep": None},
        {"id": "m1", "thr_range": "interp", "pair": None, "variant": "mono", "sweep": "S", "value": 1.0},
        {"id": "m2", "thr_range": "interp", "pair": None, "variant": "mono", "sweep": "S", "value": 9.0},
    ])
    per = [{"id": r, "pred": 1, "gold": 1, "ok": 1} for r in ("a", "b", "p1", "p2", "q1", "m2")] + \
          [{"id": r, "pred": 0, "gold": 1, "ok": 0} for r in ("q2", "m1")]
    out = e32.metric_join(per, meta)
    assert out["n"] == 8 and out["accuracy"] == 0.75
    assert out["by_range"]["interp"] == [5, 6] and out["by_range"]["extrap"] == [1, 2]
    assert out["thr_pairs_global"] == [1, 2]           # interp pair both-correct
    assert out["pure_pairs"] == {"interp": [1, 1], "extrap": [0, 1]}
    assert out["mono_sweeps"] == [1, 1]                # preds [0, 1] by value: monotone


def test_incomplete_pairs_never_count():
    meta = _meta([{"id": "p1", "thr_range": "interp", "pair": "P", "variant": "pair_thrlo"},
                  {"id": "x", "thr_range": "interp", "pair": None, "variant": "base"}])
    per = [{"id": "p1", "pred": 1, "gold": 1, "ok": 1}, {"id": "x", "pred": 0, "gold": 0, "ok": 1}]
    out = e32.metric_join(per, meta)
    assert out["thr_pairs_global"] == [0, 0] and out["pure_pairs"]["interp"] == [0, 0]


def test_frozen_model_literals_match_disk():
    import hashlib
    for key, (path, sha) in e32.JULIA_MODELS.items():
        if not path.exists():
            continue
        assert hashlib.sha256(path.read_bytes()).hexdigest() == sha, key


def test_criteria_doc_pinned_in_repo():
    doc = ROOT / "docs/E32_CRITERIA_2026-10-11.md"
    assert doc.exists()
    text = doc.read_text(encoding="utf-8")
    for needle in ("SISTEMAS COMPLETOS", "A-EMPATE", "pérdida severa", "ventaja de ARQUITECTURA"):
        assert needle in text, needle


def test_typed2000_row_construction_crosschecks_frozen_zero_shot():
    """Only runs where the parquet + frozen predictions exist; validates ids,
    types, keys and gold against the frozen zero-shot artifacts."""
    import pytest

    if not (e32.PARQUET.exists() and e32.ZERO_SHOT.exists()):
        pytest.skip("frozen typed-2000 artifacts not present")
    pytest.importorskip("pyarrow")
    rows = e32.typed2000_rows()
    assert len(rows) == 2000
    assert all(0 <= r["gold"] < len(r["keys"]) for r in rows)
    assert all(r["options"] and r["question"] for r in rows)
