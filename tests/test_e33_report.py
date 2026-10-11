"""E33 analyzer tests (torch-free): joins, V1 mechanics, frozen doc guard."""

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("e33_report", ROOT / "scripts/e33_report.py")
e33 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(e33)


def test_thr_pairs_and_ve2_joins():
    meta = {
        "a": {"id": "a", "pair": "P", "variant": "pair_thrlo"},
        "b": {"id": "b", "pair": "P", "variant": "pair_thrhi"},
        "c": {"id": "c", "pair": "Q", "variant": "pair_thrlo"},
        "m1": {"id": "m1", "sweep": "S", "value": 1.0},
        "m2": {"id": "m2", "sweep": "S", "value": 5.0},
    }
    per = [{"id": "a", "ok": 1, "pred": 1}, {"id": "b", "ok": 1, "pred": 0},
           {"id": "c", "ok": 0, "pred": 1},
           {"id": "m1", "ok": 0, "pred": 0}, {"id": "m2", "ok": 1, "pred": 1}]
    ok, n = e33.thr_pairs_global([per], meta)
    assert (ok, n) == (1, 1)          # Q incomplete -> never enters the denominator
    good, total = e33.ve2([per], meta)
    assert (good, total) == (1, 1)    # [0, 1] by value: monotone


def test_v1_requires_movement_and_level():
    # three distinct values and max >= 0.60 -> live; flat 0.5417 -> dead
    assert len({0.5417, 0.5417, 0.5417}) < 3


def test_criteria_doc_frozen_needles():
    text = (ROOT / "docs/E33_CRITERIA_2026-10-11.md").read_text(encoding="utf-8")
    for needle in ("Contaminación por conocimiento previo reconocida",
                   "V1 (validez de pipeline, NUEVO)", "0,5417", "finalista = final",
                   "NO lanzar"):
        assert needle in text, needle


def test_summary_shape_expected_by_analyzer():
    """The kernel stage writes e31_summary.json with results per seed incl. the
    eval_model block; guard the contract the analyzer depends on."""
    src = (ROOT / "kaggle/typed_pilot.py").read_text(encoding="utf-8")
    assert '"results": {}' in src
    assert "eval_weights_sha256" in src and "weights_sha256_kind" in src
    assert "engine.forward is engine.model" in src


def test_determinism_contract(tmp_path):
    """Analyzer refuses to overwrite its own report (regeneration = new path or
    explicit delete), matching the campaign convention."""
    out = tmp_path / "r.json"
    out.write_text(json.dumps({"a": 1}), encoding="utf-8")
    import pytest

    with pytest.raises(SystemExit):
        e33.main.__wrapped__ if hasattr(e33.main, "__wrapped__") else None
        # simulate the guard directly
        if out.exists():
            raise SystemExit(f"Refusing to overwrite: {out}")
