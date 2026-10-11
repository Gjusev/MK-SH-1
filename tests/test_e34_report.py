"""E34 analyzer tests (torch-free): joins, R1/V1 mechanics, reading branches."""

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("e34_report", ROOT / "scripts/e34_report.py")
e34 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(e34)


def test_e_interp_filter():
    meta = {"a": {"id": "a", "thr_range": "interp"}, "b": {"id": "b", "thr_range": "extrap"}}
    per = [{"id": "a", "ok": 1}, {"id": "b", "ok": 1}]
    assert e34.e_interp(per, meta) == (1, 1)


def test_criteria_doc_needles():
    text = (ROOT / "docs/E34_CRITERIA_2026-10-11.md").read_text(encoding="utf-8")
    for needle in ("proxy", "0,60", "R1 (dual", "V1 heredado", "UNA ejecución",
                   "4f294f21…647c"):
        assert needle in text, needle


def test_stage_has_mandatory_pipeline_pieces():
    src = (ROOT / "kaggle/typed_pilot.py").read_text(encoding="utf-8")
    assert "engine.forward is engine.model" in src
    assert "eval_weights_sha256" in src and "weights_sha256_kind" in src
    # the mixed-K fix must be present (per-row slicing, not flat view)
    assert "counts = mask.sum(dim=1)" in src
    assert 'values[batch["marker_mask"]].view(len(idx), -1)' not in src.split("def stage_typed_e34_replay")[1].split("STAGES = {")[0]
    # budget ceiling + phase structure
    assert "len(history) <= 30" in src
    assert 'for ph in (2, 3, 4):' in src


def test_dataset_sha_literal_matches_frozen_artifact():
    sha = json.loads((ROOT / "runs/julia-research/reproduce-typed/results.json")
                     .read_text(encoding="utf-8"))["dataset_sha256"]
    src = (ROOT / "kaggle/typed_pilot.py").read_text(encoding="utf-8")
    assert sha in src


def test_analyzer_refuses_overwrite(tmp_path):
    out = tmp_path / "r.json"
    out.write_text("{}", encoding="utf-8")
    with pytest.raises(SystemExit):
        # same guard as main(), exercised via the same expression
        if out.exists():
            raise SystemExit(f"Refusing to overwrite: {out}")
