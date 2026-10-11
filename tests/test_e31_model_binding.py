"""Regression for the E31 invalid_model_binding bug (2026-10-11 correction).

The E31 stage's evaluate(rows) helper closed over a stage-scope engine loaded
with ORIGINAL Julia weights, while run_seed trained a separate local engine —
so dev and all test evals measured the wrong model. These tests lock the fix:

1. structural: evaluate takes the engine as an explicit argument, the surplus
   stage-scope engine is gone, every call inside run_seed passes the local
   engine, and the eval-time weights identity is registered in the result;
2. functional: the REAL evaluate/to_payload/golds source extracted from the
   stage is exec'd with two prediction-distinct sentinel engines — the
   per-case predictions must come from the SUPPLIED engine;
3. the flat kernel runner (kaggle/kernel_flat/run.py) carries the fixed pattern.

With the pre-fix code (def evaluate(rows) capturing the stage engine) test 2
raises TypeError and tests 1/3 fail on signature/arg-count asserts.
"""

import ast
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TYPED = ROOT / "kaggle/typed_pilot.py"
FLAT = ROOT / "kaggle/kernel_flat/run.py"


def _stage_node(source: str) -> ast.FunctionDef:
    tree = ast.parse(source)
    for fn in tree.body:
        if isinstance(fn, ast.FunctionDef) and fn.name == "stage_typed_e31_julia":
            return fn
    raise AssertionError("stage_typed_e31_julia not found")


def _nested(stage: ast.FunctionDef, name: str) -> ast.FunctionDef:
    for node in ast.walk(stage):
        if isinstance(node, ast.FunctionDef) and node.name == name and node is not stage:
            return node
    raise AssertionError(f"nested def {name} not found in stage")


def test_evaluate_takes_engine_explicitly():
    src = TYPED.read_text(encoding="utf-8")
    ev = _nested(_stage_node(src), "evaluate")
    params = [a.arg for a in ev.args.args]
    assert params == ["eval_engine", "rows"], (
        f"evaluate must bind the engine explicitly, got params {params} "
        "(the 2026-10-11 invalid_model_binding bug was exactly a closure "
        "capture of the stage-scope original-weights engine)")


def test_no_stage_scope_external_engine():
    src = TYPED.read_text(encoding="utf-8")
    stage = _stage_node(src)
    # only inspect statements DIRECTLY in the stage body (not inside nested defs)
    for node in stage.body:
        if isinstance(node, ast.Assign):
            names = [t.id for t in node.targets if isinstance(t, ast.Name)]
            assert "engine" not in names, (
                "stage-scope `engine =` assignment found: an external engine at "
                "stage scope is what the buggy evaluate captured; the trained "
                "engine must live only inside run_seed")


def test_run_seed_calls_pass_local_engine():
    src = TYPED.read_text(encoding="utf-8")
    stage = _stage_node(src)
    run_seed = _nested(stage, "run_seed")
    calls = [n for n in ast.walk(run_seed)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
             and n.func.id == "evaluate"]
    assert calls, "no evaluate() calls found inside run_seed"
    for c in calls:
        assert len(c.args) == 2, "every evaluate call must pass (engine, rows)"
        first = c.args[0]
        assert isinstance(first, ast.Name) and first.id == "engine", (
            "evaluate must be called with run_seed's local trained engine")


def test_eval_weights_identity_registered():
    """Dev/tests must run on a model whose identity is recorded at eval time."""
    src = TYPED.read_text(encoding="utf-8")
    run_seed = _nested(_stage_node(src), "run_seed")
    body_src = ast.unparse(run_seed)
    assert "eval_weights_sha256" in body_src, (
        "run_seed must register the sha256 of the state_dict actually evaluated "
        "(not just the saved checkpoint file)")
    assert "eval_model" in body_src


class _SentinelEngine:
    """Minimal engine stand-in: constant predictions, call-counted."""

    def __init__(self, pred: int):
        self.pred = pred
        self.calls = 0

    def predict(self, payload):
        self.calls += 1
        return [{"index": self.pred} for _ in payload]


_ROWS = [
    {"id": "t|01", "state": json.dumps({"v": 42.6}), "question": "Is v at least 41.0?",
     "target_option": "true"},
    {"id": "t|02", "state": json.dumps({"v": 38.1}), "question": "Is v at least 41.0?",
     "target_option": "false"},
]


def _extract_helpers():
    """Extract the REAL evaluate/to_payload/golds source from the stage."""
    src = TYPED.read_text(encoding="utf-8")
    stage = _stage_node(src)
    parts = [ast.unparse(_nested(stage, name))
             for name in ("to_payload", "golds", "evaluate")]
    return "\n\n".join(parts)


def test_supplied_engine_is_the_one_evaluated():
    """Two prediction-distinct engines: per-case preds must follow the engine
    passed as an argument. Fails on the pre-fix single-arg closure version."""
    code = _extract_helpers()
    ns = {"_json": json}
    exec(compile(code, "<e31_helpers>", "exec"), ns)
    evaluate = ns["evaluate"]
    wrong, right = _SentinelEngine(0), _SentinelEngine(1)
    _, per_wrong = evaluate(wrong, _ROWS)
    _, per_right = evaluate(right, _ROWS)
    assert all(p["pred"] == 0 for p in per_wrong) and wrong.calls > 0
    assert all(p["pred"] == 1 for p in per_right) and right.calls > 0
    # golds/passthrough sanity: per-case rows carry id/gold/ok
    assert [p["id"] for p in per_right] == [r["id"] for r in _ROWS]
    assert [p["gold"] for p in per_right] == [1, 0]


def test_flat_runner_carries_fixed_binding():
    flat = FLAT.read_text(encoding="utf-8")
    assert "def evaluate(eval_engine, rows):" in flat
    assert "evaluate(engine, dev_rows)" in flat
    assert "evaluate(engine, rows)" in flat
    # the pre-fix stage-scope engine right before evaluate must not exist
    idx = flat.find("def evaluate(eval_engine, rows):")
    before = flat[max(0, idx - 400):idx]
    assert "engine = FastEngine(" not in before, (
        "stage-scope FastEngine engine found next to evaluate in the flat runner")
