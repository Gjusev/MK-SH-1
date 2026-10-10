"""Prevent response frequency being called accuracy and loader runtime drift."""
import pytest

from scripts.eval_e22_format import response_baselines
from scripts.efficiency_e22 import inference_runtime


def test_all_true_predictions_do_not_make_a_perfect_constant_baseline():
    gold = {"a": 0, "b": 1}
    result = response_baselines(gold, {k: {"pred": 1} for k in gold})
    assert result["majority_baseline"] == 0.5
    assert result["true_rate"] == 1.0
    assert result["prediction_majority_fraction"] == 1.0


def test_baseline_uses_gold_distribution_regardless_of_predictions():
    gold = {"a": 0, "b": 0, "c": 0, "d": 1}
    for answer in (0, 1):
        result = response_baselines(gold, {k: {"pred": answer} for k in gold})
        assert result["majority_baseline"] == 0.75
        assert result["constant_true_accuracy"] == 0.25
        assert result["constant_false_accuracy"] == 0.75


@pytest.mark.parametrize("gold,pred", [({}, {}), ({"a": 0}, {"b": {"pred": 0}})])
def test_baseline_rejects_empty_or_unpaired_inputs(gold, pred):
    with pytest.raises(ValueError):
        response_baselines(gold, pred)


def test_inference_context_resets_loader_threads_and_disables_graphs():
    torch = pytest.importorskip("torch")
    original_threads = torch.get_num_threads()
    original_precision = torch.get_float32_matmul_precision()
    try:
        torch.set_num_threads(4)  # Julia loader default.
        layer = torch.nn.Linear(2, 1).eval()
        with torch.enable_grad():
            assert layer(torch.ones(1, 2)).requires_grad
            with inference_runtime(8) as runtime:
                result = layer(torch.ones(1, 2))
                assert runtime["effective_threads"] == 8
                assert runtime["inference_mode"] and not runtime["grad_enabled"]
                assert not result.requires_grad
                assert result.grad_fn is None
            assert torch.is_grad_enabled()
    finally:
        torch.set_num_threads(original_threads)
        torch.set_float32_matmul_precision(original_precision)
