"""Regressions for E17 metadata and initialization; no model downloads."""

import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from mokka_decisions.arch_julia import (
    collate_julia,
    make_seeded_julia_arch,
    row_metrics,
    validate_question_types,
)


def test_row_metrics_hand_computed():
    import math

    m = row_metrics([0.2, 0.5, 0.3], 1, "score", [0.1, 0.6, 0.3])
    assert m["nll"] == -math.log(0.5)
    assert m["brier"] == 0.2**2 + 0.5**2 + 0.3**2  # one-hot at gold=1
    assert abs(m["mae"] - 0.1) < 1e-12  # |0*.2+1*.5+2*.3 - 1|
    assert abs(m["kl"] - (0.1 * math.log(0.5) + 0.6 * math.log(1.2))) < 1e-9
    assert "mae" not in row_metrics([0.5, 0.5], 0, "choice")
    assert "kl" not in row_metrics([0.5, 0.5], 0, "noul", None)


class TinyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=8)
        self.embedding = nn.Embedding(64, 8)

    def forward(self, input_ids, attention_mask):
        return SimpleNamespace(last_hidden_state=self.embedding(input_ids))


def test_missing_dev_type_fails_before_serialization():
    dev = [SimpleNamespace(id="dev:score")]
    with pytest.raises(ValueError, match="Missing question type for dev:score"):
        collate_julia(None, dev, {"train:score": "score"}, 1024, 256, "cpu")


def test_explicit_types_and_invalid_type():
    cases = [SimpleNamespace(id="c"), SimpleNamespace(id="s"), SimpleNamespace(id="b")]
    validate_question_types(cases, {"c": "choice", "s": "score", "b": "noul"})
    with pytest.raises(ValueError, match="Unsupported question type"):
        validate_question_types(cases, {"c": "choice", "s": "invalid", "b": "noul"})


def test_seed_covers_head_initialization(monkeypatch):
    from transformers import AutoModel

    monkeypatch.setattr(AutoModel, "from_pretrained", lambda *args, **kwargs: TinyEncoder())
    a = make_seeded_julia_arch("unused", seed=42)
    torch.manual_seed(999)
    torch.rand(37)
    b = make_seeded_julia_arch("unused", seed=42)
    assert all(torch.equal(v, b.state_dict()[k]) for k, v in a.state_dict().items())
    c = make_seeded_julia_arch("unused", seed=43)
    assert not torch.equal(a.type_emb.weight, c.type_emb.weight)


def test_logits_match_upstream_with_shared_weights(monkeypatch):
    """Numerical parity, including masks/types, rather than shape-only parity."""
    source = Path(__file__).resolve().parents[1] / "runs/julia-research/upstream-source/julia/model.py"
    if not source.exists():
        pytest.skip("pinned upstream source is not available")
    spec = importlib.util.spec_from_file_location("upstream_julia_parity", source)
    upstream = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(upstream)
    from transformers import AutoModel

    monkeypatch.setattr(AutoModel, "from_pretrained", lambda *args, **kwargs: TinyEncoder())
    ours = make_seeded_julia_arch("unused", seed=42).eval()
    theirs = upstream.JuliaDecisionModel(copy.deepcopy(ours.encoder)).eval()
    missing, unexpected = theirs.load_state_dict(ours.state_dict(), strict=False)
    assert not unexpected
    assert all(k == "temperature" or k.startswith("act_head.") for k in missing)
    batch = dict(
        input_ids=torch.randint(0, 64, (3, 12)),
        attention_mask=torch.tensor([[1] * 12, [1] * 9 + [0] * 3, [1] * 10 + [0] * 2]),
        marker_pos=torch.tensor([[1, 4, 7], [1, 3, 0], [1, 4, 7]]),
        marker_mask=torch.tensor([[1, 1, 1], [1, 1, 0], [1, 1, 1]], dtype=torch.bool),
        qtype=torch.tensor([0, 1, 2]),
    )
    with torch.inference_mode():
        actual = ours(**batch)
        expected = theirs(**batch)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
