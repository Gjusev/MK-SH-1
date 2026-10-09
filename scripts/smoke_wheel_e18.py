"""Wheel smoke test for the E18 bundle (run with the ISOLATED venv python).

Verifies the packaged wheel installs and supports a CPU end-to-end
train -> checkpoint -> reload -> evaluate cycle for both E18 supervision
modes, with no model downloads (tiny encoder + char tokenizer) and no GPU.

Usage: <isolated-venv-python> scripts/smoke_wheel_e18.py <wheel-or-bundle-dir>
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn


class TinyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=16)
        self.embedding = nn.Embedding(48, 16)

    def forward(self, input_ids, attention_mask):
        return SimpleNamespace(last_hidden_state=self.embedding(input_ids))


class CharTokenizer:
    mask_token = "<m>"
    cls_token_id, sep_token_id, pad_token_id, mask_token_id, unk_id = 1, 2, 0, 5, 4
    vocab = {c: 6 + i for i, c in enumerate("abcdefghijklmnopqrstuvwxyz :,.-#0123456789")}

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [self.vocab.get(ch, self.unk_id) for ch in text.lower()]}


def case(cid: str, qt: str, gold: int, k: int = 4) -> dict:
    return {
        "id": f"{cid}:{qt}",
        "group_id": cid,
        "source": "smoke", "language": "en", "domain": "smoke_family",
        "state": f"{cid} state with enough words to encode something",
        "question": f"{qt} question about {cid}?",
        "options": [{"id": f"o{j}", "description": f"option {j} of {cid}"} for j in range(k)],
        "target_kind": "single", "target_option": f"o{gold}", "split": "train",
    }


def main() -> int:
    import mokka_decisions
    from mokka_decisions.contracts import DecisionCase

    pkg = Path(mokka_decisions.__file__).resolve()
    print("mokka_decisions from:", pkg)
    assert "site-packages" in str(pkg), f"not the installed wheel copy: {pkg}"

    import transformers

    orig = transformers.AutoModel.from_pretrained
    transformers.AutoModel.from_pretrained = lambda *a, **k: TinyEncoder()
    from mokka_decisions.arch_julia import evaluate_julia_arch, make_seeded_julia_arch, train_julia_arch
    from safetensors.torch import load_file

    tok = CharTokenizer()
    train = [DecisionCase.from_dict(case(f"t{i:02d}", qt, (i + j) % 4))
             for i in range(8) for j, qt in enumerate(("score", "choice", "noul"))]
    dev = [DecisionCase.from_dict(case(f"d{i:02d}", qt, (i + j) % 4))
           for i in range(6) for j, qt in enumerate(("score", "choice", "noul"))]
    qtypes = {c.id: c.id.rsplit(":", 1)[1] for c in train + dev}
    score_train = {c.id for c in train if qtypes[c.id] == "score"}
    soft = {c.id: [0.1, 0.6, 0.2, 0.1] for c in train if qtypes[c.id] == "score"}
    dev_refs = {c.id: [0.25, 0.25, 0.25, 0.25] for c in dev}

    with tempfile.TemporaryDirectory() as tmp:
        for name, kwargs in (("ord", dict(ordinal_case_ids=score_train)),
                             ("probs", dict(soft_targets=soft))):
            model = make_seeded_julia_arch("unused", seed=42)
            out = Path(tmp) / name
            report = train_julia_arch(
                model, tok, train, dev, qtypes, output_dir=str(out),
                epochs=2, microbatch=2, grad_accumulation=2, seed=42, device="cpu",
                max_length=96, head_length=48,
                reference_probabilities=dev_refs, **kwargs,
            )
            assert report["best_dev_accuracy"] is not None
            assert (out / "checkpoint_best.safetensors").exists()
            for entry in report["history"]:
                d = entry["dev"]
                assert d["n"] == len(dev) and d["mean_kl"] is not None and d["mean_mae"] is not None
                assert set(d["by_type"]) == {"score", "choice", "noul"}
            model.load_state_dict(load_file(str(out / "checkpoint_best.safetensors")))
            full = evaluate_julia_arch(model, tok, dev, qtypes, device="cpu",
                                       reference_probabilities=dev_refs, collect_rows=True)
            assert len(full["rows"]) == len(dev)
            assert all("kl" in r and "nll" in r and "brier" in r for r in full["rows"])
            assert all(("mae" in r) == (r["type"] == "score") for r in full["rows"])
            print(name, "best_dev=", report["best_dev_accuracy"],
                  "steps=", report["final_step"], "rows=", len(full["rows"]))
    transformers.AutoModel.from_pretrained = orig
    print("SMOKE PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
