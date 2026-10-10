"""Evaluate E23 arms + references on the SEALED E23 numeric test (opened once).

References (frozen): e20_ctrl_s43 and e20_cand_s43 (best-dev prior checkpoints)
and official Julia. Same conventions as prior evaluators (CPU float32, T=1,
1024/256, batch 16; Julia via upstream engine, per-question states).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

DATA = ROOT / "data/e23-numeric"
JULIA_CKPT = ROOT / "runs/julia-research/upstream-source"
NEW_ARMS = [f"e23_{r}_s{s}" for s in (42, 43, 44) for r in ("mix", "cur")]
REF_ARMS = ["e20_ctrl_s43", "e20_cand_s43"]
CKPT_DIRS = {a: ROOT / "runs/kaggle-typed-e23" / a for a in NEW_ARMS}
CKPT_DIRS.update({a: ROOT / "runs/kaggle-typed-e20" / a for a in REF_ARMS})


def load_test_rows() -> list[dict]:
    rows = [json.loads(l) for l in (DATA / "test_rows.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    for r in rows:
        r["gold"] = next(i for i, o in enumerate(r["options"]) if o["id"] == r["target_option"])
    return rows


def evaluate_ours(arm, rows, ckpt):
    import torch
    from mokka_decisions.arch_julia import JuliaArchScorer, collate_julia, row_metrics
    from mokka_decisions.contracts import DecisionCase, Option
    from safetensors.torch import load_file
    from transformers import AutoTokenizer

    model = JuliaArchScorer("jhu-clsp/mmBERT-small")
    model.load_state_dict(load_file(str(ckpt)))
    model.to("cpu").eval()
    tok = AutoTokenizer.from_pretrained("jhu-clsp/mmBERT-small")
    qtypes = {r["id"]: r["question_type"] for r in rows}
    out = []
    with torch.inference_mode():
        for s in range(0, len(rows), 16):
            chunk = rows[s:s + 16]
            cases = [DecisionCase(id=r["id"], group_id=r["group_id"], source="e23", language="en",
                                  domain=r["domain"], state=r["state"], question=r["question"],
                                  options=tuple(Option(id=o["id"], description=o["description"]) for o in r["options"]),
                                  target_kind="single", target_option=r["target_option"])
                     for r in chunk]
            batch = collate_julia(tok, cases, qtypes, 1024, 256, "cpu")
            logits = model(**{k: batch[k] for k in ("input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype")}).float()
            probs = logits.softmax(-1)
            for r, pv_t in zip(chunk, probs):
                k = len(r["options"])
                pv = pv_t[:k].tolist()
                pred = max(range(k), key=pv.__getitem__)
                m = row_metrics(pv, r["gold"], r["question_type"])
                out.append({"id": r["id"], "group_id": r["group_id"], "domain": r["domain"],
                            "type": r["question_type"], "variant": r.get("variant"), "pair": r.get("pair"),
                            "operator": r.get("operator"), "feature": r.get("feature"),
                            "rule_in": r.get("rule_in"),
                            "gold_idx": r["gold"], "pred_idx": pred, "probabilities": pv,
                            "correct": int(pred == r["gold"]), **m})
    return out


def evaluate_julia(rows):
    ckpt = str(JULIA_CKPT)
    if ckpt not in sys.path:
        sys.path.insert(0, ckpt)
    from julia.inference import TransformerEngine
    from julia.typed import predict_typed

    engine = TransformerEngine(ckpt, device="cpu")
    from mokka_decisions.arch_julia import row_metrics
    out = []
    for r in rows:
        q = {"instructions": r["question"], "type": "noul",
             "criteria": {"false": r["options"][0]["description"], "true": r["options"][1]["description"]}}
        ans = predict_typed(engine, r["state"], {"q": q})["answers"]["q"]
        pv = [ans["probabilities"]["false"], ans["probabilities"]["true"]]
        pred = max(range(2), key=pv.__getitem__)
        m = row_metrics(pv, r["gold"], r["question_type"])
        out.append({"id": r["id"], "group_id": r["group_id"], "domain": r["domain"],
                    "type": r["question_type"], "variant": r.get("variant"), "pair": r.get("pair"),
                    "operator": r.get("operator"), "feature": r.get("feature"),
                    "rule_in": r.get("rule_in"),
                    "gold_idx": r["gold"], "pred_idx": pred, "probabilities": pv,
                    "correct": int(pred == r["gold"]), **m})
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="runs/e23-sealed-eval")
    args = parser.parse_args()
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    rows = load_test_rows()
    print(f"sealed test: {len(rows)} questions")
    for arm in NEW_ARMS + REF_ARMS:
        ckpt = CKPT_DIRS[arm] / "checkpoint_best.safetensors"
        if not ckpt.exists():
            raise SystemExit(f"missing checkpoint: {ckpt}")
        if (out / f"{arm}_percase.jsonl").exists():
            print(f"{arm}: cached")
            continue
        per = evaluate_ours(arm, rows, ckpt)
        (out / f"{arm}_percase.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in per), encoding="utf-8")
        print(f"{arm}: acc={sum(r['correct'] for r in per) / len(per):.4f}")
    if not (out / "julia_percase.jsonl").exists():
        per = evaluate_julia(rows)
        (out / "julia_percase.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in per), encoding="utf-8")
    julia = [json.loads(l) for l in (out / "julia_percase.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    print(f"julia: acc={sum(r['correct'] for r in julia) / len(julia):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
