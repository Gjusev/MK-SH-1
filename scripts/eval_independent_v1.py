"""Evaluate A0 checkpoints on the FROZEN independent set (indep-v1).

Protocol (docs/typed-independent-eval-protocol.md): state string as stored
(adapter round-trip), criteria order preserved, no none, argmax accuracy,
NLL/Brier vs one-hot gold (indep-v1 has no gold distributions — rule-certain
labels), MAE by expected value for score. T=1.0, max_length 512, batch 16,
CPU. Writes per-question rows for the paired analysis.

Usage:
  python scripts/eval_independent_v1.py --out runs/independent-v1 \
      --arm e15_probs_s44=runs/kaggle-typed-e15-seeds/e15_probs_s44/checkpoint_best.safetensors \
      --arm e15_control_s43=...
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def load_questions():
    rows = []
    for line in (ROOT / "data/independent-v1/cases.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        case = json.loads(line)
        for qid, q in case["questions"].items():
            criteria = q["criteria"]
            if criteria is None and q["type"] == "noul":
                criteria = {"false": "false", "true": "true"}
            if isinstance(criteria, list):
                criteria = {str(i): v for i, v in enumerate(criteria)}
            keys = ["false", "true"] if q["type"] == "noul" else list(criteria)
            rows.append({
                "family": case["family"], "case_id": case["id"], "qid": qid,
                "state": case["state"], "question": q["instructions"], "type": q["type"],
                "options": [criteria[k] for k in keys], "gold_idx": int(case["gold"][qid]),
            })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--arm", action="append", required=True, help="name=checkpoint.safetensors")
    args = parser.parse_args()
    import torch
    from mokka_decisions.contracts import DecisionCase, Option
    from mokka_decisions.model import OptionScorer, ScorerInference, load_checkpoint
    from transformers import AutoTokenizer

    torch.set_num_threads(8)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = load_questions()
    print(f"independent set: {len(rows)} questions")
    tokenizer = AutoTokenizer.from_pretrained("jhu-clsp/mmBERT-base")
    cases = [DecisionCase(
        id=f"{r['case_id']}:{r['qid']}", group_id=r["case_id"], source="indep-v1", language="en",
        domain=r["family"], state=r["state"], question=r["question"],
        options=tuple(Option(id=f"o{i}", description=d) for i, d in enumerate(r["options"])),
        target_kind="single", target_option=f"o{r['gold_idx']}",
    ) for r in rows]

    for spec in args.arm:
        name, ckpt = spec.split("=", 1)
        m = OptionScorer("jhu-clsp/mmBERT-base")
        load_checkpoint(m, ckpt)
        sc = ScorerInference(m, tokenizer, device="cpu", temperature=1.0, max_length=512, batch_decisions=16)
        per = []
        for s in range(0, len(cases), 16):
            for case, row, probs in zip(cases[s:s+16], rows[s:s+16], sc.score(cases[s:s+16])):
                gold = row["gold_idx"]
                pv = [probs[oid] for oid in case.option_ids]
                rec = {
                    "id": case.id, "family": row["family"], "case_id": row["case_id"], "qid": row["qid"],
                    "type": row["type"],
                    "correct": int(max(pv) == pv[gold] and pv.index(max(pv)) == gold),
                    "nll": -math.log(max(pv[gold], 1e-12)),
                    "brier": sum((p - (1.0 if i == gold else 0.0)) ** 2 for i, p in enumerate(pv)),
                }
                if row["type"] == "score":
                    rec["mae"] = abs(sum(i * p for i, p in enumerate(pv)) - gold)
                per.append(rec)
        dest = out / f"{name}_percase.jsonl"
        with dest.open("w", encoding="utf-8", newline="\n") as fh:
            for r in per:
                fh.write(json.dumps(r) + "\n")
        acc = sum(r["correct"] for r in per) / len(per)
        print(f"{name}: acc={acc:.4f} n={len(per)} -> {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
