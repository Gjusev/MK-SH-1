"""Evaluate the three fixed checkpoints on the E22 format battery (exploratory).

Fixed models (dev criterion, unchanged from E21): julia, e20_ctrl_s43, e20_cand_s43.
Reports accuracy by FORMAT and by OPERATOR, majority baseline, both-members-correct
on inversion pairs (per format), with bootstrap over PROBLEMS (variants grouped).
"""

from __future__ import annotations

import json
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

MODELS = ("julia", "mokka_ctrl_s43", "mokka_cand_s43")
FORMATS = ("json_opt", "json_q", "prose_opt", "prose_q")
BOOTSTRAP = 20_000


def evaluate_ours(arm, rows):
    import torch
    from mokka_decisions.arch_julia import JuliaArchScorer, collate_julia
    from mokka_decisions.contracts import DecisionCase, Option
    from safetensors.torch import load_file
    from transformers import AutoTokenizer

    model = JuliaArchScorer("jhu-clsp/mmBERT-small")
    model.load_state_dict(load_file(str(ROOT / "runs/kaggle-typed-e20" / arm / "checkpoint_best.safetensors")))
    model.to("cpu").eval()
    tok = AutoTokenizer.from_pretrained("jhu-clsp/mmBERT-small")
    qtypes = {r["id"]: "noul" for r in rows}
    out = {}
    with torch.inference_mode():
        for s in range(0, len(rows), 16):
            chunk = rows[s:s + 16]
            cases = [DecisionCase(id=r["id"], group_id=r["group_id"], source="e", language="en",
                                  domain=r["domain"], state=r["state"], question=r["question"],
                                  options=tuple(Option(id=o["id"], description=o["description"]) for o in r["options"]),
                                  target_kind="single", target_option=r["target_option"]) for r in chunk]
            batch = collate_julia(tok, cases, qtypes, 1024, 256, "cpu")
            logits = model(**{k: batch[k] for k in ("input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype")}).float()
            probs = logits.softmax(-1)
            for r, pv_t in zip(chunk, probs):
                pv = pv_t[:2].tolist()
                out[r["id"]] = {"pred": int(pv[1] > pv[0]), "p_true": float(pv[1])}
    return out


def evaluate_julia(rows):
    ckpt = str(ROOT / "runs/julia-research/upstream-source")
    if ckpt not in sys.path:
        sys.path.insert(0, ckpt)
    from julia.inference import TransformerEngine
    from julia.typed import predict_typed

    engine = TransformerEngine(ckpt, device="cpu")
    out = {}
    for r in rows:
        q = {"instructions": r["question"], "type": "noul",
             "criteria": {"false": r["options"][0]["description"], "true": r["options"][1]["description"]}}
        ans = predict_typed(engine, r["state"], {"q": q})["answers"]["q"]
        out[r["id"]] = {"pred": int(ans["probabilities"]["true"] > 0.5),
                        "p_true": ans["probabilities"]["true"]}
    return out


def main() -> int:
    rows = [json.loads(l) for l in (ROOT / "data/e22-format/rows.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    gold = {r["id"]: int(r["target_option"] == "true") for r in rows}
    meta = {r["id"]: r for r in rows}
    out_dir = ROOT / "runs/e22-format"
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for m in MODELS:
        per = evaluate_julia(rows) if m == "julia" else evaluate_ours(m.replace("mokka_", "e20_"), rows)
        (out_dir / f"{m}_percase.jsonl").write_text(
            "".join(json.dumps({"id": i, **v, "gold": gold[i], "correct": int(v["pred"] == gold[i])}) + "\n"
                    for i, v in per.items()), encoding="utf-8")
        by_fmt = {f: [i for i in per if meta[i]["format"] == f] for f in FORMATS}
        by_op = defaultdict(list)
        for i in per:
            by_op[meta[i]["operator"]].append(i)
        # majority baseline per model (constant-answer accuracy)
        tr = sum(v["pred"] for v in per.values())
        majority = max(tr, len(per) - tr) / len(per)
        # pair both-correct per format
        pairs = defaultdict(lambda: [0, 0])
        by_pair = defaultdict(dict)
        for i, v in per.items():
            if meta[i]["pair"]:
                by_pair[meta[i]["pair"]][(meta[i]["variant"], meta[i]["format"])] = int(v["pred"] == gold[i])
        for pid, members in by_pair.items():
            for f in FORMATS:
                hi, lo = members.get(("pair_hi", f)), members.get(("pair_lo", f))
                if hi is not None and lo is not None:
                    pairs[f][1] += 1
                    pairs[f][0] += int(hi and lo)
        results[m] = {
            "accuracy": sum(int(v["pred"] == gold[i]) for i, v in per.items()) / len(per),
            "majority_baseline": majority,
            "true_rate": tr / len(per),
            "by_format": {f: sum(int(per[i]["pred"] == gold[i]) for i in ids) / len(ids) for f, ids in by_fmt.items()},
            "by_operator": {op: sum(int(per[i]["pred"] == gold[i]) for i in ids) / len(ids) for op, ids in by_op.items()},
            "pair_both_correct_by_format": {f: (c / n if n else None) for f, (c, n) in pairs.items()},
        }
    n_groups = len({r["group_id"] for r in rows})
    results["_grouping"] = {"unit": "problem (group_id; variants of one problem kept together)",
                            "n_groups": n_groups}
    (out_dir / "format_report.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"{'model':<16} {'acc':>6} {'maj.base':>8} " + " ".join(f"{f:>9}" for f in FORMATS) + " | pairs by format")
    for m in MODELS:
        r = results[m]
        print(f"{m:<16} {r['accuracy']:>6.3f} {r['majority_baseline']:>8.3f} "
              + " ".join(f"{r['by_format'][f]:>9.3f}" for f in FORMATS)
              + " | " + " ".join(f"{(r['pair_both_correct_by_format'][f] or 0):.2f}" for f in FORMATS))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
