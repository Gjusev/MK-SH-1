"""Evaluate E19 checkpoints + Julia on the E20 diagnostic battery (exploratory).

Measures rule composition directly: accuracy on balanced noul probes and
consistency on paired interventions (factflip/ruleflip must flip when gold
flips; irrelevant/paraphrase must keep the answer).
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

ROWS = ROOT / "data/e20-diag/rows.jsonl"
ARMS = [f"e19_{r}_s{s}" for s in (42, 43) for r in ("orig", "ctrl", "cand")]


def load_rows():
    rows = [json.loads(l) for l in ROWS.read_text(encoding="utf-8").splitlines() if l.strip()]
    return rows


def evaluate_ours(arm, rows, device="cpu"):
    import torch
    from mokka_decisions.arch_julia import JuliaArchScorer, collate_julia
    from mokka_decisions.contracts import DecisionCase, Option
    from safetensors.torch import load_file
    from transformers import AutoTokenizer

    model = JuliaArchScorer("jhu-clsp/mmBERT-small")
    model.load_state_dict(load_file(str(ROOT / "runs/kaggle-typed-e19" / arm / "checkpoint_best.safetensors")))
    model.to(device).eval()
    tok = AutoTokenizer.from_pretrained("jhu-clsp/mmBERT-small")
    qtypes = {r["id"]: "noul" for r in rows}
    out = []
    with torch.inference_mode():
        for s in range(0, len(rows), 16):
            chunk = rows[s:s + 16]
            cases = [DecisionCase(id=r["id"], group_id=r["group_id"], source="e20diag", language="en",
                                  domain=r["domain"], state=r["state"], question=r["question"],
                                  options=tuple(Option(id=o["id"], description=o["description"]) for o in r["options"]),
                                  target_kind="single", target_option=r["target_option"])
                     for r in chunk]
            batch = collate_julia(tok, cases, qtypes, 1024, 256, device)
            logits = model(**{k: batch[k] for k in
                              ("input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype")}).float()
            probs = logits.softmax(-1)
            for r, pv_t in zip(chunk, probs):
                pv = pv_t[:2].tolist()
                out.append({"id": r["id"], "domain": r["domain"], "variant": r["variant"],
                            "pair": r["pair"], "gold": int(r["target_option"] == "true"),
                            "pred": int(pv[1] > pv[0]), "p_true": round(pv[1], 4),
                            "correct": int((pv[1] > pv[0]) == (r["target_option"] == "true"))})
    return out


def evaluate_julia(rows):
    ckpt = str(ROOT / "runs/julia-research/upstream-source")
    if ckpt not in sys.path:
        sys.path.insert(0, ckpt)
    from julia.inference import TransformerEngine
    from julia.typed import predict_typed

    engine = TransformerEngine(ckpt, device="cpu")
    out = []
    for r in rows:
        q = {"instructions": r["question"], "type": "noul",
             "criteria": {"false": r["options"][0]["description"], "true": r["options"][1]["description"]}}
        ans = predict_typed(engine, r["state"], {"q": q})["answers"]["q"]
        p_true = ans["probabilities"]["true"]
        out.append({"id": r["id"], "domain": r["domain"], "variant": r["variant"],
                    "pair": r["pair"], "gold": int(r["target_option"] == "true"),
                    "pred": int(p_true > 0.5), "p_true": round(p_true, 4),
                    "correct": int((p_true > 0.5) == (r["target_option"] == "true"))})
    return out


def summarize(per):
    by = defaultdict(lambda: [0, 0])
    for r in per:
        by[r["variant"]][0] += r["correct"]
        by[r["variant"]][1] += 1
    acc = sum(r["correct"] for r in per) / len(per)
    true_rate = sum(r["pred"] for r in per) / len(per)
    # paired consistencies
    indexed = {r["id"]: r for r in per}
    cons = defaultdict(lambda: [0, 0])
    for r in per:
        if r["variant"] != "base":
            continue
        cid = r["id"].rsplit(":", 1)[0]
        for variant, must_flip in (("factflip", True), ("ruleflip", True), ("irrelevant", False), ("paraphrase", False)):
            other = indexed.get(f"{cid}:{variant}")
            if not other:
                continue
            key = f"{variant}_consistency"
            cons[key][1] += 1
            if must_flip:
                ok = (r["correct"] and other["correct"]) or (r["pred"] != other["pred"] if r["gold"] != other["gold"] else r["pred"] == other["pred"])
                # strict: both members correct
                ok = r["correct"] and other["correct"]
            else:
                ok = r["correct"] and other["correct"] and r["pred"] == other["pred"]
            cons[key][0] += int(ok)
    return {"accuracy": acc, "true_rate": true_rate,
            "by_variant": {v: {"accuracy": c / n, "n": n} for v, (c, n) in sorted(by.items())},
            "consistency": {k: c / n for k, (c, n) in sorted(cons.items())}}


def main() -> int:
    rows = load_rows()
    out_dir = ROOT / "runs/e20-diag"
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for arm in ARMS:
        per = evaluate_ours(arm, rows)
        (out_dir / f"{arm}_percase.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in per), encoding="utf-8")
        results[arm] = summarize(per)
        print(f"{arm}: acc={results[arm]['accuracy']:.3f} true_rate={results[arm]['true_rate']:.2f}")
    per = evaluate_julia(rows)
    (out_dir / "julia_percase.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in per), encoding="utf-8")
    results["julia"] = summarize(per)
    print(f"julia: acc={results['julia']['accuracy']:.3f} true_rate={results['julia']['true_rate']:.2f}")
    (out_dir / "diag_report.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("\nvariant accuracies + consistencies:")
    for arm, s in results.items():
        vs = " ".join(f"{v[:4]}={d['accuracy']:.2f}" for v, d in s["by_variant"].items())
        cs = " ".join(f"{k[:12]}={v:.2f}" for k, v in s["consistency"].items())
        print(f"  {arm:<15} {vs} | {cs}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
