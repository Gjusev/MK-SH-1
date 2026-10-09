"""Evaluate E20 checkpoints (per seed) and Julia on the E21 diagnostic v2.

Reports accuracy by SKILL, by family, by seed, plus both-members-correct on
pairs. Exploratory: never a sealed confirmation.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

ARMS = [f"e20_{r}_s{s}" for s in (42, 43, 44) for r in ("ctrl", "cand")]


def skill_of(row):
    if "|numcmp_" in row["id"]:
        return "numcmp"
    if "|cat_" in row["id"]:
        return "catcond"
    if "|and4_" in row["id"]:
        return "and4"
    if "|or4_" in row["id"]:
        return "or4"
    return "pairs"


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
    out = []
    with torch.inference_mode():
        for s in range(0, len(rows), 16):
            chunk = rows[s:s + 16]
            cases = [DecisionCase(id=r["id"], group_id=r["group_id"], source="e20diag2", language="en",
                                  domain=r["domain"], state=r["state"], question=r["question"],
                                  options=tuple(Option(id=o["id"], description=o["description"]) for o in r["options"]),
                                  target_kind="single", target_option=r["target_option"]) for r in chunk]
            batch = collate_julia(tok, cases, qtypes, 1024, 256, "cpu")
            logits = model(**{k: batch[k] for k in ("input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype")}).float()
            probs = logits.softmax(-1)
            for r, pv_t in zip(chunk, probs):
                pv = pv_t[:2].tolist()
                out.append({"id": r["id"], "domain": r["domain"], "variant": r.get("variant"),
                            "skill": skill_of(r), "pair": r.get("pair"),
                            "gold": int(r["target_option"] == "true"), "pred": int(pv[1] > pv[0]),
                            "p_true": round(float(pv[1]), 4),
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
        out.append({"id": r["id"], "domain": r["domain"], "variant": r.get("variant"),
                    "skill": skill_of(r), "pair": r.get("pair"),
                    "gold": int(r["target_option"] == "true"), "pred": int(p_true > 0.5),
                    "p_true": round(float(p_true), 4),
                    "correct": int((p_true > 0.5) == (r["target_option"] == "true"))})
    return out


def summarize(per):
    by_skill = defaultdict(lambda: [0, 0])
    by_var = defaultdict(lambda: [0, 0])
    for r in per:
        by_skill[r["skill"]][0] += r["correct"]
        by_skill[r["skill"]][1] += 1
        by_var[r["variant"]][0] += r["correct"]
        by_var[r["variant"]][1] += 1
    base = {r["id"]: r for r in per if r["variant"] == "base"}
    pairs = defaultdict(lambda: [0, 0])
    for r in per:
        if r["variant"] in ("factflip", "ruleflip") and r["pair"]:
            b = base.get(r["id"])
            if b:
                k = f"{r['variant']}_both"
                pairs[k][1] += 1
                pairs[k][0] += int(r["correct"] and b["correct"])
        if r["variant"] in ("paraphrase", "irrelevant") and r["pair"]:
            b = base.get(r["id"])
            if b:
                k = f"{r['variant']}_agree"
                pairs[k][1] += 1
                pairs[k][0] += int(r["pred"] == b["pred"])
    return {"accuracy": sum(r["correct"] for r in per) / len(per),
            "true_rate": sum(r["pred"] for r in per) / len(per),
            "by_skill": {k: c / n for k, (c, n) in sorted(by_skill.items())},
            "by_variant": {k: c / n for k, (c, n) in sorted(by_var.items())},
            "pairs": {k: c / n for k, (c, n) in sorted(pairs.items())}}


def main() -> int:
    rows = [json.loads(l) for l in (ROOT / "data/e20-diag2/rows.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    out_dir = ROOT / "runs/e20-diag2"
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for arm in ARMS:
        per = evaluate_ours(arm, rows)
        (out_dir / f"{arm}_percase.jsonl").write_text("".join(json.dumps(r) + "\n" for r in per), encoding="utf-8")
        results[arm] = summarize(per)
    per = evaluate_julia(rows)
    (out_dir / "julia_percase.jsonl").write_text("".join(json.dumps(r) + "\n" for r in per), encoding="utf-8")
    results["julia"] = summarize(per)
    (out_dir / "diag2_report.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"{'model':<15} {'acc':>6} {'true%':>6} numcmp catcond  and4   or4  pairs | factflip ruleflip para irr")
    for a, s in results.items():
        bs = s["by_skill"]
        p = s["pairs"]
        print(f"{a:<15} {s['accuracy']:>6.3f} {s['true_rate']:>6.2f} {bs['numcmp']:.3f}  {bs['catcond']:.3f}  "
              f"{bs['and4']:.3f}  {bs['or4']:.3f}  {bs['pairs']:.3f} | "
              f"{p.get('factflip_both', 0):.3f}    {p.get('ruleflip_both', 0):.3f}   "
              f"{p.get('paraphrase_agree', 0):.3f} {p.get('irrelevant_agree', 0):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
