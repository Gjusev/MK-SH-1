"""E19 paired analysis: rule-variation + counterfactual data contrast.

Applies the decision rule FROZEN in docs/internal/E19_PROTOCOL_2026-10-09.md
(+ pre-launch 2-seed alignment) to the sealed-test rows produced by
scripts/eval_e19_sealed.py. Nothing here selects checkpoints or seeds.

Primary contrast (i):   cand - ctrl  (counterfactual-variation benefit)
Context contrast (ii):  ctrl - orig  (benefit of adding ordinary data)
Reference contrast (iii): cand/ctrl - Julia (advantage or absence)

Reported separately, never merged: grouped-by-case bootstrap CI (case
sampling, conditional on these trainings) and between-seed dispersion.
"""

from __future__ import annotations

import json
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/e19-sealed-eval"
KERNEL_RUN = ROOT / "runs/kaggle-typed-e19"
SEEDS = (42, 43)
BOOTSTRAP_DRAWS = 50_000
BOOTSTRAP_SEED = 20261009


def load(name: str) -> list[dict]:
    rows = [json.loads(l) for l in (RUN / f"{name}_percase.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(rows) == 750, f"{name}: {len(rows)} rows"
    return rows


def summarize(rows: list[dict]) -> dict:
    by_type = defaultdict(lambda: [0, 0])
    by_fam = defaultdict(lambda: [0, 0])
    for r in rows:
        by_type[r["type"]][0] += r["correct"]
        by_type[r["type"]][1] += 1
        by_fam[r["family"]][0] += r["correct"]
        by_fam[r["family"]][1] += 1
    mae = [r["mae"] for r in rows if "mae" in r]
    return {
        "accuracy": sum(r["correct"] for r in rows) / len(rows),
        "mean_nll": sum(r["nll"] for r in rows) / len(rows),
        "mean_brier": sum(r["brier"] for r in rows) / len(rows),
        "mean_mae": sum(mae) / len(mae) if mae else None,
        "by_type": {t: {"accuracy": c / n, "n": n} for t, (c, n) in sorted(by_type.items())},
        "by_family": {f: {"accuracy": c / n, "n": n} for f, (c, n) in sorted(by_fam.items())},
    }


def pair_consistency(rows: list[dict]) -> dict:
    """Both-correct rate on counterfactual/irrelevant structures."""
    by_pair = defaultdict(dict)
    for r in rows:
        if r.get("pair") and r["variant"] in ("base", "ruleflip", "evidenceflip"):
            by_pair[r["pair"]][r["variant"]] = r["correct"]
    flips = {"action(ruleflip)": [], "risk(evidenceflip)": []}
    for pid, members in by_pair.items():
        if "base" in members and "ruleflip" in members:
            flips["action(ruleflip)"].append(int(members["base"] and members["ruleflip"]))
        if "base" in members and "evidenceflip" in members:
            flips["risk(evidenceflip)"].append(int(members["base"] and members["evidenceflip"]))
    irr = [r["correct"] for r in rows if r.get("variant") == "irrelevant"]
    return {k: sum(v) / len(v) for k, v in flips.items() if v} | {
        "irrelevant_correct": sum(irr) / len(irr) if irr else None}


def grouped_bootstrap(delta_by_group: dict, draws=BOOTSTRAP_DRAWS, seed=BOOTSTRAP_SEED) -> list[float]:
    rng = random.Random(seed)
    groups = list(delta_by_group.values())
    out = []
    for _ in range(draws):
        sample = [groups[rng.randrange(len(groups))] for _ in range(len(groups))]
        out.append(sum(sample) / len(sample))
    out.sort()
    return [out[int(0.025 * draws)], out[int(0.975 * draws)]]


def mean_delta_by_group(rows_a: dict, rows_b: dict) -> dict:
    per = defaultdict(list)
    for rid, rb in rows_b.items():
        per[rb["group_id"]].append(rb["correct"] - rows_a[rid]["correct"])
    return {g: sum(v) / len(v) for g, v in per.items()}


def main() -> int:
    summary = json.loads((KERNEL_RUN / "e19_summary.json").read_text(encoding="utf-8"))
    arms = {f"e19_{r}_s{s}": load(f"e19_{r}_s{s}") for s in SEEDS for r in ("orig", "ctrl", "cand")}
    julia = load("julia")
    stats = {a: summarize(rs) for a, rs in arms.items()} | {"julia": summarize(julia)}
    for arm in arms:
        if set(summary["seed_hash_equal"].values()) != {True}:
            raise SystemExit("init hash mismatch in e19_summary: analysis invalid")

    def deltas(recipes):
        out = {}
        for s in SEEDS:
            a, b = f"e19_{recipes[0]}_s{s}", f"e19_{recipes[1]}_s{s}"
            out[s] = {k: stats[b][k] - stats[a][k] for k in ("accuracy", "mean_nll", "mean_brier", "mean_mae")}
            out[s]["regression_choice_noul"] = (
                (stats[b]["by_type"]["choice"]["accuracy"] - stats[a]["by_type"]["choice"]["accuracy"])
                * stats[a]["by_type"]["choice"]["n"] / 170 +
                (stats[b]["by_type"]["noul"]["accuracy"] - stats[a]["by_type"]["noul"]["accuracy"])
                * stats[a]["by_type"]["noul"]["n"] / 170)  # test mix: 2 choice + 1 noul per case
        return out

    cand_ctrl, ctrl_orig = deltas(("ctrl", "cand")), deltas(("orig", "ctrl"))
    mean_cc = {k: statistics.mean(cand_ctrl[s][k] for s in SEEDS)
               for k in ("accuracy", "regression_choice_noul", "mean_nll", "mean_brier", "mean_mae")}

    # grouped bootstrap on the pooled cand-ctrl per-case delta
    pooled = {}
    for s in SEEDS:
        pooled.update(mean_delta_by_group({r["id"]: r for r in arms[f"e19_ctrl_s{s}"]},
                                          {r["id"]: r for r in arms[f"e19_cand_s{s}"]}))
    ci_cc = grouped_bootstrap(pooled)
    pooled_cj = {}
    for s in SEEDS:
        pooled_cj.update(mean_delta_by_group({r["id"]: r for r in julia},
                                             {r["id"]: r for r in arms[f"e19_cand_s{s}"]}))
    ci_cj = grouped_bootstrap(pooled_cj)

    wins = sum(1 for s in SEEDS if cand_ctrl[s]["accuracy"] >= 0)
    losses = sum(1 for s in SEEDS if cand_ctrl[s]["accuracy"] < 0)
    if (mean_cc["accuracy"] >= 0.02 and wins == len(SEEDS) and mean_cc["regression_choice_noul"] >= -0.01):
        verdict = "improvement"
    elif losses == len(SEEDS) or mean_cc["accuracy"] <= -0.02:
        verdict = "negative"
    else:
        verdict = "inconclusive"

    report = {
        "protocol": "docs/internal/E19_PROTOCOL_2026-10-09.md (+2-seed alignment, frozen pre-results)",
        "scope": "sealed test opened once after finalist fixation; bootstrap CIs are case-sampling "
                 "conditional on these trainings; between-seed dispersion reported separately",
        "per_arm": stats,
        "pair_consistency": {a: pair_consistency(rs) for a, rs in arms.items()} | {"julia": pair_consistency(julia)},
        "cand_minus_ctrl_by_seed": {str(s): cand_ctrl[s] for s in SEEDS},
        "ctrl_minus_orig_by_seed": {str(s): ctrl_orig[s] for s in SEEDS},
        "cand_minus_ctrl_mean": mean_cc,
        "bootstrap_by_case_ci95": {"cand_minus_ctrl": ci_cc, "cand_minus_julia": ci_cj},
        "wins": f"{wins}/{len(SEEDS)}",
        "verdict": verdict,
    }
    out = RUN / "e19_report.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"{'arm':<16} {'acc':>7} {'choice':>7} {'noul':>7} {'score':>7} {'NLL':>7} {'MAE':>7} | actflip riskflip irr")
    for a in list(arms) + ["julia"]:
        s = stats[a]
        pc = report["pair_consistency"][a]
        print(f"{a:<16} {s['accuracy']:>7.4f} {s['by_type']['choice']['accuracy']:>7.3f} "
              f"{s['by_type']['noul']['accuracy']:>7.3f} {s['by_type']['score']['accuracy']:>7.3f} "
              f"{s['mean_nll']:>7.3f} {s['mean_mae']:>7.3f} | "
              f"{pc.get('action(ruleflip)') or 0:.3f} {pc.get('risk(evidenceflip)') or 0:.3f} {pc['irrelevant_correct']:.3f}")
    for fam in sorted(stats["e19_cand_s42"]["by_family"]):
        row = " ".join(f"{stats[a]['by_family'][fam]['accuracy']:.3f}" for a in ("julia", "e19_orig_s42", "e19_ctrl_s42", "e19_cand_s42"))
        print(f"  {fam:<28} julia/orig/ctrl/cand s42: {row}")
    print(f"\ncand-ctrl dAcc by seed: " + " ".join(f"s{s}={cand_ctrl[s]['accuracy']:+.4f}" for s in SEEDS)
          + f"  mean {mean_cc['accuracy']:+.4f}  CI-by-case [{ci_cc[0]:+.4f}, {ci_cc[1]:+.4f}]")
    print(f"ctrl-orig dAcc by seed: " + " ".join(f"s{s}={ctrl_orig[s]['accuracy']:+.4f}" for s in SEEDS))
    print(f"cand-julia mean dAcc CI-by-case [{ci_cj[0]:+.4f}, {ci_cj[1]:+.4f}]")
    print(f"VERDICT (frozen rule): {verdict.upper()}  -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
