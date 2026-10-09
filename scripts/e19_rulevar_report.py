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

import argparse
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


def validate_aligned_rows(arms: dict[str, list[dict]]) -> None:
    """Reject incomplete, duplicated, or differently labeled paired inputs."""
    reference = None
    fields = ("group_id", "family", "type", "gold_idx", "variant", "pair")
    for name, rows in arms.items():
        indexed = {r["id"]: tuple(r.get(k) for k in fields) for r in rows}
        if not rows or len(indexed) != len(rows):
            raise ValueError(f"{name}: empty rows or duplicate question IDs")
        if reference is None:
            reference = indexed
        elif indexed != reference:
            raise ValueError(f"{name}: question IDs, groups, labels or metadata differ")


def pooled_case_deltas(pairs: list[tuple[list[dict], list[dict]]]) -> dict:
    """Average seed-paired differences per case without overwriting a seed."""
    if not pairs:
        raise ValueError("At least one paired training is required")
    validate_aligned_rows({f"{i}:{j}": rows for i, pair in enumerate(pairs)
                           for j, rows in enumerate(pair)})
    per_seed = [mean_delta_by_group({r["id"]: r for r in a},
                                    {r["id"]: r for r in b}) for a, b in pairs]
    return {g: statistics.mean(d[g] for d in per_seed) for g in sorted(per_seed[0])}


def choice_noul_delta(a: dict, b: dict) -> float:
    kinds = ("choice", "noul")
    for kind in kinds:
        if a["by_type"][kind]["n"] != b["by_type"][kind]["n"]:
            raise ValueError(f"Different {kind} denominators in paired arms")
    count = sum(a["by_type"][t]["n"] for t in kinds)
    if not count:
        raise ValueError("No choice/noul questions")
    return sum((b["by_type"][t]["accuracy"] - a["by_type"][t]["accuracy"])
               * a["by_type"][t]["n"] for t in kinds) / count


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=RUN / "e19_report_v2.json",
                        help="New output file; refuses to overwrite a historical report")
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"Refusing to overwrite {args.out}; supply a new --out path")
    summary = json.loads((KERNEL_RUN / "e19_summary.json").read_text(encoding="utf-8"))
    arms = {f"e19_{r}_s{s}": load(f"e19_{r}_s{s}") for s in SEEDS for r in ("orig", "ctrl", "cand")}
    julia = load("julia")
    validate_aligned_rows(arms | {"julia": julia})
    expected_seeds = {f"s{s}" for s in SEEDS}
    if set(summary["seed_hash_equal"]) != expected_seeds:
        raise SystemExit("Missing or unexpected paired initialization records")
    stats = {a: summarize(rs) for a, rs in arms.items()} | {"julia": summarize(julia)}
    for arm in arms:
        if set(summary["seed_hash_equal"].values()) != {True}:
            raise SystemExit("init hash mismatch in e19_summary: analysis invalid")

    def deltas(recipes):
        out = {}
        for s in SEEDS:
            a, b = f"e19_{recipes[0]}_s{s}", f"e19_{recipes[1]}_s{s}"
            out[s] = {k: stats[b][k] - stats[a][k] for k in ("accuracy", "mean_nll", "mean_brier", "mean_mae")}
            out[s]["regression_choice_noul"] = choice_noul_delta(stats[a], stats[b])
        return out

    cand_ctrl, ctrl_orig = deltas(("ctrl", "cand")), deltas(("orig", "ctrl"))
    mean_cc = {k: statistics.mean(cand_ctrl[s][k] for s in SEEDS)
               for k in ("accuracy", "regression_choice_noul", "mean_nll", "mean_brier", "mean_mae")}

    # grouped bootstrap on the pooled cand-ctrl per-case delta
    pooled = pooled_case_deltas([(arms[f"e19_ctrl_s{s}"], arms[f"e19_cand_s{s}"])
                                 for s in SEEDS])
    ci_cc = grouped_bootstrap(pooled)
    pooled_cj = pooled_case_deltas([(julia, arms[f"e19_cand_s{s}"]) for s in SEEDS])
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
        "report_version": 2,
        "corrections": ["Average paired seeds within each case before bootstrap; v1 overwrote earlier seeds",
                        "Use the actual choice+noul count (450), not 170",
                        "irrelevant_correct is variant-only accuracy, not paired invariance"],
        "protocol": "docs/internal/E19_PROTOCOL_2026-10-09.md (+2-seed alignment, frozen pre-results)",
        "scope": "sealed test opened once after finalist fixation; bootstrap CIs are case-sampling "
                 "conditional on these trainings; between-seed dispersion reported separately",
        "per_arm": stats,
        "pair_consistency": {a: pair_consistency(rs) for a, rs in arms.items()} | {"julia": pair_consistency(julia)},
        "cand_minus_ctrl_by_seed": {str(s): cand_ctrl[s] for s in SEEDS},
        "ctrl_minus_orig_by_seed": {str(s): ctrl_orig[s] for s in SEEDS},
        "cand_minus_ctrl_mean": mean_cc,
        "cand_minus_julia_mean": statistics.mean(pooled_cj.values()),
        "between_seed_accuracy_delta_sd": {
            "cand_minus_ctrl": statistics.stdev(cand_ctrl[s]["accuracy"] for s in SEEDS),
            "cand_minus_julia": statistics.stdev(stats[f"e19_cand_s{s}"]["accuracy"]
                                                 - stats["julia"]["accuracy"] for s in SEEDS)},
        "bootstrap_by_case_ci95": {"cand_minus_ctrl": ci_cc, "cand_minus_julia": ci_cj},
        "wins": f"{wins}/{len(SEEDS)}",
        "verdict": verdict,
    }
    out = args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)

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
