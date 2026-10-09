"""E18 paired analysis: supervision contrast within JuliaArch-small.

Reads runs/kaggle-typed-e18/ (e18_summary.json + e18_{ord,probs}_s{42,43,44}/
dev_rows_best.jsonl). Decision rule is FROZEN in docs/internal/
E18_PROTOCOL_2026-10-09.md — this script only applies it.

Outputs e18_report.json + stdout table:
- per-seed paired deltas (candidate probs − control ord) for acc/NLL/Brier/KL/MAE
- grouped-by-case bootstrap CI (50k draws) on the pooled mean accuracy delta:
  case-sampling uncertainty CONDITIONAL on these trainings
- between-training dispersion (sd/range across seeds), reported separately
- kernel<->local cross-check of every arm's dev accuracy
"""

from __future__ import annotations

import json
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs" / "kaggle-typed-e18"
SEEDS = (42, 43, 44)
BOOTSTRAP_DRAWS = 50_000
BOOTSTRAP_SEED = 20261009


def load_rows(arm: str) -> list[dict]:
    path = RUN / arm / "dev_rows_best.jsonl"
    if not path.exists():
        raise SystemExit(f"missing {path}")
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    by_type = defaultdict(lambda: [0, 0])
    for r in rows:
        t = by_type[r["type"]]
        t[0] += r["correct"]
        t[1] += 1
    mae = [r["mae"] for r in rows if "mae" in r]
    return {
        "n": n,
        "accuracy": sum(r["correct"] for r in rows) / n,
        "mean_nll": sum(r["nll"] for r in rows) / n,
        "mean_brier": sum(r["brier"] for r in rows) / n,
        "mean_kl": sum(r["kl"] for r in rows if "kl" in r) / n,
        "mean_mae": sum(mae) / len(mae) if mae else None,
        "by_type": {t: {"accuracy": c / m, "n": m} for t, (c, m) in sorted(by_type.items())},
    }


def grouped_bootstrap(delta_by_group: dict[str, float], draws=BOOTSTRAP_DRAWS,
                      seed=BOOTSTRAP_SEED) -> list[float]:
    rng = random.Random(seed)
    groups = list(delta_by_group.values())
    out = []
    for _ in range(draws):
        sample = [groups[rng.randrange(len(groups))] for _ in range(len(groups))]
        out.append(sum(sample) / len(sample))
    out.sort()
    return [out[int(0.025 * draws)], out[int(0.975 * draws)]]


def main() -> int:
    summary = json.loads((RUN / "e18_summary.json").read_text(encoding="utf-8"))
    arms = summary["arms"]
    if set(arms) != {f"e18_{r}_s{s}" for r in ("ord", "probs") for s in SEEDS}:
        raise SystemExit(f"unexpected arm set: {sorted(arms)}")
    for seed, ok in summary["pair_hash_equal"].items():
        if ok is not True:
            raise SystemExit(f"pair init hash mismatch at {seed}: analysis invalid")

    rows = {arm: load_rows(arm) for arm in arms}
    stats = {arm: summarize(r) for arm, r in rows.items()}
    # kernel <-> local cross-check on every arm
    crosscheck = {}
    for arm, s in stats.items():
        claimed = arms[arm]["selected_checkpoint_dev"]["accuracy"]
        crosscheck[arm] = {"kernel": claimed, "local": s["accuracy"],
                           "match": abs(claimed - s["accuracy"]) < 1e-9}
        if not crosscheck[arm]["match"]:
            raise SystemExit(f"cross-check failed for {arm}: {claimed} vs {s['accuracy']}")

    pairs = {}
    metric_keys = ("accuracy", "mean_nll", "mean_brier", "mean_kl", "mean_mae")
    for seed in SEEDS:
        a, b = f"e18_ord_s{seed}", f"e18_probs_s{seed}"
        ids_a = [r["id"] for r in rows[a]]
        ids_b = [r["id"] for r in rows[b]]
        if ids_a != ids_b:
            raise SystemExit(f"row id mismatch between {a} and {b}")
        pairs[seed] = {k: stats[b][k] - stats[a][k] for k in metric_keys}

    mean = {k: statistics.mean(pairs[s][k] for s in SEEDS) for k in metric_keys}
    disp = {}
    for k in metric_keys:
        vals = [pairs[s][k] for s in SEEDS]
        if any(v is None for v in vals):
            disp[k] = {"sd": None, "range": None}
        else:
            disp[k] = {"sd": statistics.stdev(vals), "range": [min(vals), max(vals)]}

    # per-original-case mean delta across seeds -> grouped bootstrap (acc)
    per_group = defaultdict(list)
    for seed in SEEDS:
        a = {r["id"]: r for r in rows[f"e18_ord_s{seed}"]}
        for r in rows[f"e18_probs_s{seed}"]:
            per_group[r["group_id"]].append(r["correct"] - a[r["id"]]["correct"])
    delta_by_group = {g: sum(v) / len(v) for g, v in per_group.items()}
    ci = grouped_bootstrap(delta_by_group)

    wins = sum(1 for s in SEEDS if pairs[s]["accuracy"] >= 0)  # protocol: candidate >= control
    losses = sum(1 for s in SEEDS if pairs[s]["accuracy"] < 0)
    kl_wins = sum(1 for s in SEEDS if pairs[s]["mean_kl"] < 0)
    nll_worse = sum(1 for s in SEEDS if pairs[s]["mean_nll"] > 0)
    brier_worse = sum(1 for s in SEEDS if pairs[s]["mean_brier"] > 0)
    mae_worse = sum(1 for s in SEEDS if pairs[s]["mean_mae"] is not None and pairs[s]["mean_mae"] > 0)
    if (wins >= 2 and kl_wins >= 2 and mean["mean_kl"] < 0
            and not (nll_worse >= 2 or brier_worse >= 2 or mae_worse >= 2)):
        verdict = "improvement"
    elif losses >= 2 or kl_wins == 0:
        verdict = "negative"
    else:
        verdict = "inconclusive"

    report = {
        "protocol": "docs/internal/E18_PROTOCOL_2026-10-09.md (decision rule frozen pre-launch)",
        "scope": "supervision contrast within JuliaArch-small; not an architecture effect; "
                 "bootstrap CI covers case sampling conditional on these trainings, "
                 "NOT training-to-training variability (see dispersion)",
        "per_arm": {arm: {"best_dev_accuracy": arms[arm]["best_dev_accuracy"],
                          "best_epoch": arms[arm]["best_epoch"],
                          "final_step": arms[arm]["final_step"],
                          "metrics": {k: v for k, v in stats[arm].items() if k != "by_type"},
                          "by_type": stats[arm]["by_type"]} for arm in sorted(arms)},
        "pairs_delta_probs_minus_ord": {str(s): pairs[s] for s in SEEDS},
        "mean_delta": mean,
        "between_seed_dispersion": disp,
        "grouped_bootstrap_accuracy_delta": {"ci95": ci, "groups": len(delta_by_group),
                                             "draws": BOOTSTRAP_DRAWS, "seed": BOOTSTRAP_SEED},
        "crosscheck_kernel_local": crosscheck,
        "wins": {"accuracy": f"{wins}/3", "kl_improves": f"{kl_wins}/3"},
        "verdict": verdict,
    }
    out = RUN / "e18_report.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"{'arm':<18} {'acc':>7} {'nll':>7} {'brier':>7} {'kl':>7} {'mae':>7} epoch")
    for arm in sorted(arms):
        m = stats[arm]
        print(f"{arm:<18} {m['accuracy']:>7.4f} {m['mean_nll']:>7.4f} {m['mean_brier']:>7.4f} "
              f"{m['mean_kl']:>7.4f} {m['mean_mae']:>7.4f} {arms[arm]['best_epoch']}")
    print(f"\nper-seed dAcc (probs-ord): " + " ".join(f"s{s}={pairs[s]['accuracy']:+.4f}" for s in SEEDS))
    print(f"mean dAcc {mean['accuracy']:+.5f}  sd {disp['accuracy']['sd']:.5f}  "
          f"bootstrap-by-case CI95 [{ci[0]:+.5f}, {ci[1]:+.5f}]")
    print(f"mean dKL {mean['mean_kl']:+.5f} (kl improves {kl_wins}/3)  "
          f"mean dNLL {mean['mean_nll']:+.5f}  mean dMAE {mean['mean_mae']:+.5f}")
    print(f"VERDICT (frozen rule): {verdict.upper()}  -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
