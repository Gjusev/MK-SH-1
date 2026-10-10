"""E23 frozen-rule analysis: acquisition vs curriculum, with references.

Verdict A (acquisition): a recipe's mean vs e20_ctrl_s43 on the sealed test,
+15pp minimum and inversion pairs >=0.40 in at least two seeds.
Verdict B (curriculum): cur - mix mean and per-seed pairs, +8pp minimum,
2/3 seeds, with retention checks. Historical E23 dev selection used an
all-true new subset; conclusions must be qualified by that design defect.
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
RUN = ROOT / "runs/e23-sealed-eval"
KERNEL_RUN = ROOT / "runs/kaggle-typed-e23"
SEEDS = (42, 43, 44)
BOOTSTRAP = 20_000


def validate_aligned(arms):
    fields = ("group_id", "domain", "type", "gold_idx", "pair", "variant", "rule_in", "operator", "feature")
    reference = None
    for name, rows in arms.items():
        indexed = {r["id"]: tuple(r.get(k) for k in fields) for r in rows}
        if not rows or len(indexed) != len(rows):
            raise ValueError(f"{name}: empty or duplicate question IDs")
        if reference is not None and indexed != reference:
            raise ValueError(f"{name}: paired IDs or metadata differ")
        reference = indexed


def pooled_cluster_sums(pairs):
    """Average ALL seeds within each cluster, retaining question denominators."""
    if not pairs:
        raise ValueError("No paired seeds")
    validate_aligned({f"{i}_{j}": rows for i, pair in enumerate(pairs) for j, rows in enumerate(pair)})
    per_seed = []
    for a_rows, b_rows in pairs:
        a = {r["id"]: r for r in a_rows}
        clusters = defaultdict(lambda: [0.0, 0])
        for r in b_rows:
            gid = r.get("pair") or r["group_id"]
            clusters[gid][0] += r["correct"] - a[r["id"]]["correct"]
            clusters[gid][1] += 1
        per_seed.append(clusters)
    return {g: (statistics.mean(d[g][0] for d in per_seed), per_seed[0][g][1])
            for g in sorted(per_seed[0])}


def cluster_point(clusters):
    return sum(v[0] for v in clusters.values()) / sum(v[1] for v in clusters.values())


def cluster_bootstrap(clusters, draws=BOOTSTRAP):
    rng = random.Random(20261014)
    groups = list(clusters.values())
    values = []
    for _ in range(draws):
        sampled = rng.choices(groups, k=len(groups))
        values.append(sum(v[0] for v in sampled) / sum(v[1] for v in sampled))
    values.sort()
    return [values[int(0.025 * draws)], values[int(0.975 * draws)]]


def acquisition_pass(gain, pair_rates):
    return gain >= 0.15 and sum(v is not None and v >= 0.40 for v in pair_rates) >= 2


def load(name):
    return [json.loads(l) for l in (RUN / f"{name}_percase.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]


def summarize(rows):
    by_op = defaultdict(lambda: [0, 0])
    by_feat = defaultdict(lambda: [0, 0])
    by_rule = defaultdict(lambda: [0, 0])
    for r in rows:
        by_op[r.get("operator")][0] += r["correct"]
        by_op[r.get("operator")][1] += 1
        by_feat[r.get("feature")][0] += r["correct"]
        by_feat[r.get("feature")][1] += 1
        by_rule[r.get("rule_in")][0] += r["correct"]
        by_rule[r.get("rule_in")][1] += 1
    by_pair = defaultdict(dict)
    for r in rows:
        if r.get("pair"):
            key = (r["pair"], r["rule_in"])
            if r["variant"] in by_pair[key]:
                raise ValueError(f"Duplicate pair member: {key}, {r['variant']}")
            by_pair[key][r["variant"]] = r["correct"]
    pair_results = defaultdict(list)
    for (pid, rule_in), members in by_pair.items():
        if set(members) != {"pair_hi", "pair_lo"}:
            raise ValueError(f"Incomplete inversion pair: {pid}, {rule_in}")
        pair_results[rule_in].append(int(members["pair_hi"] and members["pair_lo"]))
    all_pairs = [v for values in pair_results.values() for v in values]
    return {"accuracy": sum(r["correct"] for r in rows) / len(rows),
            "by_operator": {op: c / n for op, (c, n) in sorted(by_op.items()) if op},
            "by_feature": {f: c / n for f, (c, n) in sorted(by_feat.items()) if f},
            "by_rule_in": {r: c / n for r, (c, n) in sorted(by_rule.items()) if r},
            "pair_both": statistics.mean(all_pairs) if all_pairs else None,
            "pair_format_count": len(all_pairs),
            "pair_both_by_rule_in": {ri: {"both_correct": sum(v), "n": len(v),
                "rate": statistics.mean(v)} for ri, v in sorted(pair_results.items())}}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=RUN / "e23_report_v2.json")
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"Refusing to overwrite an existing report: {args.out}")
    summary = json.loads((KERNEL_RUN / "e23_summary.json").read_text(encoding="utf-8"))
    if summary["seed_hash_equal"] != {f"s{s}": True for s in SEEDS}:
        raise SystemExit("init hash mismatch: analysis invalid")
    if len(set(summary["multiset_hash"] if isinstance(summary["multiset_hash"], list) else [summary["multiset_hash"]])) != 1:
        raise SystemExit("multiset hash mismatch")

    arms = {f"e23_{r}_s{s}": load(f"e23_{r}_s{s}") for s in SEEDS for r in ("mix", "cur")}
    refs = {a: load(a) for a in ("e20_ctrl_s43", "e20_cand_s43", "julia")}
    validate_aligned(arms | refs)
    stats = {a: summarize(rs) for a, rs in arms.items()} | {a: summarize(rs) for a, rs in refs.items()}

    # Verdict B: curriculum (cur - mix per seed)
    cur_mix = {s: stats[f"e23_cur_s{s}"]["accuracy"] - stats[f"e23_mix_s{s}"]["accuracy"] for s in SEEDS}
    mean_cb = statistics.mean(cur_mix.values())
    wins = sum(1 for v in cur_mix.values() if v >= 0)

    # Verdict A: acquisition (best arm family vs prior best-dev reference)
    best_new = max(statistics.mean(stats[f"e23_{r}_s{s}"]["accuracy"] for s in SEEDS) for r in ("mix", "cur"))
    ref_acc = stats["e20_ctrl_s43"]["accuracy"]
    gain = best_new - ref_acc

    clusters = pooled_cluster_sums([(arms[f"e23_mix_s{s}"], arms[f"e23_cur_s{s}"]) for s in SEEDS])
    if abs(cluster_point(clusters) - mean_cb) > 1e-12:
        raise ValueError("Bootstrap estimand differs from question-level accuracy delta")
    ci_cb = cluster_bootstrap(clusters)
    # No E23-checkpoint evaluation on the 900-row E20 retention dev was collected.
    # Missing retention cannot be replaced by the new numeric test.
    retention_pass = None
    verdict_b = "improvement" if (mean_cb >= 0.08 and wins >= 2 and retention_pass is True) else ("negative" if all(v < 0 for v in cur_mix.values()) else "inconclusive")
    acquisition = {r: {"gain": statistics.mean(stats[f"e23_{r}_s{s}"]["accuracy"] for s in SEEDS) - ref_acc,
                       "pair_rates": [stats[f"e23_{r}_s{s}"]["pair_both"] for s in SEEDS]}
                   for r in ("mix", "cur")}
    for values in acquisition.values():
        values["thresholds_met"] = acquisition_pass(values["gain"], values["pair_rates"])
    verdict_a = "acquired" if any(v["thresholds_met"] for v in acquisition.values()) else "not_acquired"

    report = {"analysis_version": 2, "protocol": "docs/internal/E23_PROTOCOL_2026-10-10.md",
              "per_arm": stats,
              "cur_minus_mix_by_seed": {str(s): v for s, v in cur_mix.items()},
              "cur_minus_mix_mean": mean_cb, "ci_by_problem": ci_cb,
              "acquisition_gain_vs_e20_ctrl": gain,
              "acquisition_by_recipe": acquisition,
              "retention_check": "not_measured_on_frozen_E20_dev",
              "seed_delta_sample_sd": statistics.stdev(cur_mix.values()),
              "ci_scope": "Paired cluster bootstrap conditional on these seeds; question denominators retained. Does not include seed or new-family uncertainty.",
              "pair_denominator": "24 pair-format instances (12 problems x 2 rule placements); per-placement rates also reported",
              "qualification": "Executed design differs from written protocol; 160/160 new dev labels are true. Generalized acquisition not demonstrated by selected checkpoints; no claim of architectural inability or absence of in-domain learning.",
              "verdict_a_acquisition": verdict_a, "verdict_b_curriculum": verdict_b,
              "wins_b": f"{wins}/3"}
    out = args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"{'arm':<16} {'acc':>6} {'pairs':>6} | by operator")
    for a in list(arms) + list(refs):
        s = stats[a]
        ops = " ".join(f"{op[:4]}={v:.2f}" for op, v in s["by_operator"].items())
        print(f"{a:<16} {s['accuracy']:>6.3f} {s['pair_both'] or 0:>6.2f} | {ops}")
    print(f"\nVerdict A (acquisition): {verdict_a.upper()} - gain {gain:+.4f} vs e20_ctrl_s43 ({ref_acc:.3f})")
    print(f"Verdict B (curriculum): {verdict_b.upper()} - cur-mix by seed: " +
          " ".join(f"s{s}={v:+.4f}" for s, v in cur_mix.items()) + f"  mean {mean_cb:+.4f}  CI {ci_cb}")
    print(f"Features (new): " + " ".join(f"{f}={v:.3f}" for f, v in stats["e23_cur_s42"]["by_feature"].items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
