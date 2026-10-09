"""E20 FASE 3 analysis: frozen rule on the sealed test (macro by reserved family).

Primary: mean accuracy across the 2 sealed families, cand vs ctrl.
Success requires all frozen regression limits, including the known-family
dev and E19 diagnostic checks. Missing checks cannot establish success.
Negative: ctrl >= cand in >=2/3 pairs or mean <= -2pp. Case-bootstrap
intervals are conditional on the observed seeds; their dispersion is separate.
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
RUN = ROOT / "runs/e20-sealed-eval"
KERNEL_RUN = ROOT / "runs/kaggle-typed-e20"
SEEDS = (42, 43, 44)
FAMILIES = ("loan_underwriting", "chemical_batch_release")
BOOTSTRAP_DRAWS = 50_000
BOOTSTRAP_SEED = 20261011
TYPES = ("choice", "noul", "score")


def validate_aligned_rows(arms):
    """Reject duplicate IDs or mismatched paired labels/metadata."""
    reference = None
    fields = ("group_id", "family", "type", "gold_idx", "variant", "pair")
    for name, rows in arms.items():
        indexed = {r["id"]: tuple(r.get(k) for k in fields) for r in rows}
        if not rows or len(indexed) != len(rows):
            raise ValueError(f"{name}: empty rows or duplicate question IDs")
        if reference is not None and indexed != reference:
            raise ValueError(f"{name}: paired IDs, labels or metadata differ")
        reference = indexed


def frozen_decision(deltas, type_deltas, known_deltas):
    """Apply mean-by-type guards, never the best seed/type difference."""
    if set(deltas) != set(SEEDS) or set(type_deltas) != set(TYPES):
        raise ValueError("Missing seeds or question types")
    mean_delta = statistics.mean(deltas.values())
    eps = 1e-12
    wins = sum(d >= -eps for d in deltas.values())
    control_nondecreases = sum(d <= eps for d in deltas.values())
    known_names = ("dev_new", "e19_diagnostic")
    missing = [name for name in known_names if known_deltas.get(name) is None]
    checks = {
        "primary_at_least_5pp": mean_delta >= 0.05 - eps,
        "candidate_nondecreasing_in_at_least_two_pairs": wins >= 2,
        "type_regressions_within_2pp": {
            t: type_deltas[t] >= -0.02 - eps for t in TYPES},
        "known_regressions_within_2pp": {
            name: None if known_deltas.get(name) is None
            else known_deltas[name] >= -0.02 - eps for name in known_names},
    }
    success = (checks["primary_at_least_5pp"]
               and checks["candidate_nondecreasing_in_at_least_two_pairs"]
               and all(checks["type_regressions_within_2pp"].values())
               and not missing
               and all(checks["known_regressions_within_2pp"].values()))
    if success:
        verdict = "improvement"
    elif control_nondecreases >= 2 or mean_delta <= -0.02 + eps:
        verdict = "negative"
    else:
        verdict = "inconclusive"
    return {"verdict": verdict, "checks": checks, "missing_checks": missing,
            "type_mean_deltas": type_deltas, "known_mean_deltas": known_deltas,
            "candidate_nondecreasing_pairs": wins,
            "control_nondecreasing_pairs": control_nondecreases}


def known_dev_deltas():
    data = ROOT / "data/e20-contrast/dev_rows.jsonl"
    expected = {json.loads(line)["id"] for line in data.read_text(encoding="utf-8").splitlines() if line.strip()}
    arms = {}
    for seed in SEEDS:
        for role in ("ctrl", "cand"):
            name = f"e20_{role}_s{seed}"
            path = KERNEL_RUN / name / "dev_rows_best.jsonl"
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
            arms[name] = [r for r in rows if r["id"] in expected]
            if {r["id"] for r in arms[name]} != expected:
                raise ValueError(f"{name}: missing known-family dev rows")
    validate_aligned_rows(arms)
    return {seed: statistics.mean(r["correct"] for r in arms[f"e20_cand_s{seed}"])
            - statistics.mean(r["correct"] for r in arms[f"e20_ctrl_s{seed}"])
            for seed in SEEDS}


def load(name):
    rows = [json.loads(l) for l in (RUN / f"{name}_percase.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(rows) == 750
    return rows


def macro_acc(rows):
    by = defaultdict(lambda: [0, 0])
    for r in rows:
        by[r["family"]][0] += r["correct"]
        by[r["family"]][1] += 1
    if set(by) != set(FAMILIES):
        raise ValueError("Reserved-family coverage differs from the frozen protocol")
    return statistics.mean(c / n for c, n in by.values())


def summarize(rows):
    by_type = defaultdict(lambda: [0, 0])
    by_fam = defaultdict(lambda: [0, 0])
    for r in rows:
        by_type[r["type"]][0] += r["correct"]
        by_type[r["type"]][1] += 1
        by_fam[r["family"]][0] += r["correct"]
        by_fam[r["family"]][1] += 1
    mae = [r["mae"] for r in rows if "mae" in r]
    return {"macro_reserved": macro_acc(rows),
            "accuracy": sum(r["correct"] for r in rows) / len(rows),
            "mean_nll": sum(r["nll"] for r in rows) / len(rows),
            "mean_brier": sum(r["brier"] for r in rows) / len(rows),
            "mean_mae": sum(mae) / len(mae) if mae else None,
            "by_type": {t: {"accuracy": c / n, "n": n} for t, (c, n) in sorted(by_type.items())},
            "by_family": {f: {"accuracy": c / n, "n": n} for f, (c, n) in sorted(by_fam.items())}}


def pair_consistency(rows):
    by_pair = defaultdict(dict)
    for r in rows:
        if r.get("pair") and r["variant"] in ("base", "factflip", "evidenceflip"):
            by_pair[r["pair"]][r["variant"]] = r["correct"]
    noul = [int(m.get("base", 0) and m.get("factflip", 1)) for m in by_pair.values()
            if "base" in m and "factflip" in m]
    risk = [int(m.get("base", 0) and m.get("evidenceflip", 1)) for m in by_pair.values()
            if "base" in m and "evidenceflip" in m]
    return {"noul_flip_both": sum(noul) / len(noul) if noul else None,
            "risk_flip_both": sum(risk) / len(risk) if risk else None}


def pooled_case_deltas(pairs):
    validate_aligned_rows({f"{i}_{side}": rows for i, pair in enumerate(pairs)
                           for side, rows in enumerate(pair)})
    per_seed = []
    for a_rows, b_rows in pairs:
        a = {r["id"]: r for r in a_rows}
        per = defaultdict(list)
        for r in b_rows:
            per[r["group_id"]].append(r["correct"] - a[r["id"]]["correct"])
        per_seed.append({g: sum(v) / len(v) for g, v in per.items()})
    return {g: statistics.mean(d[g] for d in per_seed) for g in sorted(per_seed[0])}


def grouped_bootstrap(delta_by_group, draws=BOOTSTRAP_DRAWS, seed=BOOTSTRAP_SEED):
    rng = random.Random(seed)
    groups = list(delta_by_group.values())
    out = []
    for _ in range(draws):
        sample = [groups[rng.randrange(len(groups))] for _ in range(len(groups))]
        out.append(sum(sample) / len(sample))
    out.sort()
    return [out[int(0.025 * draws)], out[int(0.975 * draws)]]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=RUN / "e20_report_v2.json")
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"Refusing to overwrite an existing report: {args.out}")
    summary = json.loads((KERNEL_RUN / "e20_summary.json").read_text(encoding="utf-8"))
    if summary["seed_hash_equal"] != {f"s{s}": True for s in SEEDS}:
        raise SystemExit("init hash mismatch: analysis invalid")
    arms = {f"e20_{r}_s{s}": load(f"e20_{r}_s{s}") for s in SEEDS for r in ("ctrl", "cand")}
    julia = load("julia")
    validate_aligned_rows(arms | {"julia": julia})
    stats = {a: summarize(rs) for a, rs in arms.items()} | {"julia": summarize(julia)}

    pairs = {s: {"macro": stats[f"e20_cand_s{s}"]["macro_reserved"] - stats[f"e20_ctrl_s{s}"]["macro_reserved"],
                 "acc": stats[f"e20_cand_s{s}"]["accuracy"] - stats[f"e20_ctrl_s{s}"]["accuracy"]}
             for s in SEEDS}
    mean_macro = statistics.mean(pairs[s]["macro"] for s in SEEDS)
    wins = sum(1 for s in SEEDS if pairs[s]["macro"] >= 0)
    type_deltas = {t: statistics.mean(
        stats[f"e20_cand_s{s}"]["by_type"][t]["accuracy"]
        - stats[f"e20_ctrl_s{s}"]["by_type"][t]["accuracy"] for s in SEEDS)
        for t in TYPES}
    dev_deltas = known_dev_deltas()
    # No E20-checkpoint evaluation of E19 diagnostic families was collected.
    # Preserve that missing check instead of substituting the noul probe set.
    decision = frozen_decision({s: pairs[s]["macro"] for s in SEEDS}, type_deltas,
                              {"dev_new": statistics.mean(dev_deltas.values()),
                               "e19_diagnostic": None})
    verdict = decision["verdict"]

    pooled = pooled_case_deltas([(arms[f"e20_ctrl_s{s}"], arms[f"e20_cand_s{s}"]) for s in SEEDS])
    ci = grouped_bootstrap(pooled)
    pooled_cj = pooled_case_deltas([(julia, arms[f"e20_cand_s{s}"]) for s in SEEDS])
    ci_cj = grouped_bootstrap(pooled_cj)

    report = {
        "analysis_version": 2,
        "protocol": "docs/internal/E20_PROTOCOL_2026-10-09.md (frozen pre-results)",
        "per_arm": stats,
        "pair_consistency": {a: pair_consistency(rs) for a, rs in arms.items()} | {"julia": pair_consistency(julia)},
        "cand_minus_ctrl_by_seed": {str(s): pairs[s] for s in SEEDS},
        "cand_minus_ctrl_mean_macro": mean_macro,
        "frozen_decision": decision,
        "known_dev_delta_by_seed": dev_deltas,
        "between_seed_dispersion": {
            "n_pairs": len(SEEDS),
            "sample_sd_macro_delta": statistics.stdev(pairs[s]["macro"] for s in SEEDS),
            "min_macro_delta": min(pairs[s]["macro"] for s in SEEDS),
            "max_macro_delta": max(pairs[s]["macro"] for s in SEEDS)},
        "interval_scope": "Pooled case bootstrap, conditional on these observed seeds and families; not seed or unseen-family uncertainty. Historical resampling method unchanged.",
        "bootstrap_by_case_ci95": {"cand_minus_ctrl": ci, "cand_minus_julia": ci_cj,
                                   "cand_minus_julia_mean": statistics.mean(pooled_cj.values())},
        "wins": f"{wins}/{len(SEEDS)}", "verdict": verdict,
    }
    out = args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    for a in list(arms) + ["julia"]:
        s = stats[a]
        print(f"{a:<15} macro={s['macro_reserved']:.4f} acc={s['accuracy']:.4f} "
              f"choice={s['by_type']['choice']['accuracy']:.3f} noul={s['by_type']['noul']['accuracy']:.3f} "
              f"score={s['by_type']['score']['accuracy']:.3f} nll={s['mean_nll']:.3f}")
    for fam in FAMILIES:
        row = " ".join(f"{stats[a]['by_family'][fam]['accuracy']:.3f}" for a in ("julia", "e20_ctrl_s42", "e20_cand_s42"))
        print(f"  {fam:<26} julia/ctrl/cand s42: {row}")
    print(f"\ncand-ctrl macro by seed: " + " ".join(f"s{s}={pairs[s]['macro']:+.4f}" for s in SEEDS)
          + f"  mean {mean_macro:+.4f}  CI-by-case [{ci[0]:+.4f}, {ci[1]:+.4f}]")
    print(f"cand-julia mean {report['bootstrap_by_case_ci95']['cand_minus_julia_mean']:+.4f} "
          f"CI [{ci_cj[0]:+.4f}, {ci_cj[1]:+.4f}]")
    print(f"VERDICT (frozen rule): {verdict.upper()}  -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
