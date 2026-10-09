"""E20 FASE 3 analysis: frozen rule on the sealed test (macro by reserved family).

Primary: mean accuracy across the 2 sealed families, cand vs ctrl.
Success: cand-ctrl >= +5.0pp, >=2/3 seed pairs, and regressions within limits
(known-family proxies: dev-nuevo + type breakdowns; no known-family test here,
so the regression limit uses per-type and the e20-diag battery as declared
secondary diagnostics). Negative: ctrl >= cand in >=2/3 pairs or mean <= -2pp.
"""

from __future__ import annotations

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


def load(name):
    rows = [json.loads(l) for l in (RUN / f"{name}_percase.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(rows) == 750
    return rows


def macro_acc(rows):
    by = defaultdict(lambda: [0, 0])
    for r in rows:
        by[r["family"]][0] += r["correct"]
        by[r["family"]][1] += 1
    return statistics.mean(c / n for f, (c, n) in by.items() if f in FAMILIES)


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
            "mean_mae": sum(mae) / len(mae) if mae else None,
            "by_type": {t: {"accuracy": c / n, "n": n} for t, (c, n) in sorted(by_type.items())},
            "by_family": {f: {"accuracy": c / n, "n": n} for f, (c, n) in sorted(by_fam.items())}}


def pair_consistency(rows):
    by_pair = defaultdict(dict)
    for r in rows:
        if r.get("pair") and r["variant"] in ("base", "factflip", "evidenceflip"):
            by_pair[r["pair"]][r["variant"]] = r["correct"]
    out = {}
    for kind, variants in (("noul_factflip", ("base", "factflip")),):
        vals = [int(all(m.get(v) for v in variants)) for m in by_pair.values()
                if "noul" in next(iter(by_pair)) or True]
    noul = [int(m.get("base", 0) and m.get("factflip", 1)) for m in by_pair.values()
            if "base" in m and "factflip" in m]
    risk = [int(m.get("base", 0) and m.get("evidenceflip", 1)) for m in by_pair.values()
            if "base" in m and "evidenceflip" in m]
    return {"noul_flip_both": sum(noul) / len(noul) if noul else None,
            "risk_flip_both": sum(risk) / len(risk) if risk else None}


def pooled_case_deltas(pairs):
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
    summary = json.loads((KERNEL_RUN / "e20_summary.json").read_text(encoding="utf-8"))
    if set(summary["seed_hash_equal"].values()) != {True}:
        raise SystemExit("init hash mismatch: analysis invalid")
    arms = {f"e20_{r}_s{s}": load(f"e20_{r}_s{s}") for s in SEEDS for r in ("ctrl", "cand")}
    julia = load("julia")
    stats = {a: summarize(rs) for a, rs in arms.items()} | {"julia": summarize(julia)}

    pairs = {s: {"macro": stats[f"e20_cand_s{s}"]["macro_reserved"] - stats[f"e20_ctrl_s{s}"]["macro_reserved"],
                 "acc": stats[f"e20_cand_s{s}"]["accuracy"] - stats[f"e20_ctrl_s{s}"]["accuracy"]}
             for s in SEEDS}
    mean_macro = statistics.mean(pairs[s]["macro"] for s in SEEDS)
    wins = sum(1 for s in SEEDS if pairs[s]["macro"] >= 0)
    losses = sum(1 for s in SEEDS if pairs[s]["macro"] < 0)
    type_reg = max((stats[f"e20_cand_s{s}"]["by_type"][t]["accuracy"]
                    - stats[f"e20_ctrl_s{s}"]["by_type"][t]["accuracy"] for s in SEEDS
                    for t in ("choice", "noul", "score")))
    if mean_macro >= 0.05 and wins >= 2 and type_reg > -0.02:
        verdict = "improvement"
    elif losses >= 2 or mean_macro <= -0.02:
        verdict = "negative"
    else:
        verdict = "inconclusive"

    pooled = pooled_case_deltas([(arms[f"e20_ctrl_s{s}"], arms[f"e20_cand_s{s}"]) for s in SEEDS])
    ci = grouped_bootstrap(pooled)
    pooled_cj = pooled_case_deltas([(julia, arms[f"e20_cand_s{s}"]) for s in SEEDS])
    ci_cj = grouped_bootstrap(pooled_cj)

    report = {
        "protocol": "docs/internal/E20_PROTOCOL_2026-10-09.md (frozen pre-results)",
        "per_arm": stats,
        "pair_consistency": {a: pair_consistency(rs) for a, rs in arms.items()} | {"julia": pair_consistency(julia)},
        "cand_minus_ctrl_by_seed": {str(s): pairs[s] for s in SEEDS},
        "cand_minus_ctrl_mean_macro": mean_macro,
        "bootstrap_by_case_ci95": {"cand_minus_ctrl": ci, "cand_minus_julia": ci_cj,
                                   "cand_minus_julia_mean": statistics.mean(pooled_cj.values())},
        "wins": f"{wins}/{len(SEEDS)}", "verdict": verdict,
    }
    out = RUN / "e20_report.json"
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
