"""E25 frozen-criteria analysis: threshold-dependence verdicts, separated.

V1 fit (train >= 0.95 final) · V2 new values at KNOWN thresholds (acc >= 0.90
AND threshold-pairs >= 0.75 per seed) · V3 NEW thresholds interp/extrap
separately (same bars) · V4 threshold-only pairs C+D aggregate (>= 0.90).
Reports accuracy/NLL/pairs per seed AND per threshold, question-ignoring
reference caps (constant rules, fixed value>=60), and grouped bootstrap CIs
(resampling base cases, so pair variants stay together).
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
KERNEL_RUN = ROOT / "runs/kaggle-typed-e25"
DATA = ROOT / "data/e25-thrvar"
TRAJECTORY_MARKS = (0.75, 0.90, 1.0)
BOOTSTRAP = 20_000


def data_meta() -> dict:
    meta = {}
    for fname in ("train_thr_rows.jsonl", "dev_thr_rows.jsonl",
                  "test_known_thr_rows.jsonl", "test_new_thr_rows.jsonl"):
        for r in [json.loads(l) for l in (DATA / fname).read_text(encoding="utf-8").splitlines() if l.strip()]:
            meta[r["id"]] = r
    return meta


def load_eval(seed: str, name: str) -> list[dict]:
    return [json.loads(l) for l in (KERNEL_RUN / f"s{seed}" / f"eval_{name}.jsonl")
            .read_text(encoding="utf-8").splitlines() if l.strip()]


def summarize(rows: list[dict], meta: dict) -> dict:
    by_thr = defaultdict(lambda: [0, 0])
    pairs = defaultdict(dict)
    for r in rows:
        m = meta[r["id"]]
        by_thr[m["threshold"]][0] += r["correct"]
        by_thr[m["threshold"]][1] += 1
        if m.get("pair"):
            pairs[m["pair"]][m["variant"]] = r["correct"]
    # only pairs whose BOTH members fall inside this subset count as pairs;
    # incomplete ones (split across subsets) are reported, never silently dropped
    complete = [v for v in pairs.values() if v.get("pair_thrlo") is not None and v.get("pair_thrhi") is not None]
    both = [int(v["pair_thrlo"] and v["pair_thrhi"]) for v in complete]
    return {"n": len(rows), "accuracy": sum(r["correct"] for r in rows) / len(rows),
            "nll": statistics.mean(r["nll"] for r in rows),
            "pairs_both": statistics.mean(both) if both else None, "pairs_n": len(both),
            "pairs_incomplete_in_subset": len(pairs) - len(complete),
            "by_threshold": {f"{t:g}": c / n for t, (c, n) in sorted(by_thr.items())}}


def grouped_bootstrap_ci(rows: list[dict], draws: int = BOOTSTRAP) -> list[float]:
    groups = defaultdict(list)
    for r in rows:
        groups[r["group_id"]].append(r["correct"])
    items = list(groups.values())
    rng = random.Random(20261015)
    values = []
    for _ in range(draws):
        sampled = rng.choices(items, k=len(items))
        values.append(sum(c for g in sampled for c in g) / sum(len(g) for g in sampled))
    values.sort()
    return [values[int(0.025 * draws)], values[int(0.975 * draws)]]


def verdict_bar(stats: dict) -> bool:
    return stats["accuracy"] >= 0.90 and stats["pairs_both"] is not None and stats["pairs_both"] >= 0.75


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "runs/e25-analysis/e25_report.json")
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"Refusing to overwrite: {args.out}")

    summary = json.loads((KERNEL_RUN / "e25_summary.json").read_text(encoding="utf-8"))
    hardware = json.loads((KERNEL_RUN / "hardware.json").read_text(encoding="utf-8"))
    meta = data_meta()
    report = {"protocol": summary["protocol"], "hardware": hardware,
              "seeds": {}, "verdicts": {}, "references": {}}

    seeds = sorted(summary["seeds"])
    for seed in seeds:
        res = summary["seeds"][seed]
        curves = [json.loads(l) for l in (KERNEL_RUN / f"s{seed}" / "curves.jsonl")
                  .read_text(encoding="utf-8").splitlines() if l.strip()]
        traj = {f"first_epoch_ge_{m:.2f}":
                next((c["epoch"] for c in curves if (c["train_acc"] or 0) >= m), None)
                for m in TRAJECTORY_MARKS}
        entry = {"fit": res["fit"], "epochs_run": res["epochs_run"],
                 "stopped_early": res["stopped_early"],
                 "attempted_steps": res["attempted_steps"], "effective_steps": res["effective_steps"],
                 "skipped_steps": res["skipped_steps"],
                 "nonfinite_grad_epochs_total": res["nonfinite_grad_epochs_total"],
                 "final_train_acc": res["final_train_acc"], "final_dev_acc": res["final_dev_acc"],
                 "final_train_nll": res["final_train_nll"], "final_dev_nll": res["final_dev_nll"],
                 "best_train_acc": res["best_train_acc"], "best_train_epoch": res["best_train_epoch"],
                 "trajectory": traj, "init_sha256": res["init_sha256"],
                 "preflight": "OK" if res.get("preflight") else ("skipped (seed 43)" if seed == "43" else None)}
        for name in ("known", "new"):
            rows = load_eval(seed, name)
            entry[f"test_{name}"] = summarize(rows, meta)
            entry[f"test_{name}"]["ci_by_base_case"] = grouped_bootstrap_ci(rows)
            if name == "new":
                for part in ("interp", "extrap"):
                    sub = [r for r in rows if meta[r["id"]]["thr_range"] == part]
                    if sub:
                        entry[f"test_new_{part}"] = summarize(sub, meta)
                        entry[f"test_new_{part}"]["ci_by_base_case"] = grouped_bootstrap_ci(sub)
        # V4: threshold-only pairs across both tests
        pair_rows = [r for name in ("known", "new") for r in load_eval(seed, name)
                     if meta[r["id"]].get("pair")]
        pair_groups = defaultdict(dict)
        for r in pair_rows:
            pair_groups[meta[r["id"]]["pair"]][meta[r["id"]]["variant"]] = r["correct"]
        both = [int(v.get("pair_thrlo") and v.get("pair_thrhi")) for v in pair_groups.values()]
        entry["pairs_cd_both"] = statistics.mean(both) if both else None
        entry["pairs_cd_n"] = len(both)
        report["seeds"][seed] = entry

    def all_seeds(pred):
        return all(pred(report["seeds"][s]) for s in seeds)

    # V3b design note: every test-D pair spans interp<->extrap thresholds
    # ((30,45) ... (85,100)); NO pure-extrap pair exists, so the pair bar is
    # not applicable there (accuracy bar stands; mixed pairs reported separately).
    report["verdicts"] = {
        "V1_fit_train_ge_0.95": all_seeds(lambda e: e["fit"]),
        "V2_new_values_known_thresholds": all_seeds(lambda e: verdict_bar(e["test_known"])),
        "V3a_new_thresholds_interp_acc_and_pairs": all_seeds(
            lambda e: verdict_bar(e.get("test_new_interp", {"accuracy": 0, "pairs_both": None}))),
        "V3b_new_thresholds_extrap_acc_only_pairs_not_applicable_by_design": all_seeds(
            lambda e: e["test_new_extrap"]["accuracy"] >= 0.90),
        "V4_threshold_only_pairs_cd_ge_0.90": all_seeds(
            lambda e: e["pairs_cd_both"] is not None and e["pairs_cd_both"] >= 0.90),
    }
    for seed in seeds:
        e = report["seeds"][seed]
        e["mixed_interp_extrap_pairs_both"] = e["test_new"]["pairs_incomplete_in_subset"]

    # question-ignoring references evaluated on the data itself
    for name, fname in (("known", "test_known_thr_rows.jsonl"), ("new", "test_new_thr_rows.jsonl")):
        rows = [json.loads(l) for l in (DATA / fname).read_text(encoding="utf-8").splitlines() if l.strip()]
        gold = [r["target_option"] == "true" for r in rows]
        pair_rows = [r for r in rows if r["pair"]]
        pgold = [r["target_option"] == "true" for r in pair_rows]
        report["references"][name] = {
            "constant_yes": sum(gold) / len(gold), "constant_no": sum(not g for g in gold) / len(gold),
            "fixed_value_ge_60": sum((r["value"] >= 60) == g for r, g in zip(rows, gold)) / len(gold),
            "on_pairs_constant_yes": sum(pgold) / len(pgold),
            "on_pairs_fixed_value_ge_60": sum((r["value"] >= 60) == g for r, g in zip(pair_rows, pgold)) / len(pgold),
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    hw = hardware.get("gpus", [])
    print(f"hardware: {[g['name'] for g in hw] or hardware['cuda_available']}")
    print(f"verdicts: {report['verdicts']}")
    for seed in seeds:
        e = report["seeds"][seed]
        print(f"  s{seed}: fit={e['fit']} train={e['final_train_acc']:.3f} dev={e['final_dev_acc']:.3f} "
              f"steps eff={e['effective_steps']}/{e['attempted_steps']} skip={e['skipped_steps']} "
              f"nonfinite_ep={e['nonfinite_grad_epochs_total']} traj={e['trajectory']}")
        for name in ("test_known", "test_new_interp", "test_new_extrap"):
            if name in e:
                s = e[name]
                print(f"    {name:<16} acc={s['accuracy']:.3f} CI={[round(v, 3) for v in s['ci_by_base_case']]} "
                      f"pairs={s['pairs_both'] if s['pairs_both'] is None else round(s['pairs_both'], 3)} "
                      f"({s['pairs_n']}) by_thr={ {k: round(v, 2) for k, v in s['by_threshold'].items()} }")
        print(f"    pairs C+D both: {e['pairs_cd_both']}")
    print(f"references: {json.dumps(report['references'], indent=1)}")
    print(f"report: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
