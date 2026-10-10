"""E26 frozen-criteria analysis: numeric order with the decade shortcut broken.

Arms: treat (non-aligned thresholds, full quadrants) vs ctrl (aligned, E25-style)
— equal rows/steps/budget. Verdicts per protocol docs/internal/E26_PROTOCOL:
V1 fit per arm · V2 testC (acc >= 0.90 AND pairs >= 0.75) TREAT · V3 testD
interp/extrap separately, pure pairs in both, same bars, TREAT · V4 pairs C+D
>= 0.90 TREAT · V5 monotone sweeps >= 0.80 (of 6) TREAT · V6 control gap with
the pre-registered reading. References (constants, fixed >=60, same-decade)
per split and arm; per-quadrant and per-threshold breakdowns; grouped
bootstrap CIs by base case.
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
KERNEL_RUN = ROOT / "runs/kaggle-typed-e26"
DATA = ROOT / "data/e26-order"
BOOTSTRAP = 20_000
THR_VARIANTS = ("pair_thrlo", "pair_thrhi")
VAL_VARIANTS = ("pair_valno", "pair_valyes")


def data_meta() -> dict:
    meta = {}
    for fname in ("train_treat_rows.jsonl", "train_ctrl_rows.jsonl", "dev_rows.jsonl",
                  "test_c_rows.jsonl", "test_d_rows.jsonl", "mono_rows.jsonl"):
        for r in [json.loads(l) for l in (DATA / fname).read_text(encoding="utf-8").splitlines() if l.strip()]:
            meta[r["id"]] = r
    return meta


def load_eval(arm_dir: str, name: str) -> list[dict]:
    return [json.loads(l) for l in (KERNEL_RUN / arm_dir / f"eval_{name}.jsonl")
            .read_text(encoding="utf-8").splitlines() if l.strip()]


def grouped_ci(rows: list[dict], draws: int = BOOTSTRAP) -> list[float]:
    groups = defaultdict(list)
    for r in rows:
        groups[r["group_id"]].append(r["correct"])
    items = list(groups.values())
    rng = random.Random(20261016)
    values = []
    for _ in range(draws):
        sampled = rng.choices(items, k=len(items))
        values.append(sum(c for g in sampled for c in g) / sum(len(g) for g in sampled))
    values.sort()
    return [values[int(0.025 * draws)], values[int(0.975 * draws)]]


def pair_stats(rows: list[dict], meta: dict, variants) -> dict:
    pairs = defaultdict(dict)
    for r in rows:
        m = meta[r["id"]]
        if m.get("pair") and m["variant"] in variants:
            pairs[(m["pair"], m["variant"].startswith("pair_thr"))][m["variant"]] = r["correct"]
    complete = [v for v in pairs.values() if len(v) == 2]
    both = [int(all(v.values())) for v in complete]
    return {"pairs_both": statistics.mean(both) if both else None, "pairs_n": len(both)}


def summarize(rows: list[dict], meta: dict) -> dict:
    by_thr = defaultdict(lambda: [0, 0])
    by_quad = defaultdict(lambda: [0, 0])
    for r in rows:
        m = meta[r["id"]]
        by_thr[m["threshold"]][0] += r["correct"]
        by_thr[m["threshold"]][1] += 1
        if m.get("quadrant"):
            by_quad[m["quadrant"]][0] += r["correct"]
            by_quad[m["quadrant"]][1] += 1
    out = {"n": len(rows), "accuracy": sum(r["correct"] for r in rows) / len(rows),
           "nll": statistics.mean(r["nll"] for r in rows),
           "by_threshold": {f"{t:g}": c / n for t, (c, n) in sorted(by_thr.items())},
           "by_quadrant": {q: {"acc": c / n, "n": n} for q, (c, n) in sorted(by_quad.items())}}
    thr = pair_stats(rows, meta, THR_VARIANTS)
    out["thr_pairs_both"], out["thr_pairs_n"] = thr["pairs_both"], thr["pairs_n"]
    val = pair_stats(rows, meta, VAL_VARIANTS)
    out["val_pairs_both"], out["val_pairs_n"] = val["pairs_both"], val["pairs_n"]
    return out


def monotone_sweeps(rows: list[dict], meta: dict) -> dict:
    sweeps = defaultdict(list)
    for r in rows:
        sweeps[meta[r["id"]]["sweep"]].append((meta[r["id"]]["value"], r["pred_idx"]))
    result = {}
    for sw, items in sorted(sweeps.items()):
        items.sort()
        preds = [p for _, p in items]
        monotone = all(preds[i] <= preds[i + 1] for i in range(len(preds) - 1)) \
            and 0 in preds and 1 in preds
        result[sw] = {"monotone": monotone, "preds": preds}
    return result


def bar(stats: dict, pairs_key: str, acc_min=0.90, pair_min=0.75) -> bool:
    return stats["accuracy"] >= acc_min and stats.get(pairs_key) is not None \
        and stats[pairs_key] >= pair_min


def same_decade_rule(row: dict) -> bool:
    return int(row["value"] // 10) == int(row["threshold"] // 10)


def reference_table(fname: str) -> dict:
    rows = [json.loads(l) for l in (DATA / fname).read_text(encoding="utf-8").splitlines() if l.strip()]
    gold = [r["target_option"] == "true" for r in rows]

    def acc(preds):
        return sum(p == g for p, g in zip(preds, gold)) / len(gold)

    out = {"constant_yes": sum(gold) / len(gold), "constant_no": acc([False] * len(rows)),
           "fixed_value_ge_60": acc([r["value"] >= 60 for r in rows]),
           "same_decade": acc([same_decade_rule(r) for r in rows])}
    pair_rows = [r for r in rows if r["pair"] and r["variant"] in THR_VARIANTS]
    if pair_rows:
        pgold = [r["target_option"] == "true" for r in pair_rows]
        out["thr_pairs_same_decade"] = sum(p == g for p, g in zip(
            [same_decade_rule(r) for r in pair_rows], pgold)) / len(pgold)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "runs/e26-analysis/e26_report.json")
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"Refusing to overwrite: {args.out}")

    summary = json.loads((KERNEL_RUN / "e26_summary.json").read_text(encoding="utf-8"))
    hardware = json.loads((KERNEL_RUN / "hardware.json").read_text(encoding="utf-8"))
    meta = data_meta()
    report = {"protocol": summary["protocol"], "hardware": hardware,
              "arms": {}, "verdicts": {}, "references": {}, "v6_reading": None}

    for arm_key, res in summary["arms"].items():
        # summary keys are "<arm>_s<seed>"; arm directories are "s<seed>_<arm>"
        name, seed = arm_key.rsplit("_s", 1)
        arm_dir = f"s{seed}_{name}"
        curves = [json.loads(l) for l in (KERNEL_RUN / arm_dir / "curves.jsonl")
                  .read_text(encoding="utf-8").splitlines() if l.strip()]
        entry = {"fit": res["fit"], "epochs_run": res["epochs_run"],
                 "stopped_early": res["stopped_early"],
                 "effective_steps": res["effective_steps"], "attempted_steps": res["attempted_steps"],
                 "skipped_steps": res["skipped_steps"],
                 "nonfinite_grad_epochs_total": res["nonfinite_grad_epochs_total"],
                 "final_train_acc": res["final_train_acc"], "final_dev_acc": res["final_dev_acc"],
                 "best_train_acc": res["best_train_acc"],
                 "best_train_epoch": res["best_train_epoch"],
                 "init_sha256": res["init_sha256"]}
        known = load_eval(arm_dir, "known")
        new = load_eval(arm_dir, "new")
        entry["test_c"] = summarize(known, meta)
        entry["test_c"]["ci_by_base_case"] = grouped_ci(known)
        for part in ("interp", "extrap"):
            sub = [r for r in new if meta[r["id"]]["thr_range"] == part]
            entry[f"test_d_{part}"] = summarize(sub, meta)
            entry[f"test_d_{part}"]["ci_by_base_case"] = grouped_ci(sub)
        # V4: pairs C+D aggregated (both kinds)
        pair_rows = [r for r in known + new if meta[r["id"]].get("pair")]
        groups = defaultdict(dict)
        for r in pair_rows:
            m = meta[r["id"]]
            groups[m["pair"]][m["variant"]] = r["correct"]
        both = []
        for v in groups.values():
            kinds = {k.startswith("pair_thr") for k in v}
            for kind in kinds:
                members = {k: val for k, val in v.items() if k.startswith("pair_thr") == kind}
                if len(members) == 2:
                    both.append(int(all(members.values())))
        entry["pairs_cd_both"] = statistics.mean(both) if both else None
        entry["pairs_cd_n"] = len(both)
        entry["monotone_sweeps"] = monotone_sweeps(load_eval(arm_dir, "mono"), meta)
        entry["monotone_fraction"] = (sum(s["monotone"] for s in entry["monotone_sweeps"].values())
                                      / len(entry["monotone_sweeps"]))
        report["arms"][arm_key] = entry

    def treat_seeds():
        return [k for k in report["arms"] if k.startswith("treat")]

    def ctrl_seeds():
        return [k for k in report["arms"] if k.startswith("ctrl")]

    def all_(keys, pred):
        return bool(keys) and all(pred(report["arms"][k]) for k in keys)

    treat, ctrl = treat_seeds(), ctrl_seeds()
    v = {"V1_fit_all_arms": all_(list(report["arms"]), lambda e: e["fit"]),
         "V1_fit_treat": all_(treat, lambda e: e["fit"]),
         "V1_fit_ctrl": all_(ctrl, lambda e: e["fit"]),
         "V2_treat_testC_acc_and_pairs": all_(
             treat, lambda e: bar(e["test_c"], "thr_pairs_both")
             and (e["test_c"]["val_pairs_both"] is None or e["test_c"]["val_pairs_both"] >= 0.75)),
         "V3a_treat_testD_interp_acc_and_pure_pairs": all_(
             treat, lambda e: bar(e["test_d_interp"], "thr_pairs_both")),
         "V3b_treat_testD_extrap_acc_and_pure_pairs": all_(
             treat, lambda e: bar(e["test_d_extrap"], "thr_pairs_both")),
         "V4_treat_pairs_cd_ge_0.90": all_(
             treat, lambda e: e["pairs_cd_both"] is not None and e["pairs_cd_both"] >= 0.90),
         "V5_treat_monotone_ge_0.80": all_(treat, lambda e: e["monotone_fraction"] >= 0.80)}
    if ctrl:
        v["V6_ctrl_testC"] = all_(ctrl, lambda e: bar(e["test_c"], "thr_pairs_both"))
        v["V6_ctrl_testD_interp"] = all_(ctrl, lambda e: bar(e["test_d_interp"], "thr_pairs_both"))
        v["V6_ctrl_testD_extrap"] = all_(ctrl, lambda e: bar(e["test_d_extrap"], "thr_pairs_both"))
        v["V6_ctrl_pairs_cd"] = all_(ctrl, lambda e: e["pairs_cd_both"] is not None
                                     and e["pairs_cd_both"] >= 0.90)
        v["V6_ctrl_monotone"] = all_(ctrl, lambda e: e["monotone_fraction"] >= 0.80)
        treat_passes = v["V2_treat_testC_acc_and_pairs"] and v["V3a_treat_testD_interp_acc_and_pure_pairs"] \
            and v["V3b_treat_testD_extrap_acc_and_pure_pairs"]
        ctrl_passes = v["V6_ctrl_testC"] and v["V6_ctrl_testD_interp"] and v["V6_ctrl_testD_extrap"]
        report["v6_reading"] = ("data_design_is_the_cause" if treat_passes and not ctrl_passes
                                else "rows_budget_sufficed" if treat_passes and ctrl_passes
                                else "capacity_or_duration_limit" if not treat_passes and not ctrl_passes
                                else "treat_fails_ctrl_passes_unexpected")
    report["verdicts"] = v

    for fname in ("train_treat_rows.jsonl", "train_ctrl_rows.jsonl",
                  "test_c_rows.jsonl", "test_d_rows.jsonl"):
        report["references"][fname.split("_rows")[0]] = reference_table(fname)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"hardware: {[g['name'] for g in hardware.get('gpus', [])]}")
    print(f"verdicts: {json.dumps(v, indent=1)}")
    print(f"V6 reading: {report['v6_reading']}")
    for arm_key, e in report["arms"].items():
        print(f"  {arm_key}: fit={e['fit']} train={e['final_train_acc']:.3f} dev={e['final_dev_acc']:.3f} "
              f"eff={e['effective_steps']}/{e['attempted_steps']} ep={e['epochs_run']}")
        for name in ("test_c", "test_d_interp", "test_d_extrap"):
            s = e[name]
            print(f"    {name:<14} acc={s['accuracy']:.3f} CI={[round(x, 3) for x in s['ci_by_base_case']]} "
                  f"thr_pairs={s['thr_pairs_both'] if s['thr_pairs_both'] is None else round(s['thr_pairs_both'], 2)} "
                  f"val_pairs={s['val_pairs_both'] if s['val_pairs_both'] is None else round(s['val_pairs_both'], 2)} "
                  f"quad={ {k: round(q['acc'], 2) for k, q in s['by_quadrant'].items()} }")
        print(f"    pairs C+D: {e['pairs_cd_both']} ({e['pairs_cd_n']}) · "
              f"monotone: {e['monotone_fraction']:.2f} "
              f"{ {k: s['monotone'] for k, s in e['monotone_sweeps'].items()} }")
    print(f"references: {json.dumps(report['references'], indent=1)}")
    print(f"report: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
