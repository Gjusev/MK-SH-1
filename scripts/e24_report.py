"""E24 frozen-rule analysis: fit verdicts + conditional contrast readouts.

Reads the kernel summary/curves/eval rows plus the frozen data metadata and
reports, SEPARATELY: per-seed fit trajectory, and (only if the fit criterion
held) the three eval contrasts — new values, new thresholds, renamed field —
each with accuracy and inversion-pair integrity. The newfield-minus-newvalues
delta isolates the field-name change by construction (same values/threshold).
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KERNEL_RUN = ROOT / "runs/kaggle-typed-e24"
DATA = ROOT / "data/e24-fit"

EVAL_FILES = {"newvalues": "eval_newvalues_rows.jsonl",
              "newthreshold": "eval_newthreshold_rows.jsonl",
              "newfield": "eval_newfield_rows.jsonl"}
TRAJECTORY_MARKS = (0.75, 0.90, 1.0)


def data_meta() -> dict:
    meta = {}
    for fname in EVAL_FILES.values():
        for r in [json.loads(l) for l in (DATA / fname).read_text(encoding="utf-8").splitlines() if l.strip()]:
            meta[r["id"]] = r
    return meta


def pair_and_group_rows(eval_rows_path: Path, meta: dict) -> dict:
    rows = [json.loads(l) for l in eval_rows_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    by_pair = defaultdict(dict)
    for r in rows:
        m = meta[r["id"]]
        if m.get("pair"):
            by_pair[m["pair"]][m["variant"]] = r["correct"]
    pairs = [int(v.get("pair_lo") and v.get("pair_hi")) for v in by_pair.values() if len(v) == 2]
    return {"n": len(rows), "accuracy": sum(r["correct"] for r in rows) / len(rows),
            "pair_both": statistics.mean(pairs) if pairs else None,
            "pair_n": len(pairs)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "runs/e24-analysis/e24_report.json")
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"Refusing to overwrite an existing report: {args.out}")

    summary = json.loads((KERNEL_RUN / "e24_summary.json").read_text(encoding="utf-8"))
    report = {"protocol": summary["protocol"], "verdicts": summary["verdicts"],
              "objective": summary["objective"], "per_seed": {}, "contrasts": {}}
    for seed, res in summary["seeds"].items():
        curves = [json.loads(l) for l in (KERNEL_RUN / f"s{seed}" / "curves.jsonl")
                  .read_text(encoding="utf-8").splitlines() if l.strip()]
        traj = {}
        for mark in TRAJECTORY_MARKS:
            hit = next((c["epoch"] for c in curves if (c["train_acc"] or 0) >= mark), None)
            traj[f"first_epoch_train_acc_ge_{mark:.2f}"] = hit
        report["per_seed"][seed] = {
            "epochs_run": res["epochs_run"], "steps": res["steps"],
            "stopped_early": res["stopped_early"], "fit": res["fit"],
            "final_train_acc": res["final_train_acc"], "final_dev_acc": res["final_dev_acc"],
            "best_train_acc": res["best_train_acc"], "best_train_epoch": res["best_train_epoch"],
            "final_train_nll": res["final_train_nll"], "final_dev_nll": res["final_dev_nll"],
            "trajectory": traj,
            "preflight": "OK" if res.get("preflight") else ("skipped (seed 43)" if seed == "43" else None),
            "init_sha256": res["init_sha256"],
        }

    meta = data_meta()
    for name in EVAL_FILES:
        eval_file = KERNEL_RUN / "s42" / f"eval_{name}.jsonl"
        seeds_eval = {}
        for seed in summary["seeds"]:
            path = KERNEL_RUN / f"s{seed}" / f"eval_{name}.jsonl"
            if path.exists():
                seeds_eval[seed] = pair_and_group_rows(path, meta)
        if not seeds_eval:
            continue
        entry = {"per_seed": seeds_eval,
                 "mean_accuracy": statistics.mean(v["accuracy"] for v in seeds_eval.values())}
        if name == "newthreshold":
            rows = [json.loads(l) for l in (KERNEL_RUN / "s42" / "eval_newthreshold.jsonl")
                    .read_text(encoding="utf-8").splitlines() if l.strip()]
            by_thr = defaultdict(lambda: [0, 0])
            for r in rows:
                by_thr[meta[r["id"]]["threshold"]][0] += r["correct"]
                by_thr[meta[r["id"]]["threshold"]][1] += 1
            entry["by_threshold_s42"] = {f"{t:g}": c / n for t, (c, n) in sorted(by_thr.items())}
        report["contrasts"][name] = entry
    if "newvalues" in report["contrasts"] and "newfield" in report["contrasts"]:
        report["contrasts"]["field_name_delta_newfield_minus_newvalues"] = (
            report["contrasts"]["newfield"]["mean_accuracy"]
            - report["contrasts"]["newvalues"]["mean_accuracy"])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"verdicts: {report['verdicts']}")
    for seed, res in report["per_seed"].items():
        print(f"  s{seed}: fit={res['fit']} final_train={res['final_train_acc']:.4f} "
              f"final_dev={res['final_dev_acc']:.4f} epochs={res['epochs_run']} "
              f"early_stop={res['stopped_early']} traj={res['trajectory']}")
    for name, entry in report["contrasts"].items():
        if isinstance(entry, dict) and "mean_accuracy" in entry:
            per = " ".join(f"s{s}={v['accuracy']:.3f}/pairs={v['pair_both']}"
                           for s, v in entry["per_seed"].items())
            print(f"  {name}: {per}")
    print(f"report: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
