"""E33 analyzer: valid re-run of E31 with the corrected pipeline (frozen criteria
docs/E33_CRITERIA_2026-10-11.md). Reads the collected kernel output
(runs/kaggle-typed-e33/: e31_summary.json + s42/s43 curves and per-case evals),
recomputes every metric from per-case predictions with the campaign joins, and
applies J1-J4 + the NEW V1 pipeline-validity criterion (live dev curve) purely
mechanically. Contamination declared: the corrective already revealed these
numbers - this is pipeline CONFIRMATION, not discovery.

Deterministic: byte-exact regenerable (verified with a second run).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/kaggle-typed-e33"
E26, E27 = ROOT / "data/e26-order", ROOT / "data/e27-grid"
SEEDS = ("s42", "s43")


def load_meta() -> dict:
    meta = {}
    for root, fname in ((E26, "test_c_rows.jsonl"), (E26, "test_d_rows.jsonl"),
                        (E26, "mono_rows.jsonl"), (E27, "test_e_rows.jsonl")):
        for l in (root / fname).read_text(encoding="utf-8").splitlines():
            if l.strip():
                r = json.loads(l)
                meta[r["id"]] = r
    return meta


def monotone_strict(preds):
    return all(a <= b for a, b in zip(preds, preds[1:])) and 0 in preds and 1 in preds


def thr_pairs_global(percases, meta):
    pairs = defaultdict(dict)
    for per in percases:
        for p in per:
            d = meta.get(p["id"])
            if d and d.get("pair") and d["variant"] in ("pair_thrlo", "pair_thrhi"):
                pairs[d["pair"]][d["variant"]] = p["ok"]
    ok = [int(all(v.values())) for v in pairs.values() if len(v) == 2]
    return sum(ok), len(ok)


def ve2(percases, meta):
    sweeps = defaultdict(list)
    for per in percases:
        for p in per:
            d = meta.get(p["id"])
            if d and d.get("sweep"):
                sweeps[d["sweep"]].append((d["value"], p["pred"]))
    good = sum(1 for items in sweeps.values()
               if monotone_strict([p for _, p in sorted(items)]))
    return good, len(sweeps)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "runs/e33-analysis/e33_report.json")
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"Refusing to overwrite: {args.out}")

    meta = load_meta()
    summary = json.loads((RUN / "e31_summary.json").read_text(encoding="utf-8"))
    per_seed = {}
    for seed in SEEDS:
        res = summary["results"][seed]
        curves = [json.loads(l) for l in (RUN / seed / "curves.jsonl").read_text(
            encoding="utf-8").splitlines() if l.strip()]
        percases = {name: [json.loads(l) for l in (RUN / seed / f"eval_{name}.jsonl")
                           .read_text(encoding="utf-8").splitlines() if l.strip()]
                    for name in ("known", "new", "mono", "e")}
        testc = sum(p["ok"] for p in percases["known"])
        thr_ok, thr_n = thr_pairs_global(list(percases.values()), meta)
        ve2_ok, ve2_n = ve2([percases["e"], percases["mono"]], meta)
        devs = [c["dev_acc"] for c in curves]
        v1 = (len(set(devs)) >= 3) and (max(devs) >= 0.60)
        per_seed[seed] = {
            "epochs": res["epochs"], "final_train_acc": res["final_train_acc"],
            "final_dev_acc": res["final_dev_acc"], "best_dev_acc": res["best_dev_acc"],
            "dev_values_distinct": len(set(devs)), "dev_max": max(devs),
            "J1_train_ge_095": res["final_train_acc"] >= 0.95,
            "testC": [testc, len(percases["known"])],
            "J2_testC_ge_090": testc / len(percases["known"]) >= 0.90,
            "thr_pairs": [thr_ok, thr_n],
            "J3_thr_pairs_ge_075": thr_n > 0 and thr_ok / thr_n >= 0.75,
            "VE2": [ve2_ok, ve2_n],
            "J4_VE2_ge_080": ve2_n > 0 and ve2_ok / ve2_n >= 0.80,
            "V1_live_dev": bool(v1),
            "eval_model": res["eval_model"],
            "evals_summary": res["evals"],
        }

    v1_all = all(per_seed[s]["V1_live_dev"] for s in SEEDS)
    j1_all = all(per_seed[s]["J1_train_ge_095"] for s in SEEDS)
    j2_all = all(per_seed[s]["J2_testC_ge_090"] for s in SEEDS)
    j3_all = all(per_seed[s]["J3_thr_pairs_ge_075"] for s in SEEDS)
    j4_all = all(per_seed[s]["J4_VE2_ge_080"] for s in SEEDS)
    if not v1_all:
        reading = "R4: pipeline STILL invalid (dev curve not live) - escalate; no J interpreted"
    elif j1_all and j2_all and j3_all and j4_all:
        reading = "R1: corrected pipeline CONFIRMED end-to-end; corrective numbers replicated as a valid run"
    elif j1_all and j2_all:
        reading = ("R2: order acquired, pipeline valid; edges differ from the corrective "
                   "- declared as run-to-run variance, descriptive only")
    else:
        reading = "R3: invalid/unstable execution (J1 failed) - investigate before any reading"

    corrective = {"testC": [1.0, 1.0], "testD": [53 / 54, 52 / 54],
                  "thr_pairs": [[17, 22], [15, 22]], "VE2": [[7, 10], [7, 10]]}
    report = {
        "protocol": "docs/E33_CRITERIA_2026-10-11.md (frozen pre-launch)",
        "nature": "PIPELINE VALIDATION, not discovery - contamination by prior "
                  "corrective knowledge declared; numeric sets already inspected",
        "single_change_vs_e31": "dev stop + all evals on the TRAINED engine "
                                "(evaluate(engine, rows), guards, at-inference weights hash)",
        "per_seed": per_seed,
        "criteria": {"V1": v1_all, "J1": j1_all, "J2": j2_all, "J3": j3_all, "J4": j4_all},
        "reading": reading,
        "corrective_reference": corrective,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    for s in SEEDS:
        p = per_seed[s]
        print(f"{s}: epochs={p['epochs']} train={p['final_train_acc']} dev={p['final_dev_acc']} "
              f"(max {p['dev_max']}, distinct {p['dev_values_distinct']}) testC={p['testC']} "
              f"thr={p['thr_pairs']} VE2={p['VE2']} V1={p['V1_live_dev']}")
    print(f"criteria: {report['criteria']}\nreading: {reading}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
