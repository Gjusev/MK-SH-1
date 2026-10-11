"""E34 analyzer: replay-for-adapted-Julia (frozen criteria
docs/E34_CRITERIA_2026-10-11.md). Recomputes every metric from the per-case
files with the campaign joins, applies V1 (live dev + eval_model hash at every
checkpoint) then R1 (dual: task-proxy >=0.60 AND testC >=0.90 AND E-interp
>=0.90 at the FINAL checkpoint, per seed, PASS only if both seeds) and the
pre-registered four-way reading incl. the proxy-ambiguity branch.
Deterministic: byte-exact regenerable via --out.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/kaggle-typed-e34"
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


def percase(seed: str, name: str, tag: str) -> list[dict]:
    return [json.loads(l) for l in (RUN / seed / f"eval_{name}__{tag}.jsonl")
            .read_text(encoding="utf-8").splitlines() if l.strip()]


def acc(per):
    return sum(p["ok"] for p in per) / len(per)


def e_interp(per, meta):
    sub = [p for p in per if meta[p["id"]].get("thr_range") == "interp"]
    return sum(p["ok"] for p in sub), len(sub)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "runs/e34-analysis/e34_report.json")
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"Refusing to overwrite: {args.out}")
    meta = load_meta()
    summary = json.loads((RUN / "e34_summary.json").read_text(encoding="utf-8"))

    per_seed, flags = {}, {}
    for seed in SEEDS:
        res = summary["results"][seed]
        hist = res["history"]
        devs = [h["dev_acc"] for h in hist]
        ckpts = [c["tag"] for c in res["checkpoints"]]
        hashes_ok = all(len(c["eval_model"].get("eval_weights_sha256", "")) == 64
                        for c in res["checkpoints"])
        v1 = len(set(devs)) >= 3 and max(devs) >= 0.60 and hashes_ok and len(ckpts) == 5
        final = "post_mix3"
        tc = percase(seed, "known", final)
        ei_n, ei_d = e_interp(percase(seed, "e", final), meta)
        t29 = percase(seed, "task29", final)
        t30 = percase(seed, "task30", final)
        t2k_t0 = acc(percase(seed, "typed2000", "t0"))
        t2k_fin = acc(percase(seed, "typed2000", final))
        per_seed[seed] = {
            "epochs_total": res["epochs_total"], "dev_distinct": len(set(devs)),
            "dev_max": max(devs), "V1": bool(v1), "checkpoints": ckpts,
            "final": {"testC": [sum(p["ok"] for p in tc), len(tc)],
                      "E_interp": [ei_n, ei_d],
                      "task29": [sum(p["ok"] for p in t29), len(t29)],
                      "task30": [sum(p["ok"] for p in t30), len(t30)],
                      "typed2000_t0": round(t2k_t0, 4),
                      "typed2000_final": round(t2k_fin, 4),
                      "typed2000_delta_pp": round((t2k_fin - t2k_t0) * 100, 2)},
            "R1_task": acc(t29) >= 0.60, "R1_testC": acc(tc) >= 0.90,
            "R1_Einterp": ei_d > 0 and ei_n / ei_d >= 0.90}
        flags[seed] = all(per_seed[seed]["R1_" + k] for k in ("task", "testC", "Einterp"))

    v1_all = all(per_seed[s]["V1"] for s in SEEDS)
    order_ok = all(per_seed[s]["R1_testC"] and per_seed[s]["R1_Einterp"] for s in SEEDS)
    task_ok = all(per_seed[s]["R1_task"] for s in SEEDS)
    t2k_improves = all(per_seed[s]["final"]["typed2000_delta_pp"] > 0 for s in SEEDS)
    if not v1_all:
        reading = "V1-FAIL: pipeline invalid (dev not live or missing eval_model hash) - no R interpreted"
    elif order_ok and task_ok:
        reading = ("R1 PASS both seeds: replay integration PORTABLE to Julia - task AND order "
                   "simultaneously; the E32 trade-off is not intrinsic to the Julia system")
    elif order_ok and not task_ok:
        reading = ("task-proxy FAILS, order kept: proxy routing not recovered"
                   + ("; typed-2000 ALSO fails to improve -> replay with the routing proxy does "
                      "not reach its real distribution" if not t2k_improves
                      else "; typed-2000 IMPROVES -> proxy trained but the routing slices do not "
                           "reflect it (proxy ambiguity declared)"))
    elif task_ok and not order_ok:
        reading = ("order FAILS, task recovered: interference on the fresh capability despite "
                   "replay - compare magnitude vs E32 (-13.5/-19.0pp)")
    else:
        reading = "BOTH FAIL at this budget: the E29b-mirror curriculum does not transfer to Julia - descriptive"

    report = {
        "protocol": "docs/E34_CRITERIA_2026-10-11.md (frozen pre-launch)",
        "nature": "replay portability test; order acquisition + solo-tuning task loss known "
                  "(E31/E33/E32), the replay outcome is the new datum; proxy task declared",
        "per_seed": per_seed, "R1_pass_both_seeds": all(flags.values()),
        "order_kept": order_ok, "task_recovered": task_ok,
        "typed2000_improves": t2k_improves, "reading": reading,
        "mokka_reference": {"task29_slice": 0.690, "task30_slice": 0.7333,
                            "note": "E29/E30 MIX finals on the same slices"},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    for s in SEEDS:
        p = per_seed[s]
        print(f"{s}: V1={p['V1']} final={p['final']} R1(task/testC/Einterp)="
              f"{p['R1_task']}/{p['R1_testC']}/{p['R1_Einterp']}")
    print(f"R1 both seeds: {all(flags.values())}\nreading: {reading}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
