"""E35 analyzer: extremes + replay protection (frozen criteria
docs/E35_CRITERIA_2026-10-11.md). Applies F1 mechanically at the FINAL
checkpoint per seed. Deterministic byte-exact via --out."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/kaggle-typed-e35"
SEEDS = ("s42", "s43")


def load_meta():
    meta = {}
    for root, fname in (("data/e26-order", "test_c_rows.jsonl"),
                        ("data/e26-order", "test_d_rows.jsonl"),
                        ("data/e26-order", "mono_rows.jsonl"),
                        ("data/e27-grid", "test_e_rows.jsonl")):
        for l in (Path(root) / fname).read_text(encoding="utf-8").splitlines():
            if l.strip():
                r = json.loads(l)
                meta[r["id"]] = r
    return meta


def analyze_set(per, meta):
    by_range = defaultdict(lambda: [0, 0])
    pairs = defaultdict(list)
    sweeps = defaultdict(list)
    for p in per:
        m = meta[p["id"]]
        ok = p["correct"]
        if m.get("thr_range"):
            by_range[m["thr_range"]][0] += ok
            by_range[m["thr_range"]][1] += 1
        if m.get("pair") and m["variant"] in ("pair_thrlo", "pair_thrhi"):
            pairs[m["pair"]].append((m["thr_range"], m["variant"], ok))
        if m.get("sweep"):
            sweeps[m["sweep"]].append((m["value"], p["pred_idx"]))
    pure = {"interp": [0, 0], "extrap": [0, 0]}
    for members in pairs.values():
        if len(members) == 2 and {v for _, v, _ in members} == {"pair_thrlo", "pair_thrhi"}:
            key = members[0][0] if members[0][0] == members[1][0] else None
            if key and key in pure:
                pure[key][1] += 1
                pure[key][0] += int(all(ok for _, _, ok in members))
    mono = sum(1 for items in sweeps.values()
               if (lambda pr: all(a <= b for a, b in zip(pr, pr[1:])) and 0 in pr and 1 in pr)([p for _, p in sorted(items)]))
    return dict(by_range=by_range, pure=pure, mono=[mono, len(sweeps)])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "runs/e35-analysis/e35_report.json")
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"Refusing to overwrite: {args.out}")
    meta = load_meta()

    def read_eval(seed, name, tag):
        return [json.loads(l) for l in (RUN / seed / f"eval_{name}__{tag}.jsonl")
                .read_text(encoding="utf-8").splitlines() if l.strip()]

    per_seed = {}
    for seed in SEEDS:
        fin, t0 = "post_ph3", "t0"
        e_fin = analyze_set(read_eval(seed, "e", fin), meta)
        e_t0 = analyze_set(read_eval(seed, "e", t0), meta)
        kn = read_eval(seed, "known", fin)
        tk = read_eval(seed, "task", fin)
        F1 = {
            "E_extrap": e_fin["by_range"].get("extrap", [0, 0]),
            "E_extrap_ge_085": e_fin["by_range"]["extrap"][0] / max(e_fin["by_range"]["extrap"][1], 1) >= 0.85,
            "testC": [sum(p["correct"] for p in kn), len(kn)],
            "testC_ge_095": sum(p["correct"] for p in kn) / len(kn) >= 0.95,
            "E_interp": e_fin["by_range"].get("interp", [0, 0]),
            "E_interp_ge_090": e_fin["by_range"]["interp"][0] / max(e_fin["by_range"]["interp"][1], 1) >= 0.90,
            "task": [sum(p["correct"] for p in tk), len(tk)],
            "task_ge_065": sum(p["correct"] for p in tk) / len(tk) >= 0.65,
            "VE2": e_fin["mono"],
            "VE2_ge_080": e_fin["mono"][0] / max(e_fin["mono"][1], 1) >= 0.80,
        }
        F1_pass = all(F1[k] for k in ("E_extrap_ge_085", "testC_ge_095", "E_interp_ge_090",
                                       "task_ge_065", "VE2_ge_080"))
        per_seed[seed] = {"F1": F1, "F1_pass": F1_pass,
                          "t0_E_extrap": e_t0["by_range"].get("extrap", [0, 0]),
                          "t0_pure": e_t0["pure"], "fin_pure": e_fin["pure"]}

    f1_all = all(per_seed[s]["F1_pass"] for s in SEEDS)
    if f1_all:
        reading = "F1 PASSES both seeds -> v1.1 candidate; report and WAIT for supervisor GO (NO HF re-export)"
    elif all(per_seed[s]["F1"]["E_extrap_ge_085"] for s in SEEDS):
        reading = "F1-falla-rompe: E-extrap >= 0.85 but another metric broke"
    elif all(per_seed[s]["F1"]["E_extrap"][0] / max(per_seed[s]["F1"]["E_extrap"][1], 1) >= 0.786 for s in SEEDS):
        reading = "F1-falla-marginal: E-extrap >= 0.784 (pre-integration level) but < 0.85"
    else:
        reading = "F1-falla-no-sube: E-extrap < 0.784"

    report = {"protocol": "docs/E35_CRITERIA_2026-10-11.md",
              "per_seed": per_seed, "F1_pass_both_seeds": f1_all, "reading": reading,
              "start_checkpoint": "E30-MIX-phase3 (sha 752770f7f6077be0a...)",
              "nature": "extremes+replay protection — the untested combination"}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    for s in SEEDS:
        f = per_seed[s]["F1"]
        print(f"{s}: E-e {f['E_extrap']} | testC {f['testC']} | E-i {f['E_interp']} | "
              f"task {f['task']} | VE2 {f['VE2']} | F1={f1_all}")
    print(f"reading: {reading}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
