"""E36 analyzer: temperature + abstention policy for MK-SH-1 v1.2 (frozen
criteria docs/E36_CRITERIA_2026-10-12.md). Deterministic byte-exact via --out.

Inputs (frozen): runs/e36-calib/logits/*.jsonl (+ artifact_hashes.json).
Procedure (all pre-declared):
- T* = golden-section NLL minimizer on the split, T in [0.05, 20]
  (src/mokka_decisions/calibrate.py::fit_temperature — the v0.1 code).
- tau* = smallest threshold in grid 0.30:0.005:0.995 with selective risk
  <= 0.05 on the split (max coverage under the risk budget; v0.1 release
  rule). DISABLED fallback if no grid point qualifies.
- Choice invariance (temperature is monotone): pred_idx(T*) == pred_idx(1)
  asserted per case on EVERY set — failure aborts (pre-registered reading 4).
- Descriptive NLL/ECE before/after on the frozen evals; agreement check of
  local choices vs the frozen Kaggle E35 s42 per-case artifacts.
- ECE: 15 equal-width bins on max-prob (same formula as evaluate.py).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOGITS = ROOT / "runs/e36-calib/logits"
KAGGLE_EVALS = {
    "known": "eval_known__post_ph3.jsonl",
    "new": "eval_new__post_ph3.jsonl",
    "mono": "eval_mono__post_ph3.jsonl",
    "e": "eval_e__post_ph3.jsonl",
    "task": "eval_task__post_ph3.jsonl",
}
KAGDIR = ROOT / "runs/kaggle-typed-e35/s42"
RISK_TARGET = 0.05
WEIGHTS_SHA = "ea66cdde2271b98efda9dbb5d4437b59aa422cb34daca85d4203508aa56264d5"
MODEL_ID = "Gjusev/MK-SH-1"


def load(name: str) -> list[dict]:
    rows = [json.loads(l) for l in (LOGITS / f"{name}.jsonl").read_text(encoding="utf-8")
            .splitlines() if l.strip()]
    if name == "task":  # e29-mix 300-id slice of dev_eval.jsonl, logits-file order
        manifest = json.loads((ROOT / "data/e29-mix/manifest.json").read_text(encoding="utf-8"))
        want = set(manifest["original_dev_ids"])
        meta = [json.loads(l) for l in (ROOT / "data/processed/v1/eval/dev_eval.jsonl")
                .read_text(encoding="utf-8").splitlines() if l.strip()]
        by_id = {m["id"]: m for m in meta if m["id"] in want}
        meta = [by_id[r["id"]] for r in rows]
    else:
        meta = [json.loads(l) for l in (ROOT / SETFILES[name]).read_text(encoding="utf-8")
                .splitlines() if l.strip()]
    if [r["id"] for r in rows] != [m["id"] for m in meta]:
        raise SystemExit(f"{name}: logits/meta id order mismatch")
    return rows


SETFILES = {
    "split": "data/e36-calib/rows.jsonl",
    "known": "data/e26-order/test_c_rows.jsonl",
    "new": "data/e26-order/test_d_rows.jsonl",
    "mono": "data/e26-order/mono_rows.jsonl",
    "e": "data/e27-grid/test_e_rows.jsonl",
    "task": None,  # e29-mix 300-id slice of dev_eval.jsonl (built in load())
}


def probs_at(logits: list[float], t: float) -> list[float]:
    scaled = [l / t for l in logits]
    m = max(scaled)
    exps = [math.exp(s - m) for s in scaled]
    z = sum(exps)
    return [e / z for e in exps]


def nll_ece(rows: list[dict], t: float) -> dict:
    """Mean NLL + 15-bin ECE at temperature t (ECE formula as evaluate.py)."""
    bins: list[list[tuple[float, int]]] = [[] for _ in range(15)]
    nll = 0.0
    n = correct = 0
    for r in rows:
        pv = probs_at(r["logits"], t)
        gold = r["gold_idx"]
        nll += -math.log(max(pv[gold], 1e-12))
        pred = max(range(len(pv)), key=lambda i: pv[i])
        ok = int(pred == gold)
        correct += ok
        n += 1
        conf = max(pv)
        bins[min(int(conf * 15), 14)].append((conf, ok))
    ece = 0.0
    for b in bins:
        if b:
            ece += len(b) / n * abs(sum(a for _, a in b) / len(b) - sum(c for c, _ in b) / len(b))
    return {"n": n, "accuracy": correct / n, "mean_nll": nll / n, "ece": ece}


def curve(rows: list[dict], t: float, grid: list[float]) -> list[dict]:
    ps = [(probs_at(r["logits"], t), r["gold_idx"]) for r in rows]
    out = []
    for tau in grid:
        acc_n = acc_c = cov = 0
        for pv, gold in ps:
            p = max(pv)
            if p >= tau:
                cov += 1
                ok = int(max(range(len(pv)), key=lambda i: pv[i]) == gold)
                acc_n += 1
                acc_c += ok
        out.append({"tau": tau, "coverage": cov / len(ps), "accepted": cov,
                    "selective_accuracy": acc_c / acc_n if acc_n else None,
                    "selective_risk": 1 - acc_c / acc_n if acc_n else None})
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "runs/e36-calib/e36_report.json")
    args = parser.parse_args()
    policy_out = args.out.parent / "inference-policy.json"
    if args.out.exists():
        raise SystemExit(f"Refusing to overwrite: {args.out}")
    hashes = json.loads((LOGITS / "artifact_hashes.json").read_text(encoding="utf-8"))
    if hashes["checkpoint"] != WEIGHTS_SHA:
        raise SystemExit("logits were not produced by the v1.1 weights — abort")

    sets = {name: load(name) for name in SETFILES}

    # --- temperature fit on the split ONLY (v0.1 code path) ---
    from mokka_decisions.calibrate import fit_temperature
    from mokka_decisions.contracts import DecisionCase

    split_meta = [json.loads(l) for l in (ROOT / "data/e36-calib/rows.jsonl")
                  .read_text(encoding="utf-8").splitlines() if l.strip()]
    cases = [DecisionCase.from_dict(r) for r in split_meta]
    logit_rows = [{opt["id"]: r["logits"][i] for i, opt in enumerate(m["options"])}
                  for r, m in zip(sets["split"], split_meta)]
    t_star = fit_temperature(cases, logit_rows)

    # --- choice invariance (monotone transform), EVERY set, per case ---
    for name, rows in sets.items():
        for r in rows:
            p1 = probs_at(r["logits"], 1.0)
            pt = probs_at(r["logits"], t_star)
            if max(range(len(p1)), key=lambda i: p1[i]) != max(range(len(pt)), key=lambda i: pt[i]):
                raise SystemExit(f"choice invariance violated: {name}/{r['id']} — bug, not a result")

    # --- split metrics before/after ---
    before = nll_ece(sets["split"], 1.0)
    after = nll_ece(sets["split"], t_star)

    # --- policy rule (pre-declared) on the split ---
    grid = [round(0.30 + 0.005 * i, 4) for i in range(int((0.995 - 0.30) / 0.005) + 1)]
    full_curve = curve(sets["split"], t_star, grid)
    chosen = next((c for c in full_curve if c["accepted"] > 0 and c["selective_risk"] <= RISK_TARGET),
                  None)
    enabled = chosen is not None

    # --- descriptive on frozen evals + agreement vs frozen Kaggle per-case ---
    descriptive = {}
    agreement = {}
    for name in ("known", "new", "mono", "e", "task"):
        descriptive[name] = {"before": nll_ece(sets[name], 1.0), "after": nll_ece(sets[name], t_star)}
        kag = KAGGLE_EVALS[name]
        if kag:
            krows = {json.loads(l)["id"]: json.loads(l) for l in (KAGDIR / kag)
                     .read_text(encoding="utf-8").splitlines() if l.strip()}
            match = sum(1 for r in sets[name]
                        if max(range(len(r["logits"])), key=lambda i: r["logits"][i])
                        == krows[r["id"]]["pred_idx"])
            agreement[name] = {"kaggle_file": kag, "match": match, "n": len(sets[name]),
                               "mismatch": len(sets[name]) - match}

    # --- pre-registered readings ---
    readings = {
        "r1_T_gt_1": t_star > 1.0,
        "r2_ece_improves": after["ece"] < before["ece"],
        "r3_policy": "enabled" if enabled else "disabled_fallback",
        "r4_choice_invariance": "verified_all_sets",
    }

    report = {
        "protocol": "docs/E36_CRITERIA_2026-10-12.md",
        "weights": {"model": MODEL_ID, "version": "1.2 (weights = v1.1, unchanged)",
                    "sha256": WEIGHTS_SHA},
        "split": {"set": "e36-calib", "n": len(sets["split"]),
                  "rows_sha256": hashes["e36cal_rows_sha256"],
                  "disjoint_from_prior_numeric_universe": True},
        "temperature": {"t_star": t_star,
                        "procedure": "golden-section NLL minimization, T in [0.05,20] "
                                     "(calibrate.fit_temperature), split only",
                        "monotone": "changes NO choices; confidences only (asserted per case, all sets)",
                        "split_before": before, "split_after": after},
        "policy": {"rule": f"smallest tau in 0.30:0.005:0.995 with selective risk <= {RISK_TARGET} "
                           "on the split (max coverage under risk budget); disabled if unmet",
                   "enabled": enabled,
                   "tau": chosen["tau"] if enabled else None,
                   "at_tau": chosen,
                   "reason_code": "below_validated_threshold"},
        "curve": full_curve,
        "evals_descriptive": descriptive,
        "agreement_vs_frozen_kaggle_s42": agreement,
        "readings": readings,
        "artifact_hashes": hashes,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    policy = {
        "format_version": 1,
        "model": {"id": MODEL_ID,
                  "weights_sha256": WEIGHTS_SHA,
                  "weights_note": "trained in E35 (E24-E35 campaign); temperature and abstention "
                                  "fitted in E36 on a frozen split, no retraining"},
        "temperature": {"value": t_star,
                        "scope": "single global scalar over option logits before softmax (all question types)",
                        "fitted_on": {"set": "e36-calib", "n": len(sets["split"]),
                                      "rows_sha256": hashes["e36cal_rows_sha256"],
                                      "procedure": "NLL golden-section minimization, T in [0.05,20]"},
                        "note": "Monotone transform: identical choices by construction; confidences only."},
        "abstention": {"enabled": enabled,
                       "metric": "max calibrated probability over the presented options",
                       "threshold": chosen["tau"] if enabled else None,
                       "reason_code": "below_validated_threshold",
                       "rule": report["policy"]["rule"],
                       "selective_risk_at_tau": chosen["selective_risk"] if enabled else None,
                       "coverage_at_tau": chosen["coverage"] if enabled else None},
        "response_semantics": {
            "probabilities": "softmax(logits / temperature.value)",
            "choice": "argmax option id when max probability >= threshold, else null",
            "abstain": "choice null + reason_code below_validated_threshold"},
        "provenance": {"experiment": "E36", "criteria": "docs/E36_CRITERIA_2026-10-12.md",
                       "report": "runs/e36-calib/e36_report.json",
                       "generator": "scripts/gen_e36_calib.py"},
    }
    policy_out.write_text(json.dumps(policy, indent=2, sort_keys=True), encoding="utf-8")

    print(f"T* = {t_star}")
    print(f"split: NLL {before['mean_nll']:.4f} -> {after['mean_nll']:.4f} | "
          f"ECE {before['ece']:.4f} -> {after['ece']:.4f} | acc {after['accuracy']:.4f}")
    if enabled:
        print(f"policy: tau={chosen['tau']} coverage={chosen['coverage']:.3f} "
              f"risk={chosen['selective_risk']:.4f} sel_acc={chosen['selective_accuracy']:.4f}")
    else:
        print("policy: DISABLED (no tau met risk<=0.05)")
    for name in descriptive:
        b, a = descriptive[name]["before"], descriptive[name]["after"]
        print(f"eval {name}: acc {a['accuracy']:.4f} | NLL {b['mean_nll']:.4f} -> {a['mean_nll']:.4f} "
              f"| ECE {b['ece']:.4f} -> {a['ece']:.4f}")
    for name, a in agreement.items():
        print(f"agreement {name}: {a['match']}/{a['n']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
