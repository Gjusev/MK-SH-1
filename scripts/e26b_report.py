"""E26b analysis report: discriminants D1/D2/D3 (frozen protocol).

Reads runs/kaggle-typed-e26b (4 arms: s44, s42_cont, s43_cont, s45), joins eval
rows by id with data/e26-order for value/threshold/pair/quadrant/sweep, and
computes per arm: testC/testD(interp,extrap) accuracy, both-correct pairs by
kind (threshold-only / value-only), strict monotone sweeps, plus D1 dev deltas
for the continuation arms, D2 seed spread vs the V2 bar, and the pre-registered
D3 decision. Writes runs/e26b-analysis/e26b_report.json + artifact_hashes.json.
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/kaggle-typed-e26b"
OUT = ROOT / "runs/e26b-analysis"
DATA = ROOT / "data/e26-order"
E26_REPORT = ROOT / "runs/e26-analysis/e26_report.json"

ARMS = ["s44", "s42_cont", "s43_cont", "s45"]


def data_rows():
    rows = {}
    for fname, split in (("test_c_rows.jsonl", "known"), ("test_d_rows.jsonl", "new"),
                         ("mono_rows.jsonl", "mono")):
        for l in (DATA / fname).read_text(encoding="utf-8").splitlines():
            r = json.loads(l)
            rows[r["id"]] = r
    return rows


def monotone_strict(preds):
    return all(a <= b for a, b in zip(preds, preds[1:])) and 0 in preds and 1 in preds


def eval_rows(arm, name):
    return [json.loads(l) for l in (RUN / arm / f"eval_{name}.jsonl").read_text(
        encoding="utf-8").splitlines() if l.strip()]


def main() -> int:
    meta = data_rows()
    summary = json.loads((RUN / "e26b_summary.json").read_text(encoding="utf-8"))

    def analyze(arm):
        out = {"arm": arm, "kind": "continuation" if arm.endswith("_cont") else "fresh"}
        # testC + testD
        for name, split in (("known", "C"), ("new", "D")):
            rows = eval_rows(arm, name)
            ok = [r["pred_idx"] == r["gold_idx"] for r in rows]
            acc = sum(ok) / len(ok)
            out[f"test{split}_acc"] = round(acc, 4)
            by_range = defaultdict(list)
            pairs = defaultdict(list)
            for r, good in zip(rows, ok):
                d = meta[r["id"]]
                by_range["interp" if d["thr_range"] == "interp" else "extrap"].append(good)
                if d["pair"]:
                    pairs[d["pair"]].append((d, good))
            if name == "new":
                for rng, goods in by_range.items():
                    out[f"D_{rng}_acc"] = round(sum(goods) / len(goods), 4)
            both = defaultdict(lambda: [0, 0])
            for pid, members in pairs.items():
                kinds = {m[0]["variant"] for m in members}
                kind = "thr" if kinds == {"pair_thrlo", "pair_thrhi"} else \
                       "val" if kinds == {"pair_valno", "pair_valyes"} else "?"
                solved = all(g for _, g in members)
                if name == "known":
                    out.setdefault(f"C_pairs_{kind}", [0, 0])
                    out[f"C_pairs_{kind}"][1] += 1
                    out[f"C_pairs_{kind}"][0] += int(solved)
                else:
                    rng = "interp" if members[0][0]["thr_range"] == "interp" else "extrap"
                    key = f"D_pairs_{kind}_{rng}"
                    out.setdefault(key, [0, 0])
                    out[key][1] += 1
                    out[key][0] += int(solved)
        # V2 on historical tests (descriptive)
        cp = out.get("C_pairs_thr", [0, 0])
        cv = out.get("C_pairs_val", [0, 0])
        pairs_tot = (cp[0] + cv[0], cp[1] + cv[1])
        out["V2_descriptive"] = {"acc_ge_090": out["testC_acc"] >= 0.90 - 1e-9,
                                 "pairs_agg": pairs_tot,
                                 "pairs_ge_075": pairs_tot[1] > 0 and pairs_tot[0] / pairs_tot[1] >= 0.75 - 1e-9}
        out["V2_pass"] = out["V2_descriptive"]["acc_ge_090"] and out["V2_descriptive"]["pairs_ge_075"]
        # mono: strict monotone sweeps
        sweeps = defaultdict(list)
        for r in eval_rows(arm, "mono"):
            sweeps[meta[r["id"]]["sweep"]].append(r)
        mono_ok = 0
        for sid, rows in sweeps.items():
            rows.sort(key=lambda r: meta[r["id"]]["value"])
            if monotone_strict([r["pred_idx"] for r in rows]):
                mono_ok += 1
        out["mono_monotone_sweeps"] = [mono_ok, len(sweeps)]
        return out

    arms = {a: analyze(a) for a in ARMS}

    # D1: continuation dev deltas (from summary d1 + curves)
    d1 = {}
    for arm in ("s42_cont", "s43_cont"):
        d = summary["results"][arm]["d1"]
        curves = [json.loads(l) for l in (RUN / arm / "curves.jsonl").read_text(
            encoding="utf-8").splitlines()]
        verdict = "limitante" if d["delta"] >= 0.05 else ("plano" if d["delta"] >= -0.05 else "sobreajuste")
        d1[arm] = {**d, "verdict": verdict,
                   "dev_first_epoch": curves[0]["dev_acc"], "epochs": len(curves),
                   "final_dev_nll": summary["results"][arm]["final_dev_nll"]}
    d1_verdict = "STOPPING_RULE_NOT_LIMITING" if all(
        v["verdict"] != "limitante" for v in d1.values()) else "STOPPING_RULE_LIMITING"

    # D2: seed spread vs V2 (fresh arms + E26 originals from the frozen report)
    e26 = json.loads(E26_REPORT.read_text(encoding="utf-8"))
    seed_testc = {}
    for arm in ("s42_treat", "s43_treat"):
        seed_testc[arm.split("_")[0]] = e26["arms"][arm]["known"]["accuracy"] \
            if "arms" in e26 and arm in e26.get("arms", {}) else None
    for arm in ("s44", "s45"):
        seed_testc[arm] = arms[arm]["testC_acc"]
    if any(v is None for v in seed_testc.values()):
        # fallback: read accuracies straight from the E26 eval files
        for arm, tag in (("s42_treat", "s42"), ("s43_treat", "s43")):
            rows = [json.loads(l) for l in (ROOT / "runs/kaggle-typed-e26" / arm /
                                            "eval_known.jsonl").read_text(encoding="utf-8").splitlines()]
            seed_testc[tag] = round(sum(r["pred_idx"] == r["gold_idx"] for r in rows) / len(rows), 4)
    any_pass = any(arms[a]["V2_pass"] for a in ("s44", "s45"))
    d2 = {"testC_by_seed": seed_testc, "range": round(max(seed_testc.values()) - min(seed_testc.values()), 4),
          "V2_pass_any_fresh_seed": any_pass,
          "verdict": "SEED_PASSES_V2" if any_pass else "NO_SEED_PASSES_V2"}
    d3 = ("E27_DENSE_COMPARISON_GRID" if d1_verdict == "STOPPING_RULE_NOT_LIMITING"
          and not any_pass else
          "RETRAIN_E26_WITH_DEV_STOP" if d1_verdict == "STOPPING_RULE_LIMITING" else
          "ADD_TWO_MORE_SEEDS_THEN_RECONSIDER")

    # hashes
    OUT.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for p in sorted(RUN.rglob("*")):
        if p.is_file():
            hashes[str(p.relative_to(RUN)).replace("\\", "/")] = {
                "sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "bytes": p.stat().st_size}

    report = {
        "protocol": "docs/internal/E26B_PROTOCOL_2026-10-10.md",
        "note": "tests C/D/mono are HISTORICAL (inspected in E26): descriptive only",
        "arms": arms,
        "D1_stopping_rule": {"detail": d1, "verdict": d1_verdict},
        "D2_seed_variance": d2,
        "D3_decision": d3,
        "quota_note": "consumption readable from the Kaggle quota API at closure",
    }
    (OUT / "e26b_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (OUT / "artifact_hashes.json").write_text(json.dumps(hashes, indent=2, sort_keys=True),
                                              encoding="utf-8")
    print(json.dumps({"D1": d1_verdict, "D1_detail": {k: {"delta": v["delta"], "verdict": v["verdict"]}
                                                      for k, v in d1.items()},
                      "D2": d2, "D3": d3,
                      "arms": {a: {"testC": arms[a]["testC_acc"], "V2_pass": arms[a]["V2_pass"],
                                   "pairs_C": {k: arms[a][k] for k in arms[a] if k.startswith("C_pairs")},
                                   "mono": arms[a]["mono_monotone_sweeps"]} for a in ARMS}},
                     indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
