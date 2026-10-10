"""E27 analysis report: frozen criteria (protocol docs/internal/E27_PROTOCOL_2026-10-10.md).

Reads runs/kaggle-typed-e27 (s42, s43), joins eval rows by id with data/e26-order
(historical dev/testC/testD/mono) and data/e27-grid (fresh testE), and computes:
V1' fit; V2-V5 historical (descriptive, same bars as E26); VE1 testE accuracy by
range + pure pairs; VE2 testE monotone sweeps; quadrant signature; references;
pre-registered readings (a)-(d). Writes runs/e27-analysis/e27_report.json +
artifact_hashes.json.
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/kaggle-typed-e27"
OUT = ROOT / "runs/e27-analysis"
E26 = ROOT / "data/e26-order"
E27 = ROOT / "data/e27-grid"
ARMS = ["s42", "s43"]


def load_meta():
    meta = {}
    for root, fname in ((E26, "test_c_rows.jsonl"), (E26, "test_d_rows.jsonl"),
                        (E26, "mono_rows.jsonl"), (E27, "test_e_rows.jsonl")):
        for l in (root / fname).read_text(encoding="utf-8").splitlines():
            r = json.loads(l)
            meta[r["id"]] = r
    return meta


def monotone_strict(preds):
    return all(a <= b for a, b in zip(preds, preds[1:])) and 0 in preds and 1 in preds


def eval_rows(arm, name):
    return [json.loads(l) for l in (RUN / arm / f"eval_{name}.jsonl").read_text(
        encoding="utf-8").splitlines() if l.strip()]


def pair_stats(rows, meta):
    pairs = defaultdict(list)
    for r, good in rows:
        d = meta[r["id"]]
        if d["pair"]:
            pairs[d["pair"]].append((d, good))
    out = {}
    for pid, members in pairs.items():
        kinds = {m[0]["variant"] for m in members}
        kind = "thr" if kinds == {"pair_thrlo", "pair_thrhi"} else \
               "val" if kinds == {"pair_valno", "pair_valyes"} else "?"
        rng = members[0][0]["thr_range"]
        key = f"{kind}_{rng}"
        s, n = out.get(key, (0, 0))
        out[key] = (s + int(all(g for _, g in members)), n + 1)
    return out


def analyze(arm, meta):
    out = {"kind": "e27-grid seed"}
    # historical testC/testD + fresh testE
    for name, split in (("known", "C"), ("new", "D"), ("e", "E")):
        rows = [(r, r["pred_idx"] == r["gold_idx"]) for r in eval_rows(arm, name)]
        acc = sum(g for _, g in rows) / len(rows)
        out[f"test{split}_acc"] = round(acc, 4)
        if name == "new":
            by = defaultdict(list)
            for r, g in rows:
                by[meta[r["id"]]["thr_range"]].append(g)
            for rng, gs in by.items():
                out[f"D_{rng}_acc"] = round(sum(gs) / len(gs), 4)
        if name == "e":
            by = defaultdict(list)
            for r, g in rows:
                by[meta[r["id"]]["thr_range"]].append(g)
            for rng, gs in by.items():
                out[f"E_{rng}_acc"] = round(sum(gs) / len(gs), 4)
        out[f"test{split}_pairs"] = {k: list(v) for k, v in pair_stats(rows, meta).items()}
    # V2 (historical, descriptive): testC acc + aggregated pairs
    cp = out["testC_pairs"]
    agg = (sum(v[0] for v in cp.values()), sum(v[1] for v in cp.values()))
    out["V2_descriptive"] = {"acc": out["testC_acc"], "acc_ge_090": out["testC_acc"] >= 0.90 - 1e-9,
                             "pairs_agg": agg, "pairs_ge_075": agg[1] > 0 and agg[0] / agg[1] >= 0.75 - 1e-9}
    out["V2_pass"] = out["V2_descriptive"]["acc_ge_090"] and out["V2_descriptive"]["pairs_ge_075"]
    # VE1: testE by range with pure pairs
    ep = out["testE_pairs"]
    ve1 = {}
    for rng in ("interp", "extrap"):
        acc = out.get(f"E_{rng}_acc")
        pk = [v for k, v in ep.items() if k.endswith(rng) and k.startswith("thr")]
        tot = (sum(v[0] for v in pk), sum(v[1] for v in pk))
        ve1[rng] = {"acc": acc, "acc_ge_090": acc is not None and acc >= 0.90 - 1e-9,
                    "pure_pairs": tot, "pairs_ge_075": tot[1] > 0 and tot[0] / tot[1] >= 0.75 - 1e-9}
    out["VE1"] = ve1
    out["VE1_pass"] = all(v["acc_ge_090"] and v["pairs_ge_075"] for v in ve1.values())
    # VE2: monotone sweeps within testE
    sweeps = defaultdict(list)
    for r in eval_rows(arm, "e"):
        if meta[r["id"]]["sweep"]:
            sweeps[meta[r["id"]]["sweep"]].append(r)
    mono_ok = 0
    for sid, rs in sweeps.items():
        rs.sort(key=lambda r: meta[r["id"]]["value"])
        if monotone_strict([r["pred_idx"] for r in rs]):
            mono_ok += 1
    out["VE2_monotone"] = [mono_ok, len(sweeps)]
    # historical mono (V5, descriptive)
    hm = defaultdict(list)
    for r in eval_rows(arm, "mono"):
        hm[meta[r["id"]]["sweep"]].append(r)
    h_ok = sum(1 for rs in hm.values()
               if monotone_strict([r["pred_idx"] for r in sorted(rs, key=lambda r: meta[r["id"]]["value"])]))
    out["V5_hist_monotone"] = [h_ok, len(hm)]
    # quadrant signature on testC+testD (historical, comparable with E26)
    quad = defaultdict(lambda: [0, 0])
    for name in ("known", "new"):
        for r, g in [(r, r["pred_idx"] == r["gold_idx"]) for r in eval_rows(arm, name)]:
            d = meta[r["id"]]
            q = d["quadrant"] or ("pair_" + ("thr" if str(d["variant"]).startswith("pair_thr") else "val"))
            quad[q][1] += 1
            quad[q][0] += int(g)
    out["quadrant_C_D"] = {k: v for k, v in sorted(quad.items())}
    return out


def main() -> int:
    meta = load_meta()
    summary = json.loads((RUN / "e27_summary.json").read_text(encoding="utf-8"))
    arms = {a: analyze(a, meta) for a in ARMS}
    both = lambda key: all(a in arms and arms[a].get(key) for a in ARMS)

    readings = {
        "a_coverage_confirmed": all(
            arms[a]["quadrant_C_D"].get("sd_n", [0, 0])[0] >= 15
            and arms[a].get("D_interp_acc", 0) >= 0.90 - 1e-9 for a in ARMS),
        "b_operator_representation_deficit": all(
            summary["results"][a]["final_train_acc"] >= 0.95
            and arms[a]["quadrant_C_D"].get("sd_n", [0, 0])[0] <= 10 for a in ARMS),
        "c_value_order_deficit": (all(arms[a]["VE1_pass"] for a in ARMS)
                                  and not both("VE2_pass")),
        "d_no_change": (not any(arms[a]["V2_pass"] for a in ARMS)
                        and max(arms[a]["testC_acc"] for a in ARMS) <= 0.896 + 1e-9),
    }
    verdicts = {
        "V1_fit": {a: summary["results"][a]["fit"] for a in ARMS},
        "V2_hist_descriptive": {a: arms[a]["V2_pass"] for a in ARMS},
        "VE1_fresh": {a: arms[a]["VE1_pass"] for a in ARMS},
        "VE2_fresh_monotone": {a: arms[a]["VE2_monotone"][0] / arms[a]["VE2_monotone"][1] >= 0.8
                               for a in ARMS},
        "readings": readings,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for p in sorted(RUN.rglob("*")):
        if p.is_file():
            hashes[str(p.relative_to(RUN)).replace("\\", "/")] = {
                "sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "bytes": p.stat().st_size}
    report = {
        "protocol": "docs/internal/E27_PROTOCOL_2026-10-10.md",
        "note": ("historical evals (dev/testC/testD/mono) descriptive; testE is the "
                 "fresh two-decimal OOD measure; E27b matched control pre-registered if treatment passes"),
        "arms": arms, "verdicts": verdicts,
        "stopping": {a: {"epochs": summary["results"][a]["epochs_run"],
                         "best_dev": summary["results"][a].get("best_dev_acc"),
                         "best_dev_epoch": summary["results"][a].get("best_dev_epoch"),
                         "final_dev": summary["results"][a]["final_dev_acc"]}
                     for a in ARMS},
    }
    (OUT / "e27_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (OUT / "artifact_hashes.json").write_text(json.dumps(hashes, indent=2, sort_keys=True),
                                              encoding="utf-8")
    print(json.dumps({"verdicts": verdicts,
                      "per_arm": {a: {"testC": arms[a]["testC_acc"],
                                      "testD_interp": arms[a].get("D_interp_acc"),
                                      "testD_extrap": arms[a].get("D_extrap_acc"),
                                      "testE_interp": arms[a]["VE1"]["interp"],
                                      "testE_extrap": arms[a]["VE1"]["extrap"],
                                      "VE2": arms[a]["VE2_monotone"],
                                      "V5_hist": arms[a]["V5_hist_monotone"],
                                      "sd_n": arms[a]["quadrant_C_D"].get("sd_n"),
                                      "stopping": report["stopping"][a]} for a in ARMS}},
                     indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
