"""E29 analysis report: forgetting/integration study (frozen protocol E29_PROTOCOL_2026-10-10.md).

Regenerates runs/e29-analysis/e29_report.json from the collected artifacts:
baseline, per-arm (MIX/ORIG) per-phase metrics with the testE interp/extrap
split and pure pairs, VE2 strict monotone sweeps, task accuracy, and the frozen
criteria F1/F2/F3 with readings (i)-(iii). Versioned post-final-review (the
closure's numbers were computed with this exact logic; artifacts untouched).
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/kaggle-typed-e29"
OUT = ROOT / "runs/e29-analysis"


def load_meta():
    meta = {}
    for root, fname in ((ROOT / "data/e26-order", "test_c_rows.jsonl"),
                        (ROOT / "data/e26-order", "test_d_rows.jsonl"),
                        (ROOT / "data/e26-order", "mono_rows.jsonl"),
                        (ROOT / "data/e27-grid", "test_e_rows.jsonl")):
        for l in (root / fname).read_text(encoding="utf-8").splitlines():
            if l.strip():
                r = json.loads(l)
                meta[r["id"]] = r
    return meta


def mono_ok(rows, meta):
    sweeps = defaultdict(list)
    for r in rows:
        if meta[r["id"]]["sweep"]:
            sweeps[meta[r["id"]]["sweep"]].append(r)
    ok = 0
    for rs in sweeps.values():
        rs.sort(key=lambda r: meta[r["id"]]["value"])
        ps = [r["pred_idx"] for r in rs]
        if all(a <= b for a, b in zip(ps, ps[1:])) and 0 in ps and 1 in ps:
            ok += 1
    return ok, len(sweeps)


def e_split(rows, meta):
    out = {}
    for rng in ("interp", "extrap"):
        rs = [r for r in rows if meta[r["id"]]["thr_range"] == rng]
        acc = sum(r["pred_idx"] == r["gold_idx"] for r in rs) / len(rs)
        pairs = defaultdict(list)
        for r in rs:
            d = meta[r["id"]]
            if d["pair"]:
                pairs[d["pair"]].append(r["pred_idx"] == r["gold_idx"])
        pure = sum(1 for v in pairs.values() if len(v) == 2 and all(v))
        out[rng] = (round(acc, 4), pure, len(pairs))
    return out


def main() -> int:
    meta = load_meta()
    s = json.load(open(RUN / "e29_summary.json", encoding="utf-8"))
    report = {"protocol": "docs/internal/E29_PROTOCOL_2026-10-10.md",
              "baseline": {k: round(v["accuracy"], 4) for k, v in s["baseline"].items()},
              "arms": {}}
    for arm in ("MIX", "ORIG"):
        phases = []
        for ph in s["results"][arm]:
            p = ph["phase"]
            e_rows = [json.loads(l) for l in (RUN / arm / f"phase{p}/eval_e.jsonl").read_text(
                encoding="utf-8").splitlines() if l.strip()]
            c_rows = [json.loads(l) for l in (RUN / arm / f"phase{p}/eval_known.jsonl").read_text(
                encoding="utf-8").splitlines() if l.strip()]
            task_rows = [json.loads(l) for l in (RUN / arm / f"phase{p}/eval_task.jsonl").read_text(
                encoding="utf-8").splitlines() if l.strip()]
            phases.append({"phase": p,
                           "testC": round(sum(r["pred_idx"] == r["gold_idx"] for r in c_rows) / len(c_rows), 4),
                           "E": e_split(e_rows, meta), "VE2": mono_ok(e_rows, meta),
                           "task": round(sum(r["pred_idx"] == r["gold_idx"] for r in task_rows) / len(task_rows), 4)})
        report["arms"][arm] = phases
    mix4, orig4 = report["arms"]["MIX"][-1], report["arms"]["ORIG"][-1]
    F1 = mix4["testC"] >= 0.90 and mix4["E"]["interp"][0] >= 0.90
    F2 = orig4["testC"] < 0.90
    reading = ("i_F1_integration_viable" if F1 else "iii_adjust_mixture") if not (F1 and not F2) \
        else "ii_order_robust_without_replay"
    report["criteria"] = {"F1_retention_with_replay": F1, "F2_catastrophic_forgetting": F2,
                          "F3_task_recovery_descriptive": {a: report["arms"][a][-1]["task"]
                                                           for a in ("MIX", "ORIG")},
                          "reading": reading}
    OUT.mkdir(exist_ok=True)
    (OUT / "e29_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["criteria"], indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
