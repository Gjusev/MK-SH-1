"""E30 analysis report: INDEPENDENT integration validation, disjoint sample (protocol E30_PROTOCOL_2026-10-11.md).

Regenerates runs/e29b-analysis/e29b_report.json: F1' at phase 3 (testC>=0.90 AND
testE-interp>=0.90) with the interp/extrap split and pure pairs, VE2, task.
Versioned post-final-review. Note (documented, artifact untouched): the frozen
e29b_summary.json carries a cosmetic "phases": 4 template leftover; results hold 3.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/kaggle-typed-e30"
OUT = ROOT / "runs/e30-analysis"


def main() -> int:
    meta = {}
    for l in open(ROOT / "data/e27-grid/test_e_rows.jsonl", encoding="utf-8"):
        if l.strip():
            r = json.loads(l)
            meta[r["id"]] = r
    c_rows = [json.loads(l) for l in open(RUN / "MIX/phase3/eval_known.jsonl", encoding="utf-8").read().splitlines() if l.strip()]
    testC = sum(r["pred_idx"] == r["gold_idx"] for r in c_rows) / len(c_rows)
    e_rows = [json.loads(l) for l in open(RUN / "MIX/phase3/eval_e.jsonl", encoding="utf-8").read().splitlines() if l.strip()]
    split = {}
    for rng in ("interp", "extrap"):
        rs = [r for r in e_rows if meta[r["id"]]["thr_range"] == rng]
        acc = sum(r["pred_idx"] == r["gold_idx"] for r in rs) / len(rs)
        pairs = defaultdict(list)
        for r in rs:
            d = meta[r["id"]]
            if d["pair"]:
                pairs[d["pair"]].append(r["pred_idx"] == r["gold_idx"])
        pure = sum(1 for v in pairs.values() if len(v) == 2 and all(v))
        split[rng] = [round(acc, 4), pure, len(pairs)]
    sweeps = defaultdict(list)
    for r in e_rows:
        if meta[r["id"]]["sweep"]:
            sweeps[meta[r["id"]]["sweep"]].append(r)
    ve2 = 0
    for rs in sweeps.values():
        rs.sort(key=lambda r: meta[r["id"]]["value"])
        ps = [r["pred_idx"] for r in rs]
        if all(a <= b for a, b in zip(ps, ps[1:])) and 0 in ps and 1 in ps:
            ve2 += 1
    task = [json.loads(l) for l in open(RUN / "MIX/phase3/eval_task.jsonl", encoding="utf-8").read().splitlines() if l.strip()]
    task_acc = sum(r["pred_idx"] == r["gold_idx"] for r in task) / len(task)
    F1p = testC >= 0.90 and split["interp"][0] >= 0.90
    report = {"protocol": "docs/internal/E30_PROTOCOL_2026-10-11.md",
              "note": "DISJOINT original sample (ranks 945-1888): outcome was NOT knowable at registration; F1pp at phase 3",
              "baseline": {k: round(v["accuracy"], 4) for k, v in json.load(
                  open(RUN / "e30_summary.json", encoding="utf-8"))["baseline"].items()},
              "phase3": {"testC": round(testC, 4), "E_interp": split["interp"], "E_extrap": split["extrap"],
                         "VE2": [ve2, len(sweeps)], "task": round(task_acc, 4)},
              "F1pp": F1p, "verdict": "INTEGRATION_INDEPENDENTLY_VALIDATED" if F1p else "sample_sensitive_integration_documented"}
    OUT.mkdir(exist_ok=True)
    (OUT / "e30_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"F1pp": F1p, "verdict": report["verdict"]}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
