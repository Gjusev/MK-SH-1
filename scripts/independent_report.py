"""Paired analysis of the independent-set evaluation (indep-v1).

Groups = 150 original cases (5 questions each) — NOT 750 independent
observations. Bootstrap resamples cases for every difference CI; exact
McNemar (per-question pairs) reported as secondary only.

Two SEPARATE conclusions requested by the 2026-10-09 commission:
 1. Does the E15 improvement (finalist vs control) hold on new cases?
 2. Any advantage vs Julia within the evaluated scope?
Julia's per-question rows carry correctness only — NLL/Brier/MAE are
reported for our arms; Julia's probabilistic metrics stay PENDING (no
comparable measurement, nothing deduced from parameters).
"""

from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/independent-v1"
FINALIST, CONTROL, JULIA = "e15_probs_s44", "e15_control_s43", "julia"


def load(name: str) -> list[dict]:
    rows = [json.loads(l) for l in (RUN / f"{name}_percase.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(rows) == 750, f"{name}: {len(rows)} rows"
    return rows


def mcnemar_exact(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, sum(math.comb(n, i) for i in range(k + 1)) / 2**n * 2)


def summarize(rows: list[dict]) -> dict:
    by_fam = defaultdict(lambda: [0, 0])
    by_type = defaultdict(lambda: [0, 0])
    for r in rows:
        by_fam[r["family"]][0] += r["correct"]
        by_fam[r["family"]][1] += 1
        by_type[r["type"]][0] += r["correct"]
        by_type[r["type"]][1] += 1
    out = {
        "accuracy": sum(r["correct"] for r in rows) / len(rows),
        "by_family": {k: v[0] / v[1] for k, v in sorted(by_fam.items())},
        "by_type": {k: v[0] / v[1] for k, v in sorted(by_type.items())},
    }
    for key in ("nll", "brier", "mae"):
        vals = [r[key] for r in rows if key in r]
        out[f"mean_{key}"] = sum(vals) / len(vals) if vals else None
    return out


def paired(rows_a: list[dict], rows_b: list[dict], label: str, n_boot=20000, seed=20261009) -> dict:
    idx = {r["id"]: r for r in rows_b}
    groups = defaultdict(list)
    b = c = 0  # a-only correct, b-only correct
    for ra in rows_a:
        rb = idx[ra["id"]]
        assert rb["id"] == ra["id"]
        groups[ra["case_id"]].append((ra["correct"], rb["correct"]))
        if ra["correct"] and not rb["correct"]:
            b += 1
        if rb["correct"] and not ra["correct"]:
            c += 1
    gids = sorted(groups)
    rng = random.Random(seed)
    diffs = []
    for _ in range(n_boot):
        sample = [groups[rng.choice(gids)] for _ in range(len(gids))]
        diffs.append((sum(x[0] for g in sample for x in g) - sum(x[1] for g in sample for x in g))
                     / sum(len(g) for g in sample))
    diffs.sort()
    point = (sum(r["correct"] for r in rows_a) - sum(r["correct"] for r in rows_b)) / len(rows_a)
    return {"pair": label, "delta_accuracy": point,
            "bootstrap95_ci_by_case": [diffs[int(0.025 * len(diffs))], diffs[int(0.975 * len(diffs))]],
            "discordant_a_only": b, "discordant_b_only": c,
            "mcnemar_exact_p_secondary": mcnemar_exact(b, c)}


def main() -> int:
    arms = {n: load(n) for n in (FINALIST, CONTROL, JULIA)}
    report = {"set": "indep-v1 (150 cases / 750 questions, frozen; groups = cases)",
              "fixation": json.loads((ROOT / "runs/typed-decisions/independent_fixation.json").read_text(encoding="utf-8"))["criterion"],
              "summaries": {}, "paired": {},
              "pending": ["Julia NLL/Brier/MAE (correctness-only rows)",
                          "comparable latency/efficiency measurements (nothing deduced from parameters)"]}
    for n, rows in arms.items():
        report["summaries"][n] = summarize(rows)
    report["paired"]["finalist_vs_control"] = paired(arms[FINALIST], arms[CONTROL], "E15 finalist vs control")
    report["paired"]["finalist_vs_julia"] = paired(arms[FINALIST], arms[JULIA], "E15 finalist vs Julia")
    report["paired"]["control_vs_julia"] = paired(arms[CONTROL], arms[JULIA], "control vs Julia")
    # context arms (never selection): recipe means over deciding seeds
    for extra in ("e15_probs_s42", "e15_probs_s43", "e15_control_s42", "e15_control_s44"):
        p = RUN / f"{extra}_percase.jsonl"
        if p.exists():
            report["summaries"][extra] = summarize(load(extra))
    dest = RUN / "independent_report.json"
    dest.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report["summaries"].items()}, indent=2)[:2000])
    for k, v in report["paired"].items():
        ci = v["bootstrap95_ci_by_case"]
        print(f"{k}: d_acc {v['delta_accuracy']*100:+.2f} pp | CI95 by case [{ci[0]*100:+.2f}, {ci[1]*100:+.2f}] | "
              f"discordant {v['discordant_a_only']}/{v['discordant_b_only']} | McNemar p={v['mcnemar_exact_p_secondary']:.4f}")
    print("saved ->", dest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
