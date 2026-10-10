"""E23 frozen-rule analysis: acquisition vs curriculum, with references.

Verdict A (acquisition): mean of the 6 new arms OR best arm group vs
e20_ctrl_s43 (prior best-dev) on the sealed test, +15pp minimum.
Verdict B (curriculum): cur - mix mean and per-seed pairs, +8pp minimum,
2/3 seeds. Retention limits vs prior checkpoints on the same sealed test.
"""

from __future__ import annotations

import json
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/e23-sealed-eval"
KERNEL_RUN = ROOT / "runs/kaggle-typed-e23"
SEEDS = (42, 43, 44)
BOOTSTRAP = 20_000


def load(name):
    return [json.loads(l) for l in (RUN / f"{name}_percase.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]


def summarize(rows):
    by_op = defaultdict(lambda: [0, 0])
    by_feat = defaultdict(lambda: [0, 0])
    by_rule = defaultdict(lambda: [0, 0])
    for r in rows:
        by_op[r.get("operator")][0] += r["correct"]
        by_op[r.get("operator")][1] += 1
        by_feat[r.get("feature")][0] += r["correct"]
        by_feat[r.get("feature")][1] += 1
        by_rule[r.get("rule_in")][0] += r["correct"]
        by_rule[r.get("rule_in")][1] += 1
    pairs = defaultdict(lambda: [0, 0])
    by_pair = defaultdict(dict)
    for r in rows:
        if r.get("pair"):
            by_pair[r["pair"]][r["variant"]] = r["correct"]
    for pid, members in by_pair.items():
        hi, lo = members.get("pair_hi"), members.get("pair_lo")
        if hi is not None and lo is not None:
            pairs["both"][1] += 1
            pairs["both"][0] += int(hi and lo)
    return {"accuracy": sum(r["correct"] for r in rows) / len(rows),
            "by_operator": {op: c / n for op, (c, n) in sorted(by_op.items()) if op},
            "by_feature": {f: c / n for f, (c, n) in sorted(by_feat.items()) if f},
            "by_rule_in": {r: c / n for r, (c, n) in sorted(by_rule.items()) if r},
            "pair_both": pairs["both"][0] / pairs["both"][1] if pairs["both"][1] else None}


def main() -> int:
    summary = json.loads((KERNEL_RUN / "e23_summary.json").read_text(encoding="utf-8"))
    if set(summary["seed_hash_equal"].values()) != {True}:
        raise SystemExit("init hash mismatch: analysis invalid")
    if len(set(summary["multiset_hash"] if isinstance(summary["multiset_hash"], list) else [summary["multiset_hash"]])) != 1:
        raise SystemExit("multiset hash mismatch")

    arms = {f"e23_{r}_s{s}": load(f"e23_{r}_s{s}") for s in SEEDS for r in ("mix", "cur")}
    refs = {a: load(a) for a in ("e20_ctrl_s43", "e20_cand_s43", "julia")}
    stats = {a: summarize(rs) for a, rs in arms.items()} | {a: summarize(rs) for a, rs in refs.items()}

    # Verdict B: curriculum (cur - mix per seed)
    cur_mix = {s: stats[f"e23_cur_s{s}"]["accuracy"] - stats[f"e23_mix_s{s}"]["accuracy"] for s in SEEDS}
    mean_cb = statistics.mean(cur_mix.values())
    wins = sum(1 for v in cur_mix.values() if v >= 0)

    # Verdict A: acquisition (best arm family vs prior best-dev reference)
    best_new = max(statistics.mean(stats[f"e23_{r}_s{s}"]["accuracy"] for s in SEEDS) for r in ("mix", "cur"))
    ref_acc = stats["e20_ctrl_s43"]["accuracy"]
    gain = best_new - ref_acc

    # paired bootstrap by problem (pair_id or group_id)
    def grouped_delta(a_rows, b_rows):
        a = {r["id"]: r for r in a_rows}
        per = defaultdict(list)
        for r in b_rows:
            gid = r.get("pair") or r["group_id"]
            per[gid].append(r["correct"] - a[r["id"]]["correct"])
        return {g: sum(v) / len(v) for g, v in per.items()}

    rng = random.Random(20261014)
    def ci(deltas):
        groups = list(deltas.values())
        vals = []
        for _ in range(BOOTSTRAP):
            sample = [groups[rng.randrange(len(groups))] for _ in range(len(groups))]
            vals.append(sum(sample) / len(sample))
        vals.sort()
        return [vals[int(0.025 * BOOTSTRAP)], vals[int(0.975 * BOOTSTRAP)]]

    mix_rows = {s: load(f"e23_mix_s{s}") for s in SEEDS}
    cur_rows = {s: load(f"e23_cur_s{s}") for s in SEEDS}
    pooled_deltas = {}
    for s in SEEDS:
        pooled_deltas.update(grouped_delta(mix_rows[s], cur_rows[s]))
    ci_cb = ci(pooled_deltas)

    verdict_b = "improvement" if (mean_cb >= 0.08 and wins >= 2) else ("negative" if all(v < 0 for v in cur_mix.values()) else "inconclusive")
    verdict_a = "acquired" if gain >= 0.15 else "not_acquired"

    report = {"protocol": "docs/internal/E23_PROTOCOL_2026-10-10.md",
              "per_arm": stats,
              "cur_minus_mix_by_seed": {str(s): v for s, v in cur_mix.items()},
              "cur_minus_mix_mean": mean_cb, "ci_by_problem": ci_cb,
              "acquisition_gain_vs_e20_ctrl": gain,
              "verdict_a_acquisition": verdict_a, "verdict_b_curriculum": verdict_b,
              "wins_b": f"{wins}/3"}
    out = RUN / "e23_report.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"{'arm':<16} {'acc':>6} {'pairs':>6} | by operator")
    for a in list(arms) + list(refs):
        s = stats[a]
        ops = " ".join(f"{op[:4]}={v:.2f}" for op, v in s["by_operator"].items())
        print(f"{a:<16} {s['accuracy']:>6.3f} {s['pair_both'] or 0:>6.2f} | {ops}")
    print(f"\nVerdict A (acquisition): {verdict_a.upper()} - gain {gain:+.4f} vs e20_ctrl_s43 ({ref_acc:.3f})")
    print(f"Verdict B (curriculum): {verdict_b.upper()} - cur-mix by seed: " +
          " ".join(f"s{s}={v:+.4f}" for s, v in cur_mix.items()) + f"  mean {mean_cb:+.4f}  CI {ci_cb}")
    print(f"Features (new): " + " ".join(f"{f}={v:.3f}" for f, v in stats["e23_cur_s42"]["by_feature"].items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
