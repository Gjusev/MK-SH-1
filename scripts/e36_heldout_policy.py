"""E36 follow-up: the frozen v1.2 policy measured on every set, not only the split.

e36_report.py fits T, sweeps tau and reports coverage/risk on the same 480
split rows, so the published 4.86 % is in-sample. This script reads the frozen
policy (runs/e36-calib/inference-policy.json) and the frozen per-case logits,
changes nothing, and reports at the shipped T and tau, per set: coverage,
kept, errors, selective risk with an exact Clopper-Pearson 95 % interval, and
whether the 5 % budget holds (point estimate and upper bound).

Raised by an external review on Hugging Face (2026-10-14). Post-hoc and
descriptive: no refit, the shipped policy is the object under test.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e36_report import LOGITS, RISK_TARGET, ROOT, SETFILES, curve, load, nll_ece  # noqa: E402

POLICY = ROOT / "runs/e36-calib/inference-policy.json"


def binom_cdf(k: int, n: int, p: float) -> float:
    return sum(math.comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(k + 1))


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Exact two-sided interval for k errors in n, by bisection on the binomial CDF."""
    def solve(f) -> float:  # f increasing in p on [0, 1]
        lo, hi = 0.0, 1.0
        for _ in range(100):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if f(mid) < 0 else (lo, mid)
        return (lo + hi) / 2
    lower = 0.0 if k == 0 else solve(lambda p: (1 - binom_cdf(k - 1, n, p)) - alpha / 2)
    upper = 1.0 if k == n else solve(lambda p: alpha / 2 - binom_cdf(k, n, p))
    return lower, upper


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "runs/e36-calib/e36_heldout_policy.json")
    args = parser.parse_args()
    policy = json.loads(POLICY.read_text(encoding="utf-8"))
    t, tau = policy["temperature"]["value"], policy["abstention"]["threshold"]
    hashes = json.loads((LOGITS / "artifact_hashes.json").read_text(encoding="utf-8"))

    per_set = {}
    for name in SETFILES:
        rows = load(name)
        at = curve(rows, t, [tau])[0]
        kept = at["accepted"]
        errors = kept - round(at["selective_accuracy"] * kept) if kept else 0
        lo, hi = clopper_pearson(errors, kept) if kept else (None, None)
        per_set[name] = {
            "n": len(rows), "accuracy_forced": nll_ece(rows, t)["accuracy"],
            "ece": nll_ece(rows, t)["ece"], "kept": kept, "coverage": at["coverage"],
            "errors_kept": errors, "selective_risk": errors / kept if kept else None,
            "risk_ci95": [lo, hi],
            "budget_point": errors / kept <= RISK_TARGET if kept else None,
            "budget_upper": hi <= RISK_TARGET if kept else None,
            "role": "fit (in-sample: T and tau chosen here)" if name == "split" else "held out",
        }

    report = {"policy": str(POLICY.relative_to(ROOT)), "temperature": t, "tau": tau,
              "risk_budget": RISK_TARGET, "interval": "Clopper-Pearson exact, 95 % two-sided",
              "logit_hashes": hashes, "sets": per_set}
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    print(f"T = {t}  tau = {tau}  budget = {RISK_TARGET}")
    for name, s in per_set.items():
        lo, hi = s["risk_ci95"]
        print(f"{name:6} n={s['n']:4} cov={s['coverage']:.3f} risk={s['errors_kept']}/{s['kept']}"
              f"={s['selective_risk']:.4f} CI95=[{lo:.4f}, {hi:.4f}] "
              f"budget point={'ok' if s['budget_point'] else 'MISS'} upper={'ok' if s['budget_upper'] else 'MISS'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
