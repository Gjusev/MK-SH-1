"""E31 analysis report: frozen criteria (protocol docs/internal/E31_PROTOCOL_2026-10-11.md).

Reads runs/kaggle-typed-e31 (s42, s43 fine-tunes of Julia-1 on our 944-row grid)
and the kernel summary, joins eval rows by id with data/e26-order (testC/testD)
and data/e27-grid (testE), and regenerates runs/e31-analysis/e31_report.json.

Per seed: J1 fit (final train >= 0.95), testC/testD/mono/testE accuracy,
threshold-only pairs (both-correct), testE interp/extrap split, VE2 monotone
sweeps, same-decade rule. NOTE on same_dec_rule (REVIEW_E31 H2): the 0.5 is the
rule's accuracy AGAINST GOLD on testC — a fixed property of the balanced grid,
not a model-dependent quantity; model<->rule agreement is reported in the
review doc (0.3750 for both seeds). Zero-shot reference read from
runs/julia-research/numeric_zeroshot.json.

Versioned post-closure to fix REVIEW_E31 H1 (the closure claimed a versioned
analyzer that did not exist); this script regenerates the frozen report
byte-exact.

INVALID_MODEL_BINDING (2026-10-11 correction): the eval/dev rows this report
summarizes measured the EXTERNAL engine with ORIGINAL Julia weights, not the
trained checkpoints — see runs/e31-analysis/INVALID_MODEL_BINDING_NOTICE.json
and the corrective re-evaluation scripts/e31_recovery_report.py. This script
reproduces the INVALID history verbatim (frozen file unchanged); it does not
endorse reading b_memorizes_without_order, which is WITHDRAWN.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/kaggle-typed-e31"
OUT = ROOT / "runs/e31-analysis"
E26 = ROOT / "data/e26-order"
E27 = ROOT / "data/e27-grid"
SEEDS = ["s42", "s43"]


def load_meta():
    meta = {}
    for root, fname in ((E26, "test_c_rows.jsonl"), (E26, "test_d_rows.jsonl"),
                        (E27, "test_e_rows.jsonl")):
        for l in (root / fname).read_text(encoding="utf-8").splitlines():
            r = json.loads(l)
            meta[r["id"]] = r
    return meta


def eval_rows(arm, name):
    return [json.loads(l) for l in (RUN / arm / f"eval_{name}.jsonl").read_text(
        encoding="utf-8").splitlines() if l.strip()]


def thr_pairs(rows, meta):
    """Threshold-only inversion pairs, both-correct, grouped per split."""
    pairs = defaultdict(list)
    for r in rows:
        d = meta[r["id"]]
        if d["pair"]:
            pairs[d["pair"]].append((d["variant"], r["pred"] == r["gold"]))
    ok = n = 0
    for members in pairs.values():
        kinds = {v for v, _ in members}
        if kinds == {"pair_thrlo", "pair_thrhi"}:  # threshold-only
            n += 1
            ok += int(all(g for _, g in members))
    return f"{ok}/{n}"


def monotone_strict(preds):
    return all(a <= b for a, b in zip(preds, preds[1:])) and 0 in preds and 1 in preds


def analyze_seed(seed, meta):
    summary = json.loads((RUN / "e31_summary.json").read_text(encoding="utf-8"))
    res = summary["results"][seed]
    krows, nrows, mrows = (eval_rows(seed, n) for n in ("known", "new", "mono"))
    erows = eval_rows(seed, "e")
    acc = lambda rows: round(sum(r["pred"] == r["gold"] for r in rows) / len(rows), 4)
    # same-decade rule vs GOLD on testC (fixed property of the balanced set; see docstring)
    sd = round(sum((int(meta[r["id"]]["value"] // 10) == int(meta[r["id"]]["threshold"] // 10))
                   == bool(r["gold"]) for r in krows) / len(krows), 4)
    sweeps = defaultdict(list)
    for r in erows:
        if meta[r["id"]]["sweep"]:
            sweeps[meta[r["id"]]["sweep"]].append(r)
    mono_ok = sum(monotone_strict([r["pred"] for r in sorted(
        rs, key=lambda r: meta[r["id"]]["value"])]) for rs in sweeps.values())
    return {
        "J1_fit": res["final_train_acc"] >= 0.95,
        "final_train": res["final_train_acc"],
        "final_dev": res["final_dev_acc"],
        "epochs": res["epochs"],
        "known": acc(krows),
        "known_thr_pairs": thr_pairs(krows, meta),
        "same_dec_rule": sd,
        "new": acc(nrows),
        "new_thr_pairs": thr_pairs(nrows, meta),
        "mono": acc(mrows),
        "e": acc(erows),
        "e_thr_pairs": thr_pairs(erows, meta),
        "e_interp": acc([r for r in erows if meta[r["id"]]["thr_range"] == "interp"]),
        "e_extrap": acc([r for r in erows if meta[r["id"]]["thr_range"] == "extrap"]),
        "VE2": [mono_ok, len(sweeps)],
    }


def main() -> int:
    meta = load_meta()
    z = json.loads((ROOT / "runs/julia-research/numeric_zeroshot.json").read_text(
        encoding="utf-8"))["results"]
    per_seed = {s: analyze_seed(s, meta) for s in SEEDS}
    report = {
        "protocol": "docs/internal/E31_PROTOCOL_2026-10-11.md",
        "subject": "Julia-1 fine-tuned on our 944-row grid (recipe mirror, seeds 42/43)",
        "zeroshot_reference": {
            "testC": z["test_c"]["accuracy"],
            "testD": z["test_d"]["accuracy"],
            "testE": z["test_e"]["accuracy"],
            "thr_pairs": "0/" + str(sum(int(z[k]["thr_pairs_both"].split("/")[1])
                                        for k in ("test_c", "test_d", "test_e"))),
        },
        "per_seed": per_seed,
        "criteria": {
            "J1_fit": all(v["J1_fit"] for v in per_seed.values()),
            "J2_order": all(v["known"] >= 0.90 for v in per_seed.values()),
            "reading": "b_memorizes_without_order",
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "e31_report.json").write_text(json.dumps(report, indent=2),
                                         encoding="utf-8")  # frozen file has no trailing newline
    print(f"wrote {OUT / 'e31_report.json'}")
    print("WARNING: this reproduces INVALID history (invalid_model_binding) — "
          "evals/dev measured the external original-weights engine; corrective "
          "re-evaluation: scripts/e31_recovery_report.py", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
