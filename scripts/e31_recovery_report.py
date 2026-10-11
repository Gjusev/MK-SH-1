"""E31 CORRECTIVE recovery report (invalid_model_binding correction, 2026-10-11).

Reads the corrective CPU re-evaluation of the SAVED E31 checkpoints (runs/e31-recovery/,
produced by the supervisor's recheck outside this repo and copied verbatim) and the
repo's own row metadata (data/e26-order, data/e27-grid), recomputes every metric from
per-case predictions with the same joins as the campaign analyzers, and derives the
pre-registered J1-J4 criteria WITHOUT imposing any reading and WITHOUT touching the
invalid frozen history (runs/e31-analysis/e31_report.json stays byte-identical; see
runs/e31-analysis/INVALID_MODEL_BINDING_NOTICE.json).

Scope labels carried in the output: this is a corrective re-evaluation of available
checkpoints — training length was selected by dev of the WRONG model (patience ran on
the external original-weights engine), so this is not a retroactively valid run of the
E31 protocol, and CPU float32 differs from the kernel's GPU fp16 runtime.
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REC = ROOT / "runs/e31-recovery"
E26 = ROOT / "data/e26-order"
E27 = ROOT / "data/e27-grid"
MODELS = ["s42", "s43"]
NOTICE = "runs/e31-analysis/INVALID_MODEL_BINDING_NOTICE.json"


def load_meta():
    meta = {}
    for root, fname in ((E26, "test_c_rows.jsonl"), (E26, "test_d_rows.jsonl"),
                        (E27, "test_e_rows.jsonl")):
        for l in (root / fname).read_text(encoding="utf-8").splitlines():
            r = json.loads(l)
            meta[r["id"]] = r
    return meta


def monotone_strict(preds):
    return all(a <= b for a, b in zip(preds, preds[1:])) and 0 in preds and 1 in preds


def thr_pairs(rows, meta):
    pairs = defaultdict(list)
    for r in rows:
        d = meta.get(r["id"])
        if d and d["pair"]:
            pairs[d["pair"]].append((d["variant"], r["pred"] == r["gold"]))
    ok = n = 0
    for members in pairs.values():
        if {v for v, _ in members} == {"pair_thrlo", "pair_thrhi"}:
            n += 1
            ok += int(all(g for _, g in members))
    return ok, n


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    h.update(p.read_bytes())
    return h.hexdigest()


def analyze_model(model: str, meta) -> dict:
    src = json.loads((REC / "corrected_checkpoint_evaluation.json").read_text(
        encoding="utf-8"))
    out = {"checkpoint_sha256": src["models"][model]["checkpoint_sha256"]}
    rows_by = {}
    for name in ("train", "dev", "known", "new", "mono", "e"):
        rows = json.loads((REC / f"{model}_{name}.json").read_text(encoding="utf-8"))
        rows_by[name] = rows
        out[name] = {"n": len(rows), "acc": round(sum(r["ok"] for r in rows) / len(rows), 4)}
    out["known"]["thr_pairs"] = list(thr_pairs(rows_by["known"], meta))
    out["new"]["thr_pairs"] = list(thr_pairs(rows_by["new"], meta))
    e = out["e"]
    for rng in ("interp", "extrap"):
        sub = [r for r in rows_by["e"] if meta[r["id"]]["thr_range"] == rng]
        e[rng] = round(sum(r["ok"] for r in sub) / len(sub), 4)
    e["thr_pairs"] = list(thr_pairs(rows_by["e"], meta))
    sweeps = defaultdict(list)
    for r in rows_by["e"]:
        if meta[r["id"]]["sweep"]:
            sweeps[meta[r["id"]]["sweep"]].append(r)
    e["VE2"] = [sum(monotone_strict([r["pred"] for r in sorted(
        rs, key=lambda r: meta[r["id"]]["value"])]) for rs in sweeps.values()),
        len(sweeps)]
    # original-weights CPU reference (same runtime), for context only
    return out


def main() -> int:
    meta = load_meta()
    src = json.loads((REC / "corrected_checkpoint_evaluation.json").read_text(
        encoding="utf-8"))
    models = {m: analyze_model(m, meta) for m in MODELS}
    # pre-registered criteria, per seed, derived — not imposed
    criteria = {}
    for m, d in models.items():
        cd = (d["known"]["thr_pairs"][0] + d["new"]["thr_pairs"][0],
              d["known"]["thr_pairs"][1] + d["new"]["thr_pairs"][1])
        d["thr_pairs_CD"] = list(cd)
        criteria[m] = {
            "J1_fit_train_ge_095": d["train"]["acc"] >= 0.95,
            "J2_order_testC_ge_090": d["known"]["acc"] >= 0.90,
            "J3_pairs_CD_ge_075": cd[1] > 0 and cd[0] / cd[1] >= 0.75,
            "J4_VE2_ge_080": d["e"]["VE2"][0] / d["e"]["VE2"][1] >= 0.80,
        }
    report = {
        "kind": "e31_corrective_recovery",
        "supersedes": {"frozen_invalid_history": "runs/e31-analysis/e31_report.json",
                       "withdrawn_reading": "b_memorizes_without_order",
                       "notice": NOTICE},
        "scope": {
            "what_this_is": "corrective CPU re-evaluation of the SAVED E31 checkpoints "
                            "(the only artifacts of the run that measured the trained model)",
            "what_this_is_not": "NOT a retroactively valid E31 protocol run: dev and "
                                "patience-10 stopping ran on the WRONG model (external "
                                "original-weights engine), so training length was not "
                                "selected by a valid signal; no retraining was done",
            "runtime": f"{src['device']}/{src['dtype']}/{src['threads']} threads "
                       f"(torch {src['torch']}, transformers {src['transformers']}); "
                       "kernel history was GPU fp16 — small runtime deltas expected",
            "e28_e30": "Mokka results E28-E30 are NOT auto-invalidated (their evals pass "
                       "the model explicitly); E30 reuses the numeric tests, so it is a "
                       "replication with a different task mix, not an independent "
                       "numeric benchmark",
            "recipe_scope": "mirror recipe declared: uniform LR 2e-5 (Mokka splits "
                            "encoder 2e-5 / head 1e-4), 30-epoch budget (Mokka 300), "
                            "4x8 vs 8x4 microbatch/accum, init from full official Julia "
                            "checkpoint (Mokka: upstream encoder + fresh head) — this is "
                            "NOT an architecture-isolating ablation",
        },
        "bug": {"defect": "evaluate(rows) closed over stage-scope engine with ORIGINAL "
                          "Julia weights; run_seed's local trained engine was never "
                          "evaluated in-kernel",
                "proof": "closure_proof.json (source sha256 "
                         "3ed85baa5fae68ccb01cf1c8d1b91b1cfc5539444566f22fe55b40c47d64bd2b)",
                "fix": "kaggle/typed_pilot.py evaluate(eval_engine, rows); external "
                       "engine removed; eval-time weights sha256 registered; regression "
                       "tests/test_e31_model_binding.py"},
        "original_reference_cpu": {k: v for k, v in src["models"]["original"].get(
            "sets", {}).items()} or "see corrected_checkpoint_evaluation.json",
        "per_model": models,
        "criteria_derived": criteria,
        "reading": None,  # deliberately unset: no reading is imposed on a corrective
        # re-evaluation; J1+J2 in both seeds is EVIDENCE FOR recipe/data transfer to
        # Julia-1, J3 splits by seed, J4 fails (VE2 7/10 < 0.80 both)
    }
    (REC / "e31_recovery_report.json").write_text(json.dumps(report, indent=2),
                                                  encoding="utf-8")
    print(f"wrote {REC / 'e31_recovery_report.json'}")
    for m in MODELS:
        print(m, criteria[m])
    return 0


if __name__ == "__main__":
    sys.exit(main())
