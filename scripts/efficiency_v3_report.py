"""Efficiency v3 formal measurement (closes the E22 -25% caveat).

Uses the rev-3 methodology (threads AFTER model load, torch.inference_mode,
fp32 weights+matmul, finiteness check outside timed region, identical timing
boundaries: pretok phase times forward only; full phase times collate+forward)
on Julia-1 official vs MOKKA-1 (E30 final = the published HF checkpoint,
sha 752770f7...). 750 typed questions (e20-contrast), batch 1 and 16,
2+ invocations in registered alternating order, raw durations saved.

Prediction verification: Julia against the saved e20-sealed percase; MOKKA-1
has no saved e20 references (different experiment family) so determinism is
verified (invocation-2 predictions identical to invocation-1).

Usage (one process per model):
  python scripts/efficiency_v3_report.py --model julia
  python scripts/efficiency_v3_report.py --model mokka1
  python scripts/efficiency_v3_report.py --merge
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
ROWS = ROOT / "data/e20-contrast/test_rows.jsonl"
JULIA_PERCASE = ROOT / "runs/e20-sealed-eval/julia_percase.jsonl"
MOKKA1_CKPT = ROOT / "runs/kaggle-typed-e30/MIX/phase3/final.safetensors"
THREADS = 8
MAXLEN, HEADLEN = 512, 256
WARMUP, TIMED = 1, 2


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def load_rows():
    rows = [json.loads(l) for l in ROWS.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(rows) == 750
    return rows


def measure(model_name: str) -> dict:
    import torch

    torch.manual_seed(0)
    t0 = time.perf_counter()
    rows = load_rows()
    qtypes = {r["id"]: r["question_type"] for r in rows}

    if model_name == "julia":
        sys.path.insert(0, str(ROOT / "runs/julia-research/upstream-source"))
        from julia.inference import TransformerEngine

        engine = TransformerEngine(str(ROOT / "runs/julia-research/upstream-source"),
                                   device="cpu", max_length=MAXLEN, head_length=HEADLEN)
        model = engine.model
        model.to(dtype=torch.float32).eval()
        from julia.cuda import move

        def collate_fn(idxs):
            br = [{"state": rows[i]["state"], "question": rows[i]["question"],
                   "type": rows[i]["question_type"],
                   "options": [o["description"] for o in rows[i]["options"]]} for i in idxs]
            return move(engine.collate(br, include_targets=False), engine.device)

        def forward_fn(c):
            return model(**c).float().cpu()

        def preds_from(out, idxs):
            preds = {}
            for i, values in zip(idxs, out):
                pv = values[: len(rows[i]["options"])].tolist()
                preds[rows[i]["id"]] = max(range(len(pv)), key=pv.__getitem__)
            return preds

        weights_path = "runs/julia-research/upstream-source/model.safetensors"
    else:
        from mokka_decisions.arch_julia import JuliaArchScorer, collate_julia
        from mokka_decisions.contracts import DecisionCase, Option
        from safetensors.torch import load_file
        from transformers import AutoTokenizer

        model = JuliaArchScorer("jhu-clsp/mmBERT-small")
        model.load_state_dict(load_file(str(MOKKA1_CKPT)))
        model.to(dtype=torch.float32).eval()
        tok = AutoTokenizer.from_pretrained("jhu-clsp/mmBERT-small")
        cases = [DecisionCase(id=r["id"], group_id=r["group_id"], source="v3", language="en",
                              domain=r["domain"], state=r["state"], question=r["question"],
                              options=tuple(Option(id=o["id"], description=o["description"])
                                            for o in r["options"]),
                              target_kind="single", target_option=r["target_option"]) for r in rows]

        def collate_fn(idxs):
            return collate_julia(tok, [cases[i] for i in idxs], qtypes, MAXLEN, HEADLEN, "cpu")

        def forward_fn(c):
            return model(**{k: c[k] for k in
                            ("input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype")}).float()

        def preds_from(out, idxs):
            preds = {}
            for i, values in zip(idxs, out):
                pv = values[: len(cases[i].options)].tolist()
                preds[cases[i].id] = max(range(len(pv)), key=pv.__getitem__)
            return preds

        weights_path = str(MOKKA1_CKPT)

    load_s = time.perf_counter() - t0
    n = len(rows)

    def scenario(batch_size, tag):
        batches = [list(range(i, min(i + batch_size, n))) for i in range(0, n, batch_size)]
        collated = [collate_fn(b) for b in batches]
        # acceptance: finite logits + predictions
        preds = {}
        for b, c in zip(batches, collated):
            out = forward_fn(c)
            if not torch.isfinite(out).all():
                raise FloatingPointError(f"nonfinite ({model_name}, {tag})")
            preds.update(preds_from(out, b))
        for _ in range(WARMUP):
            for c in collated:
                forward_fn(c)
        # rev-3: threads pinned AFTER all loading, BEFORE timing
        torch.set_num_threads(THREADS)
        pretok = []
        for _ in range(TIMED):
            t = time.perf_counter()
            for c in collated:
                forward_fn(c)
            pretok.append(time.perf_counter() - t)
        for _ in range(WARMUP):
            for b in batches:
                forward_fn(collate_fn(b))
        full = []
        for _ in range(TIMED):
            t = time.perf_counter()
            for b in batches:
                forward_fn(collate_fn(b))
            full.append(time.perf_counter() - t)
        raw = []
        for b in batches:
            t = time.perf_counter()
            forward_fn(collate_fn(b))
            raw.append({"duration_s": time.perf_counter() - t, "size": len(b)})
        return {"batch_size": batch_size, "batch_count": len(batches),
                "real_last_batch": len(batches[-1]), "pretok_pass_s": pretok,
                "full_pass_s": full, "raw_items": raw, "preds": preds}

    # inference_mode for BOTH timed phases (rev-3)
    with torch.inference_mode():
        threads_before = torch.get_num_threads()
        torch.set_num_threads(THREADS)
        effective = torch.get_num_threads()
        b1 = scenario(1, "b1")
        b16 = scenario(16, "b16")
        if torch.get_num_threads() != THREADS:
            raise RuntimeError("threads changed during measurement")

    # verification
    if model_name == "julia":
        ref = {json.loads(l)["id"]: json.loads(l)["pred_idx"]
               for l in JULIA_PERCASE.read_text(encoding="utf-8").splitlines() if l.strip()}
        mism = sum(1 for i in ref if ref[i] != b1["preds"].get(i))
        mism += sum(1 for i in ref if ref[i] != b16["preds"].get(i))
        verify = {"method": "vs saved e20-sealed percase", "mismatches": mism}
    else:
        identical = b1["preds"] == b16["preds"]
        verify = {"method": "determinism (b1 preds == b16 preds on shared rows)",
                  "identical": identical}
    import torch as _t

    return {"harness_revision": 3, "model": model_name, "load_s": round(load_s, 2),
            "threads": effective, "weight_dtype": "float32",
            "max_length": MAXLEN, "head_length": HEADLEN,
            "torch": _t.__version__, "rows": n,
            "batch1": {k: v for k, v in b1.items() if k != "preds"},
            "batch16": {k: v for k, v in b16.items() if k != "preds"},
            "verification": verify, "weights_path": weights_path,
            "weights_sha256": sha256(weights_path)}


def merge():
    out = {}
    for name in ("julia", "mokka1"):
        p = ROOT / f"runs/e22-efficiency-v3/{name}_measurement.json"
        out[name] = json.loads(p.read_text(encoding="utf-8"))
    report = {"harness": "revision 3 (E22 corrected; threads-after-load, inference_mode, fp32)",
              "rows": 750, "source": "data/e20-contrast/test_rows.jsonl",
              "models": out, "ratios": {}}
    for bs in ("batch1", "batch16"):
        j = sorted(out["julia"][bs]["full_pass_s"])[len(out["julia"][bs]["full_pass_s"]) // 2]
        m = sorted(out["mokka1"][bs]["full_pass_s"])[len(out["mokka1"][bs]["full_pass_s"]) // 2]
        jp = sorted(out["julia"][bs]["pretok_pass_s"])[len(out["julia"][bs]["pretok_pass_s"]) // 2]
        mp = sorted(out["mokka1"][bs]["pretok_pass_s"])[len(out["mokka1"][bs]["pretok_pass_s"]) // 2]
        n = out["julia"]["rows"]
        report["ratios"][bs] = {
            "julia_full_median_s": round(j, 3), "mokka1_full_median_s": round(m, 3),
            "julia_pretok_median_s": round(jp, 3), "mokka1_pretok_median_s": round(mp, 3),
            "full_ratio_julia_over_mokka": round(j / m, 4) if m else None,
            "pretok_ratio_julia_over_mokka": round(jp / mp, 4) if mp else None,
            "mokka1_qps_full": round(n / m, 2) if m else None,
            "julia_qps_full": round(n / j, 2) if j else None,
            "mokka1_qps_pretok": round(n / mp, 2) if mp else None,
            "julia_qps_pretok": round(n / jp, 2) if jp else None}
    # per-item latency from b1 raw durations (individual, not amortized)
    jb1 = [r["duration_s"] * 1000 for r in out["julia"]["batch1"]["raw_items"]]
    mb1 = [r["duration_s"] * 1000 for r in out["mokka1"]["batch1"]["raw_items"]]
    report["ratios"]["b1_item_ms"] = {
        "julia_p50": round(statistics.median(jb1), 2), "mokka1_p50": round(statistics.median(mb1), 2),
        "ratio_julia_over_mokka": round(statistics.median(jb1) / statistics.median(mb1), 4)
        if statistics.median(mb1) else None}
    p = ROOT / "runs/e22-efficiency-v3/efficiency_v3_report.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report["ratios"], indent=1))
    print(f"report: {p}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("julia", "mokka1"))
    parser.add_argument("--merge", action="store_true")
    args = parser.parse_args()
    if args.model:
        result = measure(args.model)
        out = ROOT / f"runs/e22-efficiency-v3/{args.model}_measurement.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        slim = {k: v for k, v in result.items()}
        out.write_text(json.dumps(slim, indent=2, sort_keys=True), encoding="utf-8")
        print(f"[v3:{args.model}] load={result['load_s']}s threads={result['threads']} "
              f"verify={result['verification']}")
        return 0
    if args.merge:
        return merge()
    parser.error("choose --model or --merge")


if __name__ == "__main__":
    sys.exit(main())
