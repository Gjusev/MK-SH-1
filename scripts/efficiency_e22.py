"""E22 efficiency v2: genuinely homogeneous Mokka vs Julia comparison (CPU).

Fixes every defect from the E21 review:
- TWO scenarios for ALL models: batch 1 (true individual latency) and batch 16
  (batch DURATION - never divided to fake a latency; last batch's real size
  is recorded and used for throughput).
- Julia batched via TransformerEngine.logits(rows) with PER-ROW states.
- Identical timing boundaries for both: pre-tokenized phase times the forward
  pass on already-collated inputs only; full phase times collate+forward.
  Prediction reconstruction (softmax/argmax) is OUTSIDE the timed region for
  both models.
- Predictions verified against the saved official evaluation before a
  measurement is accepted.
- Raw per-item durations, batch sizes, config and execution order are saved.
- Memory: reported as "not measured" (process-RSS syscalls fail here); the
  weight-file size is reported separately, labelled as file size.

Usage (one model per process): python scripts/efficiency_e22.py --model X
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

ROWS = ROOT / "data/e20-contrast/test_rows.jsonl"
PERCASE = ROOT / "runs/e20-sealed-eval"
THREADS = 8
BATCH = 16
MAXLEN = 512
HEADLEN = 256
WARMUP = 1
TIMED = 2


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def pct(vals, p):
    s = sorted(vals)
    return s[min(len(s) - 1, int(p / 100 * len(s)))]


def julia_rows(rows):
    out = []
    for r in rows:
        out.append({"state": r["state"], "question": r["question"],
                    "type": r["question_type"],
                    "options": [o["description"] for o in r["options"]]})
    return out


def argmax_from_logits(per_row_logits, rows):
    preds = {}
    for r, values in zip(rows, per_row_logits):
        pv = values[: len(r["options"])]
        preds[r["id"]] = max(range(len(pv)), key=pv.__getitem__)
    return preds


def load_julia():
    ckpt = str(ROOT / "runs/julia-research/upstream-source")
    if ckpt not in sys.path:
        sys.path.insert(0, ckpt)
    from julia.inference import TransformerEngine
    return TransformerEngine(ckpt, device="cpu", max_length=MAXLEN, head_length=HEADLEN)


def load_mokka(arm):
    from mokka_decisions.arch_julia import JuliaArchScorer
    from safetensors.torch import load_file
    model = JuliaArchScorer("jhu-clsp/mmBERT-small")
    model.load_state_dict(load_file(str(ROOT / "runs/kaggle-typed-e20" / arm / "checkpoint_best.safetensors")))
    model.to("cpu").eval()
    return model


def run(model_name, rows):
    import torch
    torch.set_num_threads(THREADS)
    t0 = time.perf_counter()
    if model_name == "julia":
        engine = load_julia()
        from julia.cuda import move

        def collate_fn(idxs):
            batch_rows = [{"state": rows[i]["state"], "question": rows[i]["question"],
                           "type": rows[i]["question_type"],
                           "options": [o["description"] for o in rows[i]["options"]]} for i in idxs]
            return move(engine.collate(batch_rows, include_targets=False), engine.device)

        def forward_fn(collated):
            logits = engine.model(**collated)
            if not torch.isfinite(logits).all():
                raise FloatingPointError("nonfinite")
            return logits.cpu()

        def preds_from(out, idxs):
            preds = {}
            for i, values in zip(idxs, out):
                pv = values[: len(rows[i]["options"])].tolist()
                preds[rows[i]["id"]] = max(range(len(pv)), key=pv.__getitem__)
            return preds
        weights_path = "runs/julia-research/upstream-source/model.safetensors"
    else:
        arm = model_name.replace("mokka_", "e20_")
        model = load_mokka(arm)
        from mokka_decisions.arch_julia import collate_julia
        from mokka_decisions.contracts import DecisionCase, Option
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("jhu-clsp/mmBERT-small")
        qtypes = {r["id"]: r["question_type"] for r in rows}
        cases = [DecisionCase(id=r["id"], group_id=r["group_id"], source="e", language="en", domain=r["domain"],
                              state=r["state"], question=r["question"],
                              options=tuple(Option(id=o["id"], description=o["description"]) for o in r["options"]),
                              target_kind="single", target_option=r["target_option"]) for r in rows]

        def collate_fn(idxs):
            return collate_julia(tok, [cases[i] for i in idxs], qtypes, MAXLEN, HEADLEN, "cpu")

        def forward_fn(collated):
            return model(**{k: collated[k] for k in
                            ("input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype")}).float().cpu()

        def preds_from(out, idxs):
            preds = {}
            for i, values in zip(idxs, out):
                pv = values[: len(cases[i].options)].tolist()
                preds[cases[i].id] = max(range(len(pv)), key=pv.__getitem__)
            return preds
        weights_path = f"runs/kaggle-typed-e20/{arm}/checkpoint_best.safetensors"
    load_s = time.perf_counter() - t0
    n = len(rows)

    def scenario(batch_size):
        batches = [list(range(i, min(i + batch_size, n))) for i in range(0, n, batch_size)]
        collated = [collate_fn(b) for b in batches]
        preds = {}
        for b, c in zip(batches, collated):
            preds.update(preds_from(forward_fn(c), b))
        for _ in range(WARMUP):
            for c in collated:
                forward_fn(c)
        pre_tok = []
        for _ in range(TIMED):
            t = time.perf_counter()
            for c in collated:
                forward_fn(c)
            pre_tok.append(time.perf_counter() - t)
        for _ in range(WARMUP):
            for b in batches:
                forward_fn(collate_fn(b))
        full = []
        for _ in range(TIMED):
            t = time.perf_counter()
            for b in batches:
                forward_fn(collate_fn(b))
            full.append(time.perf_counter() - t)
        raw_items = []
        for b in batches:
            t = time.perf_counter()
            forward_fn(collate_fn(b))
            raw_items.append({"duration_s": time.perf_counter() - t, "size": len(b)})
        return {"batch_size": batch_size, "batch_count": len(batches),
                "real_last_batch": len(batches[-1]),
                "pretok_pass_s": pre_tok, "full_pass_s": full,
                "raw_items": raw_items, "preds": preds}

    b1 = scenario(1)
    b16 = scenario(BATCH)
    ref_file = PERCASE / ("julia_percase.jsonl" if model_name == "julia" else f"{model_name.replace('mokka_', 'e20_')}_percase.jsonl")
    ref = {json.loads(l)["id"]: json.loads(l)["pred_idx"] for l in ref_file.read_text(encoding="utf-8").splitlines() if l.strip()}
    mism = {i for i in ref if ref[i] != b1["preds"].get(i)} | {i for i in ref if ref[i] != b16["preds"].get(i)}
    return {"model": model_name, "load_s": load_s, "threads": THREADS, "max_length": MAXLEN,
            "batch1": b1, "batch16": b16, "mismatches": len(mism), "weights_path": weights_path}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=["julia", "mokka_ctrl_s43", "mokka_cand_s43"])
    args = parser.parse_args()
    rows = [json.loads(l) for l in ROWS.read_text(encoding="utf-8").splitlines() if l.strip()]
    out = run(args.model, rows)
    if out["mismatches"]:
        print(json.dumps({"model": args.model, "REJECTED": True, "mismatches": out["mismatches"]}))
        return 1
    wp = out.pop("weights_path")
    out["weights_file"] = {"path": wp, "bytes": Path(ROOT / wp).stat().st_size,
                           "sha256": sha256(ROOT / wp)}
    out["memory"] = "not measured (process RSS syscalls fail in this environment)"
    def summary(sc):
        durs1 = [x["duration_s"] for x in sc["raw_items"]]
        passes = sc["full_pass_s"]
        n = len(rows)
        return {"item_p50_s": pct(durs1, 50), "item_p95_s": pct(durs1, 95),
                "pretok_per_item_s": (sum(sc["pretok_pass_s"]) / len(sc["pretok_pass_s"])) / n,
                "throughput_q_per_s": n / (sum(passes) / len(passes)),
                "unit": f"per-item raw durations at batch {sc['batch_size']} (item = one call for b1, one BATCH for b16)"}
    out["summary_batch1"] = summary(out["batch1"])
    out["summary_batch16"] = summary(out["batch16"])
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
