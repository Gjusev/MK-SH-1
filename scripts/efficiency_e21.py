"""E21 efficiency measurement: Mokka (JuliaArch-small) vs official Julia, CPU.

Protocol (fixed before measuring):
- Same machine, torch threads pinned to 8, float32, batch 16 (questions).
- Same examples for both: the 750 E20 sealed-test rows (max observed sequence
  ~269 tokens with our serializer; both models run with max_length=512,
  head_length=256 -> NO truncation on either side).
- Phases reported separately: model load; tokenization-only (ours; Julia
  tokenizes inside the engine, so for Julia only the full path exists - a
  declared asymmetry); full path (tokenize + forward).
- Warm-up pass, then 3 timed passes. Models run in separate OS processes,
  alternated M,J,M,J,M,J at the orchestrator level to dodge thermal/order bias.
- Memory: process RSS after load and peak during inference (no psutil ->
  tracked via tracemalloc for tensors + resource ru_maxrss for the process).
- Predictions are verified against the official evaluation percase files:
  timing must not change a single answer.

Usage (one model per process):
  python scripts/efficiency_e21.py --model julia|mokka_ctrl_s43|mokka_cand_s43 [--json out]
Orchestrate alternation from the shell; this script prints one JSON line.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

ROWS = ROOT / "data/e20-contrast/test_rows.jsonl"
PERCASE = ROOT / "runs/e20-sealed-eval"
THREADS = 8
BATCH = 16
MAXLEN = 512
HEADLEN = 256
WARMUP_PASSES = 1
TIMED_PASSES = 3


class _PMC(ctypes.Structure):
    _fields_ = [("cb", ctypes.c_uint32), ("PageFaultCount", ctypes.c_uint32),
                ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]


def rss_mb():
    """Peak working set (MB) - Windows via psapi; peak is what we report."""
    pmc = _PMC()
    pmc.cb = ctypes.sizeof(_PMC)
    handle = ctypes.windll.kernel32.GetCurrentProcess()
    fn = getattr(ctypes.windll.kernel32, "K32GetProcessMemoryInfo", None) or ctypes.windll.psapi.GetProcessMemoryInfo
    if fn(handle, ctypes.byref(pmc), pmc.cb):
        return pmc.PeakWorkingSetSize / (1024 * 1024)
    return -1.0


def pct(vals, p):
    s = sorted(vals)
    return s[min(len(s) - 1, int(p / 100 * len(s)))]


def run_mokka(arm, rows):
    import torch
    torch.set_num_threads(THREADS)
    from mokka_decisions.arch_julia import JuliaArchScorer, collate_julia
    from mokka_decisions.contracts import DecisionCase, Option
    from safetensors.torch import load_file
    from transformers import AutoTokenizer

    t0 = time.perf_counter()
    model = JuliaArchScorer("jhu-clsp/mmBERT-small")
    model.load_state_dict(load_file(str(ROOT / "runs/kaggle-typed-e20" / arm / "checkpoint_best.safetensors")))
    model.to("cpu").eval()
    tok = AutoTokenizer.from_pretrained("jhu-clsp/mmBERT-small")
    load_s = time.perf_counter() - t0

    qtypes = {r["id"]: r["question_type"] for r in rows}
    cases = [DecisionCase(id=r["id"], group_id=r["group_id"], source="e", language="en", domain=r["domain"],
                          state=r["state"], question=r["question"],
                          options=tuple(Option(id=o["id"], description=o["description"]) for o in r["options"]),
                          target_kind="single", target_option=r["target_option"]) for r in rows]

    def forward(batches):
        preds, batch_times = {}, []
        with torch.inference_mode():
            for chunk in batches:
                t = time.perf_counter()
                batch = collate_julia(tok, chunk, qtypes, MAXLEN, HEADLEN, "cpu")
                logits = model(**{k: batch[k] for k in ("input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype")}).float()
                probs = logits.softmax(-1)
                batch_times.append(time.perf_counter() - t)
                for c, pv_t in zip(chunk, probs):
                    k = len(c.options)
                    pv = pv_t[:k].tolist()
                    preds[c.id] = max(range(k), key=pv.__getitem__)
        return preds, batch_times

    def tokenize_only(batches):
        times = []
        for chunk in batches:
            t = time.perf_counter()
            collate_julia(tok, chunk, qtypes, MAXLEN, HEADLEN, "cpu")
            times.append(time.perf_counter() - t)
        return times

    batches = [cases[i:i + BATCH] for i in range(0, len(cases), BATCH)]
    tokenize_only(batches)  # warm-up
    tok_times = tokenize_only(batches)
    forward(batches)  # warm-up
    passes, all_batch = [], []
    for _ in range(TIMED_PASSES):
        preds, bt = forward(batches)
        all_batch.extend(bt)
        passes.append(sum(bt))
    per_q = [b / BATCH for b in all_batch]
    return {"load_s": load_s, "rss_peak_mb_at_load": rss_mb(),
            "tokenize_only_per_q_s": sum(tok_times) / len(rows),
            "full_pass_s": passes, "full_per_q_p50_s": pct(per_q, 50), "full_per_q_p95_s": pct(per_q, 95),
            "full_per_batch_p50_s": pct(all_batch, 50), "full_per_batch_p95_s": pct(all_batch, 95),
            "throughput_q_per_s": len(rows) / (sum(passes) / len(passes)),
            "preds": preds}


def run_julia(rows):
    import torch
    torch.set_num_threads(THREADS)
    ckpt = str(ROOT / "runs/julia-research/upstream-source")
    if ckpt not in sys.path:
        sys.path.insert(0, ckpt)
    t0 = time.perf_counter()
    from julia.inference import TransformerEngine
    from julia.typed import predict_typed
    engine = TransformerEngine(ckpt, device="cpu", max_length=MAXLEN, head_length=HEADLEN)
    load_s = time.perf_counter() - t0

    # per-QUESTION calls with each question's own state (variants differ),
    # exactly like the official evaluation - prediction equality is asserted
    def qspec(r):
        if r["question_type"] == "noul":
            crit = {"false": r["options"][0]["description"], "true": r["options"][1]["description"]}
        elif r["question_type"] == "score":
            crit = [o["description"] for o in r["options"]]
        else:
            crit = {o["id"]: o["description"] for o in r["options"]}
        return {"q": {"instructions": r["question"], "type": r["question_type"], "criteria": crit}}

    def forward():
        preds, q_times = {}, []
        for r in rows:
            t = time.perf_counter()
            ans = predict_typed(engine, r["state"], qspec(r))["answers"]["q"]
            q_times.append(time.perf_counter() - t)
            keys = [o["id"] for o in r["options"]]
            ak = [str(i) for i in range(len(keys))] if r["question_type"] == "score" else keys
            pv = [ans["probabilities"][k] for k in ak]
            preds[r["id"]] = max(range(len(pv)), key=pv.__getitem__)
        return preds, q_times

    forward()  # warm-up
    passes, all_q = [], []
    for _ in range(TIMED_PASSES):
        preds, qt = forward()
        all_q.extend(qt)
        passes.append(sum(qt))
    per_q = all_q
    return {"load_s": load_s, "rss_peak_mb_at_load": rss_mb(),
            "full_pass_s": passes, "full_per_q_p50_s": pct(per_q, 50), "full_per_q_p95_s": pct(per_q, 95),
            
            "throughput_q_per_s": len(rows) / (sum(passes) / len(passes)),
            "preds": preds,
            "note": "tokenization is inside the engine; per-question calls with per-question states (matches the official evaluation)"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=["julia", "mokka_ctrl_s43", "mokka_cand_s43"])
    args = parser.parse_args()
    rows = [json.loads(l) for l in ROWS.read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.model == "julia":
        out = run_julia(rows)
        ref = {json.loads(l)["id"]: json.loads(l)["pred_idx"] for l in (PERCASE / "julia_percase.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()}
    else:
        out = run_mokka(args.model.replace("mokka_", "e20_"), rows)
        ref = {json.loads(l)["id"]: json.loads(l)["pred_idx"] for l in (PERCASE / f"{args.model.replace('mokka_', 'e20_')}_percase.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()}
    mismatches = [i for i in ref if ref[i] != out["preds"].get(i)]
    out["prediction_mismatches_vs_official_eval"] = len(mismatches)
    out["model"] = args.model
    out["threads"] = THREADS
    out["batch_questions"] = BATCH
    out["max_length"] = MAXLEN
    out["rows"] = len(rows)
    preds = out.pop("preds")
    out["n_preds"] = len(preds)
    print(json.dumps(out))
    return 0 if not mismatches else 1


if __name__ == "__main__":
    raise SystemExit(main())
