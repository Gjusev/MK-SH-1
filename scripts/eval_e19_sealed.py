"""Evaluate E19 arms and official Julia on the SEALED test (opened once).

Protocol: docs/internal/E19_PROTOCOL_2026-10-09.md. Verifies the sealed-test
hashes against runs/e19-prep/test_freeze.json BEFORE loading any model.
Our arms: JuliaArch-small checkpoints, CPU float32, T=1, max_length 1024,
head_length 256, batch 16 (same conventions as the E17/E18 evaluators).
Julia: pinned upstream engine (runs/julia-research/upstream-source), CPU,
predict per question with its own state (variants differ per question).

Usage: python scripts/eval_e19_sealed.py [--run runs/kaggle-typed-e19]
                                        [--out runs/e19-sealed-eval]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

DATA = ROOT / "data/e19-rulevar"
FREEZE = ROOT / "runs/e19-prep/test_freeze.json"
JULIA_CKPT = ROOT / "runs/julia-research/upstream-source"
ARMS = [f"e19_{recipe}_s{seed}" for seed in (42, 43) for recipe in ("orig", "ctrl", "cand")]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_test_rows() -> list[dict]:
    rows = [json.loads(l) for l in (DATA / "test_rows.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(rows) == 750
    for r in rows:
        r["gold"] = next(i for i, o in enumerate(r["options"]) if o["id"] == r["target_option"])
    return rows


def row_metrics(pv: list[float], gold: int, qtype: str) -> dict:
    from mokka_decisions.arch_julia import row_metrics as rm

    return rm(pv, gold, qtype)


def evaluate_ours(arm: str, ckpt: Path, rows: list[dict], device: str) -> list[dict]:
    import torch
    from mokka_decisions.arch_julia import JuliaArchScorer, collate_julia
    from safetensors.torch import load_file
    from transformers import AutoTokenizer

    model = JuliaArchScorer("jhu-clsp/mmBERT-small")
    model.load_state_dict(load_file(str(ckpt)))
    model.to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained("jhu-clsp/mmBERT-small")
    qtypes = {r["id"]: r["question_type"] for r in rows}
    out = []
    with torch.inference_mode():
        for s in range(0, len(rows), 16):
            chunk = rows[s:s + 16]
            from mokka_decisions.contracts import DecisionCase, Option

            cases = [DecisionCase(id=r["id"], group_id=r["group_id"], source="e19", language="en",
                                  domain=r["domain"], state=r["state"], question=r["question"],
                                  options=tuple(Option(id=o["id"], description=o["description"]) for o in r["options"]),
                                  target_kind="single", target_option=r["target_option"])
                     for r in chunk]
            batch = collate_julia(tokenizer, cases, qtypes, 1024, 256, device)
            logits = model(**{k: batch[k] for k in
                              ("input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype")}).float()
            probs = logits.softmax(-1)
            for r, case, pv_t in zip(chunk, cases, probs):
                k = len(r["options"])
                pv = pv_t[:k].tolist()
                pred = max(range(k), key=pv.__getitem__)
                m = row_metrics(pv, r["gold"], r["question_type"])
                out.append({"id": r["id"], "group_id": r["group_id"], "family": r["domain"],
                            "type": r["question_type"], "variant": r.get("variant"), "pair": r.get("pair"),
                            "gold_idx": r["gold"], "pred_idx": pred, "probabilities": pv,
                            "correct": int(pred == r["gold"]), **m})
    return out


def evaluate_julia(rows: list[dict]) -> list[dict]:
    if str(JULIA_CKPT) not in sys.path:
        sys.path.insert(0, str(JULIA_CKPT))
    from julia.inference import TransformerEngine
    from julia.typed import predict_typed

    engine = TransformerEngine(str(JULIA_CKPT), device="cpu")
    out = []
    for r in rows:
        keys = [o["id"] for o in r["options"]]
        if r["question_type"] == "noul":
            crit = {"false": r["options"][0]["description"], "true": r["options"][1]["description"]}
        elif r["question_type"] == "score":
            crit = [o["description"] for o in r["options"]]
        else:
            crit = {o["id"]: o["description"] for o in r["options"]}
        q = {"instructions": r["question"], "type": r["question_type"], "criteria": crit}
        ans = predict_typed(engine, r["state"], {"q": q})["answers"]["q"]
        ans_keys = [str(i) for i in range(len(keys))] if r["question_type"] == "score" else keys
        pv = [ans["probabilities"][k] for k in ans_keys]
        pred = max(range(len(pv)), key=pv.__getitem__)
        m = row_metrics(pv, r["gold"], r["question_type"])
        out.append({"id": r["id"], "group_id": r["group_id"], "family": r["domain"],
                    "type": r["question_type"], "variant": r.get("variant"), "pair": r.get("pair"),
                    "gold_idx": r["gold"], "pred_idx": pred, "probabilities": pv,
                    "correct": int(pred == r["gold"]), **m})
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="runs/kaggle-typed-e19")
    parser.add_argument("--out", default="runs/e19-sealed-eval")
    args = parser.parse_args()
    run, out = ROOT / args.run, ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)

    freeze = json.loads(FREEZE.read_text(encoding="utf-8"))
    for name, expect in freeze["files_sha256"].items():
        if name.startswith("test"):
            actual = sha256(DATA / name)
            assert actual == expect, f"sealed test changed: {name}"
    print("sealed-test hashes match the freeze")

    rows = load_test_rows()
    device = "cpu"
    for arm in ARMS:
        ckpt = run / arm / "checkpoint_best.safetensors"
        if not ckpt.exists():
            raise SystemExit(f"missing checkpoint: {ckpt}")
        if (out / f"{arm}_percase.jsonl").exists():
            print(f"{arm}: cached")
            continue
        per = evaluate_ours(arm, ckpt, rows, device)
        with (out / f"{arm}_percase.jsonl").open("w", encoding="utf-8", newline="\n") as f:
            for r in per:
                f.write(json.dumps(r) + "\n")
        acc = sum(r["correct"] for r in per) / len(per)
        print(f"{arm}: acc={acc:.4f} n={len(per)} sha={sha256(ckpt)[:12]}")
    julia = evaluate_julia(rows)
    with (out / "julia_percase.jsonl").open("w", encoding="utf-8", newline="\n") as f:
        for r in julia:
            f.write(json.dumps(r) + "\n")
    print(f"julia: acc={sum(r['correct'] for r in julia) / len(julia):.4f} n={len(julia)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
