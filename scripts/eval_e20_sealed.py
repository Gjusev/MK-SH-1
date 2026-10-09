"""Evaluate E20 arms and official Julia on the SEALED E20 test (opened once).

Protocol: docs/internal/E20_PROTOCOL_2026-10-09.md. Verifies sealed hashes
against runs/e20-prep/test_freeze.json BEFORE loading models. Same
conventions as the E19 evaluator (CPU float32, T=1, 1024/256, batch 16;
Julia via the pinned upstream engine, per-question states).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

DATA = ROOT / "data/e20-contrast"
FREEZE = ROOT / "runs/e20-prep/test_freeze.json"
JULIA_CKPT = ROOT / "runs/julia-research/upstream-source"
ARMS = [f"e20_{r}_s{s}" for s in (42, 43, 44) for r in ("ctrl", "cand")]


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


def evaluate_ours(arm: str, ckpt: Path, rows: list[dict], device="cpu"):
    import torch
    from mokka_decisions.arch_julia import JuliaArchScorer, collate_julia, row_metrics
    from mokka_decisions.contracts import DecisionCase, Option
    from safetensors.torch import load_file
    from transformers import AutoTokenizer

    model = JuliaArchScorer("jhu-clsp/mmBERT-small")
    model.load_state_dict(load_file(str(ckpt)))
    model.to(device).eval()
    tok = AutoTokenizer.from_pretrained("jhu-clsp/mmBERT-small")
    qtypes = {r["id"]: r["question_type"] for r in rows}
    out = []
    with torch.inference_mode():
        for s in range(0, len(rows), 16):
            chunk = rows[s:s + 16]
            cases = [DecisionCase(id=r["id"], group_id=r["group_id"], source="e20", language="en",
                                  domain=r["domain"], state=r["state"], question=r["question"],
                                  options=tuple(Option(id=o["id"], description=o["description"]) for o in r["options"]),
                                  target_kind="single", target_option=r["target_option"])
                     for r in chunk]
            batch = collate_julia(tok, cases, qtypes, 1024, 256, device)
            logits = model(**{k: batch[k] for k in
                              ("input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype")}).float()
            probs = logits.softmax(-1)
            for r, pv_t in zip(chunk, probs):
                k = len(r["options"])
                pv = pv_t[:k].tolist()
                pred = max(range(k), key=pv.__getitem__)
                m = row_metrics(pv, r["gold"], r["question_type"])
                out.append({"id": r["id"], "group_id": r["group_id"], "family": r["domain"],
                            "type": r["question_type"], "variant": r.get("variant"), "pair": r.get("pair"),
                            "gold_idx": r["gold"], "pred_idx": pred, "probabilities": pv,
                            "correct": int(pred == r["gold"]), **m})
    return out


def evaluate_julia(rows: list[dict]):
    ckpt = str(JULIA_CKPT)
    if ckpt not in sys.path:
        sys.path.insert(0, ckpt)
    from julia.inference import TransformerEngine
    from julia.typed import predict_typed

    engine = TransformerEngine(ckpt, device="cpu")
    out = []
    for r in rows:
        if r["question_type"] == "noul":
            crit = {"false": r["options"][0]["description"], "true": r["options"][1]["description"]}
        elif r["question_type"] == "score":
            crit = [o["description"] for o in r["options"]]
        else:
            crit = {o["id"]: o["description"] for o in r["options"]}
        q = {"instructions": r["question"], "type": r["question_type"], "criteria": crit}
        ans = predict_typed(engine, r["state"], {"q": q})["answers"]["q"]
        keys = [o["id"] for o in r["options"]]
        ans_keys = [str(i) for i in range(len(keys))] if r["question_type"] == "score" else keys
        pv = [ans["probabilities"][k] for k in ans_keys]
        pred = max(range(len(pv)), key=pv.__getitem__)
        from mokka_decisions.arch_julia import row_metrics
        m = row_metrics(pv, r["gold"], r["question_type"])
        out.append({"id": r["id"], "group_id": r["group_id"], "family": r["domain"],
                    "type": r["question_type"], "variant": r.get("variant"), "pair": r.get("pair"),
                    "gold_idx": r["gold"], "pred_idx": pred, "probabilities": pv,
                    "correct": int(pred == r["gold"]), **m})
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="runs/kaggle-typed-e20")
    parser.add_argument("--out", default="runs/e20-sealed-eval")
    args = parser.parse_args()
    run, out = ROOT / args.run, ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    freeze = json.loads(FREEZE.read_text(encoding="utf-8"))
    for name, expect in freeze["files_sha256"].items():
        if name.startswith("test"):
            assert sha256(DATA / name) == expect, f"sealed test changed: {name}"
    print("sealed-test hashes match the freeze")
    rows = load_test_rows()
    for arm in ARMS:
        ckpt = run / arm / "checkpoint_best.safetensors"
        if not ckpt.exists():
            raise SystemExit(f"missing checkpoint: {ckpt}")
        if (out / f"{arm}_percase.jsonl").exists():
            print(f"{arm}: cached")
            continue
        per = evaluate_ours(arm, ckpt, rows)
        (out / f"{arm}_percase.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in per), encoding="utf-8")
        print(f"{arm}: acc={sum(r['correct'] for r in per) / len(per):.4f} sha={sha256(ckpt)[:12]}")
    if not (out / "julia_percase.jsonl").exists():
        julia = evaluate_julia(rows)
        (out / "julia_percase.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in julia), encoding="utf-8")
    julia = [json.loads(l) for l in (out / "julia_percase.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    print(f"julia: acc={sum(r['correct'] for r in julia) / len(julia):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
