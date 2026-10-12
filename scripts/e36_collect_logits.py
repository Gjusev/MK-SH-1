"""E36 logits collection: pass MK-SH-1 v1.1 (weights unchanged) over the
e36-calib split + the frozen eval sets, CPU float32, saving logits per case.

Guards (criteria docs/E36_CRITERIA_2026-10-12.md):
- checkpoint sha256 verified against the frozen literal (v1.1 = ea66cdde…)
- e36-calib manifest: generator sha + rows sha verified before anything runs
- float32 CPU, 8 threads after model load, torch.inference_mode

Outputs: runs/e36-calib/logits/{split,known,new,mono,e,task}.jsonl
(+ artifact_hashes.json). Deterministic order; logits saved raw (float32 ->
python float).
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CKPT = ROOT / "runs/hf-export/mokka-1-v1.1/model.safetensors"
CKPT_SHA = "ea66cdde2271b98efda9dbb5d4437b59aa422cb34daca85d4203508aa56264d5"
OUT = ROOT / "runs/e36-calib/logits"
SETS = (
    ("split", ("data/e36-calib", "rows.jsonl")),
    ("known", ("data/e26-order", "test_c_rows.jsonl")),
    ("new", ("data/e26-order", "test_d_rows.jsonl")),
    ("mono", ("data/e26-order", "mono_rows.jsonl")),
    ("e", ("data/e27-grid", "test_e_rows.jsonl")),
)


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def load_task_rows() -> list[dict]:
    """Task-E29 slice: 300 dev rows by frozen id list (e29-mix manifest)."""
    manifest = json.loads((ROOT / "data/e29-mix/manifest.json").read_text(encoding="utf-8"))
    want = set(manifest["original_dev_ids"])
    path = ROOT / "data/processed/v1/eval/dev_eval.jsonl"
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    picked = [r for r in rows if r["id"] in want]
    if len(picked) != len(want):
        raise SystemExit(f"task slice mismatch: {len(picked)}/{len(want)} from {path}")
    return picked


def main() -> int:
    import torch

    calib_manifest = json.loads((ROOT / "data/e36-calib/manifest.json").read_text(encoding="utf-8"))
    if calib_manifest["generator_sha256"] != sha256(ROOT / "scripts/gen_e36_calib.py"):
        raise SystemExit("e36-calib manifest generator sha mismatch vs committed script")
    if calib_manifest["rows_sha256"] != sha256(ROOT / "data/e36-calib/rows.jsonl"):
        raise SystemExit("e36-calib rows.jsonl sha mismatch vs manifest")
    got = sha256(CKPT)
    if got != CKPT_SHA:
        raise SystemExit(f"checkpoint sha mismatch: {got}")

    torch.set_num_threads(8)
    from mokka_decisions.arch_julia import JuliaArchScorer, collate_julia
    from mokka_decisions.contracts import DecisionCase
    from safetensors.torch import load_file
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("jhu-clsp/mmBERT-small")
    model = JuliaArchScorer("jhu-clsp/mmBERT-small")
    model.load_state_dict(load_file(str(CKPT)), strict=True)
    model.eval()

    sets = {name: [json.loads(l) for l in (ROOT / d / f).read_text(encoding="utf-8").splitlines()
                   if l.strip()] for name, (d, f) in SETS}
    sets["task"] = load_task_rows()

    OUT.mkdir(parents=True, exist_ok=True)
    hashes = {"checkpoint": CKPT_SHA, "torch": torch.__version__,
              "threads": torch.get_num_threads(), "device": "cpu", "dtype": "float32",
              "e36cal_rows_sha256": calib_manifest["rows_sha256"]}
    with torch.inference_mode():
        for name, rows in sets.items():
            cases = [DecisionCase.from_dict(r) for r in rows]
            qtypes = {r["id"]: r.get("question_type") or "choice" for r in rows}
            out_rows = []
            for s in range(0, len(cases), 16):
                chunk = cases[s:s + 16]
                batch = collate_julia(tokenizer, chunk, qtypes, 1024, 256, "cpu")
                logits = model(**{k: batch[k] for k in
                                  ("input_ids", "attention_mask", "marker_pos", "marker_mask",
                                   "qtype")}).float()
                for r, case in enumerate(chunk):
                    k = len(case.options)
                    out_rows.append({"id": case.id, "type": qtypes[case.id],
                                     "gold_idx": int(batch["targets"][r].item()),
                                     "k": k,
                                     "logits": logits[r, :k].tolist()})
            path = OUT / f"{name}.jsonl"
            path.write_text("".join(json.dumps(r) + "\n" for r in out_rows), encoding="utf-8")
            hashes[f"{name}_sha256"] = sha256(path)
            hashes[f"{name}_n"] = len(out_rows)
            print(f"{name}: {len(out_rows)} rows -> {path}")

    (OUT / "artifact_hashes.json").write_text(json.dumps(hashes, indent=2, sort_keys=True),
                                              encoding="utf-8")
    print("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
