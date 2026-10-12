"""Assemble the MK-SH-1 v1.2 export: v1.1 weights BYTE-IDENTICAL + tokenizer
+ inference-policy.json + card (docs/E36_CRITERIA_2026-10-12.md section 4).

v1.2 = v1.1 weights + temperature + abstention policy. NO retraining; not a
single weight changes. --verify (default on) then loads the package in a
FRESH isolated interpreter from the export directory ONLY and checks:
- choices identical to raw (T=1) on a smoke slice of the calibration split
- calibrated confidences differ from raw
- abstention fires exactly when max calibrated prob < tau (reason_code
  below_validated_threshold, choice null)

Usage: python scripts/export_mksh_v12.py [--no-verify]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "runs/hf-export/mokka-1-v1.1"
DST = ROOT / "runs/hf-export/mksh-1-v1.2"
WEIGHTS_SHA = "ea66cdde2271b98efda9dbb5d4437b59aa422cb34daca85d4203508aa56264d5"
COPY = ("model.safetensors", "tokenizer.json", "tokenizer_config.json", "ARCHITECTURE.md")

VERIFY = r"""
import json, math, sys
from pathlib import Path
d = Path(sys.argv[1])
policy = json.loads((d / "inference-policy.json").read_text(encoding="utf-8"))
T = policy["temperature"]["value"]
tau = policy["abstention"]["threshold"]
assert policy["abstention"]["enabled"] and tau is not None, "policy disabled"

from mokka_decisions.arch_julia import JuliaArchScorer, collate_julia
from mokka_decisions.contracts import DecisionCase
from mokka_decisions.policy import apply_policy
from safetensors.torch import load_file
from transformers import AutoTokenizer
import torch

torch.set_num_threads(8)
tok = AutoTokenizer.from_pretrained(str(d))  # tokenizer from the package ONLY
model = JuliaArchScorer("jhu-clsp/mmBERT-small")
model.load_state_dict(load_file(str(d / "model.safetensors")), strict=True)
model.eval()

meta = [json.loads(l) for l in (Path(sys.argv[2]) / "data/e36-calib/rows.jsonl")
        .read_text(encoding="utf-8").splitlines() if l.strip()][:24]
cases = [DecisionCase.from_dict(r) for r in meta]
qtypes = {r["id"]: r.get("question_type") or "choice" for r in meta}
n_abst = 0
with torch.inference_mode():
    for s in range(0, len(cases), 16):
        chunk = cases[s:s+16]
        batch = collate_julia(tok, chunk, qtypes, 1024, 256, "cpu")
        logits = model(**{k: batch[k] for k in
            ("input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype")}).float()
        for r, case in enumerate(chunk):
            lg = logits[r, :len(case.options)].tolist()
            raw = [math.exp(x) for x in lg]; zr = sum(raw); raw = [x/zr for x in raw]
            cal = [math.exp(x / T) for x in lg]; zc = sum(cal); cal = [x/zc for x in cal]
            assert max(range(len(raw)), key=lambda i: raw[i]) == \
                   max(range(len(cal)), key=lambda i: cal[i]), case.id  # identical choice
            assert abs(raw[0] - cal[0]) > 1e-9, case.id                # confidence moved
            probs = {opt.id: p for opt, p in zip(case.options, cal)}
            resp = apply_policy(case, probs, threshold=tau)
            mx = max(probs.values())
            assert resp.abstain == (mx < tau), case.id                  # policy consistent
            if resp.abstain:
                n_abst += 1
                assert resp.choice is None and \
                    resp.reason_code == "below_validated_threshold", case.id
print(f"VERIFY OK: {len(cases)} cases, choices identical raw/calibrated, "
      f"confidences calibrated, {n_abst} abstained (reason below_validated_threshold)")
"""


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-verify", action="store_true")
    args = parser.parse_args()
    if DST.exists():
        raise SystemExit(f"Refusing to overwrite existing export: {DST}")
    if sha256(SRC / "model.safetensors") != WEIGHTS_SHA:
        raise SystemExit("v1.1 weights sha mismatch — abort before export")

    DST.mkdir(parents=True)
    hashes = {}
    for name in COPY:
        shutil.copy2(SRC / name, DST / name)
        hashes[name] = sha256(DST / name)
    if hashes["model.safetensors"] != WEIGHTS_SHA:
        raise SystemExit("copied weights sha mismatch — abort")
    for name in ("tokenizer.json", "tokenizer_config.json", "ARCHITECTURE.md"):
        if hashes[name] != sha256(SRC / name):
            raise SystemExit(f"copied {name} differs from v1.1 — abort")

    policy = json.loads((ROOT / "runs/e36-calib/inference-policy.json").read_text(encoding="utf-8"))
    (DST / "inference-policy.json").write_text(json.dumps(policy, indent=2, sort_keys=True),
                                               encoding="utf-8")
    hashes["inference-policy.json"] = sha256(DST / "inference-policy.json")
    print(f"export assembled: {DST}")
    for k, v in hashes.items():
        print(f"  {k}: {v[:16]}…")

    if not args.no_verify:
        r = subprocess.run([sys.executable, "-I", "-c", VERIFY, str(DST), str(ROOT)],
                           capture_output=True, text=True)
        sys.stdout.write(r.stdout)
        sys.stderr.write(r.stderr)
        if r.returncode != 0:
            raise SystemExit("fresh-process verification FAILED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
