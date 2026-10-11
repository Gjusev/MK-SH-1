"""Julia-1 zero-shot on OUR frozen numeric evals (official engine, CPU).

The missing head-to-head: Julia-1 never ran on the E26-E28 numeric tasks. Same
frozen analyses as ours: testC/testD accuracy by range, threshold-only pairs
both-correct, strict monotone sweeps, AND the same-decade shortcut reference on
Julia's predictions (does Julia-1 implement order or the decade heuristic?).
Uses the verified weights and isolated .venv-julia. Writes
runs/julia-research/numeric_zeroshot.json + percase jsonl.
"""

from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CKPT = ROOT / "runs/julia-research/upstream-source"
E26 = ROOT / "data/e26-order"
E27 = ROOT / "data/e27-grid"
OUT = ROOT / "runs/julia-research"


def load(path):
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def monotone_strict(preds):
    return all(a <= b for a, b in zip(preds, preds[1:])) and 0 in preds and 1 in preds


def analyze(rows, preds, meta, name):
    golds = [r["target_option"] == "true" for r in rows]
    hits = [p == g for p, g in zip(preds, golds)]
    out = {"n": len(rows), "accuracy": round(sum(hits) / len(hits), 4)}
    if name == "test_c":
        by_thr = defaultdict(list)
        for r, h in zip(rows, hits):
            by_thr[r["threshold"]].append(h)
        out["by_threshold"] = {f"{t:g}": round(sum(v) / len(v), 4) for t, v in sorted(by_thr.items())}
    if name == "test_d":
        by = defaultdict(list)
        for r, h in zip(rows, hits):
            by[r["thr_range"]].append(h)
        out["by_range"] = {k: round(sum(v) / len(v), 4) for k, v in by.items()}
    # threshold-only pairs both-correct
    pairs = defaultdict(list)
    for r, h in zip(rows, hits):
        if r["pair"] and r["variant"] in ("pair_thrlo", "pair_thrhi"):
            pairs[r["pair"]].append(h)
    if pairs:
        solved = sum(1 for v in pairs.values() if len(v) == 2 and all(v))
        out["thr_pairs_both"] = f"{solved}/{len(pairs)}"
    # same-decade rule vs julia agreement + rule accuracy
    sd_rule = [int(r["value"] // 10) == int(r["threshold"] // 10) for r in rows]
    out["same_decade_rule_acc"] = round(sum(sr == g for sr, g in zip(sd_rule, golds)) / len(rows), 4)
    out["julia_agrees_with_rule"] = round(sum(sr == p for sr, p in zip(sd_rule, preds)) / len(rows), 4)
    # monotone sweeps (mono + testE sweeps)
    sweeps = defaultdict(list)
    for r, p in zip(rows, preds):
        if r.get("sweep"):
            sweeps[r["sweep"]].append((r["value"], p))
    if sweeps:
        ok = 0
        for sid, members in sweeps.items():
            members.sort()
            if monotone_strict([p for _, p in members]):
                ok += 1
        out["monotone_sweeps"] = f"{ok}/{len(sweeps)}"
    return out


def main() -> int:
    sys.path.insert(0, str(CKPT))
    sys.path.insert(0, str(ROOT / "src"))
    from julia.router.engine import FastEngine
    import torch

    torch.set_num_threads(8)
    engine = FastEngine(CKPT, device="cpu", transformer_backend="torch",
                        strict_encoding=True, max_length=1024, head_length=512,
                        batch_size=16, marker_only_head=False)
    specs = {"test_c": E26 / "test_c_rows.jsonl", "test_d": E26 / "test_d_rows.jsonl",
             "mono": E26 / "mono_rows.jsonl", "test_e": E27 / "test_e_rows.jsonl"}
    results, percase = {}, []
    for name, path in specs.items():
        rows = load(path)
        payload = [dict(state=json.loads(r["state"]), question=r["question"],
                        type="noul", options=["No", "Yes"]) for r in rows]
        t0 = time.monotonic()
        preds = []
        for i in range(0, len(payload), 32):
            for ans in engine.predict(payload[i:i + 32]):
                preds.append(ans["index"] == 1)
        dt = time.monotonic() - t0
        results[name] = analyze(rows, preds, None, name)
        results[name]["elapsed_s"] = round(dt, 1)
        for r, p in zip(rows, preds):
            percase.append({"set": name, "id": r["id"], "pred": int(p),
                            "gold": int(r["target_option"] == "true")})
        print(name, json.dumps(results[name]), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "numeric_zeroshot.json").write_text(json.dumps(
        {"runner": "Julia-1 official FastEngine (CPU, 8 threads), zero-shot",
         "weights_sha256": "df853bf7fe424420011f3d0c47a05d7341aa9eefa7fb9f203ea4aada4ad95b72",
         "protocol": "our frozen E26/E27 numeric evals, same analyses as the Mokka closures",
         "results": results}, indent=2), encoding="utf-8")
    (OUT / "numeric_zeroshot_percase.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in percase), encoding="utf-8")
    print("saved ->", OUT / "numeric_zeroshot.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
