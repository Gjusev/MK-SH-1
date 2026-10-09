"""Evaluate saved JuliaArch checkpoints with explicit types; never train.

The independent-v1 set was already exposed by E16. These results are
exploratory. E17 checkpoints were selected using a dev type-map bug; this
script cannot recover the checkpoint that correct historical selection
would have chosen. Frozen datasets and old outputs are never overwritten.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def load_sets():
    from mokka_decisions.contracts import DecisionCase, Option

    manifest_path = ROOT / "data/independent-v1/manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    data_path = manifest_path.parent / "cases.jsonl"
    if sha256(data_path) != manifest["files"]["cases.jsonl"]["sha256"]:
        raise ValueError("Frozen independent-v1 file hash mismatch")
    if sha256(ROOT / manifest["generator"]) != manifest["generator_sha256"]:
        raise ValueError("Frozen generator hash mismatch")
    dev_raw = jsonl(ROOT / "data/typed-v1/cases/dev.jsonl")
    dev_cases = [DecisionCase.from_dict(row) for row in dev_raw]
    dev_types = {row["id"]: row["question_type"] for row in dev_raw}
    indep_cases, indep_types = [], {}
    originals = jsonl(data_path)
    for case in originals:
        state = case["state"]
        if not isinstance(state, str):
            state = json.dumps(state, ensure_ascii=False)
        for qid, question in case["questions"].items():
            kind = question["type"]
            criteria = question["criteria"]
            if criteria is None and kind == "noul":
                criteria = {"false": "false", "true": "true"}
            if isinstance(criteria, list):
                criteria = {str(i): value for i, value in enumerate(criteria)}
            keys = ["false", "true"] if kind == "noul" else list(criteria)
            gold = int(case["gold"][qid])
            item = DecisionCase(
                id=f"{case['id']}:{qid}", group_id=case["id"], source="indep-v1",
                language="en", domain=case["family"], state=state,
                question=question["instructions"],
                options=tuple(Option(id=f"o{i}", description=criteria[key]) for i, key in enumerate(keys)),
                target_kind="single", target_option=f"o{gold}",
            )
            indep_cases.append(item)
            indep_types[item.id] = kind
    if len(originals) != manifest["cases"] or len(indep_cases) != manifest["questions"]:
        raise ValueError("Frozen independent-v1 counts mismatch")
    sets = {"dev_correct_types": (dev_cases, dev_types), "independent_v1_exploratory": (indep_cases, indep_types)}
    for name, (cases, types) in sets.items():
        if len({c.id for c in cases}) != len(cases) or len(types) != len(cases):
            raise ValueError(f"{name}: duplicate ids or missing types")
    return sets


def summarize(rows):
    def grouped(field):
        counts = defaultdict(lambda: [0, 0])
        for row in rows:
            counts[row[field]][0] += row["correct"]
            counts[row[field]][1] += 1
        return {key: {"correct": val[0], "n": val[1], "accuracy": val[0] / val[1]}
                for key, val in sorted(counts.items())}
    result = {
        "n": len(rows), "groups": len({row["case_id"] for row in rows}),
        "correct": sum(row["correct"] for row in rows),
        "accuracy": sum(row["correct"] for row in rows) / len(rows),
        "by_type": grouped("type"), "by_family": grouped("family"),
    }
    for metric in ("nll", "brier", "mae"):
        values = [row[metric] for row in rows if metric in row]
        result["mean_" + metric] = sum(values) / len(values) if values else None
    return result


def paired(a, b):
    import numpy as np

    ia, ib = {r["id"]: r for r in a}, {r["id"]: r for r in b}
    if len(ia) != len(a) or len(ib) != len(b) or ia.keys() != ib.keys():
        raise ValueError("Paired comparisons require identical unique question ids")
    groups = defaultdict(list)
    for key in sorted(ia):
        x, y = ia[key], ib[key]
        if x["case_id"] != y["case_id"] or x["type"] != y["type"]:
            raise ValueError("Group/type mismatch in paired rows")
        groups[x["case_id"]].append(x["correct"] - y["correct"])
    if len({len(values) for values in groups.values()}) != 1:
        raise ValueError("This bootstrap implementation expects equal question counts per case")
    values = np.array([sum(v) / len(v) for v in groups.values()])
    rng = np.random.default_rng(20261009)
    boot = np.concatenate([values[rng.integers(0, len(values), (1000, len(values)))].mean(axis=1)
                           for _ in range(50)])
    return {"delta_accuracy": float(values.mean()), "cases": len(values),
            "bootstrap95_by_case": np.quantile(boot, [0.025, 0.975]).tolist(),
            "bootstrap_draws": len(boot), "bootstrap_seed": 20261009,
            "scope": "conditional on these checkpoints and exposed families; not training-seed uncertainty"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--batch", type=int, default=16)
    args = parser.parse_args()
    import torch
    from safetensors.torch import load_file
    from transformers import AutoTokenizer
    from mokka_decisions.arch_julia import JuliaArchScorer, collate_julia, validate_question_types
    from julia.data import sequence as upstream_sequence

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sets = load_sets()
    checkpoints = {name: ROOT / f"runs/kaggle-typed-julia-small/{name}/checkpoint_best.safetensors"
                   for name in ("julias_ordinal", "julias_probs")}
    sources = [ROOT / "data/independent-v1/cases.jsonl",
               ROOT / "data/independent-v1/manifest.json",
               ROOT / "data/typed-v1/cases/dev.jsonl", Path(__file__),
               ROOT / "src/mokka_decisions/arch_julia.py",
               ROOT / "scripts/gen_independent_v1.py"]
    hashes = {str(path.relative_to(ROOT)): sha256(path) for path in sources}
    references = {name: ROOT / f"runs/independent-v1/{name}_percase.jsonl"
                  for name in ("julia", "e15_probs_s44", "e15_control_s43")}
    plan = {
        "frozen_at": datetime.now(timezone.utc).isoformat(), "purpose": "E17 saved-checkpoint diagnostic",
        "evaluation_status": "exploratory: independent-v1 was exposed by E16",
        "checkpoint_selection_limit": "E17 best checkpoints selected using all-choice dev metadata; historical selection cannot be repaired without rerunning training",
        "initialization_limit": "E17 training seeded after model construction; heads not proven matched",
        "device": "cpu", "dtype": "float32", "threads": args.threads, "batch": args.batch,
        "temperature": 1.0, "max_length": 1024, "head_length": 256,
        "checkpoints": {name: {"path": str(path.relative_to(ROOT)), "sha256": sha256(path)}
                        for name, path in checkpoints.items()},
        "reference_sha256": {name: sha256(path) for name, path in references.items()},
        "source_sha256": hashes,
        "environment": {**{p: importlib.metadata.version(p) for p in ("torch", "transformers", "safetensors")},
                        "python": platform.python_version()},
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
    }
    with (out / "evaluation_plan.json").open("x", encoding="utf-8") as f:
        json.dump(plan, f, indent=2)
    tokenizer = AutoTokenizer.from_pretrained("jhu-clsp/mmBERT-small", local_files_only=True)
    # Check every input against upstream strict encoding. Refuse silent truncation.
    for set_name, (cases, types) in sets.items():
        validate_question_types(cases, types)
        for case in cases:
            expected = upstream_sequence(tokenizer, {
                "state": case.state, "question": case.question,
                "options": [o.description for o in case.options], "type": types[case.id],
            }, 1024, 256, strict=True)
            from mokka_decisions.arch_julia import julia_sequence
            actual = julia_sequence(tokenizer, case.state, case.question,
                                    [o.description for o in case.options], types[case.id], 1024, 256)
            if any(actual[key] != expected[key] for key in ("ids", "markers", "qtype")):
                raise ValueError(f"Serialization parity failure: {case.id}")
        print(f"preflight {set_name}: {len(cases)} rows, strict encoding and upstream parity PASS", flush=True)
    report = {"plan": plan, "results": {}, "comparisons": {}}
    all_rows = {}
    for name, checkpoint in checkpoints.items():
        print(f"loading {name}", flush=True)
        model = JuliaArchScorer("jhu-clsp/mmBERT-small").float().eval()
        model.load_state_dict(load_file(str(checkpoint)), strict=True)
        report["results"][name] = {}
        for set_name, (cases, types) in sets.items():
            per = []
            dest = out / f"{name}_{set_name}_percase.jsonl"
            started = time.perf_counter()
            with dest.open("x", encoding="utf-8") as f, torch.inference_mode():
                for start in range(0, len(cases), args.batch):
                    chunk = cases[start:start + args.batch]
                    batch = collate_julia(tokenizer, chunk, types, 1024, 256, "cpu")
                    inputs = {k: batch[k] for k in ("input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype")}
                    probs = model(**inputs).softmax(-1).tolist()
                    for case, pv in zip(chunk, probs):
                        pv = pv[:len(case.options)]
                        if not all(math.isfinite(v) for v in pv) or abs(sum(pv) - 1.0) > 1e-5:
                            raise ValueError(f"Invalid probability vector: {case.id}")
                        gold = case.option_ids.index(case.target_option)
                        pred = max(range(len(pv)), key=pv.__getitem__)
                        row = {"id": case.id, "case_id": case.group_id, "family": case.domain,
                               "type": types[case.id], "gold_idx": gold, "pred_idx": pred,
                               "probabilities": pv, "correct": int(pred == gold),
                               "nll": -math.log(max(pv[gold], 1e-12)),
                               "brier": sum((v - float(i == gold)) ** 2 for i, v in enumerate(pv))}
                        if types[case.id] == "score":
                            row["mae"] = abs(sum(i * v for i, v in enumerate(pv)) - gold)
                        per.append(row)
                        f.write(json.dumps(row) + "\n")
                    f.flush()
                    print(f"{name} {set_name} {len(per)}/{len(cases)}", flush=True)
            summary = summarize(per)
            summary["elapsed_s"] = time.perf_counter() - started
            summary["rows_sha256"] = sha256(dest)
            report["results"][name][set_name] = summary
            if set_name == "independent_v1_exploratory":
                all_rows[name] = per
            print(f"RESULT {name} {set_name}: {json.dumps(summary)}", flush=True)
        del model
        import gc
        gc.collect()
    for name, per in all_rows.items():
        for ref, path in references.items():
            report["comparisons"][f"{name}_minus_{ref}"] = paired(per, jsonl(path))
    report["comparisons"]["julias_probs_minus_julias_ordinal"] = paired(all_rows["julias_probs"], all_rows["julias_ordinal"])
    for path in sources:
        if sha256(path) != hashes[str(path.relative_to(ROOT))]:
            raise ValueError(f"Source changed during evaluation: {path}")
    for name, path in checkpoints.items():
        if sha256(path) != plan["checkpoints"][name]["sha256"]:
            raise ValueError(f"Checkpoint changed during evaluation: {name}")
    with (out / "report.json").open("x", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Completed: {out / 'report.json'}", flush=True)


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT / "runs/julia-research/upstream-source"))
    main()

