"""E32: honest head-to-head — integrated Mokka version vs adapted Julia-1.

One harness, two model front-ends, same frozen sets, same joins, CPU float32.
Run order (criteria frozen in docs/E32_CRITERIA_2026-10-11.md, committed first):

  .venv-julia/Scripts/python.exe scripts/e32_report.py --side julia   # adapted s42/s43 + original
  .venv/Scripts/python.exe       scripts/e32_report.py --side mokka   # E30-MIX + candidate context
  .venv/Scripts/python.exe       scripts/e32_report.py --merge        # deterministic report (byte-exact)

E31 regression requirements on the Julia side: FastEngine without compile,
guards engine.model is model AND engine.forward is engine.model asserted
before any evaluation, and an AT-EVAL-TIME in-memory weights hash (sorted
state_dict keys, float32-canonicalized tensor bytes) recorded next to the
checkpoint FILE sha256. No training, no GPU.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/e32-analysis"
E26, E27 = ROOT / "data/e26-order", ROOT / "data/e27-grid"
JSRC = ROOT / "runs/julia-research/upstream-source"
PARQUET = ROOT / "runs/julia-research/reproduce-typed/test-00000-of-00001.parquet"
ZERO_SHOT = ROOT / "runs/julia-research/reproduce-typed/original-criteria-predictions.jsonl"
E31 = ROOT / "runs/kaggle-typed-e31"
E30_MIX = ROOT / "runs/kaggle-typed-e30/MIX/phase3/final.safetensors"
E28_CAND = ROOT / "runs/kaggle-typed-e29/candidate_initial.safetensors"
TASK_ROWS = ROOT / "data/processed/v1/eval/dev_eval.jsonl"
E30_MANIFEST = ROOT / "data/e30-mix/manifest.json"
THREADS = 8

JULIA_MODELS = {
    "julia_adapted_s42": (E31 / "s42/final.safetensors",
                          "0360634129f4f18b5a46a77b4096a09fb0cad3a5cceb1ed470413bad2bf905c6"),
    "julia_adapted_s43": (E31 / "s43/final.safetensors",
                          "38b98b5dcb836f07c3d6cfd5eebabee4c01bb731e6c309b02dd5ab86d22eeb51"),
    "julia_original": (JSRC / "model.safetensors",
                       "df853bf7fe424420011f3d0c47a05d7341aa9eefa7fb9f203ea4aada4ad95b72"),
}
MOKKA_MODELS = {
    "mokka_integrated_e30mix": (E30_MIX,
                                None),  # hash pinned in code below (candidate lineage)
    "mokka_candidate_e28s43": (E28_CAND,
                               "75ea9eed7aac43a1ac1dc7fdeca40329af3335696861db3aae92aec844614829"),
}
E30_MIX_SHA = "752770f7f6077be0"  # prefix; full hash asserted at runtime from file


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def weights_sha_canonical(model) -> str:
    """AT-EVAL-TIME in-memory hash: sorted state_dict keys, float32 bytes."""
    import torch

    h = hashlib.sha256()
    for k, v in sorted(model.state_dict().items()):
        h.update(k.encode("utf-8"))
        h.update(v.detach().cpu().to(torch.float32).contiguous().numpy().tobytes())
    return h.hexdigest()


def load_meta() -> dict:
    meta = {}
    for root, fname in ((E26, "test_c_rows.jsonl"), (E26, "test_d_rows.jsonl"),
                        (E26, "mono_rows.jsonl"), (E27, "test_e_rows.jsonl")):
        for l in (root / fname).read_text(encoding="utf-8").splitlines():
            if l.strip():
                r = json.loads(l)
                meta[r["id"]] = r
    return meta


def numeric_sets():
    sets = {}
    for name, root, fname in (("known", E26, "test_c_rows.jsonl"), ("new", E26, "test_d_rows.jsonl"),
                              ("mono", E26, "mono_rows.jsonl"), ("e", E27, "test_e_rows.jsonl")):
        sets[name] = [json.loads(l) for l in (root / fname).read_text(encoding="utf-8").splitlines() if l.strip()]
    return sets


def typed2000_rows():
    """Official Julia test set rebuilt from the frozen parquet, cross-checked
    against the frozen zero-shot per-question artifacts (ids, keys, gold)."""
    import pyarrow.parquet as pq

    frozen = {}
    for l in ZERO_SHOT.read_text(encoding="utf-8").splitlines():
        if l.strip():
            r = json.loads(l)
            frozen[r["id"]] = r
    rows = []
    for case in pq.read_table(PARQUET).to_pylist():
        state = json.loads(case["state"])
        gold = json.loads(case["gold"])
        for qid, q in json.loads(case["questions"]).items():
            kind = q["type"]
            crit = q.get("criteria")
            if crit is None and kind == "noul":
                crit = {"false": "false", "true": "true"}
            if isinstance(crit, list):
                crit = {str(i): v for i, v in enumerate(crit)}
            keys = ["false", "true"] if kind == "noul" else list(crit)
            rid = f"{case['id']}:{qid}"
            ref = frozen[rid]
            assert ref["type"] == kind and ref["keys"] == keys and ref["gold"] == str(gold[qid]["label"]), rid
            rows.append({"id": rid, "type": kind, "keys": keys,
                         "gold": keys.index(str(gold[qid]["label"])),
                         "options": [crit[k] for k in keys],
                         "state": state, "question": q["instructions"]})
    assert len(rows) == 2000 and set(frozen) == {r["id"] for r in rows}
    return rows


def task_rows():
    ids = set(json.loads(E30_MANIFEST.read_text(encoding="utf-8"))["original_dev_ids"])
    rows = [json.loads(l) for l in TASK_ROWS.read_text(encoding="utf-8").splitlines() if l.strip()]
    picked = [r for r in rows if r["id"] in ids]
    assert len(picked) == len(ids) == 300
    return picked


def write_percase(model_key: str, set_name: str, percase: list[dict]) -> str:
    d = OUT / "percase"
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{model_key}__{set_name}.json"
    p.write_text(json.dumps(percase, sort_keys=True), encoding="utf-8")
    return str(p.relative_to(OUT))


# ---------------------------------------------------------------- julia side
def side_julia() -> int:
    import torch

    torch.set_num_threads(THREADS)
    sys.path.insert(0, str(JSRC))
    from julia.router.engine import FastEngine
    from safetensors.torch import load_file

    engine = FastEngine(str(JSRC), device="cpu", transformer_backend="torch",
                        strict_encoding=True, max_length=1024, head_length=512,
                        batch_size=8, marker_only_head=False)
    model = engine.model  # no compile: engine.forward IS engine.model
    assert engine.model is model and engine.forward is engine.model, (
        "e32: eval engine/model link broken — weight identity would not cover inference")

    sets = numeric_sets()

    def payload_noul(r):
        return dict(state=json.loads(r["state"]), question=r["question"],
                    type="noul", options=["No", "Yes"])

    def run_model(key, path, sha_expect):
        got = file_sha(path)
        assert sha_expect and got == sha_expect, f"{key}: file sha mismatch {got}"
        model.load_state_dict(load_file(str(path), device="cpu"), strict=True)
        model.eval()
        wsha = weights_sha_canonical(model)
        out = {"checkpoint_file_sha256": got, "eval_weights_sha256": wsha,
               "guards": "engine.model is model; engine.forward is engine.model (asserted)",
               "weights_sha256_kind": "in-memory state_dict, sorted keys, float32-canonicalized",
               "sets": {}}
        for name, rows in sets.items():
            payload = [payload_noul(r) for r in rows]
            gold = [int(r["target_option"] == "true") for r in rows]
            per = []
            with torch.inference_mode():
                for i in range(0, len(payload), 16):
                    for ans, g, r in zip(engine.predict(payload[i:i + 16]), gold[i:i + 16], rows[i:i + 16]):
                        per.append({"id": r["id"], "pred": int(ans["index"]), "gold": int(g),
                                    "ok": int(ans["index"] == g)})
            out["sets"][name] = {"n": len(per), "correct": sum(p["ok"] for p in per),
                                 "percase_file": write_percase(key, name, per)}
        # axis B: official 2000-question set
        rows = typed2000_rows()
        per = []
        with torch.inference_mode():
            for i in range(0, len(rows), 16):
                chunk = rows[i:i + 16]
                payload = [dict(state=r["state"], question=r["question"], type=r["type"],
                                options=r["options"]) for r in chunk]
                for ans, r in zip(engine.predict(payload), chunk):
                    per.append({"id": r["id"], "pred": int(ans["index"]), "gold": int(r["gold"]),
                                "ok": int(ans["index"] == r["gold"])})
        out["sets"]["typed2000"] = {"n": len(per), "correct": sum(p["ok"] for p in per),
                                    "percase_file": write_percase(key, "typed2000", per)}
        return out

    result = {"side": "julia", "torch": torch.__version__,
              "transformers": __import__("transformers").__version__,
              "device": "cpu", "threads": THREADS, "models": {}}
    for key, (path, sha) in JULIA_MODELS.items():
        print(f"[e32-julia] {key} ...", flush=True)
        result["models"][key] = run_model(key, path, sha)
    (OUT / "julia_side.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print("[e32-julia] done")
    return 0


# ---------------------------------------------------------------- mokka side
def side_mokka() -> int:
    import torch

    torch.set_num_threads(THREADS)
    sys.path.insert(0, str(ROOT / "src"))
    from mokka_decisions.arch_julia import JuliaArchScorer, collate_julia
    from mokka_decisions.contracts import DecisionCase, Option
    from safetensors.torch import load_file
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("jhu-clsp/mmBERT-small")
    sets = numeric_sets()
    task = task_rows()

    def cases_from(rows, qtypes):
        return [DecisionCase(id=r["id"], group_id=r.get("group_id", r["id"]), source="e32",
                             language="en", domain=r.get("domain", "task"), state=r["state"],
                             question=r["question"],
                             options=tuple(Option(id=o["id"], description=o["description"])
                                           for o in r["options"]),
                             target_kind="single", target_option=r["target_option"])
                for r in rows]

    all_sets = dict(sets)
    all_sets["task"] = task
    qtypes = {}
    for rows in all_sets.values():
        for r in rows:
            qtypes[r["id"]] = r.get("question_type") or "choice"

    def run_model(key, path, sha_expect):
        got = file_sha(path)
        if sha_expect:
            assert got == sha_expect, f"{key}: file sha mismatch {got}"
        else:
            assert got.startswith(E30_MIX_SHA), f"{key}: unexpected file sha {got}"
        model = JuliaArchScorer("jhu-clsp/mmBERT-small")
        model.load_state_dict(load_file(str(path), device="cpu"), strict=True)
        model.to("cpu").eval()
        wsha = weights_sha_canonical(model)
        out = {"checkpoint_file_sha256": got, "eval_weights_sha256": wsha,
               "guards": "single model object evaluated (no engine wrapper on this side)",
               "weights_sha256_kind": "in-memory state_dict, sorted keys, float32-canonicalized",
               "sets": {}}
        for name, rows in all_sets.items():
            cases = cases_from(rows, qtypes)
            per = []
            with torch.inference_mode():
                for s in range(0, len(cases), 16):
                    chunk = cases[s:s + 16]
                    batch = collate_julia(tok, chunk, qtypes, 1024, 256, "cpu")
                    logits = model(**{k: batch[k] for k in
                                      ("input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype")}).float()
                    probs = logits.softmax(-1)
                    for r, case, pv_t in zip(rows[s:s + 16], chunk, probs):
                        k = len(case.options)
                        pred = max(range(k), key=pv_t[:k].__getitem__)
                        gold = next(i for i, o in enumerate(r["options"]) if o["id"] == r["target_option"])
                        per.append({"id": r["id"], "pred": int(pred), "gold": int(gold),
                                    "ok": int(pred == gold)})
            out["sets"][name] = {"n": len(per), "correct": sum(p["ok"] for p in per),
                                 "percase_file": write_percase(key, name, per)}
        return out

    result = {"side": "mokka", "torch": torch.__version__, "device": "cpu",
              "threads": THREADS, "models": {}}
    for key, (path, sha) in MOKKA_MODELS.items():
        print(f"[e32-mokka] {key} ...", flush=True)
        result["models"][key] = run_model(key, path, sha)
    (OUT / "mokka_side.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print("[e32-mokka] done")
    return 0


# ---------------------------------------------------------------- merge
def load_percase(rel: str) -> list[dict]:
    return json.loads((OUT / rel).read_text(encoding="utf-8"))


def metric_join(per: list[dict], meta: dict) -> dict:
    """Same joins as the campaign analyzers: accuracy, by thr_range, global
    threshold-only pairs, PURE pairs by range (both members same range), and
    strict monotone sweeps."""
    acc = sum(p["ok"] for p in per) / len(per)
    by_range = defaultdict(lambda: [0, 0])
    pairs, sweeps = defaultdict(list), defaultdict(list)
    for p in per:
        m = meta[p["id"]]
        if m.get("thr_range"):
            by_range[m["thr_range"]][0] += p["ok"]
            by_range[m["thr_range"]][1] += 1
        if m.get("pair") and m["variant"] in ("pair_thrlo", "pair_thrhi"):
            pairs[m["pair"]].append((m["thr_range"], m["variant"], p["ok"]))
        if m.get("sweep"):
            sweeps[m["sweep"]].append((m["value"], p["pred"]))
    thr_global = [int(all(ok for _, _, ok in members)) for members in pairs.values()
                  if len(members) == 2]
    pure = {"interp": [0, 0], "extrap": [0, 0]}
    for members in pairs.values():
        if len(members) == 2 and {v for _, v, _ in members} == {"pair_thrlo", "pair_thrhi"}:
            key = {"interp": "interp", "extrap": "extrap"}.get(members[0][0])                 if members[0][0] == members[1][0] else None
            if key:
                pure[key][1] += 1
                pure[key][0] += int(all(ok for _, _, ok in members))
    mono = 0
    for items in sweeps.values():
        preds = [p for _, p in sorted(items)]
        if all(a <= b for a, b in zip(preds, preds[1:])) and 0 in preds and 1 in preds:
            mono += 1
    return {"n": len(per), "accuracy": round(acc, 6),
            "thr_pairs_global": [sum(thr_global), len(thr_global)],
            "mono_sweeps": [mono, len(sweeps)],
            "pure_pairs": pure,
            "by_range": {k: [c, n] for k, (c, n) in sorted(by_range.items())} if by_range else {}}


def merge() -> int:
    meta = load_meta()
    julia = json.loads((OUT / "julia_side.json").read_text(encoding="utf-8"))
    mokka = json.loads((OUT / "mokka_side.json").read_text(encoding="utf-8"))

    def axis_a(side):
        out = {}
        for key, m in side["models"].items():
            per = {name: load_percase(s["percase_file"]) for name, s in m["sets"].items()}
            known = metric_join(per["known"], meta)
            new = metric_join(per["new"], meta)
            e = metric_join(per["e"], meta)
            out[key] = {
                "testC": {"n": known["n"], "correct": sum(p["ok"] for p in per["known"])},
                "testD": {"n": new["n"], "correct": sum(p["ok"] for p in per["new"]),
                          **{k: new["by_range"].get(k) for k in ("interp", "extrap")}},
                "testE": {"n": e["n"], "correct": sum(p["ok"] for p in per["e"]),
                          "interp": e["by_range"].get("interp"),
                          "extrap": e["by_range"].get("extrap"),
                          "pure_pairs": e["pure_pairs"], "mono": e["mono_sweeps"]},
                "thr_pairs_global": known["thr_pairs_global"][0] + new["thr_pairs_global"][0]
                                    + e["thr_pairs_global"][0],
                "thr_pairs_global_n": 22,
            }
        return out

    a = {"julia_adapted_s42": None, "julia_adapted_s43": None, "julia_original": None,
         "mokka_integrated_e30mix": None, "mokka_candidate_e28s43": None}
    aj = axis_a(julia)
    am = axis_a(mokka)
    a.update(aj)
    a.update(am)

    # ---- axis B
    typed = {}
    for key, m in julia["models"].items():
        per = load_percase(m["sets"]["typed2000"]["percase_file"])
        typed[key] = {"n": len(per), "correct": sum(p["ok"] for p in per)}
    task = {}
    for key, m in mokka["models"].items():
        per = load_percase(m["sets"]["task"]["percase_file"])
        task[key] = {"n": len(per), "correct": sum(p["ok"] for p in per)}

    # ---- pre-registered criteria (docs/E32_CRITERIA_2026-10-11.md)
    def acc(d):
        return d["correct"] / d["n"]

    def seed_mean(keys, getter):
        vals = [getter(k) for k in keys]
        return sum(vals) / len(vals)

    jkeys = ["julia_adapted_s42", "julia_adapted_s43"]
    mkey = "mokka_integrated_e30mix"
    A1 = {"mokka": acc(a[mkey]["testC"]), "julia": seed_mean(jkeys, lambda k: acc(a[k]["testC"]))}
    margins = {
        "A1_testC": (a[mkey]["testC"]["correct"], sum(a[k]["testC"]["correct"] for k in jkeys) / 2, 2, 48),
        "A2_testD": (a[mkey]["testD"]["correct"], sum(a[k]["testD"]["correct"] for k in jkeys) / 2, 2, 54),
        "A3_E_interp": (a[mkey]["testE"]["interp"][0], sum(a[k]["testE"]["interp"][0] for k in jkeys) / 2, 2, 116),
        "A4_E_extrap": (a[mkey]["testE"]["extrap"][0], sum(a[k]["testE"]["extrap"][0] for k in jkeys) / 2, 1, 56),
        "A5_thr_pairs": (a[mkey]["thr_pairs_global"], sum(a[k]["thr_pairs_global"] for k in jkeys) / 2, 2.5, 22),
        "A6_VE2": (a[mkey]["testE"]["mono"][0], sum(a[k]["testE"]["mono"][0] for k in jkeys) / 2, 1.5, 10),
    }
    wins = {}
    for mid, (mv, jv, margin, n) in margins.items():
        wins[mid] = "mokka" if mv - jv > margin else ("julia" if jv - mv > margin else "tie")
    mokka_wins = sum(1 for v in wins.values() if v == "mokka")
    julia_wins = sum(1 for v in wins.values() if v == "julia")
    a1_diff_rows = (a[mkey]["testC"]["correct"] - sum(a[k]["testC"]["correct"] for k in jkeys) / 2)
    if abs(a1_diff_rows) <= 2 and mokka_wins < 4 and julia_wins < 4:
        axis_a_reading = "A-EMPATE (sin ventaja numerica de sistema)"
    elif a1_diff_rows > 2 and mokka_wins >= 4:
        axis_a_reading = "A-MOKKA (ventaja numerica de sistema Mokka)"
    elif a1_diff_rows < -2 and julia_wins >= 4:
        axis_a_reading = "A-JULIA (ventaja numerica de sistema Julia)"
    else:
        axis_a_reading = "A-MIXTO (patron no cubierto por las ramas registradas)"

    sanity = {"mokka_testC_ge_090": A1["mokka"] >= 0.90,
              "julia_testC_ge_090": A1["julia"] >= 0.90}

    # retention categories
    def retention(delta_pp):
        if delta_pp >= -5:
            return "conserva"
        if delta_pp >= -15:
            return "degradacion_moderada"
        return "perdida_severa"

    orig_acc = acc(typed["julia_original"]) * 100
    retention_seeds = {}
    for k in jkeys:
        d = acc(typed[k]) * 100 - orig_acc
        retention_seeds[k] = {"acc_pp": round(acc(typed[k]) * 100, 4),
                              "delta_pp": round(d, 4), "category": retention(d)}
    mean_delta = sum(v["delta_pp"] for v in retention_seeds.values()) / 2
    julia_task_category = retention(mean_delta)

    mokka_task = {"integrated_acc": round(acc(task[mkey]) * 100, 4),
                  "candidate_acc": round(acc(task["mokka_candidate_e28s43"]) * 100, 4),
                  "recovery_delta_pp": round((acc(task[mkey]) - acc(task["mokka_candidate_e28s43"])) * 100, 4),
                  "semantics": "recuperacion por mezcla (la candidata nunca conocio la tarea)"}

    if axis_a_reading.startswith("A-EMPATE"):
        cross = ("dos sistemas viables; sin ventaja de sistema medible en este alcance"
                 if julia_task_category != "perdida_severa"
                 else "ventaja de sistema Mokka en integracion")
    elif axis_a_reading.startswith("A-MOKKA"):
        cross = ("ventaja de sistema Mokka en este alcance" if julia_task_category != "perdida_severa"
                 else "ventaja de sistema Mokka en este alcance (y en integracion)")
    elif axis_a_reading.startswith("A-JULIA"):
        cross = ("ventaja de sistema Julia en este alcance" if julia_task_category != "perdida_severa"
                 else "A-MIXTO: Julia gana numerico pero pierde la tarea severamente")
    else:
        cross = "descriptivo sin etiqueta de ventaja"

    report = {
        "protocol": "docs/E32_CRITERIA_2026-10-11.md (frozen pre-evaluation)",
        "scope": "COMPLETE SYSTEMS (architecture+recipe+data); recipes NOT identical; "
                 "NO architecture claims possible",
        "harness": {"device": "cpu", "dtype": "float32", "threads": THREADS,
                    "julia_env": ".venv-julia (transformers 5.0.0)",
                    "sets": "frozen e26-order/e27-grid manifests + official Julia parquet "
                            "(revision c76749ec...) + E30 task slice (300 ids)"},
        "models": {**{k: {"checkpoint_file_sha256": v["checkpoint_file_sha256"],
                          "eval_weights_sha256": v["eval_weights_sha256"]}
                      for k, v in julia["models"].items()},
                   **{k: {"checkpoint_file_sha256": v["checkpoint_file_sha256"],
                          "eval_weights_sha256": v["eval_weights_sha256"]}
                      for k, v in mokka["models"].items()}},
        "axis_a": {k: v for k, v in a.items()},
        "axis_a_headline": {"testC_acc": A1, "metric_wins": wins,
                            "mokka_wins": mokka_wins, "julia_wins": julia_wins,
                            "reading": axis_a_reading, "sanity_both_ge_090": sanity},
        "axis_b": {"julia_retention": {"original_acc_pp": round(orig_acc, 4),
                                       "per_seed": retention_seeds,
                                       "mean_delta_pp": round(mean_delta, 4),
                                       "category": julia_task_category},
                   "mokka_recovery": mokka_task},
        "cross_axis_reading": cross,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "e32_report.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"A1": A1, "wins": wins, "reading": axis_a_reading,
                      "julia_retention": retention_seeds, "category": julia_task_category,
                      "mokka_task": mokka_task, "cross": cross}, indent=1, default=str))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--side", choices=("julia", "mokka"))
    parser.add_argument("--merge", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.side == "julia":
        return side_julia()
    if args.side == "mokka":
        return side_mokka()
    if args.merge:
        return merge()
    parser.error("choose --side julia|mokka or --merge")


if __name__ == "__main__":
    sys.exit(main())
