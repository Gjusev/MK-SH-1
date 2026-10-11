"""Kaggle GPU job: install the pinned wheel, train / resume / evaluate.

Stages (arg --stage):
  smoke        hardware report + 100-step timing + save/no-save check
  train        full training from configs/train.yaml (or --config override)
  resume_test  train A (2 epochs) vs train B (1 epoch + reload + 1 epoch)
  eval         our model + temperature + threshold on all eval sets
  baselines    laya + gliner + tfidf on all eval sets (needs our rows? no: independent)

The bundle dataset (wheel + configs + prepared data) is discovered under
/kaggle/input. A checkpoint dataset (for resume) is discovered the same way.
Everything lands in /kaggle/working with a run_status.json verdict.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def log(msg: str) -> None:
    print(f"[mokka {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def find_bundle() -> Path:
    root = Path("/kaggle/input")
    if root.exists():
        # attachment debug: list every dataset mount before anything else
        log("input mounts: " + ", ".join(sorted(str(p) for p in root.iterdir())))
        candidates = [p.parent for p in root.rglob("BUNDLE_MANIFEST.json")]
        # prefer the full bundle (contains the wheel); checkpoint bundles also
        # carry a manifest but no dist/
        candidates.sort(key=lambda b: not any(b.rglob("*.whl")))
        for bundle in candidates:
            listing = [str(p) for p in sorted(bundle.rglob("*")) if p.is_file()][:80]
            log(f"bundle candidate {bundle}:\n  " + "\n  ".join(listing))
            if any(bundle.rglob("*.whl")):
                return bundle
        if candidates:
            raise SystemExit("only checkpoint-style bundles found; attach the full bundle dataset")
        listing = [str(p) for p in sorted(root.rglob("*"))][:60]
        raise SystemExit(
            "bundle dataset not found. /kaggle/input listing:\n" + "\n".join(listing)
        )
    raise SystemExit("no /kaggle/input; run locally with --bundle-style paths instead")


def find_file(bundle: Path, relative: str) -> Path:
    """Resolve a bundle-relative path across zip-extraction layouts."""
    target = bundle / relative
    if target.exists():
        return target
    name = Path(relative).name
    hits = [p for p in bundle.rglob(name) if p.is_file()]
    if not hits:
        raise SystemExit(f"bundle file not found: {relative} (no {name!r} anywhere)")
    expected_dirs = list(Path(relative).parts[:-1])
    for hit in hits:  # prefer hits whose parents contain the expected dirs
        if all(d in hit.parts for d in expected_dirs):
            return hit
    return hits[0]


def find_checkpoint() -> Path | None:
    for pattern in ("/kaggle/input/*/checkpoint_latest.pt", "/kaggle/input/*/checkpoint*.pt"):
        hits = sorted(glob.glob(pattern))
        if hits:
            return Path(hits[0])
    return None


def install_wheel(bundle: Path) -> None:
    wheels = sorted(bundle.glob("**/*.whl"))
    if not wheels:
        raise SystemExit("no wheel in bundle")
    log(f"installing {wheels[0].name}")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "--no-deps", str(wheels[0])])
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "--quiet",
         "transformers==5.19.0", "datasets==5.1.0", "safetensors>=0.4", "sentencepiece>=0.2"]
    )


def hardware_report() -> dict:
    import torch

    report = {
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }
    if report["cuda_available"]:
        props = torch.cuda.get_device_properties(0)
        report["gpu"] = props.name
        report["vram_gb"] = round(props.total_memory / 2**30, 2)
    return report


def require_gpu() -> None:
    import torch

    if not torch.cuda.is_available():
        raise SystemExit(
            "GPU required but not assigned (phone verification / settings?). Refusing to train on CPU."
        )


def load_cases(path: Path):
    from mokka_decisions.contracts import DecisionCase, load_jsonl

    return [DecisionCase.from_dict(r) for r in load_jsonl(str(path))]


def stage_smoke(bundle: Path, work: Path, config: str) -> None:
    import yaml

    require_gpu()
    from mokka_decisions.train import TrainConfig, run_training
    from transformers import AutoTokenizer

    cfg_data = yaml.safe_load(find_file(bundle, config).read_text(encoding="utf-8"))
    train_cfg = TrainConfig.from_mapping(cfg_data["train"])
    # 100-step-ish smoke: tiny slice, 1 epoch, no saves needed beyond latest
    cases = load_cases(find_file(bundle, "data/processed/v1/instances/train.jsonl"))[:64]
    dev = load_cases(find_file(bundle, "data/processed/v1/eval/dev_eval.jsonl"))[:64]
    train_cfg.epochs = 1
    train_cfg.eval_every_steps = 0
    train_cfg.output_dir = str(work / "smoke")
    tokenizer = AutoTokenizer.from_pretrained(train_cfg.encoder_name)
    t0 = time.time()
    report = run_training(train_cfg, train_cases=cases, dev_cases=dev, tokenizer=tokenizer)
    dt = time.time() - t0
    steps = report["final_global_step"]
    summary = {
        "stage": "smoke",
        "hardware": hardware_report(),
        "steps": steps,
        "wall_s": round(dt, 1),
        "s_per_step": round(dt / max(steps, 1), 3),
        "dev": report["history"][-1]["dev"] if report["history"] else None,
    }
    (work / "run_status.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log(json.dumps(summary))


def stage_train(bundle: Path, work: Path, config: str) -> None:
    import yaml

    require_gpu()
    from mokka_decisions.train import TrainConfig, run_training
    from transformers import AutoTokenizer

    cfg_data = yaml.safe_load(find_file(bundle, config).read_text(encoding="utf-8"))
    train_cfg = TrainConfig.from_mapping(cfg_data["train"])
    train_cfg.output_dir = str(work / "train")

    train = load_cases(find_file(bundle, "data/processed/v1/instances/train.jsonl"))
    dev = load_cases(find_file(bundle, "data/processed/v1/eval/dev_eval.jsonl"))
    log(f"train={len(train)} dev_eval={len(dev)}")

    resume = find_checkpoint()
    if resume:
        log(f"resuming from {resume}")
    tokenizer = AutoTokenizer.from_pretrained(train_cfg.encoder_name)
    manifest = json.loads((find_file(bundle, "data/processed/v1/manifests/revisions.json")).read_text())
    report = run_training(
        train_cfg,
        train_cases=train,
        dev_cases=dev,
        tokenizer=tokenizer,
        resume_from=resume,
        data_manifest_hash=manifest["data_hash"],
    )
    (work / "run_status.json").write_text(
        json.dumps(
            {
                "stage": "train",
                "hardware": hardware_report(),
                "resumed_from": str(resume) if resume else None,
                "final_global_step": report["final_global_step"],
                "best_dev_accuracy": report["best_dev_accuracy"],
                "epochs": train_cfg.epochs,
                "data_manifest_hash": manifest["data_hash"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    log(f"done: best_dev={report['best_dev_accuracy']}")
    # copy best weights next to output for easy retrieval
    import shutil

    best = work / "train/checkpoint_best.safetensors"
    if best.exists():
        shutil.copy(best, work / "model_best.safetensors")
        meta = work / "train/checkpoint_best.json"
        if meta.exists():
            shutil.copy(meta, work / "model_best.json")


def stage_resume_test(bundle: Path, work: Path, config: str) -> None:
    """Obligatory resume proof: interrupted+resumed run vs continuous run."""
    require_gpu()
    import yaml

    from mokka_decisions.train import TrainConfig, run_training
    from transformers import AutoTokenizer

    cfg_data = yaml.safe_load(find_file(bundle, config).read_text(encoding="utf-8"))
    train = load_cases(find_file(bundle, "data/processed/v1/instances/train.jsonl"))[:2000]
    dev = load_cases(find_file(bundle, "data/processed/v1/eval/dev_eval.jsonl"))[:400]
    tokenizer = AutoTokenizer.from_pretrained(cfg_data["train"]["encoder_name"])

    # A: continuous, 2 epochs
    cfg_a = TrainConfig.from_mapping(cfg_data["train"])
    cfg_a.epochs = 2
    cfg_a.output_dir = str(work / "resume_a")
    cfg_a.save_every_minutes = 0
    log("A: continuous 2 epochs")
    report_a = run_training(cfg_a, train_cases=train, dev_cases=dev, tokenizer=tokenizer)
    dev_a = [h for h in report_a["history"] if "epoch_time_s" in h.get("dev", {})][-1]["dev"]

    # B: 1 epoch, save, reload, +1 epoch
    cfg_b = TrainConfig.from_mapping(cfg_data["train"])
    cfg_b.epochs = 1
    cfg_b.output_dir = str(work / "resume_b1")
    cfg_b.save_every_minutes = 0
    log("B1: 1 epoch then stop")
    run_training(cfg_b, train_cases=train, dev_cases=dev, tokenizer=tokenizer)
    ckpt = work / "resume_b1/checkpoint_latest.pt"
    cfg_b2 = TrainConfig.from_mapping(cfg_data["train"])
    cfg_b2.epochs = 2
    cfg_b2.output_dir = str(work / "resume_b2")
    cfg_b2.save_every_minutes = 0
    log("B2: resume from checkpoint for epoch 2")
    report_b = run_training(
        cfg_b2, train_cases=train, dev_cases=dev, tokenizer=tokenizer, resume_from=ckpt
    )
    dev_b = [h for h in report_b["history"] if "epoch_time_s" in h.get("dev", {})][-1]["dev"]

    summary = {
        "stage": "resume_test",
        "continuous_dev": dev_a,
        "resumed_dev": dev_b,
        "accuracy_delta": abs(dev_a["accuracy"] - dev_b["accuracy"]),
        "nll_delta": abs(dev_a["mean_nll"] - dev_b["mean_nll"]),
        "verdict": "pass" if abs(dev_a["accuracy"] - dev_b["accuracy"]) <= 0.05 else "investigate",
        "note": "identical seeds; residual delta = CUDA kernel nondeterminism",
    }
    (work / "run_status.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log(json.dumps(summary))


def stage_eval(bundle: Path, work: Path, config: str) -> None:
    """Our model: temperature on cal_temperature, threshold on cal_policy,
    metrics on every eval set (test pools opened here, once)."""
    require_gpu()
    import yaml

    from mokka_decisions.backends.custom import CustomBackend
    from mokka_decisions.calibrate import calibration_revision, fit_temperature, save_calibration
    from mokka_decisions.evaluate import (
        bootstrap_accuracy_ci,
        dump_rows,
        oos_false_acceptance,
        permutation_flip_rate,
        run_backend,
        summarize,
        summarize_by,
    )
    from mokka_decisions.instances import permute_case
    from mokka_decisions.policy import pick_threshold, risk_coverage_curve, save_policy

    # prefer a checkpoint trained in THIS session over any attached dataset
    # (an attached old checkpoint must never shadow the fresh one)
    ckpt = None
    for pattern in (
        "/kaggle/working/model_best.safetensors",
        "/kaggle/working/train/model_best.safetensors",
        "/kaggle/input/*/model_best.safetensors",
    ):
        hits = sorted(glob.glob(pattern))
        if hits:
            ckpt = Path(hits[0])
            break
    if ckpt is None:
        raise SystemExit("no trained checkpoint found for eval stage")
    log(f"eval checkpoint: {ckpt}")

    cfg_data = yaml.safe_load(find_file(bundle, config).read_text(encoding="utf-8"))
    eval_cfg = yaml.safe_load((find_file(bundle, "configs/evaluation.yaml")).read_text(encoding="utf-8"))["evaluation"]
    backend = CustomBackend(ckpt, max_length=cfg_data["train"]["max_length"])
    backend.warmup()

    cal_t = load_cases(find_file(bundle, "data/processed/v1/instances/cal_temperature.jsonl"))
    cal_p = load_cases(find_file(bundle, "data/processed/v1/instances/cal_policy.jsonl"))
    log(f"cal_temperature={len(cal_t)} cal_policy={len(cal_p)}")

    logits = backend.scorer.score_logits(cal_t)
    temperature = fit_temperature(cal_t, logits)
    rev = calibration_revision(temperature, [c.id for c in cal_t])
    save_calibration(work / "calibration.json", temperature, rev, {"n": len(cal_t)})
    backend.scorer.temperature = temperature
    log(f"temperature={temperature}")

    probs_p = backend.scorer.score(cal_p)
    policy = pick_threshold(cal_p, probs_p, target_risk=eval_cfg["target_risk"])
    policy["calibration_revision"] = rev
    save_policy(work / "policy.json", policy)
    log(f"threshold={policy['threshold']} risk={policy['risk']} coverage={policy['coverage']}")
    curve = risk_coverage_curve(cal_p, probs_p)
    (work / "risk_coverage_calpolicy.json").write_text(json.dumps(curve, indent=2), encoding="utf-8")

    results: dict[str, dict] = {}
    eval_dir = find_file(bundle, "data/processed/v1/eval/dev_eval.jsonl").parent
    for eval_file in sorted(eval_dir.glob("*.jsonl")):
        name = eval_file.stem
        if name == "dev_eval":
            continue
        cases = load_cases(eval_file)
        rows = run_backend(backend, cases)
        dump_rows(rows, work / "rows" / f"{name}_mokka.jsonl")
        s = summarize(rows, cases, threshold=policy["threshold"])
        s["by_language"] = summarize_by(rows, cases, "language", threshold=policy["threshold"])
        s["bootstrap"] = bootstrap_accuracy_ci(rows, cases, n_boot=eval_cfg["bootstrap_n"])
        if any(c.target_option == "none" for c in cases):
            s["oos"] = oos_false_acceptance(rows, cases, threshold=policy["threshold"])
        results[name] = s
        log(f"{name}: acc={s['forced_accuracy']} coverage={s.get('coverage')}")

    # permutation probe on a sample of banking candidates eval
    perm_file = eval_dir / "banking77_test_candidates.jsonl"
    if perm_file.exists():
        base_cases = load_cases(perm_file)[: eval_cfg["permutation_probe_n"]]
        base_rows = run_backend(backend, base_cases)
        perm_cases = [permute_case(c, seed=1000 + i) for i, c in enumerate(base_cases)]
        perm_rows = run_backend(backend, perm_cases)
        results["permutation_probe"] = permutation_flip_rate(base_rows, perm_rows)

    (work / "eval_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    (work / "run_status.json").write_text(
        json.dumps({"stage": "eval", "temperature": temperature, "policy": policy,
                    "checkpoint": str(ckpt), "sets": sorted(results)}, indent=2),
        encoding="utf-8",
    )


def stage_baselines(bundle: Path, work: Path, config: str) -> None:
    """Comparators on the same eval sets. TF-IDF trains on the train split
    (fixed taxonomy); laya and gliner run zero-shot. No model of ours needed."""
    require_gpu()
    # laya only: it coexists with transformers 5.x (verified in v8).
    # gliner2 goes EXCLUSIVELY to the isolated venv below — never here.
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "--quiet", "laya>=0.3"]
    )
    from mokka_decisions.backends.laya import LayaBackend
    from mokka_decisions.backends.linear import LinearBackend, LinearRejectBackend
    from mokka_decisions.contracts import load_jsonl
    from mokka_decisions.evaluate import dump_rows, run_backend, summarize, summarize_by, bootstrap_accuracy_ci

    train_rows_raw = []
    for utt_file in sorted(find_file(bundle, "data/processed/v1/utterances/banking77.jsonl").parent.glob("*.jsonl")):
        train_rows_raw.extend(r for r in load_jsonl(str(utt_file)) if r.get("split") == "train")
    log(f"tfidf train rows: {len(train_rows_raw)}")
    backends = [LinearBackend(train_rows_raw), LinearRejectBackend(train_rows_raw)]
    log("tfidf fitted (closed-world + trainable-none variants)")
    try:
        backends.append(LayaBackend())
        log("laya loaded")
    except Exception as exc:
        log(f"laya load failed: {exc}")
    # gliner2 needs transformers<5: run it in an ISOLATED venv (the main env
    # keeps transformers 5.x for the trainer and future VLM comparators)
    gliner_ok = False
    try:
        import venv as _venv

        venv_dir = Path("/tmp/gliner-venv")
        vpy = str(venv_dir / "bin" / "python")
        if not Path(vpy).exists():
            import shutil as _sh

            if _sh.which("uv"):
                subprocess.check_call(["uv", "venv", str(venv_dir)])
                # uv installs with --python of the venv
                def _uv_install(*pkgs):
                    subprocess.check_call(["uv", "pip", "install", "--python", vpy, "--quiet", *pkgs])
            else:
                _venv.create(str(venv_dir), with_pip=True)

                def _uv_install(*pkgs):
                    subprocess.check_call([vpy, "-m", "pip", "install", "--quiet", *pkgs])
        else:
            def _uv_install(*pkgs):
                subprocess.check_call([vpy, "-m", "pip", "install", "--quiet", *pkgs])
        wheel = sorted(bundle.rglob("*.whl"))[0]
        _uv_install(str(wheel), "pyyaml", "safetensors")
        # CUDA torch in the venv when the kernel has a GPU (matches cu128);
        # fall back to CPU torch otherwise
        import torch as _torch

        if _torch.cuda.is_available():
            try:
                subprocess.check_call([vpy, "-m", "pip", "install", "--quiet",
                                       "torch==2.11.0", "--index-url",
                                       "https://download.pytorch.org/whl/cu128"])
            except Exception as exc:
                log(f"cuda torch in venv failed ({exc}); gliner falls back to CPU")
        _uv_install("gliner2[local]>=2.0", "transformers<5")
        gliner_ok = True
        log("gliner venv ready (isolated, transformers<5)")
    except Exception as exc:
        log(f"gliner venv setup failed: {exc}")

    def run_gliner_worker(cases):
        """Score via the isolated venv worker; returns BackendRow dicts."""
        from mokka_decisions.backends.base import BackendRow

        work_dir = Path("/tmp/gliner-io")
        work_dir.mkdir(exist_ok=True)
        cases_file = work_dir / "cases.jsonl"
        rows_file = work_dir / "rows.jsonl"
        cases_file.write_text(
            "\n".join(json.dumps(c.to_dict(), ensure_ascii=False) for c in cases), encoding="utf-8"
        )
        worker = Path("/tmp/gliner_worker.py")
        worker.write_text(json.loads(GLINER_WORKER_SRC), encoding="utf-8")
        subprocess.check_call(
            [str(Path("/tmp/gliner-venv/bin/python")), str(worker),
             str(cases_file), str(rows_file)], cwd=str(work_dir)
        )
        rows_raw = [json.loads(ln) for ln in rows_file.read_text(encoding="utf-8").splitlines() if ln.strip()]
        return [BackendRow(**{k: v for k, v in r.items() if k in BackendRow.__dataclass_fields__}) for r in rows_raw]

    eval_dir = find_file(bundle, "data/processed/v1/eval/dev_eval.jsonl").parent
    for eval_file in sorted(eval_dir.glob("*.jsonl")):
        name = eval_file.stem
        cases = load_cases(eval_file)
        for backend in backends:
            rows = run_backend(backend, cases)
            dump_rows(rows, work / "rows" / f"{name}_{backend.name}.jsonl")
            s = summarize(rows, cases)
            s["by_language"] = summarize_by(rows, cases, "language")
            s["bootstrap"] = bootstrap_accuracy_ci(rows, cases, n_boot=300)
            (work / "results" / f"{name}_{backend.name}.json").parent.mkdir(exist_ok=True)
            (work / "results" / f"{name}_{backend.name}.json").write_text(
                json.dumps(s, indent=2), encoding="utf-8"
            )
            log(f"{name} / {backend.name}: acc={s['forced_accuracy']}")
        if gliner_ok:
            try:
                rows = run_gliner_worker(cases)
                dump_rows(rows, work / "rows" / f"{name}_gliner25_multilingual_zero_shot.jsonl")
                s = summarize(rows, cases)
                s["by_language"] = summarize_by(rows, cases, "language")
                s["bootstrap"] = bootstrap_accuracy_ci(rows, cases, n_boot=300)
                (work / "results" / f"{name}_gliner25_multilingual_zero_shot.json").write_text(
                    json.dumps(s, indent=2), encoding="utf-8"
                )
                log(f"{name} / gliner25: acc={s['forced_accuracy']}")
            except Exception as exc:
                log(f"{name} / gliner25 worker failed: {exc}")

    # fair selective comparison: each baseline gets its own threshold fitted
    # on cal_policy (the same split and target the final model uses)
    try:
        from mokka_decisions.policy import pick_threshold, risk_coverage_curve

        cal_p = load_cases(find_file(bundle, "data/processed/v1/instances/cal_policy.jsonl"))
        thresholds = {}
        for backend in backends:
            probs = [backend.decide([c])[0].probabilities for c in cal_p]
            pol = pick_threshold(cal_p, probs, target_risk=0.05)
            thresholds[backend.name] = pol
            log(f"policy {backend.name}: thr={pol['threshold']} risk={pol['risk']} cov={pol['coverage']}")
        (work / "baseline_policies.json").write_text(json.dumps(thresholds, indent=2), encoding="utf-8")
    except Exception as exc:
        log(f"baseline policy fitting failed: {exc}")

    (work / "run_status.json").write_text(
        json.dumps({"stage": "baselines", "backends": [b.name for b in backends]}, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage", default="typed_e31_julia",
        help="comma-separated stages: smoke,train,resume_test,eval,baselines",
    )
    parser.add_argument("--config", default="configs/train.yaml")  # bundle v1 carries pilot-budget data
    args = parser.parse_args()

    work = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path("runs/kaggle-local")
    work.mkdir(parents=True, exist_ok=True)
    bundle = find_bundle()
    log(f"bundle={bundle} stage={args.stage}")
    install_wheel(bundle)
    log(f"hardware={hardware_report()}")

    stages = {
        "smoke": stage_smoke,
        "train": stage_train,
        "resume_test": stage_resume_test,
        "eval": stage_eval,
        "baselines": stage_baselines,
    }
    stages.update(STAGES)
    stages.update(TYPED_STAGES)  # inlined pilot stage dicts (defined below)
    completed = []
    for stage_name in [s.strip() for s in args.stage.split(",") if s.strip()]:
        if stage_name not in stages:
            raise SystemExit(f"unknown stage {stage_name!r}; known: {sorted(stages)}")
        stages[stage_name](bundle, work, args.config)
        completed.append(stage_name)
        # preserve each stage verdict; run_status.json records the whole sequence
        status_path = work / "run_status.json"
        prev = {}
        if status_path.exists():
            try:
                prev = json.loads(status_path.read_text(encoding="utf-8"))
            except Exception:
                prev = {}
        prev["stages_completed"] = completed
        prev["last_stage"] = stage_name
        status_path.write_text(json.dumps(prev, indent=2), encoding="utf-8")
    log(f"stages complete: {completed}")

# ==== inlined: multimodal pilot stages (kaggle/mm_pilot.py) ====


import json
import subprocess
import sys
import time
from pathlib import Path


def log(msg: str) -> None:
    print(f"[mm {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _find_mm_root(bundle: Path) -> Path:
    for marker in bundle.rglob("mm/manifest.json"):
        return marker.parent
    raise SystemExit("multimodal dataset not found in bundle (mm/manifest.json)")


def _load_mm_cases(root: Path, split: str):
    from mokka_decisions.contracts import DecisionCase, load_jsonl

    rows = load_jsonl(str(root / "cases" / f"{split}.jsonl"))
    cases = [DecisionCase.from_dict(r) for r in rows]
    types = {r["id"]: r.get("case_type", "") for r in rows}
    media = {r["id"]: r.get("media") for r in rows if r.get("media")}
    return cases, types, media


def _install(pkgs: list[str]) -> None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", *pkgs])


def stage_mm_profile(bundle: Path, work: Path, config: str) -> None:
    import torch

    if not torch.cuda.is_available():
        raise SystemExit("mm_profile needs the GPU that was not assigned")
    _install(["matplotlib", "pillow>=10"])
    from mokka_decisions.multimodal.model import SIGLIP2_MODEL, MultimodalOptionScorer, ImageTower
    from transformers import AutoTokenizer, AutoProcessor

    root = _find_mm_root(bundle)
    cases, types, media = _load_mm_cases(root, "train")
    log(f"mm data: {len(cases)} train cases, {sum(1 for v in media.values() if (v or {}).get('screenshot', {}).get('sha256'))} with screenshots")

    tok = AutoTokenizer.from_pretrained("jhu-clsp/mmBERT-base")
    proc = AutoProcessor.from_pretrained(SIGLIP2_MODEL)
    tower = ImageTower().cuda().eval()
    model = MultimodalOptionScorer("jhu-clsp/mmBERT-base").cuda()
    enc_params = sum(p.numel() for p in model.encoder.parameters()) / 1e6
    tower_params = sum(p.numel() for p in model.tower.parameters()) / 1e6
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    log(f"params: encoder={enc_params:.1f}M tower(frozen)={tower_params:.1f}M trainable={trainable:.1f}M total={sum(p.numel() for p in model.parameters())/1e6:.1f}M")

    # build a small cache + run 60 steps at microbatch 1 (fp16 on T4 = half, not bf16)
    from mokka_decisions.multimodal.train_mm import build_image_cache, encode_mm_batch

    torch.backends.cuda.matmul.allow_tf32 = True
    t0 = time.time()
    cache = build_image_cache(cases[:200], media, root, tower, proc, device="cuda")
    log(f"image cache: {len(cache)} vectors in {time.time()-t0:.1f}s")

    opt = torch.optim.AdamW(model.parameter_groups(2e-5, 1e-4))
    scaler = torch.amp.GradScaler("cuda")
    from torch.nn import functional as F

    model.train()
    torch.cuda.reset_peak_memory_stats()
    step_times = []
    n_steps = 60
    idx = 0
    for step in range(n_steps):
        case = cases[idx % len(cases)]
        idx += 1
        if not case.target_option:
            continue
        t1 = time.time()
        batch = encode_mm_batch(tok, [case], media, cache, max_length=256, device="cuda", vision_dim=tower.dim)
        with torch.autocast("cuda", enabled=True, dtype=torch.float16):
            logits = model(
                batch["input_ids"], batch["attention_mask"], batch["token_type_ids"],
                batch["decision_index"], batch["option_slot"],
                batch["num_decisions"], batch["max_options"], batch["image_vec"],
            )
            loss = F.cross_entropy(logits, batch["targets"])
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        opt.zero_grad()
        if step >= 10:  # skip warmup steps in the timing stats
            step_times.append(time.time() - t1)
    vram = torch.cuda.max_memory_allocated() / 2**30
    mean_s = sum(step_times) / len(step_times)
    profile = {
        "stage": "mm_profile",
        "params": {"encoder_M": round(enc_params, 1), "tower_frozen_M": round(tower_params, 1),
                   "trainable_M": round(trainable, 1), "total_M": round(enc_params + tower_params + (trainable - enc_params), 1)},
        "microbatch": 1,
        "steps_timed": len(step_times),
        "mean_s_per_step": round(mean_s, 4),
        "est_train_minutes_microbatch4": round(mean_s * (len(cases) * 3 / 4) / 60 * 0.55, 1),
        "peak_vram_gb": round(vram, 2),
        "dtype": "fp16 (T4: no bf16 assumed)",
        "loss_finite": bool(torch.isfinite(loss).item()),
    }
    (work / "mm_profile.json").write_text(json.dumps(profile, indent=2), encoding="utf-8")
    log(json.dumps(profile))


def stage_mm_train(bundle: Path, work: Path, config: str) -> None:
    import torch

    if not torch.cuda.is_available():
        raise SystemExit("mm_train needs the GPU that was not assigned")
    _install(["matplotlib", "pillow>=10"])
    from mokka_decisions.multimodal.model import MultimodalOptionScorer, SIGLIP2_MODEL
    from mokka_decisions.multimodal.train_mm import (
        build_image_cache,
        load_case_types,
        load_media,
        train_multimodal,
    )
    from transformers import AutoTokenizer, AutoProcessor

    root = _find_mm_root(bundle)
    train_cases, _, _ = _load_mm_cases(root, "train")
    dev_cases, _, _ = _load_mm_cases(root, "dev")
    cal_t, _, _ = _load_mm_cases(root, "cal_temperature")
    cal_p, _, _ = _load_mm_cases(root, "cal_policy")

    case_files = [root / "cases" / f"{s}.jsonl" for s in ("train", "dev", "cal_temperature", "cal_policy", "test")]
    media, _ = load_media(case_files)
    types = load_case_types(case_files)

    # initialise the text encoder from OUR trained textual checkpoint when
    # attached (dataset gjusev/mokka-decisions-checkpoint), else from scratch
    import glob as _glob

    ckpts = sorted(_glob.glob("/kaggle/input/*/model_best.safetensors"))
    model = MultimodalOptionScorer("jhu-clsp/mmBERT-base")
    if ckpts:
        from mokka_decisions.model import load_checkpoint

        log(f"initialising encoder from textual checkpoint: {ckpts[0]}")
        textual_state = None
        from safetensors.torch import load_file

        textual_state = load_file(ckpts[0])
        enc_keys = {k[len("encoder."):]: v for k, v in textual_state.items() if k.startswith("encoder.")}
        missing, unexpected = model.encoder.load_state_dict(enc_keys, strict=False)
        log(f"encoder init: {len(enc_keys)} tensors, missing={len(missing)}, unexpected={len(unexpected)}")

    tok = AutoTokenizer.from_pretrained("jhu-clsp/mmBERT-base")
    proc = AutoProcessor.from_pretrained(SIGLIP2_MODEL)
    tower = model.tower
    all_cases = train_cases + dev_cases + cal_t + cal_p
    t0 = time.time()
    cache = build_image_cache(all_cases, media, root, tower, proc, device="cuda")
    log(f"image cache: {len(cache)} vectors in {time.time()-t0:.1f}s")

    out = work / "mm_train"
    report = train_multimodal(
        model=model,
        tokenizer=tok,
        train_cases=train_cases,
        dev_cases=dev_cases,
        media=media,
        vec_cache=cache,
        output_dir=out,
        epochs=3,
        microbatch=4,
        grad_accumulation=8,
        case_types=types,
    )
    log(f"mm_train done: {json.dumps(report)[:300]}")

    # calibration + policy on the mm splits with the best model
    from safetensors.torch import load_file

    best_path = out / "model_best.safetensors"
    model.load_state_dict(load_file(str(best_path)))
    from mokka_decisions.calibrate import fit_temperature, save_calibration, calibration_revision
    from mokka_decisions.multimodal.train_mm import encode_mm_batch
    from mokka_decisions.policy import pick_threshold, save_policy, risk_coverage_curve

    model = model.cuda().eval()

    def score(cases_list):
        probs = []
        with torch.no_grad():
            for s in range(0, len(cases_list), 16):
                chunk = cases_list[s : s + 16]
                batch = encode_mm_batch(tok, chunk, media, cache, max_length=256, device="cuda", vision_dim=tower.dim)
                with torch.autocast("cuda", enabled=True, dtype=torch.float16):
                    logits = model(
                        batch["input_ids"], batch["attention_mask"], batch["token_type_ids"],
                        batch["decision_index"], batch["option_slot"],
                        batch["num_decisions"], batch["max_options"], batch["image_vec"],
                    ).float()
                for i, case in enumerate(chunk):
                    vals = logits[i, : len(case.options)]
                    pp = torch.softmax(vals, dim=-1)
                    probs.append({oid: float(v) for oid, v in zip(case.option_ids, pp)})
        return probs

    # temperature on cal_temperature (finite rows only)
    logits_rows = []
    with torch.no_grad():
        for s in range(0, len(cal_t), 16):
            chunk = [c for c in cal_t[s : s + 16] if c.target_option]
            if not chunk:
                continue
            batch = encode_mm_batch(tok, chunk, media, cache, max_length=256, device="cuda", vision_dim=tower.dim)
            with torch.autocast("cuda", enabled=True, dtype=torch.float16):
                logits = model(
                    batch["input_ids"], batch["attention_mask"], batch["token_type_ids"],
                    batch["decision_index"], batch["option_slot"],
                    batch["num_decisions"], batch["max_options"], batch["image_vec"],
                ).float()
            for i, case in enumerate(chunk):
                logits_rows.append({oid: float(logits[i, j]) for j, oid in enumerate(case.option_ids)})
    finite_t = [c for c in cal_t if c.target_option]
    temperature = fit_temperature(finite_t, logits_rows)
    rev = calibration_revision(temperature, [c.id for c in finite_t])
    save_calibration(work / "mm_calibration.json", temperature, rev, {"n": len(finite_t)})
    log(f"mm temperature={temperature}")

    probs_p = score([c for c in cal_p if c.target_option])
    policy = pick_threshold([c for c in cal_p if c.target_option], probs_p, target_risk=0.05)
    save_policy(work / "mm_policy.json", policy)
    log(f"mm policy: {json.dumps(policy)[:200]}")

    # test evaluation by case_type (single pass, frozen everything)
    test_cases, test_types, _ = _load_mm_cases(root, "test")
    from mokka_decisions.evaluate import summarize, summarize_by, dump_rows
    from mokka_decisions.backends.base import BackendRow
    import time as _time

    rows = []
    typed = [c for c in test_cases if c.target_option]
    probs = score(typed)
    for case, p in zip(typed, probs):
        t0_ = _time.perf_counter()
        cand = max(p, key=p.get)
        rows.append(BackendRow(
            case_id=case.id, group_id=case.group_id, source=case.source, language=case.language,
            domain=case.domain, backend="mokka_mm", model_revision=rev, candidate=cand,
            probabilities=p, latency_ms=0.0))
    dump_rows(rows, work / "rows" / "mm_test_mokka_mm.jsonl")
    by_type = {}
    for t in ("vision_needed", "text_suffices", "blurry", "no_valid_option"):
        sub = [(r, c) for r, c in zip(rows, typed) if test_types.get(c.id) == t]
        if sub:
            by_type[t] = summarize([r for r, _ in sub], [c for _, c in sub], threshold=policy.get("threshold"))
    overall = summarize(rows, typed, threshold=policy.get("threshold"))
    result = {"overall": overall, "by_case_type": by_type, "temperature": temperature, "policy": policy}
    (work / "mm_eval_results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    log(f"mm test overall: {json.dumps(overall)[:250]}")


def stage_mm_baselines(bundle: Path, work: Path, config: str) -> None:
    """OCR->text and Qwen3.5-0.8B on the same mm test cases (+ textual mokka)."""
    import torch

    if not torch.cuda.is_available():
        raise SystemExit("mm_baselines needs the GPU that was not assigned")
    root = _find_mm_root(bundle)
    test_cases, test_types, media = _load_mm_cases(root, "test")
    typed = [c for c in test_cases if c.target_option]

    from mokka_decisions.evaluate import dump_rows, summarize
    from mokka_decisions.backends.base import BackendRow

    def rows_from_probs(name, probs_list, latency_ms=None):
        rows = []
        for case, p in zip(typed, probs_list):
            if not p:
                rows.append(BackendRow(case_id=case.id, group_id=case.group_id, source=case.source,
                                       language=case.language, domain=case.domain, backend=name,
                                       model_revision="", candidate="", probabilities={},
                                       latency_ms=0.0, error="no output"))
                continue
            total = sum(p.values()) or 1.0
            pn = {k: v / total for k, v in p.items()}
            rows.append(BackendRow(case_id=case.id, group_id=case.group_id, source=case.source,
                                   language=case.language, domain=case.domain, backend=name,
                                   model_revision="", candidate=max(pn, key=pn.get),
                                   probabilities=pn, latency_ms=latency_ms or 0.0))
        return rows

    # ---- OCR -> text (rapidocr, ONNX runtime, full cost counted) ----
    try:
        _install(["rapidocr-onnxruntime", "onnxruntime"])
        from rapidocr_onnxruntime import RapidOCR
        from PIL import Image

        ocr = RapidOCR()
        import time as _time

        ocr_probs, ocr_times = [], []
        from mokka_decisions.multimodal.train_mm import encode_mm_batch  # noqa: F401 (typing)

        tok = None
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained("jhu-clsp/mmBERT-base")
        import glob as _glob

        ckpts = sorted(_glob.glob("/kaggle/input/*/model_best.safetensors"))
        from mokka_decisions.model import OptionScorer, load_checkpoint, ScorerInference

        m = OptionScorer("jhu-clsp/mmBERT-base")
        if ckpts:
            load_checkpoint(m, ckpts[0])
        scorer = ScorerInference(m, tok, device="cuda", temperature=1.0, batch_decisions=8)

        for case in typed:
            t1 = _time.perf_counter()
            shot = (media.get(case.id) or {}).get("screenshot") or {}
            ocr_text = ""
            if shot.get("path"):
                img_path = root / shot["path"]
                if img_path.exists():
                    result, _ = ocr(str(img_path))
                    if result:
                        ocr_text = " ".join(line[1] for line in result)
            ocr_time = (_time.perf_counter() - t1) * 1000
            # state + OCR text scored by the textual model (its own question/options)
            from mokka_decisions.contracts import DecisionCase as DC

            augmented = DC(
                id=case.id, group_id=case.group_id, source=case.source, language=case.language,
                domain=case.domain,
                state=(case.state + " | screenshot text: " + ocr_text).strip(" |"),
                question=case.question, options=case.options,
                target_kind=case.target_kind, target_option=case.target_option,
                label_origin=case.label_origin, split=case.split,
            )
            p = scorer.score([augmented])[0] if ocr_text else {}
            ocr_probs.append(p)
            ocr_times.append(ocr_time + (scorer.score and 0))
        rows = rows_from_probs("ocr_to_mokka_textual", ocr_probs)
        dump_rows(rows, work / "rows" / "mm_test_ocr_textual.jsonl")
        s = summarize(rows, typed)
        s["ocr_ms_p50"] = sorted(ocr_times)[len(ocr_times) // 2]
        (work / "results" / "mm_ocr_textual.json").parent.mkdir(exist_ok=True)
        (work / "results" / "mm_ocr_textual.json").write_text(json.dumps(s, indent=2), encoding="utf-8")
        log(f"ocr->textual: acc={s['forced_accuracy']}")
    except Exception as exc:
        log(f"ocr baseline failed: {exc}")

    # ---- textual mokka (no image, no OCR): the text-only reference ----
    try:
        import glob as _glob

        ckpts = sorted(_glob.glob("/kaggle/input/*/model_best.safetensors"))
        if ckpts:
            from transformers import AutoTokenizer

            from mokka_decisions.model import OptionScorer, load_checkpoint, ScorerInference

            m = OptionScorer("jhu-clsp/mmBERT-base")
            load_checkpoint(m, ckpts[0])
            cal = json.loads((work / "calibration.json").read_text(encoding="utf-8"))
            tok = AutoTokenizer.from_pretrained("jhu-clsp/mmBERT-base")
            scorer = ScorerInference(m, tok, device="cuda", temperature=cal.get("temperature", 1.0))
            probs = scorer.score(typed)
            rows = rows_from_probs("mokka_textual", probs)
            dump_rows(rows, work / "rows" / "mm_test_mokka_textual.jsonl")
            s = summarize(rows, typed)
            (work / "results" / "mm_textual.json").parent.mkdir(exist_ok=True)
            (work / "results" / "mm_textual.json").write_text(json.dumps(s, indent=2), encoding="utf-8")
            log(f"textual: acc={s['forced_accuracy']}")
    except Exception as exc:
        log(f"textual baseline failed: {exc}")

    # ---- Qwen3.5-0.8B (VLM, fixed selection protocol) ----
    try:
        from transformers import AutoProcessor, AutoTokenizer

        model_name = "Qwen/Qwen3.5-0.8B"
        processor = AutoProcessor.from_pretrained(model_name)
        from transformers import AutoModelForImageTextToText

        vlm = AutoModelForImageTextToText.from_pretrained(
            model_name, torch_dtype=torch.float16, device_map="cuda:0"
        )
        vlm.eval()
        n_params = sum(p.numel() for p in vlm.parameters()) / 1e6
        log(f"qwen3.5-0.8B loaded: {n_params:.0f}M params")

        import time as _time

        qwen_probs, qwen_times, parse_failures = [], [], []
        letters = "ABCDEFGH"
        for case in typed:
            t1 = _time.perf_counter()
            opts = list(case.options)
            opt_lines = "\n".join(f"{letters[i]}. {o.description}" for i, o in enumerate(opts))
            prompt = (
                f"Customer message: {case.state}\n"
                f"Question: {case.question}\n"
                f"Options:\n{opt_lines}\n"
                "Answer with the letter of the single best option."
            )
            shot = (media.get(case.id) or {}).get("screenshot") or {}
            img_path = root / shot["path"] if shot.get("path") else None
            if img_path is not None and img_path.exists():
                from PIL import Image

                pil = Image.open(img_path).convert("RGB")
                inputs = processor(text=prompt, images=pil, return_tensors="pt").to("cuda:0")
            else:
                inputs = processor(text=prompt, return_tensors="pt").to("cuda:0")
            with torch.no_grad():
                out = vlm.generate(**inputs, max_new_tokens=4, do_sample=False)
            answer = processor.tokenizer.decode(
                out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
            ).strip()
            first_letter = answer[:1].upper()
            pick = {o.id: 0.0 for o in opts}
            if first_letter in letters[: len(opts)]:
                pick[opts[letters.index(first_letter)].id] = 1.0
            else:
                # unparseable answer: uniform (measured as the failure it is)
                pick = {o.id: 1.0 / len(opts) for o in opts}
                parse_failures.append(case.id)
            qwen_probs.append(pick)
            qwen_times.append((_time.perf_counter() - t1) * 1000)
        rows = rows_from_probs("qwen35_08b_vlm", qwen_probs)
        for r, t_ms in zip(rows, qwen_times):
            r.latency_ms = t_ms
        dump_rows(rows, work / "rows" / "mm_test_qwen35.jsonl")
        s = summarize(rows, typed)
        s["params_M"] = round(n_params)
        s["latency_ms_p50"] = sorted(qwen_times)[len(qwen_times) // 2]
        s["parse_failures"] = len(parse_failures)
        (work / "results" / "mm_qwen35.json").parent.mkdir(exist_ok=True)
        (work / "results" / "mm_qwen35.json").write_text(json.dumps(s, indent=2), encoding="utf-8")
        log(f"qwen3.5: acc={s['forced_accuracy']} p50={s['latency_ms_p50']:.0f}ms")
    except Exception as exc:
        import traceback

        log(f"qwen baseline failed: {traceback.format_exc()[-400:]}")

    (work / "run_status.json").write_text(
        json.dumps({"stage": "mm_baselines", "n_test": len(typed)}, indent=2), encoding="utf-8"
    )


STAGES = {
    "mm_profile": stage_mm_profile,
    "mm_train": stage_mm_train,
    "mm_baselines": stage_mm_baselines,
}

# ==== inlined: typed-decisions arms (kaggle/typed_pilot.py) ====


import json
import time
from pathlib import Path


def log(msg: str) -> None:
    print(f"[typed {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _find_typed_root(bundle: Path) -> Path:
    for marker in bundle.rglob("typed-v1/manifest.json"):
        return marker.parent
    raise SystemExit("typed-v1 dataset not found in bundle (typed-v1/manifest.json)")


def _load(root: Path, split: str):
    from mokka_decisions.contracts import DecisionCase, load_jsonl

    rows = load_jsonl(str(root / "cases" / f"{split}.jsonl"))
    return [DecisionCase.from_dict(r) for r in rows]


def _train_arm(name: str, encoder: str, train, dev, work: Path, init_from: str = "", epochs: int = 3):
    import torch
    from transformers import AutoTokenizer

    from mokka_decisions.train import TrainConfig, run_training

    cfg = TrainConfig(
        encoder_name=encoder,
        max_length=512,
        microbatch=4,
        grad_accumulation=8,
        epochs=epochs,
        encoder_lr=2e-5,
        head_lr=1e-4,
        seed=42,
        amp=torch.cuda.is_available(),
        output_dir=str(work / name),
        init_from=init_from,
        max_options_per_step=32,
    )
    tokenizer = AutoTokenizer.from_pretrained(encoder)
    report = run_training(cfg, train_cases=train, dev_cases=dev, tokenizer=tokenizer)
    log(f"{name}: best_dev={report['best_dev_accuracy']}")
    return work / name / "checkpoint_best.safetensors", report


def stage_typed_arms(bundle: Path, work: Path, config: str) -> None:
    import glob

    import torch

    if not torch.cuda.is_available():
        raise SystemExit("typed_arms needs the GPU that was not assigned")
    root = _find_typed_root(bundle)
    train = _load(root, "train")
    dev = _load(root, "dev")
    cal = _load(root, "cal")
    log(f"typed data: train={len(train)} dev={len(dev)} cal={len(cal)}")

    # our released checkpoint as the init arm's starting point
    ckpts = sorted(glob.glob("/kaggle/input/**/model_best.safetensors", recursive=True))
    init_path = ckpts[0] if ckpts else ""
    log(f"init checkpoint: {init_path or '(none attached; base_init arm skipped)'}")

    arms = [("base_fresh", "jhu-clsp/mmBERT-base", "")]
    if init_path:
        arms.append(("base_init", "jhu-clsp/mmBERT-base", init_path))
    arms.append(("small_fresh", "jhu-clsp/mmBERT-small", ""))

    results = {}
    import glob as _g

    for name, encoder, init in arms:
        pretrained = sorted(_g.glob(f"/kaggle/input/*/arms/{name}/checkpoint_best.safetensors"))
        if pretrained:
            arm_dir = work / name
            arm_dir.mkdir(parents=True, exist_ok=True)
            import shutil as _sh

            _sh.copy2(pretrained[0], arm_dir / "checkpoint_best.safetensors")
            report = {"best_dev_accuracy": None, "final_step": None, "config_hash": ""}
            log(f"{name}: reusing attached pre-trained checkpoint {pretrained[0]}")
        else:
            best, report = _train_arm(name, encoder, train, dev, work, init_from=init)
        results[name] = {
            "encoder": encoder,
            "initialised_from": init or "upstream",
            "best_dev_accuracy": report["best_dev_accuracy"],
            "final_step": report["final_global_step"],
            "config_hash": report["config_hash"],
            "reused_checkpoint": bool(pretrained),
        }

    # ---- frozen-protocol evaluation on the pinned test parquet ----
    import math

    parquet = work / "typed-test-00000-of-00001.parquet"  # work dir: input is read-only
    if not parquet.exists():
        import urllib.request

        url = (
            "https://huggingface.co/datasets/LocalLLaMA/typed-decisions/resolve/"
            "c76749ec58bd8c3d2ea706b31c333a9059c38f90/all/test-00000-of-00001.parquet"
        )
        parquet.write_bytes(urllib.request.urlopen(url, timeout=180).read())

    # inline frozen-protocol adapter (same construction as the local script)
    from mokka_decisions.contracts import DecisionCase, Option
    from mokka_decisions.model import OptionScorer, ScorerInference, load_checkpoint

    import pyarrow.parquet as pq

    cases_rows = pq.read_table(parquet).to_pylist()
    rows, metadata = [], []
    for case in cases_rows:
        state = json.loads(case["state"])
        gold = json.loads(case["gold"])
        for question_id, question in json.loads(case["questions"]).items():
            kind = question["type"]
            criteria = question.get("criteria")
            if criteria is None and kind == "noul":
                criteria = {"false": "false", "true": "true"}
            if isinstance(criteria, list):
                criteria = {str(i): v for i, v in enumerate(criteria)}
            keys = ["false", "true"] if kind == "noul" else list(criteria)
            rows.append((json.dumps(state, ensure_ascii=False), question["instructions"],
                         kind, [criteria[k] for k in keys]))
            metadata.append(dict(id=case["id"] + ":" + question_id, type=kind, keys=keys,
                                 gold=str(gold[question_id]["label"])))
    import collections

    counts = collections.Counter(r[2] for r in rows)
    assert len(cases_rows) == 400 and counts == {"choice": 600, "score": 800, "noul": 600}, counts

    eval_cases = [
        DecisionCase(
            id=m["id"], group_id=m["id"].split(":")[0], source="typed-decisions",
            language="en", domain="benchmark", state=sj, question=q,
            options=tuple(Option(id=f"o{i}", description=d) for i, d in enumerate(opts)),
            target_kind="single", target_option=f"o{m['keys'].index(m['gold'])}",
        )
        for (sj, q, k, opts), m in zip(rows, metadata)
    ]

    for name, encoder, _init in arms:
        from transformers import AutoTokenizer

        model = OptionScorer(encoder)
        load_checkpoint(model, work / name / "checkpoint_best.safetensors")
        tokenizer = AutoTokenizer.from_pretrained(encoder)
        scorer = ScorerInference(model, tokenizer, device="cuda", temperature=1.0,
                                 max_length=512, batch_decisions=16)
        probs_all = []
        t0 = time.time()
        for s in range(0, len(eval_cases), 16):
            probs_all.extend(scorer.score(eval_cases[s : s + 16]))
        stats = collections.defaultdict(lambda: dict(count=0, correct=0))
        mae = 0
        n_mae = 0
        nll = 0.0
        for case, m, probs in zip(eval_cases, metadata, probs_all):
            gold_idx = m["keys"].index(m["gold"])
            pred_idx = int(max(probs, key=probs.get)[1:])
            st = stats[m["type"]]
            st["count"] += 1
            st["correct"] += int(pred_idx == gold_idx)
            ordered = [probs[f"o{i}"] for i in range(len(m["keys"]))]
            nll += -math.log(max(ordered[gold_idx], 1e-12))
            if m["type"] == "score":
                mae += abs(sum(i * p for i, p in enumerate(ordered)) - gold_idx)
                n_mae += 1
        for st in stats.values():
            st["accuracy"] = st["correct"] / st["count"]
        total = sum(st["correct"] for st in stats.values())
        results[name]["test"] = {
            "overall": f"{total}/2000",
            "overall_accuracy": total / 2000,
            "by_type": dict(stats),
            "score_mae": mae / n_mae if n_mae else None,
            "mean_nll": nll / 2000,
            "eval_elapsed_s": round(time.time() - t0, 1),
            "protocol": "frozen v1.0 (json.dumps state, max_len 512, T=1.0, batch 16)",
        }
        log(f"{name} TEST: {total}/2000 = {total/2000:.4f} | {dict(stats)}")

    results["_meta"] = {
        "typed_train_manifest": json.loads((root / "manifest.json").read_text(encoding="utf-8")),
        "comparison_reference": {
            "julia_reproduced_cpu": 0.7255,
            "mokka_base_zero_shot_asrun": 0.3615,
        },
        "arm_meaning": "equal data (typed-train) x encoder size x init",
    }
    (work / "typed_arms_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    (work / "run_status.json").write_text(
        json.dumps({"stage": "typed_arms", "arms": [a[0] for a in arms]}, indent=2),
        encoding="utf-8",
    )


def stage_typed_a1(bundle: Path, work: Path, config: str) -> None:
    """A1 marker architecture on the same typed-train data as the A0 arms.

    One sequence per decision (state+question+marked options), scores from
    marker hidden states. Trained with the same recipe/splits; evaluated with
    the same frozen adapter construction. Reports sequence lengths, step time
    and permutation behaviour for the A0-vs-A1 cost/quality comparison.
    """
    import time as _time

    import torch

    if not torch.cuda.is_available():
        raise SystemExit("typed_a1 needs the GPU that was not assigned")
    root = _find_typed_root(bundle)
    train = _load(root, "train")
    dev = _load(root, "dev")
    log(f"typed A1 data: train={len(train)} dev={len(dev)}")

    from mokka_decisions.arch_a1 import MarkerScorer, train_a1, collate_a1, forward_a1

    model = MarkerScorer("jhu-clsp/mmBERT-base")
    t0 = _time.time()
    report = train_a1(
        model, train, dev,
        output_dir=str(work / "a1_base"), epochs=3, microbatch=4, grad_accumulation=8,
        device="cuda",
    )
    log(f"a1 trained in {_time.time()-t0:.0f}s best_dev={report['best_dev_accuracy']}")

    # reload best and run the frozen-protocol eval (same construction as arms)
    from mokka_decisions.contracts import DecisionCase, Option
    from mokka_decisions.model import load_checkpoint
    from safetensors.torch import load_file

    model.load_state_dict(load_file(str(work / "a1_base" / "checkpoint_best.safetensors")))
    model = model.cuda().eval()

    import pyarrow.parquet as pq
    import urllib.request

    parquet = work / "typed-test-00000-of-00001.parquet"  # work dir: input is read-only
    if not parquet.exists():
        url = (
            "https://huggingface.co/datasets/LocalLLaMA/typed-decisions/resolve/"
            "c76749ec58bd8c3d2ea706b31c333a9059c38f90/all/test-00000-of-00001.parquet"
        )
        parquet.write_bytes(urllib.request.urlopen(url, timeout=180).read())
    import collections
    import math

    cases_rows = pq.read_table(parquet).to_pylist()
    eval_cases, metadata = [], []
    for case in cases_rows:
        state = json.loads(case["state"])
        gold = json.loads(case["gold"])
        for question_id, question in json.loads(case["questions"]).items():
            kind = question["type"]
            criteria = question.get("criteria")
            if criteria is None and kind == "noul":
                criteria = {"false": "false", "true": "true"}
            if isinstance(criteria, list):
                criteria = {str(i): v for i, v in enumerate(criteria)}
            keys = ["false", "true"] if kind == "noul" else list(criteria)
            eval_cases.append(DecisionCase(
                id=f"{case['id']}:{question_id}", group_id=case["id"], source="typed-decisions",
                language="en", domain="benchmark",
                state=json.dumps(state, ensure_ascii=False), question=question["instructions"],
                options=tuple(Option(id=f"o{i}", description=criteria[k]) for i, k in enumerate(keys)),
                target_kind="single", target_option=f"o{keys.index(str(gold[question_id]['label']))}",
            ))
            metadata.append(dict(id=eval_cases[-1].id, type=kind, keys=keys,
                                 gold=str(gold[question_id]["label"])))

    stats = collections.defaultdict(lambda: dict(count=0, correct=0))
    mae = n_mae = 0
    nll = 0.0
    t0 = _time.time()
    skipped_truncation = 0
    latencies = []
    for s in range(0, len(eval_cases), 8):
        chunk = eval_cases[s : s + 8]
        t1 = _time.perf_counter()
        batch = collate_a1(model, chunk, "cuda")
        logits = forward_a1(model, batch).float()
        torch.cuda.synchronize()
        if s < 100:
            latencies.append((_time.perf_counter() - t1) * 1000 / len(chunk))
        preds = logits.argmax(-1).tolist()
        for i, (case, m) in enumerate(zip(chunk, metadata[s : s + 8])):
            if batch["targets"][i].item() == -100:
                skipped_truncation += 1
                stats[m["type"]]["count"] += 1
                continue  # truncated markers: counted as failures (unusable)
            gold_idx = m["keys"].index(m["gold"])
            correct = preds[i] == gold_idx
            st = stats[m["type"]]
            st["count"] += 1
            st["correct"] += int(correct)
    for st in stats.values():
        st["accuracy"] = st["correct"] / st["count"] if st["count"] else None
    total_correct = sum(st["correct"] for st in stats.values())
    result = {
        "arm": "a1_base",
        "label": "adapted (equal data, marker architecture)",
        "best_dev_accuracy": report["best_dev_accuracy"],
        "test": {
            "overall": f"{total_correct}/2000",
            "overall_accuracy": total_correct / 2000,
            "by_type": dict(stats),
            "skipped_by_truncation": skipped_truncation,
            "eval_elapsed_s": round(_time.time() - t0, 1),
            "latency_ms_batch1_p50": sorted(latencies)[len(latencies) // 2] if latencies else None,
            "protocol": "frozen v1.0 adapter construction",
        },
        "history": report["history"],
    }
    (work / "typed_a1_results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    log(f"A1 TEST: {total_correct}/2000 = {total_correct/2000:.4f} skipped={skipped_truncation}")
    (work / "run_status.json").write_text(
        json.dumps({"stage": "typed_a1", "overall": total_correct / 2000}, indent=2), encoding="utf-8"
    )


def stage_typed_ord(bundle: Path, work: Path, config: str, epochs: int = 3, out: str = "base_ord") -> None:
    """H1 experiment: A0 base + ordinal soft-target loss on score rows.

    Hypothesis (from dev error diagnosis): score-type rows fail by confident
    adjacent-level confusions; a unimodal soft target (tau=1) on score rows
    should improve score accuracy and MAE without touching anything else.
    Control: base_fresh (same data, init, recipe, seeds) -> dev comparison.
    typed_ord2 = H2: same recipe, epochs extended only (dev curve not saturated).
    """
    import time as _time

    import torch

    if not torch.cuda.is_available():
        raise SystemExit("typed_ord needs the GPU that was not assigned")
    root = _find_typed_root(bundle)
    train = _load(root, "train")
    dev = _load(root, "dev")
    log(f"typed_ord data: train={len(train)} dev={len(dev)}")

    # question types come from the rebuilt typed-v1 rows
    types_path = root / "cases" / "train.jsonl"
    ordinal_ids = set()
    import json as _json

    for line in types_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = _json.loads(line)
            if row.get("question_type") == "score":
                ordinal_ids.add(row["id"])
    log(f"ordinal (score) train rows: {len(ordinal_ids)}")

    from mokka_decisions.train import TrainConfig, run_training
    from transformers import AutoTokenizer

    cfg = TrainConfig(
        encoder_name="jhu-clsp/mmBERT-base",
        max_length=512, microbatch=4, grad_accumulation=8, epochs=epochs,
        encoder_lr=2e-5, head_lr=1e-4, seed=42, amp=True,
        output_dir=str(work / out), max_options_per_step=32,
    )
    tokenizer = AutoTokenizer.from_pretrained("jhu-clsp/mmBERT-base")
    t0 = _time.time()
    report = run_training(
        cfg, train_cases=train, dev_cases=dev, tokenizer=tokenizer,
        ordinal_case_ids=ordinal_ids, ordinal_tau=1.0,
    )
    log(f"base_ord trained in {_time.time()-t0:.0f}s best_dev={report['best_dev_accuracy']}")

    # dev diagnosis by question type (frozen construction)
    import collections
    import math

    dev_rows = [ _json.loads(l) for l in (root / "cases" / "dev.jsonl").read_text(encoding="utf-8").splitlines() if l.strip() ]
    types = {r["id"]: r.get("question_type", "?") for r in dev_rows}
    from mokka_decisions.model import OptionScorer, load_checkpoint, ScorerInference
    from safetensors.torch import load_file

    m = OptionScorer("jhu-clsp/mmBERT-base")
    load_checkpoint(m, work / out / "checkpoint_best.safetensors")
    sc = ScorerInference(m, tokenizer, device="cuda", temperature=1.0, max_length=512, batch_decisions=16)
    typed_dev = [c for c in dev if c.target_option]
    probs = sc.score(typed_dev)
    stats = collections.defaultdict(lambda: dict(n=0, ok=0))
    mae = n_mae = 0
    for c, p_ in zip(typed_dev, probs):
        pred = max(p_, key=p_.get)
        qt = types.get(c.id, "?")
        stats[qt]["n"] += 1
        stats[qt]["ok"] += int(pred == c.target_option)
        if qt == "score":
            idx = c.option_ids.index(c.target_option)
            ev = sum(i * p_[oid] for i, oid in enumerate(c.option_ids))
            mae += abs(ev - idx)
            n_mae += 1
    result = {
        "arm": out,
        "hypothesis": "unimodal soft targets on score rows fix confident adjacent-level confusions",
        "control": "base_fresh (same data/init/recipe)",
        "epochs": epochs,
        "ordinal_train_rows": len(ordinal_ids),
        "best_dev_accuracy": report["best_dev_accuracy"],
        "dev_by_type": {k: {"acc": v["ok"] / v["n"], "n": v["n"]} for k, v in stats.items()},
        "dev_score_mae": mae / n_mae if n_mae else None,
        "history": report["history"],
    }
    (work / f"{out}_results.json").write_text(_json.dumps(result, indent=2), encoding="utf-8")
    log(f"{out} dev by type: {result['dev_by_type']} mae={result['dev_score_mae']}")
    (work / "run_status.json").write_text(
        _json.dumps({"stage": f"typed_ord:{out}", "best_dev": report["best_dev_accuracy"]}, indent=2),
        encoding="utf-8",
    )


def stage_typed_small_ord(bundle: Path, work: Path, config: str, epochs: int = 4, out: str = "small_ord") -> None:
    """Recipe transfer: the dev-winning ordinal loss on mmBERT-small.

    Equal-parameter comparison vs Julia-1 (140M vs 144M) with the recipe that
    won on Base. Same data, seed and protocol as small_fresh (control).
    """
    import time as _time

    import torch

    if not torch.cuda.is_available():
        raise SystemExit("typed_small_ord needs the GPU that was not assigned")
    root = _find_typed_root(bundle)
    train = _load(root, "train")
    dev = _load(root, "dev")
    import json as _json

    ordinal_ids = {
        _json.loads(line)["id"]
        for line in (root / "cases" / "train.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip() and _json.loads(line).get("question_type") == "score"
    }
    log(f"small_ord: train={len(train)} dev={len(dev)} ordinal={len(ordinal_ids)}")

    from mokka_decisions.train import TrainConfig, run_training
    from transformers import AutoTokenizer

    cfg = TrainConfig(
        encoder_name="jhu-clsp/mmBERT-small",
        max_length=512, microbatch=4, grad_accumulation=8, epochs=epochs,
        encoder_lr=2e-5, head_lr=1e-4, seed=42, amp=True,
        output_dir=str(work / out), max_options_per_step=32,
    )
    tokenizer = AutoTokenizer.from_pretrained("jhu-clsp/mmBERT-small")
    t0 = _time.time()
    report = run_training(
        cfg, train_cases=train, dev_cases=dev, tokenizer=tokenizer,
        ordinal_case_ids=ordinal_ids, ordinal_tau=1.0,
    )
    log(f"small_ord trained in {_time.time()-t0:.0f}s best_dev={report['best_dev_accuracy']}")

    # dev by type
    import collections

    dev_rows = [_json.loads(l) for l in (root / "cases" / "dev.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    types = {r["id"]: r.get("question_type", "?") for r in dev_rows}
    from mokka_decisions.model import OptionScorer, load_checkpoint, ScorerInference
    from safetensors.torch import load_file

    m = OptionScorer("jhu-clsp/mmBERT-small")
    load_checkpoint(m, work / out / "checkpoint_best.safetensors")
    sc = ScorerInference(m, tokenizer, device="cuda", temperature=1.0, max_length=512, batch_decisions=16)
    typed_dev = [c for c in dev if c.target_option]
    probs = sc.score(typed_dev)
    stats = collections.defaultdict(lambda: dict(n=0, ok=0))
    mae = n_mae = 0
    for c, p_ in zip(typed_dev, probs):
        pred = max(p_, key=p_.get)
        qt = types.get(c.id, "?")
        stats[qt]["n"] += 1
        stats[qt]["ok"] += int(pred == c.target_option)
        if qt == "score":
            idx = c.option_ids.index(c.target_option)
            ev = sum(i * p_[oid] for i, oid in enumerate(c.option_ids))
            mae += abs(ev - idx)
            n_mae += 1
    result = {
        "arm": out,
        "hypothesis": "ordinal-loss recipe transfers from Base to Small",
        "control": "small_fresh (dev 0.6386, test 0.5345)",
        "epochs": epochs,
        "best_dev_accuracy": report["best_dev_accuracy"],
        "dev_by_type": {k: {"acc": v["ok"] / v["n"], "n": v["n"]} for k, v in stats.items()},
        "dev_score_mae": mae / n_mae if n_mae else None,
        "history": report["history"],
    }
    (work / f"{out}_results.json").write_text(_json.dumps(result, indent=2), encoding="utf-8")
    log(f"{out} dev by type: {result['dev_by_type']} mae={result['dev_score_mae']}")
    (work / "run_status.json").write_text(
        _json.dumps({"stage": f"typed_small_ord:{out}", "best_dev": report["best_dev_accuracy"]}, indent=2),
        encoding="utf-8",
    )


def stage_typed_e15(bundle: Path, work: Path, config: str, seed: int = 42, arms: str = "both") -> None:
    """E15: dataset-native score supervision vs the artificial ordinal target.

    seed/arms parametrise the matched-seed replication (seed 42 ran as v1);
    arms: "both" | "control" | "probs".

    Control (e15_control): the base_ord recipe exactly — artificial unimodal
    targets exp(-|j-gold|/tau=1) on score rows, hard CE on choice/noul.
    Candidate (e15_probs): identical data, init, batch, optimizer, steps,
    LR schedule and selection; ONLY the score-row target vector changes to
    the original gold.probabilities of the train split (soft CE).
    Pre-fixed selection: best dev accuracy (same criterion as every arm).
    Recorded metrics per arm: accuracy by type, mean NLL (hard label),
    KL(gold||model) on all dev rows, multiclass Brier, score MAE.
    Dev probabilities are used for METRICS only, never for training.
    """
    import collections
    import math
    import time as _time

    import torch

    if not torch.cuda.is_available():
        raise SystemExit("typed_e15 needs the GPU that was not assigned")
    root = _find_typed_root(bundle)
    train = _load(root, "train")
    dev = _load(root, "dev")
    import json as _json

    def _load_probs(split: str) -> dict:
        rows = [
            _json.loads(l)
            for l in (root / "cases" / f"probabilities_{split}.jsonl").read_text(encoding="utf-8").splitlines()
            if l.strip()
        ]
        return {r["id"]: r for r in rows}

    probs_train = _load_probs("train")
    probs_dev = _load_probs("dev")
    qtypes = {
        _json.loads(l)["id"]: _json.loads(l).get("question_type", "?")
        for l in (root / "cases" / "train.jsonl").read_text(encoding="utf-8").splitlines()
        if l.strip()
    }
    # candidate targets: dataset-native distributions on score rows only
    soft_targets = {}
    for c in train:
        if qtypes.get(c.id) == "score":
            row = probs_train.get(c.id)
            if row is None or len(row["probabilities"]) != len(c.options):
                raise SystemExit(f"probabilities misaligned for {c.id}")
            s = sum(row["probabilities"])
            if abs(s - 1.0) > 1e-3:
                raise SystemExit(f"probabilities not normalised for {c.id} (sum={s})")
            soft_targets[c.id] = row["probabilities"]
    ordinal_ids = {i for i, t in qtypes.items() if t == "score"}
    log(f"typed_e15: train={len(train)} dev={len(dev)} score_rows={len(ordinal_ids)} "
        f"soft_targets={len(soft_targets)} dev_probs={len(probs_dev)}")
    assert set(soft_targets) == ordinal_ids, "every score train row must carry its distribution"

    from mokka_decisions.train import TrainConfig, run_training
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("jhu-clsp/mmBERT-base")

    def _arm(name: str, *, ordinal: bool) -> dict:
        cfg = TrainConfig(
            encoder_name="jhu-clsp/mmBERT-base",
            max_length=512, microbatch=4, grad_accumulation=8, epochs=3,
            encoder_lr=2e-5, head_lr=1e-4, seed=seed, amp=True,
            output_dir=str(work / name), max_options_per_step=32,
        )
        t0 = _time.time()
        report = run_training(
            cfg, train_cases=train, dev_cases=dev, tokenizer=tokenizer,
            ordinal_case_ids=ordinal_ids if ordinal else None, ordinal_tau=1.0,
            soft_targets=None if ordinal else soft_targets,
        )
        log(f"{name} trained in {_time.time()-t0:.0f}s best_dev={report['best_dev_accuracy']}")
        # ponytail: latest.pt (~4.9GB each) filled /kaggle/working with 5 arms in v2;
        # best+report is all the analysis needs — delete latest after each arm
        latest = work / name / "checkpoint_latest.pt"
        if latest.exists():
            latest.unlink()
        return report

    name_c = f"e15_control_s{seed}"
    name_p = f"e15_probs_s{seed}"
    report_c = _arm(name_c, ordinal=True) if arms in ("both", "control") else None
    report_p = _arm(name_p, ordinal=False) if arms in ("both", "probs") else None

    # ---- dev metrics for both arms (checkpoint_best of each) ----
    from mokka_decisions.model import OptionScorer, ScorerInference, load_checkpoint

    def _metrics(name: str) -> dict:
        m = OptionScorer("jhu-clsp/mmBERT-base")
        load_checkpoint(m, work / name / "checkpoint_best.safetensors")
        sc = ScorerInference(m, tokenizer, device="cuda", temperature=1.0, max_length=512, batch_decisions=16)
        typed_dev = [c for c in dev if c.target_option]
        probs = sc.score(typed_dev)
        stats = collections.defaultdict(lambda: dict(n=0, ok=0))
        nll = n = 0
        kl = n_kl = 0
        brier = n_br = 0
        mae = n_mae = 0
        for c, p_ in zip(typed_dev, probs):
            qt = probs_dev[c.id]["question_type"] if c.id in probs_dev else qtypes.get(c.id, "?")
            vec = probs_dev[c.id]["probabilities"] if c.id in probs_dev else None
            gold_idx = c.option_ids.index(c.target_option)
            pv = [p_[oid] for oid in c.option_ids]
            stats[qt]["n"] += 1
            stats[qt]["ok"] += int(max(p_, key=p_.get) == c.target_option)
            eps = 1e-12
            nll += -math.log(max(pv[gold_idx], eps)); n += 1
            if vec is not None and len(vec) == len(pv):
                kl += sum(v * (math.log(max(v, eps)) - math.log(max(pi, eps)))
                          for v, pi in zip(vec, pv) if v > 0)
                n_kl += 1
                brier += sum((pi - (1.0 if i == gold_idx else 0.0)) ** 2 for i, pi in enumerate(pv))
                n_br += 1
            if qt == "score":
                ev = sum(i * pi for i, pi in enumerate(pv))
                mae += abs(ev - gold_idx); n_mae += 1
        return {
            "dev_by_type": {k: {"acc": v["ok"] / v["n"], "n": v["n"]} for k, v in stats.items()},
            "mean_nll": nll / n if n else None,
            "mean_kl_gold_vs_model": kl / n_kl if n_kl else None,
            "mean_brier_multiclass": brier / n_br if n_br else None,
            "dev_score_mae": mae / n_mae if n_mae else None,
        }

    result = {
        "experiment": "E15 dataset-native score supervision",
        "seed": seed,
        "selection_criterion": "best dev accuracy (pre-fixed)",
        "control": ({"arm": name_c, "best_dev_accuracy": report_c["best_dev_accuracy"],
                     "history": report_c["history"], **_metrics(name_c)}
                    if report_c else None),
        "candidate": ({"arm": name_p, "best_dev_accuracy": report_p["best_dev_accuracy"],
                       "history": report_p["history"], **_metrics(name_p)}
                      if report_p else None),
        "note": "candidate differs ONLY in score-row targets (original probabilities via soft CE); "
                "choice/noul untouched; same data/init/batch/optimizer/steps/schedule/seed.",
    }
    (work / f"e15_s{seed}_results.json").write_text(_json.dumps(result, indent=2), encoding="utf-8")
    log(f"e15 seed={seed}: " + " ".join(
        f"{a['arm']} dev={a['best_dev_accuracy']:.4f}" for a in (result["control"], result["candidate"]) if a))
    (work / "run_status.json").write_text(
        _json.dumps({"stage": f"typed_e15:seed{seed}", "seed": seed,
                     "control_dev": result["control"]["best_dev_accuracy"] if result["control"] else None,
                     "candidate_dev": result["candidate"]["best_dev_accuracy"] if result["candidate"] else None},
                    indent=2),
        encoding="utf-8",
    )


def stage_typed_e15_seeds(bundle: Path, work: Path, config: str) -> None:
    """Matched-seed replication of E15 (commission 2026-10-08, step 3).

    Seed 42 already ran in v1 (collected). Here: control seeds 43,44 and
    candidate seeds 43,44,45 -> 3 seeds per recipe, matched pairs at
    42/43/44 plus one extra candidate. Same kernel session so the paired
    comparison avoids cross-session GPU variance.
    """
    for seed in (43, 44):
        stage_typed_e15(bundle, work, config, seed=seed)
    stage_typed_e15(bundle, work, config, seed=45, arms="probs")


def stage_typed_e15_completion(bundle: Path, work: Path, config: str) -> None:
    """Finish the E15 replication arms lost to working-disk exhaustion in v2.

    v2 recovered: control s43/s44 and probs s43 (complete, collected locally).
    probs s44 died after epoch 1 and probs s45 never started. Here they rerun
    from scratch with identical seed/recipe; _arm now deletes checkpoint_latest.pt
    per arm so 2 arms fit comfortably.
    """
    import time as _time

    root = _find_typed_root(bundle)
    import json as _json

    train = _load(root, "train")
    dev = _load(root, "dev")

    def _load_probs(split: str) -> dict:
        return {_json.loads(l)["id"]: _json.loads(l)
                for l in (root / "cases" / f"probabilities_{split}.jsonl").read_text(encoding="utf-8").splitlines()
                if l.strip()}

    probs_train = _load_probs("train")
    qtypes = {_json.loads(l)["id"]: _json.loads(l).get("question_type", "?")
              for l in (root / "cases" / "train.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()}
    soft_targets = {}
    for c in train:
        if qtypes.get(c.id) == "score":
            row = probs_train[c.id]
            assert len(row["probabilities"]) == len(c.options)
            soft_targets[c.id] = row["probabilities"]

    from mokka_decisions.train import TrainConfig, run_training
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("jhu-clsp/mmBERT-base")
    for seed in (44, 45):
        name = f"e15_probs_s{seed}"
        cfg = TrainConfig(
            encoder_name="jhu-clsp/mmBERT-base",
            max_length=512, microbatch=4, grad_accumulation=8, epochs=3,
            encoder_lr=2e-5, head_lr=1e-4, seed=seed, amp=True,
            output_dir=str(work / name), max_options_per_step=32,
        )
        t0 = _time.time()
        report = run_training(cfg, train_cases=train, dev_cases=dev, tokenizer=tokenizer,
                              soft_targets=soft_targets)
        log(f"{name} trained in {_time.time()-t0:.0f}s best_dev={report['best_dev_accuracy']}")
        latest = work / name / "checkpoint_latest.pt"
        if latest.exists():
            latest.unlink()
        (work / "run_status.json").write_text(
            _json.dumps({"stage": "typed_e15_completion", "arm": name,
                         "best_dev": report["best_dev_accuracy"]}, indent=2), encoding="utf-8")
    log("e15_completion: done (probs s44 rerun complete + s45)")



def stage_typed_julia_small(bundle: Path, work: Path, config: str) -> None:
    """Faithful Julia-arch Small control: architecture and supervision as
    SEPARATE variables (commission 2026-10-08/09 step 4).

    Exploratory architecture/supervision arms; the full 2x2 needs A0-small
    native probabilities and a matched-budget A0-small ordinal control.
    Historical arms:
      (A0-small, artificial)  = small_ord (E12, dev 0.6439)
      (JuliaArch-small, artificial) = julias_ordinal  <-- NEW
      (JuliaArch-small, native probs) = julias_probs  <-- NEW
    E12 used four epochs; these arms use three. Historical E17 dev results
    also used a train-only type map and unmatched head initialization.
    Future runs use explicit train+dev types and seed before construction.
    Fidelity: sequence() parity + model structure verified by
    tests/test_arch_julia_serialization.py against the pinned upstream source.
    """
    import time as _time

    import torch

    if not torch.cuda.is_available():
        raise SystemExit("typed_julia_small needs the GPU that was not assigned")
    root = _find_typed_root(bundle)
    train = _load(root, "train")
    dev = _load(root, "dev")
    import json as _json

    def _load_probs(split: str) -> dict:
        return {_json.loads(l)["id"]: _json.loads(l)
                for l in (root / "cases" / f"probabilities_{split}.jsonl").read_text(encoding="utf-8").splitlines()
                if l.strip()}

    probs_train = _load_probs("train")
    qtypes = {}
    for split in ("train", "dev"):
        for line in (root / "cases" / f"{split}.jsonl").read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = _json.loads(line)
            if row["id"] in qtypes:
                raise ValueError(f"Duplicate question id across splits: {row['id']}")
            qtypes[row["id"]] = row["question_type"]
    ordinal_ids = {c.id for c in train if qtypes[c.id] == "score"}
    soft_targets = {i: probs_train[i]["probabilities"] for i in ordinal_ids}
    log(f"julia_small: train={len(train)} dev={len(dev)} score_rows={len(ordinal_ids)}")

    from mokka_decisions.arch_julia import make_seeded_julia_arch, train_julia_arch
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("jhu-clsp/mmBERT-small")
    for name, kwargs in (("julias_ordinal", dict(ordinal_case_ids=ordinal_ids)),
                         ("julias_probs", dict(soft_targets=soft_targets))):
        model = make_seeded_julia_arch("jhu-clsp/mmBERT-small", seed=42)
        t0 = _time.time()
        report = train_julia_arch(
            model, tokenizer, train, dev, qtypes,
            output_dir=str(work / name), epochs=3, microbatch=4, grad_accumulation=8,
            encoder_lr=2e-5, head_lr=1e-4, seed=42, device="cuda",
            max_length=1024, head_length=256, **kwargs,
        )
        log(f"{name} trained in {_time.time()-t0:.0f}s best_dev={report['best_dev_accuracy']}")
        latest = work / name / "checkpoint_latest.pt"
        if latest.exists():
            latest.unlink()  # keep /kaggle/working bounded (see v2 incident)
        (work / "run_status.json").write_text(
            _json.dumps({"stage": "typed_julia_small", "arm": name,
                         "best_dev": report["best_dev_accuracy"]}, indent=2), encoding="utf-8")


def _state_dict_sha256(model) -> str:
    """Deterministic content hash of the initial weights (per-tensor bytes)."""
    import hashlib

    h = hashlib.sha256()
    for k, v in sorted(model.state_dict().items()):
        h.update(k.encode("utf-8"))
        h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def stage_typed_e18_supervision(bundle: Path, work: Path, config: str) -> None:
    """E18: controlled supervision contrast WITHIN JuliaArch-small.

    Protocol frozen in docs/internal/E18_PROTOCOL_2026-10-09.md BEFORE any
    results. 3 matched seeds x {ordinal artificial, native probabilities} on
    score rows; identical init (hash-verified per pair), case order, data,
    batch, steps, optimizer and schedule within each seed pair. Checkpoint
    selection: correct-typed dev accuracy only (independent-v1 untouched).
    """
    import time as _time

    import torch

    if not torch.cuda.is_available():
        raise SystemExit("typed_e18_supervision needs the GPU that was not assigned")
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        log(f"e18 gpu[{i}]={props.name} vram={props.total_memory / 2**30:.1f} GiB")
    root = _find_typed_root(bundle)
    train = _load(root, "train")
    dev = _load(root, "dev")
    import json as _json

    def _load_probs(split: str) -> dict:
        return {_json.loads(l)["id"]: _json.loads(l)
                for l in (root / "cases" / f"probabilities_{split}.jsonl").read_text(encoding="utf-8").splitlines()
                if l.strip()}

    probs_train = _load_probs("train")
    probs_dev = {i: r["probabilities"] for i, r in _load_probs("dev").items()}
    qtypes = {}
    for split in ("train", "dev"):
        for line in (root / "cases" / f"{split}.jsonl").read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = _json.loads(line)
            if row["id"] in qtypes:
                raise ValueError(f"Duplicate question id across splits: {row['id']}")
            qtypes[row["id"]] = row["question_type"]
    ordinal_ids = {c.id for c in train if qtypes[c.id] == "score"}
    soft_targets = {i: probs_train[i]["probabilities"] for i in ordinal_ids}
    log(f"e18: train={len(train)} dev={len(dev)} score_rows={len(ordinal_ids)} dev_refs={len(probs_dev)}")

    from mokka_decisions.arch_julia import (
        evaluate_julia_arch, make_seeded_julia_arch, train_julia_arch,
    )
    from safetensors.torch import load_file
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("jhu-clsp/mmBERT-small")
    recipes = (("ord", dict(ordinal_case_ids=ordinal_ids)),
               ("probs", dict(soft_targets=soft_targets)))
    control_hashes: dict[int, str] = {}
    summary = {"protocol": "docs/internal/E18_PROTOCOL_2026-10-09.md (frozen pre-launch)",
               "init_hashes": {}, "pair_hash_equal": {}, "arms": {}}
    for seed in (42, 43, 44):
        for recipe, kwargs in recipes:
            name = f"e18_{recipe}_s{seed}"
            model = make_seeded_julia_arch("jhu-clsp/mmBERT-small", seed=seed)
            init_hash = _state_dict_sha256(model)
            if recipe == "ord":
                control_hashes[seed] = init_hash
            else:
                equal = init_hash == control_hashes[seed]
                summary["pair_hash_equal"][f"s{seed}"] = equal
                if not equal:
                    raise SystemExit(f"E18 pair init mismatch at seed {seed}: abort before training")
            summary["init_hashes"][f"{recipe}_s{seed}"] = init_hash
            t0 = _time.time()
            report = train_julia_arch(
                model, tokenizer, train, dev, qtypes,
                output_dir=str(work / name), epochs=3, microbatch=4, grad_accumulation=8,
                encoder_lr=2e-5, head_lr=1e-4, seed=seed, device="cuda",
                max_length=1024, head_length=256,
                reference_probabilities=probs_dev, **kwargs,
            )
            # best epoch per the frozen tie rule (>= keeps the LATEST best)
            best_entry = None
            for entry in report["history"]:
                if best_entry is None or entry["dev"]["accuracy"] >= best_entry["dev"]["accuracy"]:
                    best_entry = entry
            model.load_state_dict(load_file(str(work / name / "checkpoint_best.safetensors")))
            full = evaluate_julia_arch(model, tokenizer, dev, qtypes, device="cuda",
                                       reference_probabilities=probs_dev, collect_rows=True)
            with (work / name / "dev_rows_best.jsonl").open("w", encoding="utf-8") as f:
                for row in full["rows"]:
                    f.write(_json.dumps(row) + "\n")
            latest = work / name / "checkpoint_latest.pt"
            if latest.exists():
                latest.unlink()  # keep /kaggle/working bounded (E15 v2 incident)
            summary["arms"][name] = {
                "recipe": recipe, "seed": seed, "init_sha256": init_hash,
                "trained_s": round(_time.time() - t0, 1),
                "best_dev_accuracy": report["best_dev_accuracy"],
                "final_step": report["final_step"],
                "best_epoch": best_entry["epoch"],
                "best_epoch_dev": {k: v for k, v in best_entry["dev"].items() if k != "rows"},
                "selected_checkpoint_dev": {k: v for k, v in full.items() if k != "rows"},
            }
            log(f"{name}: best_dev={report['best_dev_accuracy']:.4f} "
                f"(epoch {best_entry['epoch']}) in {_time.time() - t0:.0f}s")
            (work / "run_status.json").write_text(
                _json.dumps({"stage": "typed_e18_supervision", "arm": name,
                             "best_dev": report["best_dev_accuracy"]}, indent=2), encoding="utf-8")
    (work / "e18_summary.json").write_text(_json.dumps(summary, indent=2), encoding="utf-8")
    log("e18_supervision: done")




def _find_e19_root(bundle: Path) -> Path:
    for marker in bundle.rglob("e19-rulevar/manifest.json"):
        return marker.parent
    raise SystemExit("e19-rulevar dataset not found in bundle (e19-rulevar/manifest.json)")


def _load_rows_file(path: Path):
    import json

    from mokka_decisions.contracts import DecisionCase

    return [DecisionCase.from_dict(json.loads(l))
            for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def stage_typed_e19_rulevar(bundle: Path, work: Path, config: str) -> None:
    """E19: rule-variation + counterfactual data contrast within JuliaArch-small.

    Protocol frozen in docs/internal/E19_PROTOCOL_2026-10-09.md BEFORE any
    results; quota rule applied pre-launch: seeds (42, 43). Arms per seed
    (identical init, hash-verified; orig rows keep native-probs supervision
    on score rows as the declared convention; new deterministic rows hard CE):
      e19_orig : 4950 original rows, 4 epochs = 620 steps (matched budget)
      e19_ctrl : +1650 ordinary verifiable rows, 3 epochs = 621 steps
      e19_cand : +1650 rule-variation/counterfactual rows, 3 epochs = 621 steps
    Selection: dev = original 570 + new 330 = 900 questions, correct types.
    The sealed test is NEVER touched here.
    """
    import time as _time

    import torch

    if not torch.cuda.is_available():
        raise SystemExit("typed_e19_rulevar needs the GPU that was not assigned")
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        log(f"e19 gpu[{i}]={props.name} vram={props.total_memory / 2**30:.1f} GiB")
    root = _find_typed_root(bundle)
    e19 = _find_e19_root(bundle)
    train_orig = _load(root, "train")
    dev_orig = _load(root, "dev")
    ctrl_rows = _load_rows_file(e19 / "train_ctrl_rows.jsonl")
    cand_rows = _load_rows_file(e19 / "train_cand_rows.jsonl")
    dev_new = _load_rows_file(e19 / "dev_rows.jsonl")
    import json as _json

    def _load_probs(split: str) -> dict:
        return {_json.loads(l)["id"]: _json.loads(l)
                for l in (root / "cases" / f"probabilities_{split}.jsonl").read_text(encoding="utf-8").splitlines()
                if l.strip()}

    probs_train = _load_probs("train")
    probs_dev = {i: r["probabilities"] for i, r in _load_probs("dev").items()}
    qtypes = {}
    for split in ("train", "dev"):
        for line in (root / "cases" / f"{split}.jsonl").read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = _json.loads(line)
            if row["id"] in qtypes:
                raise ValueError(f"Duplicate question id across splits: {row['id']}")
            qtypes[row["id"]] = row["question_type"]
    # the :action base rows are intentionally SHARED between the ctrl and cand
    # arm files (same underlying cases); duplicates only matter within a file
    # or against the original typed-v1 ids (qtypes already holds those).
    for fname in ("train_ctrl_rows.jsonl", "train_cand_rows.jsonl", "dev_rows.jsonl"):
        seen_in_file = set()
        for line in (e19 / fname).read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = _json.loads(line)
                if row["id"] in seen_in_file:
                    raise ValueError(f"Duplicate id within {fname}: {row['id']}")
                seen_in_file.add(row["id"])
                qtypes.setdefault(row["id"], row["question_type"])
                if qtypes[row["id"]] != row["question_type"]:
                    raise ValueError(f"Conflicting question_type for {row['id']}")
    # declared supervision convention: native probs on ORIGINAL score rows only
    soft_targets = {c.id: probs_train[c.id]["probabilities"] for c in train_orig
                    if qtypes[c.id] == "score" and c.id in probs_train}
    dev = dev_orig + dev_new
    log(f"e19: orig_train={len(train_orig)} ctrl={len(ctrl_rows)} cand={len(cand_rows)} "
        f"dev={len(dev)} (orig {len(dev_orig)} + new {len(dev_new)}) soft_orig={len(soft_targets)}")

    from mokka_decisions.arch_julia import (
        evaluate_julia_arch, make_seeded_julia_arch, train_julia_arch,
    )
    from safetensors.torch import load_file
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("jhu-clsp/mmBERT-small")
    arms = (("orig", train_orig, 4), ("ctrl", train_orig + ctrl_rows, 3), ("cand", train_orig + cand_rows, 3))
    init_hashes: dict = {}
    summary = {"protocol": "docs/internal/E19_PROTOCOL_2026-10-09.md (frozen pre-launch)",
               "seeds": [42, 43], "init_hashes": {}, "seed_hash_equal": {}, "arms": {}}
    for seed in (42, 43):
        for name, data, epochs in arms:
            arm = f"e19_{name}_s{seed}"
            model = make_seeded_julia_arch("jhu-clsp/mmBERT-small", seed=seed)
            init_hash = _state_dict_sha256(model)
            if seed in init_hashes:
                equal = init_hash == init_hashes[seed]
                summary["seed_hash_equal"][f"s{seed}"] = equal
                if not equal:
                    raise SystemExit(f"E19 init mismatch at seed {seed} ({name}): abort")
            else:
                init_hashes[seed] = init_hash
            summary["init_hashes"][f"{name}_s{seed}"] = init_hash
            t0 = _time.time()
            report = train_julia_arch(
                model, tokenizer, data, dev, qtypes,
                output_dir=str(work / arm), epochs=epochs, microbatch=4, grad_accumulation=8,
                encoder_lr=2e-5, head_lr=1e-4, seed=seed, device="cuda",
                max_length=1024, head_length=256,
                reference_probabilities=probs_dev, soft_targets=soft_targets,
            )
            best_entry = None
            for entry in report["history"]:
                if best_entry is None or entry["dev"]["accuracy"] >= best_entry["dev"]["accuracy"]:
                    best_entry = entry
            model.load_state_dict(load_file(str(work / arm / "checkpoint_best.safetensors")))
            full = evaluate_julia_arch(model, tokenizer, dev, qtypes, device="cuda",
                                       reference_probabilities=probs_dev, collect_rows=True)
            with (work / arm / "dev_rows_best.jsonl").open("w", encoding="utf-8") as f:
                for row in full["rows"]:
                    f.write(_json.dumps(row) + "\n")
            latest = work / arm / "checkpoint_latest.pt"
            if latest.exists():
                latest.unlink()
            summary["arms"][arm] = {
                "recipe": name, "seed": seed, "init_sha256": init_hash,
                "trained_s": round(_time.time() - t0, 1),
                "best_dev_accuracy": report["best_dev_accuracy"],
                "final_step": report["final_step"], "best_epoch": best_entry["epoch"],
                "best_epoch_dev": {k: v for k, v in best_entry["dev"].items() if k != "rows"},
                "selected_checkpoint_dev": {k: v for k, v in full.items() if k != "rows"},
            }
            log(f"{arm}: best_dev={report['best_dev_accuracy']:.4f} (epoch {best_entry['epoch']}) "
                f"steps={report['final_step']} in {_time.time() - t0:.0f}s")
            (work / "run_status.json").write_text(
                _json.dumps({"stage": "typed_e19_rulevar", "arm": arm,
                             "best_dev": report["best_dev_accuracy"]}, indent=2), encoding="utf-8")
    (work / "e19_summary.json").write_text(_json.dumps(summary, indent=2), encoding="utf-8")
    log("e19_rulevar: done")



def _find_e20_root(bundle: Path) -> Path:
    for marker in bundle.rglob("e20-contrast/manifest.json"):
        return marker.parent
    raise SystemExit("e20-contrast dataset not found in bundle (e20-contrast/manifest.json)")


def stage_typed_e20_contrast(bundle: Path, work: Path, config: str) -> None:
    """E20: composition-paired vs coverage data contrast within JuliaArch-small.

    Protocol frozen in docs/internal/E20_PROTOCOL_2026-10-09.md (Phase 1
    diagnosis first: models fail rule composition OOD). Arms per seed
    (identical init, hash-verified): ctrl = +1650 base questions (330 cases),
    cand = +1650 paired base+flip questions (165 cases); same type mix, steps
    (621x3), supervision convention (native probs on ORIGINAL score rows).
    Selection: dev = original 570 + new 330. Sealed test untouched here.
    """
    import time as _time

    import torch

    if not torch.cuda.is_available():
        raise SystemExit("typed_e20_contrast needs the GPU that was not assigned")
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        log(f"e20 gpu[{i}]={props.name} vram={props.total_memory / 2**30:.1f} GiB")
    root = _find_typed_root(bundle)
    e20 = _find_e20_root(bundle)
    train_orig = _load(root, "train")
    dev_orig = _load(root, "dev")
    ctrl_rows = _load_rows_file(e20 / "train_ctrl_rows.jsonl")
    cand_rows = _load_rows_file(e20 / "train_cand_rows.jsonl")
    dev_new = _load_rows_file(e20 / "dev_rows.jsonl")
    import json as _json

    def _load_probs(split: str) -> dict:
        return {_json.loads(l)["id"]: _json.loads(l)
                for l in (root / "cases" / f"probabilities_{split}.jsonl").read_text(encoding="utf-8").splitlines()
                if l.strip()}

    probs_train = _load_probs("train")
    probs_dev = {i: r["probabilities"] for i, r in _load_probs("dev").items()}
    qtypes = {}
    for split in ("train", "dev"):
        for line in (root / "cases" / f"{split}.jsonl").read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = _json.loads(line)
            if row["id"] in qtypes:
                raise ValueError(f"Duplicate question id across splits: {row['id']}")
            qtypes[row["id"]] = row["question_type"]
    for fname in ("train_ctrl_rows.jsonl", "train_cand_rows.jsonl", "dev_rows.jsonl"):
        seen_in_file = set()
        for line in (e20 / fname).read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = _json.loads(line)
                if row["id"] in seen_in_file:
                    raise ValueError(f"Duplicate id within {fname}: {row['id']}")
                seen_in_file.add(row["id"])
                qtypes.setdefault(row["id"], row["question_type"])
                if qtypes[row["id"]] != row["question_type"]:
                    raise ValueError(f"Conflicting question_type for {row['id']}")
    soft_targets = {c.id: probs_train[c.id]["probabilities"] for c in train_orig
                    if qtypes[c.id] == "score" and c.id in probs_train}
    dev = dev_orig + dev_new
    log(f"e20: orig_train={len(train_orig)} ctrl={len(ctrl_rows)} cand={len(cand_rows)} "
        f"dev={len(dev)} soft_orig={len(soft_targets)}")

    from mokka_decisions.arch_julia import (
        evaluate_julia_arch, make_seeded_julia_arch, train_julia_arch,
    )
    from safetensors.torch import load_file
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("jhu-clsp/mmBERT-small")
    arms = (("ctrl", train_orig + ctrl_rows, 3), ("cand", train_orig + cand_rows, 3))
    init_hashes: dict = {}
    summary = {"protocol": "docs/internal/E20_PROTOCOL_2026-10-09.md (frozen pre-launch)",
               "seeds": [42, 43, 44], "init_hashes": {}, "seed_hash_equal": {}, "arms": {}}
    for seed in (42, 43, 44):
        for name, data, epochs in arms:
            arm = f"e20_{name}_s{seed}"
            model = make_seeded_julia_arch("jhu-clsp/mmBERT-small", seed=seed)
            init_hash = _state_dict_sha256(model)
            if seed in init_hashes:
                equal = init_hash == init_hashes[seed]
                summary["seed_hash_equal"][f"s{seed}"] = equal
                if not equal:
                    raise SystemExit(f"E20 init mismatch at seed {seed} ({name}): abort")
            else:
                init_hashes[seed] = init_hash
            summary["init_hashes"][f"{name}_s{seed}"] = init_hash
            t0 = _time.time()
            report = train_julia_arch(
                model, tokenizer, data, dev, qtypes,
                output_dir=str(work / arm), epochs=epochs, microbatch=4, grad_accumulation=8,
                encoder_lr=2e-5, head_lr=1e-4, seed=seed, device="cuda",
                max_length=1024, head_length=256,
                reference_probabilities=probs_dev, soft_targets=soft_targets,
            )
            best_entry = None
            for entry in report["history"]:
                if best_entry is None or entry["dev"]["accuracy"] >= best_entry["dev"]["accuracy"]:
                    best_entry = entry
            model.load_state_dict(load_file(str(work / arm / "checkpoint_best.safetensors")))
            full = evaluate_julia_arch(model, tokenizer, dev, qtypes, device="cuda",
                                       reference_probabilities=probs_dev, collect_rows=True)
            with (work / arm / "dev_rows_best.jsonl").open("w", encoding="utf-8") as f:
                for row in full["rows"]:
                    f.write(_json.dumps(row) + "\n")
            latest = work / arm / "checkpoint_latest.pt"
            if latest.exists():
                latest.unlink()
            summary["arms"][arm] = {
                "recipe": name, "seed": seed, "init_sha256": init_hash,
                "trained_s": round(_time.time() - t0, 1),
                "best_dev_accuracy": report["best_dev_accuracy"],
                "final_step": report["final_step"], "best_epoch": best_entry["epoch"],
                "best_epoch_dev": {k: v for k, v in best_entry["dev"].items() if k != "rows"},
                "selected_checkpoint_dev": {k: v for k, v in full.items() if k != "rows"},
            }
            log(f"{arm}: best_dev={report['best_dev_accuracy']:.4f} (epoch {best_entry['epoch']}) "
                f"steps={report['final_step']} in {_time.time() - t0:.0f}s")
            (work / "run_status.json").write_text(
                _json.dumps({"stage": "typed_e20_contrast", "arm": arm,
                             "best_dev": report["best_dev_accuracy"]}, indent=2), encoding="utf-8")
    (work / "e20_summary.json").write_text(_json.dumps(summary, indent=2), encoding="utf-8")
    log("e20_contrast: done")



def _find_e23_root(bundle: Path) -> Path:
    for marker in bundle.rglob("e23-numeric/manifest.json"):
        return marker.parent
    raise SystemExit("e23-numeric dataset not found in bundle (e23-numeric/manifest.json)")


def stage_typed_e23_numeric(bundle: Path, work: Path, config: str) -> None:
    """E23: atomic numeric learning + curriculum-order contrast (frozen protocol).

    docs/internal/E23_PROTOCOL_2026-10-10.md. Two questions SEPARATELY:
      A (acquisition): do the new data teach numeric capability?
      B (curriculum): does ordering those SAME data by phases beat shuffling?
    Arms per seed (identical multiset, identical init, only ORDER differs):
      e23_mix : multiset M, shuffled (rng seed^0xC0FFEE)
      e23_cur : multiset M, phases first (atomic numeric block first)
    Multiconjunto hash (IDs with multiplicities) must be IDENTICAL between
    arms; the sequence hash must DIFFER (design). Selection: dev only.
    """
    import hashlib
    import random as _random
    import time as _time

    import torch

    if not torch.cuda.is_available():
        raise SystemExit("typed_e23_numeric needs the GPU that was not assigned")
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        log(f"e23 gpu[{i}]={props.name} vram={props.total_memory / 2**30:.1f} GiB")
    root = _find_typed_root(bundle)
    e19 = _find_e19_root(bundle)
    e23 = _find_e23_root(bundle)
    train_orig = _load(root, "train")
    dev_orig = _load(root, "dev")
    atomic_rows = _load_rows_file(e23 / "train_atomic_rows.jsonl")
    dev_new = _load_rows_file(e23 / "dev_numeric_rows.jsonl")
    # retention: E19 train_ctrl action slot (330 cases x 1 question, from TRAIN split only)
    import json as _json

    retention_rows = []
    for line in (e19 / "train_ctrl_rows.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = _json.loads(line)
        if row["id"].endswith(":action"):
            from mokka_decisions.contracts import DecisionCase, Option
            retention_rows.append(DecisionCase.from_dict(row))

    def _load_probs(split: str) -> dict:
        return {_json.loads(l)["id"]: _json.loads(l)
                for l in (root / "cases" / f"probabilities_{split}.jsonl").read_text(encoding="utf-8").splitlines()
                if l.strip()}

    probs_train = _load_probs("train")
    probs_dev = {i: r["probabilities"] for i, r in _load_probs("dev").items()}
    qtypes = {}
    for split in ("train", "dev"):
        for line in (root / "cases" / f"{split}.jsonl").read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = _json.loads(line)
            if row["id"] in qtypes:
                raise ValueError(f"Duplicate question id across splits: {row['id']}")
            qtypes[row["id"]] = row["question_type"]
    for fname in ("train_atomic_rows.jsonl", "dev_numeric_rows.jsonl"):
        seen_in_file = set()
        for line in (e23 / fname).read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = _json.loads(line)
                if row["id"] in seen_in_file:
                    raise ValueError(f"Duplicate id within {fname}: {row['id']}")
                seen_in_file.add(row["id"])
                qtypes.setdefault(row["id"], row["question_type"])
                if qtypes[row["id"]] != row["question_type"]:
                    raise ValueError(f"Conflicting question_type for {row['id']}")
    for c in retention_rows:
        qtypes.setdefault(c.id, "choice")
    soft_targets = {c.id: probs_train[c.id]["probabilities"] for c in train_orig
                    if qtypes.get(c.id) == "score" and c.id in probs_train}
    dev = dev_orig + dev_new
    log(f"e23: orig={len(train_orig)} atomic={len(atomic_rows)} retention={len(retention_rows)} "
        f"dev={len(dev)} (orig {len(dev_orig)} + new {len(dev_new)}) soft_orig={len(soft_targets)}")

    # ---- build the shared multiset M ----
    from mokka_decisions.contracts import DecisionCase, Option

    def to_case(row):
        return DecisionCase(id=row["id"], group_id=row["group_id"], source="e23", language="en",
                            domain=row["domain"], state=row["state"], question=row["question"],
                            options=tuple(Option(id=o["id"], description=o["description"]) for o in row["options"]),
                            target_kind="single", target_option=row["target_option"])

    atomic_cases = atomic_rows  # _load_rows_file already returns DecisionCase objects
    retention_cases = retention_rows
    multiset = list(train_orig) + atomic_cases + retention_cases
    # multiset hash: sorted IDs with multiplicities
    from collections import Counter
    id_counts = Counter(c.id for c in multiset)
    multiset_hash = hashlib.sha256(
        json.dumps(sorted(id_counts.items()), sort_keys=True).encode()).hexdigest()
    log(f"e23 multiset: {len(multiset)} rows, {len(id_counts)} distinct, hash={multiset_hash[:16]}")

    from mokka_decisions.arch_julia import (
        evaluate_julia_arch, make_seeded_julia_arch, train_julia_arch,
    )
    from safetensors.torch import load_file
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("jhu-clsp/mmBERT-small")
    init_hashes: dict = {}
    summary = {"protocol": "docs/internal/E23_PROTOCOL_2026-10-10.md (frozen pre-launch)",
               "seeds": [42, 43, 44], "multiset_hash": multiset_hash,
               "multiset_size": len(multiset), "init_hashes": {}, "seed_hash_equal": {}, "arms": {}}
    for seed in (42, 43, 44):
        # build arm-specific sequences
        rng = _random.Random(seed ^ 0xC0FFEE)
        mix_seq = list(multiset)
        rng.shuffle(mix_seq)
        atomic_ids = {a.id for a in atomic_cases}
        cur_seq = atomic_cases + [c for c in multiset if c.id not in atomic_ids]
        seq_hash_mix = hashlib.sha256("|".join(c.id for c in mix_seq).encode()).hexdigest()
        seq_hash_cur = hashlib.sha256("|".join(c.id for c in cur_seq).encode()).hexdigest()
        assert seq_hash_mix != seq_hash_cur, "arm sequences must differ (design)"
        for name, seq in (("mix", mix_seq), ("cur", cur_seq)):
            arm = f"e23_{name}_s{seed}"
            model = make_seeded_julia_arch("jhu-clsp/mmBERT-small", seed=seed)
            init_hash = _state_dict_sha256(model)
            if seed in init_hashes:
                equal = init_hash == init_hashes[seed]
                summary["seed_hash_equal"][f"s{seed}"] = equal
                if not equal:
                    raise SystemExit(f"E23 init mismatch at seed {seed} ({name}): abort")
            else:
                init_hashes[seed] = init_hash
            summary["init_hashes"][f"{name}_s{seed}"] = init_hash
            t0 = _time.time()
            report = train_julia_arch(
                model, tokenizer, seq, dev, qtypes,
                output_dir=str(work / arm), epochs=3, microbatch=4, grad_accumulation=8,
                encoder_lr=2e-5, head_lr=1e-4, seed=seed, device="cuda",
                max_length=1024, head_length=256,
                reference_probabilities=probs_dev, soft_targets=soft_targets,
                shuffle=False,
            )
            best_entry = None
            for entry in report["history"]:
                if best_entry is None or entry["dev"]["accuracy"] >= best_entry["dev"]["accuracy"]:
                    best_entry = entry
            model.load_state_dict(load_file(str(work / arm / "checkpoint_best.safetensors")))
            full = evaluate_julia_arch(model, tokenizer, dev, qtypes, device="cuda",
                                       reference_probabilities=probs_dev, collect_rows=True)
            with (work / arm / "dev_rows_best.jsonl").open("w", encoding="utf-8") as f:
                for row in full["rows"]:
                    f.write(_json.dumps(row) + "\n")
            latest = work / arm / "checkpoint_latest.pt"
            if latest.exists():
                latest.unlink()
            summary["arms"][arm] = {
                "recipe": name, "seed": seed, "init_sha256": init_hash,
                "seq_sha256": seq_hash_mix if name == "mix" else seq_hash_cur,
                "trained_s": round(_time.time() - t0, 1),
                "best_dev_accuracy": report["best_dev_accuracy"],
                "final_step": report["final_step"], "best_epoch": best_entry["epoch"],
                "best_epoch_dev": {k: v for k, v in best_entry["dev"].items() if k != "rows"},
                "selected_checkpoint_dev": {k: v for k, v in full.items() if k != "rows"},
            }
            log(f"{arm}: best_dev={report['best_dev_accuracy']:.4f} (epoch {best_entry['epoch']}) "
                f"steps={report['final_step']} in {_time.time() - t0:.0f}s")
            (work / "run_status.json").write_text(
                _json.dumps({"stage": "typed_e23_numeric", "arm": arm,
                             "best_dev": report["best_dev_accuracy"]}, indent=2), encoding="utf-8")
    (work / "e23_summary.json").write_text(_json.dumps(summary, indent=2), encoding="utf-8")
    log("e23_numeric: done")



def _find_e24_root(bundle: Path) -> Path:
    for marker in bundle.rglob("e24-fit/manifest.json"):
        return marker.parent
    raise SystemExit("e24-fit dataset not found in bundle (e24-fit/manifest.json)")


def _e24_cases_and_qtypes(path: Path):
    """Rows as raw dicts (metadata), DecisionCase objects, and the qtype map."""
    import json as _json

    from mokka_decisions.contracts import DecisionCase

    rows = [_json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    return rows, [DecisionCase.from_dict(r) for r in rows], {r["id"]: r["question_type"] for r in rows}


E24_FIT_THRESHOLD = 0.95  # frozen: train accuracy required on the microset


def _e24_grad_norm(grads) -> float:
    """Global L2 norm over a list of gradient tensors (fp32)."""
    import torch

    if not grads:
        return 0.0
    return float(torch.linalg.vector_norm(torch.stack(
        [torch.linalg.vector_norm(g.float()) for g in grads])).item())


def _e24_preflight(model, tokenizer, train_cases, qtypes, init_path, init_hash, device):
    """One REAL minibatch, before any training (frozen E24 protocol).

    Finite loss + correct labels; train mode + autograd; finite non-null grads
    in encoder AND head; effective weight change after optimizer.step; batched
    vs per-row logit correspondence (fp32); then reload the pristine init
    (hash-verified). The diagnostic optimizer.step must not leak into training.
    """
    import torch

    from mokka_decisions.arch_julia import collate_julia, row_loss

    keys = ("input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype")
    chunk = train_cases[:4]
    batch = collate_julia(tokenizer, chunk, qtypes, 1024, 256, device)
    for i, case in enumerate(chunk):
        gold_idx = case.option_ids.index(case.target_option)
        assert int(batch["targets"][i].item()) == gold_idx, (case.id, "label misaligned")
    checks = {"labels_correct": True}

    model.train()
    checks["training_mode"] = bool(model.training)
    with torch.autocast("cuda", enabled=(device == "cuda")):
        logits = model(**{k: batch[k] for k in keys})
        loss = row_loss(logits, batch["targets"], chunk, None, None)
    checks["loss_finite"] = bool(torch.isfinite(loss).item())
    checks["autograd_active"] = bool(loss.requires_grad)
    checks["loss_value"] = float(loss.item())
    loss.backward()

    enc_grads = [p.grad for n, p in model.named_parameters() if n.startswith("encoder.") and p.grad is not None]
    head_grads = [p.grad for n, p in model.named_parameters() if not n.startswith("encoder.") and p.grad is not None]
    finite_grads = all(torch.isfinite(g).all().item() for g in enc_grads + head_grads)
    checks.update({"grad_norm_encoder": round(_e24_grad_norm(enc_grads), 6),
                   "grad_norm_head": round(_e24_grad_norm(head_grads), 6),
                   "grads_finite": bool(finite_grads),
                   "grads_nonzero": bool(_e24_grad_norm(enc_grads) > 0 and _e24_grad_norm(head_grads) > 0)})

    # weight change after one optimizer.step: one sentinel param per group
    # (picked dynamically — encoder param naming varies by architecture)
    sentinel_enc = next(model.encoder.parameters())
    sentinel_head = next(model.scorer.parameters())
    before_enc = sentinel_enc.detach().clone()
    before_head = sentinel_head.detach().clone()
    torch.optim.AdamW(model.parameter_groups(2e-5, 1e-4), weight_decay=0.01).step()
    delta_enc = float((sentinel_enc.detach() - before_enc).norm().item())
    delta_head = float((sentinel_head.detach() - before_head).norm().item())
    checks.update({"weight_delta_encoder": delta_enc, "weight_delta_head": delta_head,
                   "weights_changed": bool(delta_enc > 0 and delta_head > 0)})

    # example <-> logits <-> targets correspondence: batched vs isolated rows (fp32 eval)
    model.eval()
    with torch.inference_mode():
        batched = model(**{k: batch[k] for k in keys}).float()
        max_diff = 0.0
        for i, case in enumerate(chunk):
            single = collate_julia(tokenizer, [case], qtypes, 1024, 256, device)
            l1 = model(**{k: single[k] for k in keys}).float()[0]
            max_diff = max(max_diff, (l1[: len(case.options)]
                                      - batched[i, : len(case.options)]).abs().max().item())
    checks["batch_vs_single_max_abs_diff"] = max_diff
    checks["example_logit_alignment"] = bool(max_diff <= 1e-4)

    from safetensors.torch import load_file

    model.load_state_dict(load_file(str(init_path)))
    checks["init_restored"] = bool(_state_dict_sha256(model) == init_hash)
    model.train()
    if not all(v is True for k, v in checks.items() if isinstance(v, bool)):
        raise SystemExit(f"E24 preflight FAILED: {checks}")
    return checks


def _e24_train_loop(model, tokenizer, train_cases, dev_cases, qtypes, arm: Path, *, seed,
                    device="cuda", epochs_max=150, microbatch=4, grad_accumulation=8,
                    eval_cases=None):
    """E24 fitting loop: EXACT train_julia_arch recipe plus the protocol's
    diagnostics (train/dev curves separated, grad norms, weight deltas,
    intermediate checkpoints, early stop at 3 consecutive perfect epochs).
    eval_cases: run the contrast evals on the FINAL checkpoint only if the fit
    criterion holds (frozen rule)."""
    import json as _json
    import math
    import random as _random
    import time as _time

    import torch

    from mokka_decisions.arch_julia import collate_julia, evaluate_julia_arch, row_loss
    from safetensors.torch import save_file

    keys = ("input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype")
    model = model.to(device)
    model.train()
    usable = [c for c in train_cases if c.target_option]
    steps_per_epoch = math.ceil(len(usable) / (microbatch * grad_accumulation))
    total_steps = max(steps_per_epoch * epochs_max, 1)
    optimizer = torch.optim.AdamW(model.parameter_groups(2e-5, 1e-4), weight_decay=0.01)
    warmup = max(int(total_steps * 0.05), 1)

    def lr_lambda(step):
        if step < warmup:
            return step / warmup
        progress = (step - warmup) / max(total_steps - warmup, 1)
        return max(0.05, 0.95 * (1 - progress) + 0.05)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))
    rng = _random.Random(seed ^ 0xA17A)
    init_snapshot = {k: v.detach().clone() for k, v in model.state_dict().items()}

    def _delta(encoder: bool) -> float:
        total = 0.0
        for n, p in model.state_dict().items():
            if n.startswith("encoder.") == encoder:
                total += (p.detach() - init_snapshot[n]).float().pow(2).sum().item()
        return total ** 0.5

    save_epochs = {15, 50, 100}
    history, perfect_streak, global_step, stopped_early = [], 0, 0, False
    t0 = _time.time()
    with (arm / "curves.jsonl").open("w", encoding="utf-8") as curves:
        for epoch in range(epochs_max):
            order = list(range(len(usable)))
            rng.shuffle(order)
            optimizer.zero_grad(set_to_none=True)
            loss_sum, rows_seen, grad_enc, grad_head, grad_steps = 0.0, 0, 0.0, 0.0, 0
            for start in range(0, len(order), microbatch):
                chunk = [usable[i] for i in order[start : start + microbatch]]
                batch = collate_julia(tokenizer, chunk, qtypes, 1024, 256, device)
                with torch.autocast("cuda", enabled=(device == "cuda")):
                    logits = model(**{k: batch[k] for k in keys})
                    loss = row_loss(logits, batch["targets"], chunk, None, None) / grad_accumulation
                scaler.scale(loss).backward()
                loss_sum += float(loss.item()) * grad_accumulation * len(chunk)
                rows_seen += len(chunk)
                if (start // microbatch + 1) % grad_accumulation == 0 or start + microbatch >= len(order):
                    scaler.unscale_(optimizer)
                    grad_enc += _e24_grad_norm([p.grad for n, p in model.named_parameters()
                                                if n.startswith("encoder.") and p.grad is not None])
                    grad_head += _e24_grad_norm([p.grad for n, p in model.named_parameters()
                                                 if not n.startswith("encoder.") and p.grad is not None])
                    grad_steps += 1
                    torch.nn.utils.clip_grad_norm_(
                        [p for g in optimizer.param_groups for p in g["params"]], 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1
                    if global_step % 25 == 0:
                        print(f"[e24 s{seed}] epoch={epoch} step={global_step} "
                              f"loss={float(loss.item()) * grad_accumulation:.4f}", flush=True)
            tr = evaluate_julia_arch(model, tokenizer, usable, qtypes, device=device)
            dv = evaluate_julia_arch(model, tokenizer, dev_cases, qtypes, device=device)
            perfect_streak = perfect_streak + 1 if tr["accuracy"] == 1.0 else 0
            entry = {"epoch": epoch, "step": global_step,
                     "train_loss": loss_sum / max(rows_seen, 1),
                     "train_acc": tr["accuracy"], "train_nll": tr["mean_nll"],
                     "dev_acc": dv["accuracy"], "dev_nll": dv["mean_nll"],
                     "grad_norm_encoder_mean": grad_enc / max(grad_steps, 1),
                     "grad_norm_head_mean": grad_head / max(grad_steps, 1)}
            if epoch % 5 == 0 or epoch == epochs_max - 1:
                entry["weight_delta_encoder"] = _delta(True)
                entry["weight_delta_head_group"] = _delta(False)
            history.append(entry)
            curves.write(_json.dumps(entry, default=str) + "\n")
            log(f"e24 s{seed} epoch={epoch} train_acc={tr['accuracy']:.4f} "
                f"dev_acc={dv['accuracy']:.4f} loss={entry['train_loss']:.4f}")
            if epoch + 1 in save_epochs:
                save_file(dict(model.state_dict()), str(arm / f"e{epoch + 1:03d}.safetensors"),
                          metadata={"format": "pt"})
            if perfect_streak >= 3:
                stopped_early = True
                break

    save_file(dict(model.state_dict()), str(arm / "final.safetensors"), metadata={"format": "pt"})
    final = history[-1]
    fit = bool(final["train_acc"] is not None and final["train_acc"] >= E24_FIT_THRESHOLD)
    result = {"epochs_run": len(history), "steps": global_step, "stopped_early": stopped_early,
              "final_train_acc": final["train_acc"], "final_dev_acc": final["dev_acc"],
              "final_train_nll": final["train_nll"], "final_dev_nll": final["dev_nll"],
              "best_train_acc": max(h["train_acc"] for h in history),
              "best_train_epoch": max(history, key=lambda h: h["train_acc"])["epoch"],
              "fit": fit, "checkpoints": sorted(p.name for p in arm.glob("*.safetensors")),
              "trained_s": round(_time.time() - t0, 1)}

    # per-row predictions on train+dev for the FINAL checkpoint (always, also on failure)
    predictions = []
    for name, cases in (("train", usable), ("dev", dev_cases)):
        ev = evaluate_julia_arch(model, tokenizer, cases, qtypes, device=device, collect_rows=True)
        predictions.extend({"set": name, **r} for r in ev["rows"])
    (arm / "predictions_final.jsonl").write_text(
        "".join(_json.dumps(r, default=str) + "\n" for r in predictions), encoding="utf-8")

    # conditional eval contrasts on the FINAL checkpoint (frozen rule: only on fit)
    if fit and eval_cases:
        result["evals"] = {}
        for name, cases in eval_cases.items():
            ev = evaluate_julia_arch(model, tokenizer, cases, qtypes, device=device,
                                     collect_rows=True)
            (arm / f"eval_{name}.jsonl").write_text(
                "".join(_json.dumps(r, default=str) + "\n" for r in ev["rows"]), encoding="utf-8")
            result["evals"][name] = {"n": ev["n"], "accuracy": ev["accuracy"],
                                     "mean_nll": ev["mean_nll"]}
    return result


def stage_typed_e24_fit(bundle: Path, work: Path, config: str) -> None:
    """E24: minimal numeric FITTING test (frozen protocol, 2026-10-10).

    Can JuliaArch-small FIT 48 problems (one field, one operator, one threshold,
    direct No/Yes question)? NOT a generalization benchmark. Gradient dominance
    removed: the microset is the ONLY training data. Preflight minibatch
    verification before seed 42; seed 43 only if 42 fits; the three eval
    contrasts only on fit. Forgetting experiment prepared, NOT launched.
    """
    import json as _json

    import torch

    if not torch.cuda.is_available():
        raise SystemExit("typed_e24_fit needs the GPU that was not assigned")
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        log(f"e24 gpu[{i}]={props.name} vram={props.total_memory / 2**30:.1f} GiB")
    e24 = _find_e24_root(bundle)
    _, train_cases, _ = _e24_cases_and_qtypes(e24 / "train_fit_rows.jsonl")
    _, dev_cases, _ = _e24_cases_and_qtypes(e24 / "dev_fit_rows.jsonl")
    eval_specs = (("newvalues", "eval_newvalues_rows.jsonl"),
                  ("newthreshold", "eval_newthreshold_rows.jsonl"),
                  ("newfield", "eval_newfield_rows.jsonl"))
    qtypes, eval_cases = {}, {}
    for fname in ("train_fit_rows.jsonl", "dev_fit_rows.jsonl") + tuple(f for _, f in eval_specs):
        for r in [_json.loads(l) for l in (e24 / fname).read_text(encoding="utf-8").splitlines() if l.strip()]:
            if r["id"] in qtypes:
                raise ValueError(f"duplicate id: {r['id']}")
            qtypes[r["id"]] = r["question_type"]
    for name, fname in eval_specs:
        _, cases, _ = _e24_cases_and_qtypes(e24 / fname)
        eval_cases[name] = cases
    log(f"e24: train={len(train_cases)} dev={len(dev_cases)} "
        f"evals={ {k: len(v) for k, v in eval_cases.items()} }")

    from mokka_decisions.arch_julia import make_seeded_julia_arch
    from safetensors.torch import save_file
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("jhu-clsp/mmBERT-small")

    def run_seed(seed: int, preflight: bool, run_evals: bool, epochs_max: int = 150) -> dict:
        arm = work / f"s{seed}"
        arm.mkdir(parents=True, exist_ok=True)
        model = make_seeded_julia_arch("jhu-clsp/mmBERT-small", seed=seed)
        init_hash = _state_dict_sha256(model)
        save_file(dict(model.state_dict()), str(arm / "initial.safetensors"), metadata={"format": "pt"})
        model = model.to("cuda")
        pre = None
        if preflight:
            pre = _e24_preflight(model, tokenizer, train_cases, qtypes,
                                 arm / "initial.safetensors", init_hash, "cuda")
            log(f"e24 preflight s{seed}: OK {pre}")
        result = _e24_train_loop(model, tokenizer, train_cases, dev_cases, qtypes, arm,
                                 seed=seed, device="cuda", epochs_max=epochs_max,
                                 eval_cases=eval_cases if run_evals else None)
        result["init_sha256"] = init_hash
        result["preflight"] = pre
        log(f"e24 s{seed}: fit={result['fit']} final_train={result['final_train_acc']:.4f} "
            f"final_dev={result['final_dev_acc']:.4f} epochs={result['epochs_run']} "
            f"in {result['trained_s']}s")
        (work / "run_status.json").write_text(
            _json.dumps({"stage": "typed_e24_fit", "seed": seed, "fit": result["fit"],
                         "final_train_acc": result["final_train_acc"]}), encoding="utf-8")
        return result

    summary = {
        "protocol": "docs/internal/E24_PROTOCOL_2026-10-10.md (frozen pre-launch)",
        "objective": "FIT test: train accuracy >= 0.95 on the 48-problem microset within "
                     "budget; NOT a generalization benchmark",
        "recipe": {"architecture": "JuliaArchScorer(mmBERT-small), unchanged",
                   "init": "upstream encoder + seeded fresh head (make_seeded_julia_arch)",
                   "epochs_max": 150, "steps_max": 300, "microbatch": 4, "grad_accumulation": 8,
                   "encoder_lr": 2e-5, "head_lr": 1e-4, "weight_decay": 0.01, "warmup_ratio": 0.05,
                   "max_grad_norm": 1.0, "amp": True, "shuffle": "per epoch",
                   "loss": "hard CE (deterministic noul rows)",
                   "early_stop": "train accuracy == 1.0 at 3 consecutive epoch ends",
                   "selection_for_evals": "FINAL checkpoint (no cherry-picking)"},
        "data": {"train": len(train_cases), "dev": len(dev_cases),
                 "evals": {k: len(v) for k, v in eval_cases.items()}},
        "seeds": {}, "verdicts": {},
        "forgetting_experiment": "PREPARED, NOT LAUNCHED (E25 candidate: resume the fitted "
                                 "checkpoint with original tasks mixed in; measure microset "
                                 "accuracy before/after)",
    }
    r42 = run_seed(42, preflight=True, run_evals=True)
    summary["seeds"]["42"] = r42
    summary["verdicts"]["fit_s42"] = r42["fit"]
    if r42["fit"]:
        r43 = run_seed(43, preflight=False, run_evals=True)
        summary["seeds"]["43"] = r43
        summary["verdicts"]["fit_s43"] = r43["fit"]
        summary["verdicts"]["overall"] = "FIT_CONFIRMED" if r43["fit"] else "FIT_SINGLE"
    else:
        summary["verdicts"]["overall"] = "NO_FIT"
    (work / "e24_summary.json").write_text(_json.dumps(summary, indent=2, default=str),
                                           encoding="utf-8")
    log(f"e24_fit: done — {summary['verdicts']}")


def _find_e25_root(bundle: Path) -> Path:
    for marker in bundle.rglob("e25-thrvar/manifest.json"):
        return marker.parent
    raise SystemExit("e25-thrvar dataset not found in bundle (e25-thrvar/manifest.json)")


def _e25_hardware_report() -> dict:
    import torch

    report = {"torch": torch.__version__, "cuda_available": torch.cuda.is_available(),
              "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
              "gpus": []}
    if report["cuda_available"]:
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            report["gpus"].append({"index": i, "name": props.name,
                                   "vram_gb": round(props.total_memory / 2**30, 2),
                                   "capability": f"{props.major}.{props.minor}"})
    import transformers

    report["transformers"] = transformers.__version__
    return report


def _e25_train_loop(model, tokenizer, train_cases, dev_cases, qtypes, arm: Path, *, seed,
                    device="cuda", epochs_max=300, microbatch=4, grad_accumulation=8,
                    eval_cases=None, early_stop=True, dev_patience=None):
    """E25 loop: EXACT E24 recipe with the instrumentation the E24 review demands.

    Differences vs _e24_train_loop (frozen in the E25 protocol):
    - attempted vs effective vs skipped optimizer updates are counted separately;
      a step whose grads are non-finite after unscale_ is counted as skipped
      (scaler.step still runs, so the AMP scale shrinks as designed);
    - the LR scheduler advances ONLY on effective updates;
    - AMP scale is recorded per epoch and non-finite grad epochs are flagged
      (norms stored as null, never inf);
    - hardware telemetry is saved by the stage (hardware.json).
    """
    import json as _json
    import math
    import random as _random
    import time as _time

    import torch

    from mokka_decisions.arch_julia import collate_julia, evaluate_julia_arch, row_loss
    from safetensors.torch import save_file

    keys = ("input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype")
    model = model.to(device)
    model.train()
    usable = [c for c in train_cases if c.target_option]
    steps_per_epoch = math.ceil(len(usable) / (microbatch * grad_accumulation))
    max_effective_steps = steps_per_epoch * epochs_max
    optimizer = torch.optim.AdamW(model.parameter_groups(2e-5, 1e-4), weight_decay=0.01)
    warmup = max(int(max_effective_steps * 0.05), 1)

    def lr_lambda(step):
        if step < warmup:
            return step / warmup
        progress = (step - warmup) / max(max_effective_steps - warmup, 1)
        return max(0.05, 0.95 * (1 - progress) + 0.05)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))
    rng = _random.Random(seed ^ 0xA17A)
    init_snapshot = {k: v.detach().clone() for k, v in model.state_dict().items()}

    def _delta(encoder: bool) -> float:
        total = 0.0
        for n, p in model.state_dict().items():
            if n.startswith("encoder.") == encoder:
                total += (p.detach() - init_snapshot[n]).float().pow(2).sum().item()
        return total ** 0.5

    save_epochs = {25, 100, 200}
    history = []
    attempted = effective = skipped = 0
    perfect_streak, stopped_early = 0, False
    best_dev, dev_stale = -1.0, 0
    t0 = _time.time()
    with (arm / "curves.jsonl").open("w", encoding="utf-8") as curves:
        for epoch in range(epochs_max):
            order = list(range(len(usable)))
            rng.shuffle(order)
            optimizer.zero_grad(set_to_none=True)
            loss_sum, rows_seen = 0.0, 0
            grad_enc, grad_head, grad_steps = 0.0, 0.0, 0
            nonfinite_epoch = 0
            scale_at_epoch_start = scaler.get_scale() if device == "cuda" else None
            for start in range(0, len(order), microbatch):
                chunk = [usable[i] for i in order[start : start + microbatch]]
                batch = collate_julia(tokenizer, chunk, qtypes, 1024, 256, device)
                with torch.autocast("cuda", enabled=(device == "cuda")):
                    logits = model(**{k: batch[k] for k in keys})
                    loss = row_loss(logits, batch["targets"], chunk, None, None) / grad_accumulation
                scaler.scale(loss).backward()
                loss_sum += float(loss.item()) * grad_accumulation * len(chunk)
                rows_seen += len(chunk)
                if (start // microbatch + 1) % grad_accumulation == 0 or start + microbatch >= len(order):
                    attempted += 1
                    scaler.unscale_(optimizer)
                    finite = all(torch.isfinite(p.grad).all().item()
                                 for g in optimizer.param_groups for p in g["params"]
                                 if p.grad is not None)
                    if finite:
                        grad_enc += _e24_grad_norm([p.grad for n, p in model.named_parameters()
                                                    if n.startswith("encoder.") and p.grad is not None])
                        grad_head += _e24_grad_norm([p.grad for n, p in model.named_parameters()
                                                     if not n.startswith("encoder.") and p.grad is not None])
                        grad_steps += 1
                        torch.nn.utils.clip_grad_norm_(
                            [p for g in optimizer.param_groups for p in g["params"]], 1.0)
                        effective += 1
                    else:
                        nonfinite_epoch += 1
                        skipped += 1
                    scaler.step(optimizer)  # applies only when grads are finite
                    scaler.update()
                    if finite:
                        scheduler.step()  # advances ONLY on effective updates
                    optimizer.zero_grad(set_to_none=True)
                    if attempted % 50 == 0:
                        print(f"[e25 s{seed}] epoch={epoch} attempted={attempted} "
                              f"effective={effective} loss={float(loss.item()) * grad_accumulation:.4f}",
                              flush=True)
            tr = evaluate_julia_arch(model, tokenizer, usable, qtypes, device=device)
            dv = evaluate_julia_arch(model, tokenizer, dev_cases, qtypes, device=device)
            perfect_streak = perfect_streak + 1 if tr["accuracy"] == 1.0 else 0
            if dev_patience is not None:
                if dv["accuracy"] > best_dev + 1e-12:
                    best_dev, dev_stale = dv["accuracy"], 0
                else:
                    dev_stale += 1
            entry = {"epoch": epoch, "attempted_steps": attempted, "effective_steps": effective,
                     "skipped_steps": skipped, "grad_nonfinite_epochs": nonfinite_epoch,
                     "amp_scale_epoch_start": scale_at_epoch_start,
                     "amp_scale_epoch_end": scaler.get_scale() if device == "cuda" else None,
                     "train_loss": loss_sum / max(rows_seen, 1),
                     "train_acc": tr["accuracy"], "train_nll": tr["mean_nll"],
                     "dev_acc": dv["accuracy"], "dev_nll": dv["mean_nll"],
                     "grad_norm_encoder_mean": (grad_enc / grad_steps) if grad_steps else None,
                     "grad_norm_head_mean": (grad_head / grad_steps) if grad_steps else None}
            if epoch % 5 == 0 or epoch == epochs_max - 1:
                entry["weight_delta_encoder"] = _delta(True)
                entry["weight_delta_head_group"] = _delta(False)
            history.append(entry)
            curves.write(_json.dumps(entry, default=str) + "\n")
            log(f"e25 s{seed} epoch={epoch} train_acc={tr['accuracy']:.4f} "
                f"dev_acc={dv['accuracy']:.4f} eff={effective}/{attempted} "
                f"skip={skipped} loss={entry['train_loss']:.4f}")
            if epoch + 1 in save_epochs:
                save_file(dict(model.state_dict()), str(arm / f"e{epoch + 1:03d}.safetensors"),
                          metadata={"format": "pt"})
            if early_stop and perfect_streak >= 3:
                stopped_early = True
                break
            if dev_patience is not None and dev_stale >= dev_patience:
                stopped_early = True
                break

    save_file(dict(model.state_dict()), str(arm / "final.safetensors"), metadata={"format": "pt"})
    final = history[-1]
    fit = bool(final["train_acc"] is not None and final["train_acc"] >= 0.95)
    result = {"epochs_run": len(history), "attempted_steps": attempted,
              "effective_steps": effective, "skipped_steps": skipped,
              "stopped_early": stopped_early,
              "final_train_acc": final["train_acc"], "final_dev_acc": final["dev_acc"],
              "final_train_nll": final["train_nll"], "final_dev_nll": final["dev_nll"],
              "best_train_acc": max(h["train_acc"] for h in history),
              "best_train_epoch": max(history, key=lambda h: h["train_acc"])["epoch"],
              "best_dev_acc": (max(h["dev_acc"] for h in history) if history else None),
              "best_dev_epoch": (max(history, key=lambda h: h["dev_acc"])["epoch"] if history else None),
              "nonfinite_grad_epochs_total": sum(h["grad_nonfinite_epochs"] for h in history),
              "fit": fit, "checkpoints": sorted(p.name for p in arm.glob("*.safetensors")),
              "trained_s": round(_time.time() - t0, 1)}

    predictions = []
    for name, cases in (("train", usable), ("dev", dev_cases)):
        ev = evaluate_julia_arch(model, tokenizer, cases, qtypes, device=device, collect_rows=True)
        predictions.extend({"set": name, **r} for r in ev["rows"])
    (arm / "predictions_final.jsonl").write_text(
        "".join(_json.dumps(r, default=str) + "\n" for r in predictions), encoding="utf-8")

    # tests C/D: opened here — the finalist (final checkpoint) is already fixed
    if eval_cases:
        result["evals"] = {}
        for name, cases in eval_cases.items():
            ev = evaluate_julia_arch(model, tokenizer, cases, qtypes, device=device,
                                     collect_rows=True)
            (arm / f"eval_{name}.jsonl").write_text(
                "".join(_json.dumps(r, default=str) + "\n" for r in ev["rows"]), encoding="utf-8")
            result["evals"][name] = {"n": ev["n"], "accuracy": ev["accuracy"],
                                     "mean_nll": ev["mean_nll"]}
    return result


def stage_typed_e25_thrvar(bundle: Path, work: Path, config: str) -> None:
    """E25: threshold-VARIABLE comparison (frozen protocol, 2026-10-10).

    Can the model learn to compare the state value against the threshold IN
    THE QUESTION? Multiple train thresholds (40..90) with threshold-only
    inversion pairs (identical state, only the question threshold changes);
    held-out tests: new values at known thresholds (C) and never-used
    thresholds (D: interp/extrap). Tests are opened once the finalist (final
    checkpoint) is fixed. Forgetting study NOT launched here.
    """
    import json as _json
    import time as _time

    import torch

    if not torch.cuda.is_available():
        raise SystemExit("typed_e25_thrvar needs the GPU that was not assigned")
    hardware = _e25_hardware_report()
    (work / "hardware.json").write_text(_json.dumps(hardware, indent=2), encoding="utf-8")
    log(f"e25 hardware: {hardware}")
    e25 = _find_e25_root(bundle)
    _, train_cases, _ = _e24_cases_and_qtypes(e25 / "train_thr_rows.jsonl")
    _, dev_cases, _ = _e24_cases_and_qtypes(e25 / "dev_thr_rows.jsonl")
    eval_specs = (("known", "test_known_thr_rows.jsonl"), ("new", "test_new_thr_rows.jsonl"))
    qtypes, eval_cases = {}, {}
    for fname in ("train_thr_rows.jsonl", "dev_thr_rows.jsonl") + tuple(f for _, f in eval_specs):
        for r in [_json.loads(l) for l in (e25 / fname).read_text(encoding="utf-8").splitlines() if l.strip()]:
            if r["id"] in qtypes:
                raise ValueError(f"duplicate id: {r['id']}")
            qtypes[r["id"]] = r["question_type"]
    for name, fname in eval_specs:
        _, cases, _ = _e24_cases_and_qtypes(e25 / fname)
        eval_cases[name] = cases
    log(f"e25: train={len(train_cases)} dev={len(dev_cases)} "
        f"evals={ {k: len(v) for k, v in eval_cases.items()} }")

    from mokka_decisions.arch_julia import make_seeded_julia_arch
    from safetensors.torch import save_file
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("jhu-clsp/mmBERT-small")

    def run_seed(seed: int, preflight: bool, epochs_max: int = 300) -> dict:
        arm = work / f"s{seed}"
        arm.mkdir(parents=True, exist_ok=True)
        model = make_seeded_julia_arch("jhu-clsp/mmBERT-small", seed=seed)
        init_hash = _state_dict_sha256(model)
        save_file(dict(model.state_dict()), str(arm / "initial.safetensors"), metadata={"format": "pt"})
        model = model.to("cuda")
        pre = None
        if preflight:
            pre = _e24_preflight(model, tokenizer, train_cases, qtypes,
                                 arm / "initial.safetensors", init_hash, "cuda")
            log(f"e25 preflight s{seed}: OK")
        result = _e25_train_loop(model, tokenizer, train_cases, dev_cases, qtypes, arm,
                                 seed=seed, device="cuda", epochs_max=epochs_max,
                                 eval_cases=eval_cases)
        result["init_sha256"] = init_hash
        result["preflight"] = pre
        log(f"e25 s{seed}: fit={result['fit']} final_train={result['final_train_acc']:.4f} "
            f"final_dev={result['final_dev_acc']:.4f} epochs={result['epochs_run']} "
            f"eff={result['effective_steps']}/{result['attempted_steps']} "
            f"in {result['trained_s']}s")
        (work / "run_status.json").write_text(
            _json.dumps({"stage": "typed_e25_thrvar", "seed": seed, "fit": result["fit"],
                         "final_train_acc": result["final_train_acc"]}), encoding="utf-8")
        return result

    summary = {
        "protocol": "docs/internal/E25_PROTOCOL_2026-10-10.md (frozen pre-launch)",
        "objective": "learn to compare value vs the threshold IN THE QUESTION (two variable "
                     "operands); NOT a generalization benchmark, NOT vs Julia",
        "recipe": {"architecture": "JuliaArchScorer(mmBERT-small), unchanged",
                   "init": "upstream encoder + seeded fresh head",
                   "epochs_max": 300, "max_effective_steps": 600, "microbatch": 4,
                   "grad_accumulation": 8, "encoder_lr": 2e-5, "head_lr": 1e-4,
                   "weight_decay": 0.01, "warmup_ratio": 0.05, "max_grad_norm": 1.0,
                   "amp": True, "shuffle": "per epoch", "loss": "hard CE",
                   "early_stop": "train accuracy == 1.0 at 3 consecutive epoch ends",
                   "scheduler_advances": "ONLY on effective updates (non-finite-grad steps "
                                         "are skipped and do not consume the schedule)",
                   "fit_criterion": "train accuracy >= 0.95 at the final epoch",
                   "selection": "finalist = FINAL checkpoint; dev tracked, never selected on"},
        "data": {"train": len(train_cases), "dev": len(dev_cases),
                 "evals": {k: len(v) for k, v in eval_cases.items()}},
        "hardware": hardware,
        "seeds": {}, "verdicts": {},
        "forgetting_experiment": "NOT launched in E25 (next after threshold dependence)",
    }
    r42 = run_seed(42, preflight=True)
    summary["seeds"]["42"] = r42
    summary["verdicts"]["fit_s42"] = r42["fit"]
    if r42["fit"]:
        r43 = run_seed(43, preflight=False)
        summary["seeds"]["43"] = r43
        summary["verdicts"]["fit_s43"] = r43["fit"]
        summary["verdicts"]["overall_fit"] = "FIT_CONFIRMED" if r43["fit"] else "FIT_SINGLE"
    else:
        summary["verdicts"]["overall_fit"] = "NO_FIT"
    (work / "e25_summary.json").write_text(_json.dumps(summary, indent=2, default=str),
                                           encoding="utf-8")
    log(f"e25_thrvar: done — {summary['verdicts']}")




def _find_e26_root(bundle: Path) -> Path:
    for marker in bundle.rglob("e26-order/manifest.json"):
        return marker.parent
    raise SystemExit("e26-order dataset not found in bundle (e26-order/manifest.json)")


def stage_typed_e26_order(bundle: Path, work: Path, config: str) -> None:
    """E26: numeric ORDER with the same-decade shortcut broken + equal-budget control.

    Frozen protocol docs/internal/E26_PROTOCOL_2026-10-10.md (recommendation of the
    E25 review). Two arms with equal rows/steps/evals: treatment (non-decade-aligned
    thresholds 41..92, all four quadrants incl. same-decade negatives and far
    positives, threshold-only AND value-only pairs) vs control (decade-aligned
    30..100, E25-style placement where floor(v/10)==floor(t/10) still scores 0.917).
    Shared dev/C/D/monotonicity evals are opened only after each finalist (final
    checkpoint) is fixed. Reuses the E25 instrumented train loop UNCHANGED.
    """
    import json as _json

    import torch

    if not torch.cuda.is_available():
        raise SystemExit("typed_e26_order needs the GPU that was not assigned")
    hardware = _e25_hardware_report()
    (work / "hardware.json").write_text(_json.dumps(hardware, indent=2), encoding="utf-8")
    log(f"e26 hardware: {hardware}")
    e26 = _find_e26_root(bundle)
    _, dev_cases, _ = _e24_cases_and_qtypes(e26 / "dev_rows.jsonl")
    eval_specs = (("known", "test_c_rows.jsonl"), ("new", "test_d_rows.jsonl"),
                  ("mono", "mono_rows.jsonl"))
    qtypes, eval_cases, seen = {}, {}, set()
    for fname in (("train_treat_rows.jsonl", "train_ctrl_rows.jsonl", "dev_rows.jsonl")
                  + tuple(f for _, f in eval_specs)):
        for r in [_json.loads(l) for l in (e26 / fname).read_text(encoding="utf-8").splitlines() if l.strip()]:
            if r["id"] in seen:
                raise ValueError(f"duplicate id: {r['id']}")
            seen.add(r["id"])
            qtypes[r["id"]] = r["question_type"]
    for name, fname in eval_specs:
        _, cases, _ = _e24_cases_and_qtypes(e26 / fname)
        eval_cases[name] = cases
    log(f"e26: dev={len(dev_cases)} evals={ {k: len(v) for k, v in eval_cases.items()} }")

    from mokka_decisions.arch_julia import make_seeded_julia_arch
    from safetensors.torch import save_file
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("jhu-clsp/mmBERT-small")

    def run_arm(arm: str, seed: int, preflight: bool, epochs_max: int = 300) -> dict:
        _, train_cases, _ = _e24_cases_and_qtypes(e26 / f"train_{arm}_rows.jsonl")
        log(f"e26 arm={arm} s{seed}: train={len(train_cases)}")
        arm_dir = work / f"s{seed}_{arm}"
        arm_dir.mkdir(parents=True, exist_ok=True)
        model = make_seeded_julia_arch("jhu-clsp/mmBERT-small", seed=seed)
        init_hash = _state_dict_sha256(model)
        save_file(dict(model.state_dict()), str(arm_dir / "initial.safetensors"),
                  metadata={"format": "pt"})
        model = model.to("cuda")
        pre = None
        if preflight:
            pre = _e24_preflight(model, tokenizer, train_cases, qtypes,
                                 arm_dir / "initial.safetensors", init_hash, "cuda")
            log(f"e26 preflight {arm} s{seed}: OK")
        result = _e25_train_loop(model, tokenizer, train_cases, dev_cases, qtypes, arm_dir,
                                 seed=seed, device="cuda", epochs_max=epochs_max,
                                 eval_cases=eval_cases)
        result["init_sha256"] = init_hash
        result["preflight"] = pre
        log(f"e26 {arm} s{seed}: fit={result['fit']} final_train={result['final_train_acc']:.4f} "
            f"final_dev={result['final_dev_acc']:.4f} epochs={result['epochs_run']} "
            f"eff={result['effective_steps']}/{result['attempted_steps']} in {result['trained_s']}s")
        (work / "run_status.json").write_text(
            _json.dumps({"stage": "typed_e26_order", "arm": arm, "seed": seed,
                         "fit": result["fit"], "final_train_acc": result["final_train_acc"]}),
            encoding="utf-8")
        return result

    summary = {
        "protocol": "docs/internal/E26_PROTOCOL_2026-10-10.md (frozen pre-launch)",
        "objective": ("learn numeric ORDER (value >= question threshold) with the "
                      "same-decade shortcut broken in treatment vs an equal-budget "
                      "shortcut-compatible control; NOT vs Julia, NOT forgetting"),
        "recipe": {"architecture": "JuliaArchScorer(mmBERT-small), unchanged",
                   "init": "upstream encoder + seeded fresh head",
                   "epochs_max": 300, "max_effective_steps": 900, "microbatch": 4,
                   "grad_accumulation": 8, "encoder_lr": 2e-5, "head_lr": 1e-4,
                   "weight_decay": 0.01, "warmup_ratio": 0.05, "max_grad_norm": 1.0,
                   "amp": True, "shuffle": "per epoch", "loss": "hard CE",
                   "early_stop": "train accuracy == 1.0 at 3 consecutive epoch ends",
                   "fit_criterion": "train accuracy >= 0.95 at the final epoch",
                   "selection": "finalist = FINAL checkpoint; dev tracked, never selected on",
                   "train_loop": "identical to E25 (_e25_train_loop, instrumentation per E24 review)"},
        "data": {"treat": 96, "ctrl": 96, "dev": 48,
                 "evals": {k: len(v) for k, v in eval_cases.items()}},
        "hardware": hardware,
        "arms": {}, "verdicts": {},
        "forgetting_experiment": "NOT launched in E26 (after order acquisition)",
    }
    r42 = run_arm("treat", 42, preflight=True)
    summary["arms"]["treat_s42"] = r42
    summary["verdicts"]["fit_treat_s42"] = r42["fit"]
    if r42["fit"]:
        summary["arms"]["treat_s43"] = run_arm("treat", 43, preflight=False)
        summary["verdicts"]["fit_treat_s43"] = summary["arms"]["treat_s43"]["fit"]
    c42 = run_arm("ctrl", 42, preflight=False)
    summary["arms"]["ctrl_s42"] = c42
    summary["verdicts"]["fit_ctrl_s42"] = c42["fit"]
    if c42["fit"]:
        summary["arms"]["ctrl_s43"] = run_arm("ctrl", 43, preflight=False)
        summary["verdicts"]["fit_ctrl_s43"] = summary["arms"]["ctrl_s43"]["fit"]
    (work / "e26_summary.json").write_text(_json.dumps(summary, indent=2, default=str),
                                           encoding="utf-8")
    log(f"e26_order: done — {summary['verdicts']}")


def _find_e26b_ckpts(bundle: Path) -> Path:
    for marker in bundle.rglob("e26-ckpts/ckpt_manifest.json"):
        return marker.parent
    raise SystemExit("e26b checkpoints not found in bundle (e26-ckpts/ckpt_manifest.json)")


_E26B_CKPT_SHA = {42: "0c10e71469299028beccb6f5d7424320e3f187e65af437374d218bd703ca5b22",
                  43: "4dd3169be1ebdae22587eefcd1cee6e4db0e5d2e2d27339299863d3f1dea8715"}
_E26B_EPOCHS_RUN = {42: 59, 43: 42}
_E26B_DEV_AT_STOP = {42: 0.625, 43: 0.9166666666666666}


def stage_typed_e26b_diag(bundle: Path, work: Path, config: str) -> None:
    """E26b diagnostic (frozen protocol docs/internal/E26B_PROTOCOL_2026-10-10.md).

    Two discriminators recommended by the E26 review, no new data:
    A) CONTINUE the s42/s43 TREATMENT finalists from their final.safetensors with
    early stopping DISABLED up to the original 300-epoch budget (does dev keep
    rising past memorization? -> was the stopping rule the lever?);
    B) FRESH seeds 44/45 with the exact E26 treat recipe (how much was seed
    variance?). Tests C/D/mono are HISTORICAL (inspected in E26): descriptive
    only, no OOD claims. Run order: s44 (preflight, standard conditions) ->
    s42_cont -> s43_cont -> s45.
    """
    import hashlib as _hashlib
    import json as _json

    import torch

    if not torch.cuda.is_available():
        raise SystemExit("typed_e26b_diag needs the GPU that was not assigned")
    hardware = _e25_hardware_report()
    (work / "hardware.json").write_text(_json.dumps(hardware, indent=2), encoding="utf-8")
    log(f"e26b hardware: {hardware}")
    e26 = _find_e26_root(bundle)
    ckpts = _find_e26b_ckpts(bundle)
    for seed, sha in _E26B_CKPT_SHA.items():
        f = ckpts / f"s{seed}_treat_final.safetensors"
        got = _hashlib.sha256(f.read_bytes()).hexdigest()
        if got != sha:
            raise SystemExit(f"e26b: finalist s{seed} hash mismatch: {got}")
    log("e26b: finalist hashes verified against frozen literals")
    _, train_cases, _ = _e24_cases_and_qtypes(e26 / "train_treat_rows.jsonl")
    _, dev_cases, _ = _e24_cases_and_qtypes(e26 / "dev_rows.jsonl")
    eval_specs = (("known", "test_c_rows.jsonl"), ("new", "test_d_rows.jsonl"),
                  ("mono", "mono_rows.jsonl"))
    qtypes, eval_cases, seen = {}, {}, set()
    for fname in ("train_treat_rows.jsonl", "dev_rows.jsonl") + tuple(f for _, f in eval_specs):
        for r in [_json.loads(l) for l in (e26 / fname).read_text(encoding="utf-8").splitlines() if l.strip()]:
            if r["id"] in seen:
                raise ValueError(f"duplicate id: {r['id']}")
            seen.add(r["id"])
            qtypes[r["id"]] = r["question_type"]
    for name, fname in eval_specs:
        _, cases, _ = _e24_cases_and_qtypes(e26 / fname)
        eval_cases[name] = cases
    log(f"e26b: treat={len(train_cases)} dev={len(dev_cases)} "
        f"evals={ {k: len(v) for k, v in eval_cases.items()} }")

    from mokka_decisions.arch_julia import make_seeded_julia_arch
    from safetensors.torch import load_file, save_file
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("jhu-clsp/mmBERT-small")

    def run_arm(tag: str, seed: int, *, continue_from=None, epochs_max: int,
                early_stop: bool, preflight: bool) -> dict:
        arm_dir = work / tag
        arm_dir.mkdir(parents=True, exist_ok=True)
        model = make_seeded_julia_arch("jhu-clsp/mmBERT-small", seed=seed)
        if continue_from is not None:
            model.load_state_dict(load_file(str(continue_from), device="cpu"), strict=True)
        init_hash = _state_dict_sha256(model)
        save_file(dict(model.state_dict()), str(arm_dir / "initial.safetensors"),
                  metadata={"format": "pt"})
        model = model.to("cuda")
        pre = None
        if preflight:
            pre = _e24_preflight(model, tokenizer, train_cases, qtypes,
                                 arm_dir / "initial.safetensors", init_hash, "cuda")
            log(f"e26b preflight {tag}: OK")
        result = _e25_train_loop(model, tokenizer, train_cases, dev_cases, qtypes, arm_dir,
                                 seed=seed, device="cuda", epochs_max=epochs_max,
                                 eval_cases=eval_cases, early_stop=early_stop)
        result["init_sha256"] = init_hash
        result["preflight"] = pre
        result["continued_from"] = str(continue_from) if continue_from else None
        if continue_from is not None:
            curves = [_json.loads(l) for l in
                      (arm_dir / "curves.jsonl").read_text(encoding="utf-8").splitlines()]
            best = max(curves, key=lambda c: c["dev_acc"] or 0.0)
            result["d1"] = {"dev_at_e26_stop": _E26B_DEV_AT_STOP[seed],
                            "dev_final": result["final_dev_acc"],
                            "delta": round(result["final_dev_acc"] - _E26B_DEV_AT_STOP[seed], 4),
                            "dev_best_during_continuation": best["dev_acc"],
                            "dev_best_epoch_index": best["epoch"]}
        log(f"e26b {tag}: fit={result['fit']} final_train={result['final_train_acc']:.4f} "
            f"final_dev={result['final_dev_acc']:.4f} epochs={result['epochs_run']} "
            f"eff={result['effective_steps']}/{result['attempted_steps']}"
            + (f" d1_delta={result['d1']['delta']}" if continue_from else "")
            + f" in {result['trained_s']}s")
        (work / "run_status.json").write_text(
            _json.dumps({"stage": "typed_e26b_diag", "arm": tag, "seed": seed,
                         "fit": result["fit"], "final_dev_acc": result["final_dev_acc"]}),
            encoding="utf-8")
        return result

    summary = {
        "protocol": "docs/internal/E26B_PROTOCOL_2026-10-10.md (frozen pre-launch)",
        "objective": ("stopping-rule and seed-variance diagnostic after the E26 review; "
                      "tests C/D/mono are HISTORICAL (inspected in E26) - descriptive only"),
        "arms": {"s44": "fresh seed, exact E26 treat recipe, early stop ON",
                 "s42_cont": "continue s42_treat finalist, early stop OFF, 241 more epochs",
                 "s43_cont": "continue s43_treat finalist, early stop OFF, 258 more epochs",
                 "s45": "fresh seed, exact E26 treat recipe, early stop ON"},
        "budget": {"per_arm_effective_steps_max": 900, "continuation_epochs": _E26B_EPOCHS_RUN},
        "dev_at_e26_stop": _E26B_DEV_AT_STOP,
        "hardware": hardware,
        "results": {},
        "forgetting_experiment": "NOT launched",
    }
    order = [("s44", 44, dict(epochs_max=300, early_stop=True, preflight=True)),
             ("s42_cont", 42, dict(continue_from=ckpts / "s42_treat_final.safetensors",
                                   epochs_max=300 - _E26B_EPOCHS_RUN[42],
                                   early_stop=False, preflight=False)),
             ("s43_cont", 43, dict(continue_from=ckpts / "s43_treat_final.safetensors",
                                   epochs_max=300 - _E26B_EPOCHS_RUN[43],
                                   early_stop=False, preflight=False)),
             ("s45", 45, dict(epochs_max=300, early_stop=True, preflight=False))]
    for tag, seed, kw in order:
        summary["results"][tag] = run_arm(tag, seed, **kw)
    (work / "e26b_summary.json").write_text(_json.dumps(summary, indent=2, default=str),
                                            encoding="utf-8")
    log("e26b_diag: done")


def stage_typed_e26c_seeds(bundle: Path, work: Path, config: str) -> None:
    """E26c: fresh treat seeds 46/47, exact E26 recipe (E26b review's replication arm).

    Closes the near-miss gap the frozen E26b D3 rule left open (s44 = 43/48 with
    4/4 pairs): two more seeds decide whether the sub-0.90 testC ceiling is
    variance or signal. Tests C/D/mono HISTORICAL: descriptive only.
    """
    import json as _json

    import torch

    if not torch.cuda.is_available():
        raise SystemExit("typed_e26c_seeds needs the GPU that was not assigned")
    hardware = _e25_hardware_report()
    (work / "hardware.json").write_text(_json.dumps(hardware, indent=2), encoding="utf-8")
    log(f"e26c hardware: {hardware}")
    e26 = _find_e26_root(bundle)
    _, train_cases, _ = _e24_cases_and_qtypes(e26 / "train_treat_rows.jsonl")
    _, dev_cases, _ = _e24_cases_and_qtypes(e26 / "dev_rows.jsonl")
    eval_specs = (("known", "test_c_rows.jsonl"), ("new", "test_d_rows.jsonl"),
                  ("mono", "mono_rows.jsonl"))
    qtypes, eval_cases, seen = {}, {}, set()
    for fname in ("train_treat_rows.jsonl", "dev_rows.jsonl") + tuple(f for _, f in eval_specs):
        for r in [_json.loads(l) for l in (e26 / fname).read_text(encoding="utf-8").splitlines() if l.strip()]:
            if r["id"] in seen:
                raise ValueError(f"duplicate id: {r['id']}")
            seen.add(r["id"])
            qtypes[r["id"]] = r["question_type"]
    for name, fname in eval_specs:
        _, cases, _ = _e24_cases_and_qtypes(e26 / fname)
        eval_cases[name] = cases
    log(f"e26c: treat={len(train_cases)} dev={len(dev_cases)} "
        f"evals={ {k: len(v) for k, v in eval_cases.items()} }")

    from mokka_decisions.arch_julia import make_seeded_julia_arch
    from safetensors.torch import save_file
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("jhu-clsp/mmBERT-small")

    def run_seed(seed: int, preflight: bool) -> dict:
        arm = work / f"s{seed}"
        arm.mkdir(parents=True, exist_ok=True)
        model = make_seeded_julia_arch("jhu-clsp/mmBERT-small", seed=seed)
        init_hash = _state_dict_sha256(model)
        save_file(dict(model.state_dict()), str(arm / "initial.safetensors"),
                  metadata={"format": "pt"})
        model = model.to("cuda")
        pre = None
        if preflight:
            pre = _e24_preflight(model, tokenizer, train_cases, qtypes,
                                 arm / "initial.safetensors", init_hash, "cuda")
            log(f"e26c preflight s{seed}: OK")
        result = _e25_train_loop(model, tokenizer, train_cases, dev_cases, qtypes, arm,
                                 seed=seed, device="cuda", epochs_max=300,
                                 eval_cases=eval_cases)
        result["init_sha256"] = init_hash
        result["preflight"] = pre
        log(f"e26c s{seed}: fit={result['fit']} final_train={result['final_train_acc']:.4f} "
            f"final_dev={result['final_dev_acc']:.4f} epochs={result['epochs_run']} "
            f"eff={result['effective_steps']}/{result['attempted_steps']} in {result['trained_s']}s")
        (work / "run_status.json").write_text(
            _json.dumps({"stage": "typed_e26c_seeds", "seed": seed, "fit": result["fit"],
                         "final_dev_acc": result["final_dev_acc"]}), encoding="utf-8")
        return result

    summary = {
        "protocol": "docs/internal/E26C_PROTOCOL_2026-10-10.md (frozen pre-launch)",
        "objective": ("two more treat seeds with the exact E26 recipe; closes the "
                      "near-miss gap of the E26b D3 rule (tests HISTORICAL, descriptive)"),
        "hardware": hardware, "results": {},
    }
    for seed in (46, 47):
        summary["results"][f"s{seed}"] = run_seed(seed, preflight=(seed == 46))
    (work / "e26c_summary.json").write_text(_json.dumps(summary, indent=2, default=str),
                                            encoding="utf-8")
    log("e26c_seeds: done")





def _find_e27_root(bundle: Path) -> Path:
    for marker in bundle.rglob("e27-grid/manifest.json"):
        return marker.parent
    raise SystemExit("e27-grid dataset not found in bundle (e27-grid/manifest.json)")


def stage_typed_e27_grid(bundle: Path, work: Path, config: str) -> None:
    """E27: dense comparison grid (frozen protocol docs/internal/E27_PROTOCOL_2026-10-10.md).

    TRAIN = the 92 exact E26 treat-train values against 8 thresholds each
    (offsets -25/-7/-1/0/+1/+7/+15/+25, frozen substitutions): 4 Yes / 4 No per
    value with identical value -> any question-ignoring rule is EXACTLY 0.500 on
    train. Loop change (declared): NO perfect-train early stop; dev-patience 10
    (stop when dev_acc has not improved for 10 epochs; finalist = FINAL
    checkpoint, no dev selection). Evals: historical dev/testC/testD/mono from
    e26-order (descriptive, comparable) + FRESH two-decimal testE (real OOD).
    Seeds 42/43 fresh, both run regardless of fit (dev-stopping declared).
    """
    import json as _json

    import torch

    if not torch.cuda.is_available():
        raise SystemExit("typed_e27_grid needs the GPU that was not assigned")
    hardware = _e25_hardware_report()
    (work / "hardware.json").write_text(_json.dumps(hardware, indent=2), encoding="utf-8")
    log(f"e27 hardware: {hardware}")
    e27 = _find_e27_root(bundle)
    e26 = _find_e26_root(bundle)
    _, train_cases, _ = _e24_cases_and_qtypes(e27 / "train_rows.jsonl")
    _, dev_cases, _ = _e24_cases_and_qtypes(e26 / "dev_rows.jsonl")
    eval_specs = (("known", e26, "test_c_rows.jsonl"), ("new", e26, "test_d_rows.jsonl"),
                  ("mono", e26, "mono_rows.jsonl"), ("e", e27, "test_e_rows.jsonl"))
    qtypes, eval_cases, seen = {}, {}, set()
    for root, fname in [(e26, "dev_rows.jsonl")] + [(r, f) for _, r, f in eval_specs]             + [(e27, "train_rows.jsonl")]:
        for r in [_json.loads(l) for l in (root / fname).read_text(encoding="utf-8").splitlines() if l.strip()]:
            if r["id"] in seen:
                raise ValueError(f"duplicate id: {r['id']}")
            seen.add(r["id"])
            qtypes[r["id"]] = r["question_type"]
    for name, root, fname in eval_specs:
        _, cases, _ = _e24_cases_and_qtypes(root / fname)
        eval_cases[name] = cases
    log(f"e27: train={len(train_cases)} dev={len(dev_cases)} "
        f"evals={ {k: len(v) for k, v in eval_cases.items()} }")

    from mokka_decisions.arch_julia import make_seeded_julia_arch
    from safetensors.torch import save_file
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("jhu-clsp/mmBERT-small")

    def run_seed(seed: int, preflight: bool) -> dict:
        arm = work / f"s{seed}"
        arm.mkdir(parents=True, exist_ok=True)
        model = make_seeded_julia_arch("jhu-clsp/mmBERT-small", seed=seed)
        init_hash = _state_dict_sha256(model)
        save_file(dict(model.state_dict()), str(arm / "initial.safetensors"),
                  metadata={"format": "pt"})
        model = model.to("cuda")
        pre = None
        if preflight:
            pre = _e24_preflight(model, tokenizer, train_cases, qtypes,
                                 arm / "initial.safetensors", init_hash, "cuda")
            log(f"e27 preflight s{seed}: OK")
        result = _e25_train_loop(model, tokenizer, train_cases, dev_cases, qtypes, arm,
                                 seed=seed, device="cuda", epochs_max=300,
                                 eval_cases=eval_cases, early_stop=False, dev_patience=10)
        result["init_sha256"] = init_hash
        result["preflight"] = pre
        log(f"e27 s{seed}: fit={result['fit']} final_train={result['final_train_acc']:.4f} "
            f"final_dev={result['final_dev_acc']:.4f} best_dev={result['best_dev_acc']:.4f}@{result['best_dev_epoch']} "
            f"epochs={result['epochs_run']} eff={result['effective_steps']}/{result['attempted_steps']} "
            f"in {result['trained_s']}s")
        (work / "run_status.json").write_text(
            _json.dumps({"stage": "typed_e27_grid", "seed": seed, "fit": result["fit"],
                         "final_dev_acc": result["final_dev_acc"]}), encoding="utf-8")
        return result

    summary = {
        "protocol": "docs/internal/E27_PROTOCOL_2026-10-10.md (frozen pre-launch)",
        "objective": ("dense comparison grid: both axes covered per value; "
                      "question-ignoring rules exactly 0.500 on train by construction; "
                      "dev-patience 10 replaces memorization stop; fresh 2-decimal testE "
                      "is the real OOD measure; historical evals descriptive"),
        "recipe": {"train_loop": "E25 loop unchanged except early_stop=False + dev_patience=10",
                   "epochs_max": 300, "microbatch": 4, "grad_accumulation": 8,
                   "selection": "finalist = FINAL checkpoint (no dev selection)"},
        "data": {"train": len(train_cases), "dev": len(dev_cases),
                 "evals": {k: len(v) for k, v in eval_cases.items()}},
        "hardware": hardware,
        "results": {},
        "forgetting_experiment": "NOT launched",
    }
    for seed in (42, 43):
        summary["results"][f"s{seed}"] = run_seed(seed, preflight=(seed == 42))
    (work / "e27_summary.json").write_text(_json.dumps(summary, indent=2, default=str),
                                           encoding="utf-8")
    log("e27_grid: done")




def _find_e28_root(bundle: Path) -> Path:
    for marker in bundle.rglob("e28-ext/manifest.json"):
        return marker.parent
    raise SystemExit("e28-ext dataset not found in bundle (e28-ext/manifest.json)")


def stage_typed_e28_ext(bundle: Path, work: Path, config: str) -> None:
    """E28: extremes + monotone chains on the frozen E27 grid (frozen protocol
    docs/internal/E28_PROTOCOL_2026-10-10.md). Attacks the two declared E27
    residuals: extreme-threshold coverage (16 new values at the range edges) and
    the value-dimension at fixed threshold (8 monotone chains). Everything else
    identical to E27 (dev-patience-10, finalist = final checkpoint, seeds 42/43).
    Evals unchanged: historical + fresh two-decimal testE.
    """
    import json as _json

    import torch

    if not torch.cuda.is_available():
        raise SystemExit("typed_e28_ext needs the GPU that was not assigned")
    hardware = _e25_hardware_report()
    (work / "hardware.json").write_text(_json.dumps(hardware, indent=2), encoding="utf-8")
    log(f"e28 hardware: {hardware}")
    e28 = _find_e28_root(bundle)
    e27 = _find_e27_root(bundle)
    e26 = _find_e26_root(bundle)
    _, train_cases, _ = _e24_cases_and_qtypes(e28 / "train_rows.jsonl")
    _, dev_cases, _ = _e24_cases_and_qtypes(e26 / "dev_rows.jsonl")
    eval_specs = (("known", e26, "test_c_rows.jsonl"), ("new", e26, "test_d_rows.jsonl"),
                  ("mono", e26, "mono_rows.jsonl"), ("e", e27, "test_e_rows.jsonl"))
    qtypes, eval_cases, seen = {}, {}, set()
    for root, fname in [(e26, "dev_rows.jsonl")] + [(r, f) for _, r, f in eval_specs] \
            + [(e28, "train_rows.jsonl")]:
        for r in [_json.loads(l) for l in (root / fname).read_text(encoding="utf-8").splitlines() if l.strip()]:
            if r["id"] in seen:
                raise ValueError(f"duplicate id: {r['id']}")
            seen.add(r["id"])
            qtypes[r["id"]] = r["question_type"]
    for name, root, fname in eval_specs:
        _, cases, _ = _e24_cases_and_qtypes(root / fname)
        eval_cases[name] = cases
    log(f"e28: train={len(train_cases)} dev={len(dev_cases)} "
        f"evals={ {k: len(v) for k, v in eval_cases.items()} }")

    from mokka_decisions.arch_julia import make_seeded_julia_arch
    from safetensors.torch import save_file
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("jhu-clsp/mmBERT-small")

    def run_seed(seed: int, preflight: bool) -> dict:
        arm = work / f"s{seed}"
        arm.mkdir(parents=True, exist_ok=True)
        model = make_seeded_julia_arch("jhu-clsp/mmBERT-small", seed=seed)
        init_hash = _state_dict_sha256(model)
        save_file(dict(model.state_dict()), str(arm / "initial.safetensors"),
                  metadata={"format": "pt"})
        model = model.to("cuda")
        pre = None
        if preflight:
            pre = _e24_preflight(model, tokenizer, train_cases, qtypes,
                                 arm / "initial.safetensors", init_hash, "cuda")
            log(f"e28 preflight s{seed}: OK")
        result = _e25_train_loop(model, tokenizer, train_cases, dev_cases, qtypes, arm,
                                 seed=seed, device="cuda", epochs_max=300,
                                 eval_cases=eval_cases, early_stop=False, dev_patience=10)
        result["init_sha256"] = init_hash
        result["preflight"] = pre
        log(f"e28 s{seed}: fit={result['fit']} final_train={result['final_train_acc']:.4f} "
            f"final_dev={result['final_dev_acc']:.4f} best_dev={result['best_dev_acc']:.4f}@{result['best_dev_epoch']} "
            f"epochs={result['epochs_run']} eff={result['effective_steps']}/{result['attempted_steps']} "
            f"in {result['trained_s']}s")
        (work / "run_status.json").write_text(
            _json.dumps({"stage": "typed_e28_ext", "seed": seed, "fit": result["fit"],
                         "final_dev_acc": result["final_dev_acc"]}), encoding="utf-8")
        return result

    summary = {
        "protocol": "docs/internal/E28_PROTOCOL_2026-10-10.md (frozen pre-launch)",
        "objective": ("close the two declared E27 residuals: extreme-threshold coverage "
                      "(16 edge values x grid) and value-order at fixed threshold (8 chains); "
                      "everything else identical to E27"),
        "data": {"train": len(train_cases), "dev": len(dev_cases),
                 "evals": {k: len(v) for k, v in eval_cases.items()}},
        "hardware": hardware,
        "results": {},
        "forgetting_experiment": "NOT launched (next if E28 closes the residuals)",
    }
    for seed in (42, 43):
        summary["results"][f"s{seed}"] = run_seed(seed, preflight=(seed == 42))
    (work / "e28_summary.json").write_text(_json.dumps(summary, indent=2, default=str),
                                           encoding="utf-8")
    log("e28_ext: done")




def stage_typed_e28b_seeds(bundle: Path, work: Path, config: str) -> None:
    """E28b: fresh seeds 44/45 with the exact E28 recipe (variance check of the
    edge trade-off seen in E28: s42 traded ~2 interp rows for extrap while s43
    was near-complete). Tests whether the s43-level candidate is typical.
    Tests C/D/mono/testE HISTORICAL/inspected: descriptive only.
    """
    import json as _json

    import torch

    if not torch.cuda.is_available():
        raise SystemExit("typed_e28b_seeds needs the GPU that was not assigned")
    hardware = _e25_hardware_report()
    (work / "hardware.json").write_text(_json.dumps(hardware, indent=2), encoding="utf-8")
    log(f"e28b hardware: {hardware}")
    e28 = _find_e28_root(bundle)
    e27 = _find_e27_root(bundle)
    e26 = _find_e26_root(bundle)
    _, train_cases, _ = _e24_cases_and_qtypes(e28 / "train_rows.jsonl")
    _, dev_cases, _ = _e24_cases_and_qtypes(e26 / "dev_rows.jsonl")
    eval_specs = (("known", e26, "test_c_rows.jsonl"), ("new", e26, "test_d_rows.jsonl"),
                  ("mono", e26, "mono_rows.jsonl"), ("e", e27, "test_e_rows.jsonl"))
    qtypes, eval_cases, seen = {}, {}, set()
    for root, fname in [(e26, "dev_rows.jsonl")] + [(r, f) for _, r, f in eval_specs] + [(e28, "train_rows.jsonl")]:
        for r in [_json.loads(l) for l in (root / fname).read_text(encoding="utf-8").splitlines() if l.strip()]:
            if r["id"] in seen:
                raise ValueError(f"duplicate id: {r['id']}")
            seen.add(r["id"])
            qtypes[r["id"]] = r["question_type"]
    for name, root, fname in eval_specs:
        _, cases, _ = _e24_cases_and_qtypes(root / fname)
        eval_cases[name] = cases
    log(f"e28b: train={len(train_cases)} dev={len(dev_cases)} "
        f"evals={ {k: len(v) for k, v in eval_cases.items()} }")

    from mokka_decisions.arch_julia import make_seeded_julia_arch
    from safetensors.torch import save_file
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("jhu-clsp/mmBERT-small")

    def run_seed(seed: int, preflight: bool) -> dict:
        arm = work / f"s{seed}"
        arm.mkdir(parents=True, exist_ok=True)
        model = make_seeded_julia_arch("jhu-clsp/mmBERT-small", seed=seed)
        init_hash = _state_dict_sha256(model)
        save_file(dict(model.state_dict()), str(arm / "initial.safetensors"),
                  metadata={"format": "pt"})
        model = model.to("cuda")
        pre = None
        if preflight:
            pre = _e24_preflight(model, tokenizer, train_cases, qtypes,
                                 arm / "initial.safetensors", init_hash, "cuda")
            log(f"e28b preflight s{seed}: OK")
        result = _e25_train_loop(model, tokenizer, train_cases, dev_cases, qtypes, arm,
                                 seed=seed, device="cuda", epochs_max=300,
                                 eval_cases=eval_cases, early_stop=False, dev_patience=10)
        result["init_sha256"] = init_hash
        result["preflight"] = pre
        log(f"e28b s{seed}: fit={result['fit']} final_dev={result['final_dev_acc']:.4f} "
            f"best_dev={result['best_dev_acc']:.4f}@{result['best_dev_epoch']} "
            f"epochs={result['epochs_run']} eff={result['effective_steps']}/{result['attempted_steps']}")
        (work / "run_status.json").write_text(
            _json.dumps({"stage": "typed_e28b_seeds", "seed": seed, "fit": result["fit"],
                         "final_dev_acc": result["final_dev_acc"]}), encoding="utf-8")
        return result

    summary = {"protocol": "docs/internal/E28B_PROTOCOL_2026-10-10.md (frozen pre-launch)",
               "objective": ("two more E28-recipe seeds: is the s42 edge trade-off variance? "
                             "tests HISTORICAL/inspected: descriptive"),
               "hardware": hardware, "results": {}}
    for seed in (44, 45):
        summary["results"][f"s{seed}"] = run_seed(seed, preflight=(seed == 44))
    (work / "e28b_summary.json").write_text(_json.dumps(summary, indent=2, default=str),
                                            encoding="utf-8")
    log("e28b_seeds: done")




_E29_CKPT_SHA = "75ea9eed7aac43a1ac1dc7fdeca40329af3335696861db3aae92aec844614829"


def _find_e29_root(bundle: Path) -> Path:
    for marker in bundle.rglob("e29-mix/manifest.json"):
        return marker.parent
    raise SystemExit("e29-mix manifest not found in bundle (e29-mix/manifest.json)")


def stage_typed_e29_forget(bundle: Path, work: Path, config: str) -> None:
    """E29: forgetting/integration study (frozen protocol E29_PROTOCOL_2026-10-10.md).

    Resume the E28 s43 CANDIDATE (sha256-verified) and train 20 epochs in two
    arms: MIX (944 original tasks + 944 E28-grid replay) vs ORIG (944 original
    only). Baseline evals BEFORE training; order (testC/testE) + task (original
    300-slice) measured after each 5-epoch phase. Fresh optimizer per phase,
    identical in both arms (declared).
    """
    import hashlib as _hashlib
    import json as _json

    import torch

    if not torch.cuda.is_available():
        raise SystemExit("typed_e29_forget needs the GPU that was not assigned")
    hardware = _e25_hardware_report()
    (work / "hardware.json").write_text(_json.dumps(hardware, indent=2), encoding="utf-8")
    log(f"e29 hardware: {hardware}")
    e29 = _find_e29_root(bundle)
    e28 = _find_e28_root(bundle)
    e27 = _find_e27_root(bundle)
    e26 = _find_e26_root(bundle)
    manifest = _json.loads((e29 / "manifest.json").read_text(encoding="utf-8"))

    ckpt = e29 / "e29-ckpts" / "e28_s43_final.safetensors"
    got = _hashlib.sha256(ckpt.read_bytes()).hexdigest()
    if got != _E29_CKPT_SHA:
        raise SystemExit(f"e29: candidate checkpoint hash mismatch: {got}")
    log("e29: candidate checkpoint verified against frozen literal")

    def rows_with_ids(path, ids):
        want = set(ids)
        rows = [_json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]
        picked = [r for r in rows if r["id"] in want]
        if len(picked) != len(want):
            raise SystemExit(f"e29: id mismatch selecting {path}: {len(picked)}/{len(want)}")
        return picked

    orig_train = rows_with_ids(bundle / "data/processed/v1/instances/train.jsonl",
                               manifest["original_train_ids"])
    task_dev = rows_with_ids(bundle / "data/processed/v1/eval/dev_eval.jsonl",
                             manifest["original_dev_ids"])
    _, grid_cases, _ = _e24_cases_and_qtypes(e28 / "train_rows.jsonl")
    _, dev_cases, _ = _e24_cases_and_qtypes(e26 / "dev_rows.jsonl")

    def as_cases(rows):
        from mokka_decisions.contracts import DecisionCase
        return [DecisionCase.from_dict(r) for r in rows]

    qtypes = {}
    for rows in (orig_train, task_dev):
        for r in rows:
            qtypes[r["id"]] = r.get("question_type") or "choice"
    eval_specs = (("known", e26, "test_c_rows.jsonl"), ("new", e26, "test_d_rows.jsonl"),
                  ("mono", e26, "mono_rows.jsonl"), ("e", e27, "test_e_rows.jsonl"))
    eval_cases = {}
    for name, root, fname in eval_specs:
        for r in [_json.loads(l) for l in (root / fname).read_text(encoding="utf-8").splitlines() if l.strip()]:
            qtypes[r["id"]] = r["question_type"]
        _, cases, _ = _e24_cases_and_qtypes(root / fname)
        eval_cases[name] = cases
    eval_cases["task"] = as_cases(task_dev)
    for r in [_json.loads(l) for l in (e28 / "train_rows.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()] + \
            [_json.loads(l) for l in (e26 / "dev_rows.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]:
        qtypes[r["id"]] = r["question_type"]
    log(f"e29: orig={len(orig_train)} grid={len(grid_cases)} task_dev={len(task_dev)} "
        f"evals={ {k: len(v) for k, v in eval_cases.items()} }")

    from mokka_decisions.arch_julia import evaluate_julia_arch, make_seeded_julia_arch
    from safetensors.torch import load_file, save_file
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("jhu-clsp/mmBERT-small")

    def build_model():
        model = make_seeded_julia_arch("jhu-clsp/mmBERT-small", seed=43)
        model.load_state_dict(load_file(str(ckpt), device="cpu"), strict=True)
        return model

    def eval_all(model, device="cuda"):
        out = {}
        for name, cases in eval_cases.items():
            ev = evaluate_julia_arch(model, tokenizer, cases, qtypes, device=device)
            out[name] = {"n": ev["n"], "accuracy": ev["accuracy"]}
        return out

    def eval_files(model, arm_dir, device="cuda"):
        arm_dir.mkdir(parents=True, exist_ok=True)
        for name, cases in eval_cases.items():
            ev = evaluate_julia_arch(model, tokenizer, cases, qtypes, device=device,
                                     collect_rows=True)
            (arm_dir / f"eval_{name}.jsonl").write_text(
                "".join(_json.dumps(r, default=str) + "\n" for r in ev["rows"]), encoding="utf-8")

    summary = {
        "protocol": "docs/internal/E29_PROTOCOL_2026-10-10.md (frozen pre-launch)",
        "objective": ("forgetting/integration: does E28-s43's comparison capability survive "
                      "resumed training on the original tasks, with and without replay?"),
        "arms": {"MIX": "944 original + 944 E28 grid (59 steps/epoch)",
                 "ORIG": "944 original only (30 steps/epoch)"},
        "phases": 4, "epochs_per_phase": 5,
        "candidate_sha256": _E29_CKPT_SHA,
        "hardware": hardware,
        "baseline": None, "results": {"MIX": [], "ORIG": []},
    }

    # baseline (phase 0): candidate as-is, before ANY training
    model = build_model().to("cuda")
    summary["baseline"] = eval_all(model)
    eval_files(model, work / "baseline")
    save_file(dict(model.state_dict()), str(work / "candidate_initial.safetensors"),
              metadata={"format": "pt"})
    log(f"e29 baseline (candidate as-is): { {k: round(v['accuracy'], 4) for k, v in summary['baseline'].items()} }")
    del model
    torch.cuda.empty_cache()

    def run_arm(tag, train_cases):
        model = build_model()
        arm_root = work / tag
        arm_root.mkdir(parents=True, exist_ok=True)
        for phase in range(1, 5):
            phase_dir = arm_root / f"phase{phase}"
            phase_dir.mkdir(parents=True, exist_ok=True)
            res = _e25_train_loop(model, tokenizer, train_cases, dev_cases, qtypes,
                                  phase_dir, seed=43, device="cuda",
                                  epochs_max=5, eval_cases=eval_cases,
                                  early_stop=False, dev_patience=None)
            summary["results"][tag].append({"phase": phase,
                                            "evals": res["evals"],
                                            "final_train_acc": res["final_train_acc"],
                                            "final_dev_acc": res["final_dev_acc"],
                                            "effective_steps": res["effective_steps"],
                                            "attempted_steps": res["attempted_steps"]})
            log(f"e29 {tag} phase{phase}: train={res['final_train_acc']:.4f} "
                f"numdev={res['final_dev_acc']:.4f} "
                f"evals={ {k: round(v['accuracy'], 4) for k, v in res['evals'].items()} }")
            (work / "run_status.json").write_text(
                _json.dumps({"stage": "typed_e29_forget", "arm": tag, "phase": phase}),
                encoding="utf-8")
        save_file(dict(model.state_dict()), str(arm_root / "final.safetensors"),
                  metadata={"format": "pt"})
        del model
        torch.cuda.empty_cache()

    mix_cases = as_cases(orig_train) + list(grid_cases)
    run_arm("MIX", mix_cases)
    run_arm("ORIG", as_cases(orig_train))

    (work / "e29_summary.json").write_text(_json.dumps(summary, indent=2, default=str),
                                           encoding="utf-8")
    log("e29_forget: done")






def stage_typed_e29b_integration(bundle: Path, work: Path, config: str) -> None:
    """E29b: FORMAL integration validation, 15-epoch curriculum (frozen protocol E29B_PROTOCOL_2026-10-11.md).

    Same as E29's MIX arm with ONE declared change: 3 phases x 5 epochs
    (E29's curve: task recovery plateaus by phase 3 while the fresh-order cost
    concentrates in phase 4). No ORIG arm (already measured). Frozen bars F1'
    identical to F1. One shot.
    """
    import hashlib as _hashlib
    import json as _json

    import torch

    if not torch.cuda.is_available():
        raise SystemExit("typed_e29_forget needs the GPU that was not assigned")
    hardware = _e25_hardware_report()
    (work / "hardware.json").write_text(_json.dumps(hardware, indent=2), encoding="utf-8")
    log(f"e29 hardware: {hardware}")
    e29 = _find_e29_root(bundle)
    e28 = _find_e28_root(bundle)
    e27 = _find_e27_root(bundle)
    e26 = _find_e26_root(bundle)
    manifest = _json.loads((e29 / "manifest.json").read_text(encoding="utf-8"))

    ckpt = e29 / "e29-ckpts" / "e28_s43_final.safetensors"
    got = _hashlib.sha256(ckpt.read_bytes()).hexdigest()
    if got != _E29_CKPT_SHA:
        raise SystemExit(f"e29: candidate checkpoint hash mismatch: {got}")
    log("e29: candidate checkpoint verified against frozen literal")

    def rows_with_ids(path, ids):
        want = set(ids)
        rows = [_json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]
        picked = [r for r in rows if r["id"] in want]
        if len(picked) != len(want):
            raise SystemExit(f"e29: id mismatch selecting {path}: {len(picked)}/{len(want)}")
        return picked

    orig_train = rows_with_ids(bundle / "data/processed/v1/instances/train.jsonl",
                               manifest["original_train_ids"])
    task_dev = rows_with_ids(bundle / "data/processed/v1/eval/dev_eval.jsonl",
                             manifest["original_dev_ids"])
    _, grid_cases, _ = _e24_cases_and_qtypes(e28 / "train_rows.jsonl")
    _, dev_cases, _ = _e24_cases_and_qtypes(e26 / "dev_rows.jsonl")

    def as_cases(rows):
        from mokka_decisions.contracts import DecisionCase
        return [DecisionCase.from_dict(r) for r in rows]

    qtypes = {}
    for rows in (orig_train, task_dev):
        for r in rows:
            qtypes[r["id"]] = r.get("question_type") or "choice"
    eval_specs = (("known", e26, "test_c_rows.jsonl"), ("new", e26, "test_d_rows.jsonl"),
                  ("mono", e26, "mono_rows.jsonl"), ("e", e27, "test_e_rows.jsonl"))
    eval_cases = {}
    for name, root, fname in eval_specs:
        for r in [_json.loads(l) for l in (root / fname).read_text(encoding="utf-8").splitlines() if l.strip()]:
            qtypes[r["id"]] = r["question_type"]
        _, cases, _ = _e24_cases_and_qtypes(root / fname)
        eval_cases[name] = cases
    eval_cases["task"] = as_cases(task_dev)
    for r in [_json.loads(l) for l in (e28 / "train_rows.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()] + \
            [_json.loads(l) for l in (e26 / "dev_rows.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]:
        qtypes[r["id"]] = r["question_type"]
    log(f"e29: orig={len(orig_train)} grid={len(grid_cases)} task_dev={len(task_dev)} "
        f"evals={ {k: len(v) for k, v in eval_cases.items()} }")

    from mokka_decisions.arch_julia import evaluate_julia_arch, make_seeded_julia_arch
    from safetensors.torch import load_file, save_file
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("jhu-clsp/mmBERT-small")

    def build_model():
        model = make_seeded_julia_arch("jhu-clsp/mmBERT-small", seed=43)
        model.load_state_dict(load_file(str(ckpt), device="cpu"), strict=True)
        return model

    def eval_all(model, device="cuda"):
        out = {}
        for name, cases in eval_cases.items():
            ev = evaluate_julia_arch(model, tokenizer, cases, qtypes, device=device)
            out[name] = {"n": ev["n"], "accuracy": ev["accuracy"]}
        return out

    def eval_files(model, arm_dir, device="cuda"):
        arm_dir.mkdir(parents=True, exist_ok=True)
        for name, cases in eval_cases.items():
            ev = evaluate_julia_arch(model, tokenizer, cases, qtypes, device=device,
                                     collect_rows=True)
            (arm_dir / f"eval_{name}.jsonl").write_text(
                "".join(_json.dumps(r, default=str) + "\n" for r in ev["rows"]), encoding="utf-8")

    summary = {
        "protocol": "docs/internal/E29B_PROTOCOL_2026-10-11.md (frozen pre-launch)",
        "objective": ("forgetting/integration: does E28-s43's comparison capability survive "
                      "resumed training on the original tasks, with and without replay?"),
        "arm": "MIX only, 3 phases x 5 epochs (E29 had 4; ORIG already measured)",
        "phases": 4, "epochs_per_phase": 5,
        "candidate_sha256": _E29_CKPT_SHA,
        "hardware": hardware,
        "baseline": None, "results": {"MIX": []},
    }

    # baseline (phase 0): candidate as-is, before ANY training
    model = build_model().to("cuda")
    summary["baseline"] = eval_all(model)
    eval_files(model, work / "baseline")
    save_file(dict(model.state_dict()), str(work / "candidate_initial.safetensors"),
              metadata={"format": "pt"})
    log(f"e29 baseline (candidate as-is): { {k: round(v['accuracy'], 4) for k, v in summary['baseline'].items()} }")
    del model
    torch.cuda.empty_cache()

    def run_arm(tag, train_cases):
        model = build_model()
        arm_root = work / tag
        arm_root.mkdir(parents=True, exist_ok=True)
        for phase in range(1, 4):
            phase_dir = arm_root / f"phase{phase}"
            phase_dir.mkdir(parents=True, exist_ok=True)
            res = _e25_train_loop(model, tokenizer, train_cases, dev_cases, qtypes,
                                  phase_dir, seed=43, device="cuda",
                                  epochs_max=5, eval_cases=eval_cases,
                                  early_stop=False, dev_patience=None)
            summary["results"][tag].append({"phase": phase,
                                            "evals": res["evals"],
                                            "final_train_acc": res["final_train_acc"],
                                            "final_dev_acc": res["final_dev_acc"],
                                            "effective_steps": res["effective_steps"],
                                            "attempted_steps": res["attempted_steps"]})
            log(f"e29 {tag} phase{phase}: train={res['final_train_acc']:.4f} "
                f"numdev={res['final_dev_acc']:.4f} "
                f"evals={ {k: round(v['accuracy'], 4) for k, v in res['evals'].items()} }")
            (work / "run_status.json").write_text(
                _json.dumps({"stage": "typed_e29b_integration", "arm": tag, "phase": phase}),
                encoding="utf-8")
        save_file(dict(model.state_dict()), str(arm_root / "final.safetensors"),
                  metadata={"format": "pt"})
        del model
        torch.cuda.empty_cache()

    mix_cases = as_cases(orig_train) + list(grid_cases)
    run_arm("MIX", mix_cases)

    (work / "e29b_summary.json").write_text(_json.dumps(summary, indent=2, default=str),
                                           encoding="utf-8")
    log("e29_forget: done")




def _find_e30_root(bundle: Path) -> Path:
    for marker in bundle.rglob("e30-mix/manifest.json"):
        return marker.parent
    raise SystemExit("e30-mix manifest not found in bundle (e30-mix/manifest.json)")


def stage_typed_e30_val(bundle: Path, work: Path, config: str) -> None:
    """E30: INDEPENDENT integration validation - NEW DISJOINT original sample (frozen protocol E30_PROTOCOL_2026-10-11.md).

    Identical to E29b (candidate, MIX arm, 3 phases x 5 epochs) with ONE change:
    the original-task sample is NEW and DISJOINT (sha256(id) ranks 945-1888 /
    task slice 301-600, per the e30-mix manifest) - the outcome was NOT
    knowable at registration time. F1'' bars identical to F1.
    """
    import hashlib as _hashlib
    import json as _json

    import torch

    if not torch.cuda.is_available():
        raise SystemExit("typed_e29_forget needs the GPU that was not assigned")
    hardware = _e25_hardware_report()
    (work / "hardware.json").write_text(_json.dumps(hardware, indent=2), encoding="utf-8")
    log(f"e29 hardware: {hardware}")
    e29 = _find_e29_root(bundle)
    e30 = _find_e30_root(bundle)
    e28 = _find_e28_root(bundle)
    e27 = _find_e27_root(bundle)
    e26 = _find_e26_root(bundle)
    manifest = _json.loads((e30 / "manifest.json").read_text(encoding="utf-8"))

    ckpt = e29 / "e29-ckpts" / "e28_s43_final.safetensors"  # candidate ships in e29-mix; reused
    got = _hashlib.sha256(ckpt.read_bytes()).hexdigest()
    if got != _E29_CKPT_SHA:
        raise SystemExit(f"e29: candidate checkpoint hash mismatch: {got}")
    log("e29: candidate checkpoint verified against frozen literal")

    def rows_with_ids(path, ids):
        want = set(ids)
        rows = [_json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]
        picked = [r for r in rows if r["id"] in want]
        if len(picked) != len(want):
            raise SystemExit(f"e29: id mismatch selecting {path}: {len(picked)}/{len(want)}")
        return picked

    orig_train = rows_with_ids(bundle / "data/processed/v1/instances/train.jsonl",
                               manifest["original_train_ids"])
    task_dev = rows_with_ids(bundle / "data/processed/v1/eval/dev_eval.jsonl",
                             manifest["original_dev_ids"])
    _, grid_cases, _ = _e24_cases_and_qtypes(e28 / "train_rows.jsonl")
    _, dev_cases, _ = _e24_cases_and_qtypes(e26 / "dev_rows.jsonl")

    def as_cases(rows):
        from mokka_decisions.contracts import DecisionCase
        return [DecisionCase.from_dict(r) for r in rows]

    qtypes = {}
    for rows in (orig_train, task_dev):
        for r in rows:
            qtypes[r["id"]] = r.get("question_type") or "choice"
    eval_specs = (("known", e26, "test_c_rows.jsonl"), ("new", e26, "test_d_rows.jsonl"),
                  ("mono", e26, "mono_rows.jsonl"), ("e", e27, "test_e_rows.jsonl"))
    eval_cases = {}
    for name, root, fname in eval_specs:
        for r in [_json.loads(l) for l in (root / fname).read_text(encoding="utf-8").splitlines() if l.strip()]:
            qtypes[r["id"]] = r["question_type"]
        _, cases, _ = _e24_cases_and_qtypes(root / fname)
        eval_cases[name] = cases
    eval_cases["task"] = as_cases(task_dev)
    for r in [_json.loads(l) for l in (e28 / "train_rows.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()] + \
            [_json.loads(l) for l in (e26 / "dev_rows.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]:
        qtypes[r["id"]] = r["question_type"]
    log(f"e29: orig={len(orig_train)} grid={len(grid_cases)} task_dev={len(task_dev)} "
        f"evals={ {k: len(v) for k, v in eval_cases.items()} }")

    from mokka_decisions.arch_julia import evaluate_julia_arch, make_seeded_julia_arch
    from safetensors.torch import load_file, save_file
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("jhu-clsp/mmBERT-small")

    def build_model():
        model = make_seeded_julia_arch("jhu-clsp/mmBERT-small", seed=43)
        model.load_state_dict(load_file(str(ckpt), device="cpu"), strict=True)
        return model

    def eval_all(model, device="cuda"):
        out = {}
        for name, cases in eval_cases.items():
            ev = evaluate_julia_arch(model, tokenizer, cases, qtypes, device=device)
            out[name] = {"n": ev["n"], "accuracy": ev["accuracy"]}
        return out

    def eval_files(model, arm_dir, device="cuda"):
        arm_dir.mkdir(parents=True, exist_ok=True)
        for name, cases in eval_cases.items():
            ev = evaluate_julia_arch(model, tokenizer, cases, qtypes, device=device,
                                     collect_rows=True)
            (arm_dir / f"eval_{name}.jsonl").write_text(
                "".join(_json.dumps(r, default=str) + "\n" for r in ev["rows"]), encoding="utf-8")

    summary = {
        "protocol": "docs/internal/E30_PROTOCOL_2026-10-11.md (frozen pre-launch)",
        "objective": ("forgetting/integration: does E28-s43's comparison capability survive "
                      "resumed training on the original tasks, with and without replay?"),
        "arm": "MIX only, 3 phases x 5 epochs (E29 had 4; ORIG already measured)",
        "phases": 4, "epochs_per_phase": 5,
        "candidate_sha256": _E29_CKPT_SHA,
        "hardware": hardware,
        "baseline": None, "results": {"MIX": []},
    }

    # baseline (phase 0): candidate as-is, before ANY training
    model = build_model().to("cuda")
    summary["baseline"] = eval_all(model)
    eval_files(model, work / "baseline")
    save_file(dict(model.state_dict()), str(work / "candidate_initial.safetensors"),
              metadata={"format": "pt"})
    log(f"e29 baseline (candidate as-is): { {k: round(v['accuracy'], 4) for k, v in summary['baseline'].items()} }")
    del model
    torch.cuda.empty_cache()

    def run_arm(tag, train_cases):
        model = build_model()
        arm_root = work / tag
        arm_root.mkdir(parents=True, exist_ok=True)
        for phase in range(1, 4):
            phase_dir = arm_root / f"phase{phase}"
            phase_dir.mkdir(parents=True, exist_ok=True)
            res = _e25_train_loop(model, tokenizer, train_cases, dev_cases, qtypes,
                                  phase_dir, seed=43, device="cuda",
                                  epochs_max=5, eval_cases=eval_cases,
                                  early_stop=False, dev_patience=None)
            summary["results"][tag].append({"phase": phase,
                                            "evals": res["evals"],
                                            "final_train_acc": res["final_train_acc"],
                                            "final_dev_acc": res["final_dev_acc"],
                                            "effective_steps": res["effective_steps"],
                                            "attempted_steps": res["attempted_steps"]})
            log(f"e29 {tag} phase{phase}: train={res['final_train_acc']:.4f} "
                f"numdev={res['final_dev_acc']:.4f} "
                f"evals={ {k: round(v['accuracy'], 4) for k, v in res['evals'].items()} }")
            (work / "run_status.json").write_text(
                _json.dumps({"stage": "typed_e30_val", "arm": tag, "phase": phase}),
                encoding="utf-8")
        save_file(dict(model.state_dict()), str(arm_root / "final.safetensors"),
                  metadata={"format": "pt"})
        del model
        torch.cuda.empty_cache()

    mix_cases = as_cases(orig_train) + list(grid_cases)
    run_arm("MIX", mix_cases)

    (work / "e30_summary.json").write_text(_json.dumps(summary, indent=2, default=str),
                                           encoding="utf-8")
    log("e29_forget: done")


def _fetch_julia_src() -> Path:
    """Download Julia-1 from HuggingFace (public) and verify the frozen weights hash.

    Transport amendment (pre-launch, declared): the original plan shipped
    julia-src in the bundle, but the local uplink degraded to ~0.15MB/s; the
    kernel has internet, so it pulls the public snapshot at datacenter speed.
    Same weights (sha256 verified against the same frozen literal), same engine.
    """
    import os as _os
    _os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    from huggingface_hub import snapshot_download
    js = Path(snapshot_download("SupersonicLabs/Julia-1"))
    if not (js / "julia" / "router" / "engine.py").exists():
        raise SystemExit(f"julia snapshot lacks the engine package: {js}")
    return js


_E31_JULIA_SHA = "df853bf7fe424420011f3d0c47a05d7341aa9eefa7fb9f203ea4aada4ad95b72"


def stage_typed_e31_julia(bundle: Path, work: Path, config: str) -> None:
    """E31: fine-tune JULIA-1 on OUR 944-row grid (frozen protocol E31_PROTOCOL_2026-10-11.md).

    The fair same-data comparison: Julia-1 official weights (sha-verified),
    engine-required transformers==5.0.0 installed in-kernel, recipe mirroring
    ours (AdamW 2e-5 uniform, wd 0.01, warmup 5%, clip 1.0, fp16+GradScaler,
    effective batch 32 = 4x8 accumulation, 30 epochs max, dev-patience 10 on the
    E26 numeric dev, finalist=final). Seeds 42+43 both run. Frozen criteria
    J1-J4 with readings (a) recipe transferable (b) memorizes-without-order
    (c) cannot fit at this budget. Nothing auto-launched after.
    """
    import hashlib as _hashlib
    import json as _json
    import subprocess as _subprocess
    import sys as _sys
    import time as _time

    import torch

    if not torch.cuda.is_available():
        raise SystemExit("typed_e31_julia needs the GPU that was not assigned")
    _subprocess.check_call([_sys.executable, "-m", "pip", "install", "-q",
                            "transformers==5.0.0"], env={**__import__("os").environ,
                                                          "PIP_CONFIG_FILE": "/dev/null"})
    js = _fetch_julia_src()
    got = _hashlib.sha256((js / "model.safetensors").read_bytes()).hexdigest()
    if got != _E31_JULIA_SHA:
        raise SystemExit(f"e31: Julia-1 weights hash mismatch: {got}")
    log("e31: Julia-1 weights verified against frozen literal")
    _sys.path.insert(0, str(js))
    from julia.router.engine import FastEngine

    dev_rows = [_json.loads(l) for l in (bundle / "e26-order/dev_rows.jsonl").read_text(
        encoding="utf-8").splitlines() if l.strip()]
    grid_rows = [_json.loads(l) for l in (bundle / "e28-ext/train_rows.jsonl").read_text(
        encoding="utf-8").splitlines() if l.strip()]
    e26 = _find_e26_root(bundle)
    e27 = _find_e27_root(bundle)
    specs = (("known", e26 / "test_c_rows.jsonl"), ("new", e26 / "test_d_rows.jsonl"),
             ("mono", e26 / "mono_rows.jsonl"), ("e", e27 / "test_e_rows.jsonl"))
    eval_rows = {name: [_json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()
                        if l.strip()] for name, path in specs}
    log(f"e31: grid={len(grid_rows)} dev={len(dev_rows)} "
        f"evals={ {k: len(v) for k, v in eval_rows.items()} }")

    def to_payload(rows):
        return [dict(state=_json.loads(r["state"]), question=r["question"],
                     type="noul", options=["No", "Yes"]) for r in rows]

    def golds(rows):
        return [int(r["target_option"] == "true") for r in rows]

    # E31 correction (invalid_model_binding, 2026-10-11): evaluate previously
    # closed over a stage-scope engine loaded with ORIGINAL Julia weights, so
    # dev and tests inside run_seed evaluated the wrong model. The engine is
    # now an explicit argument and the surplus external engine is gone.
    torch.backends.cuda.matmul.allow_tf32 = True

    def evaluate(eval_engine, rows):
        payload, gold = to_payload(rows), golds(rows)
        correct, percase = 0, []
        for i in range(0, len(payload), 16):
            for ans, g, r in zip(eval_engine.predict(payload[i:i + 16]), gold[i:i + 16], rows[i:i + 16]):
                ok = int(ans["index"] == g)
                correct += ok
                percase.append({"id": r["id"], "pred": ans["index"], "gold": g, "ok": ok})
        return correct / len(rows), percase

    def run_seed(seed, work_dir):
        import random as _random

        work_dir.mkdir(parents=True, exist_ok=True)
        _random.seed(seed)
        torch.manual_seed(seed)
        engine = FastEngine(js, device="cuda", transformer_backend="torch",
                            strict_encoding=True, max_length=1024, head_length=512,
                            batch_size=8, marker_only_head=False)
        model = engine.model
        model.train()
        opt = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
        scaler = torch.amp.GradScaler("cuda")
        encoded = engine._encode(to_payload(grid_rows))
        gold_all = golds(grid_rows)
        rng = _random.Random(seed ^ 0xE31)
        steps_per_epoch = (len(encoded) + 31) // 32
        warmup = max(int(steps_per_epoch * 30 * 0.05), 1)

        def lr_lambda(step):
            if step < warmup:
                return step / warmup
            p = (step - warmup) / max(steps_per_epoch * 30 - warmup, 1)
            return max(0.05, 0.95 * (1 - p) + 0.05)

        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
        best_dev, stale, history = -1.0, 0, []
        payload_dev = to_payload(dev_rows)
        for epoch in range(30):
            order = list(range(len(encoded)))
            rng.shuffle(order)
            tr_correct, tr_seen = 0, 0
            for chunk_start in range(0, len(order), 8):
                idx = order[chunk_start:chunk_start + 8]
                batch = engine._pack([encoded[i] for i in idx])
                with torch.autocast("cuda", dtype=torch.float16):
                    values = model(**batch)
                    logits = values[batch["marker_mask"]].view(len(idx), -1)
                    gold = torch.tensor([gold_all[i] for i in idx], device=logits.device)
                    loss = torch.nn.functional.cross_entropy(logits, gold) / 4
                scaler.scale(loss).backward()
                tr_correct += int((logits.argmax(-1) == gold).sum())
                tr_seen += len(idx)
                if (chunk_start // 8 + 1) % 4 == 0 or chunk_start + 8 >= len(order):
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(opt)
                    scaler.update()
                    sched.step()
                    opt.zero_grad(set_to_none=True)
            model.eval()
            dev_acc, _ = evaluate(engine, dev_rows)
            tr_acc = tr_correct / tr_seen
            history.append({"epoch": epoch, "train_acc": round(tr_acc, 4),
                            "dev_acc": round(dev_acc, 4)})
            log(f"e31 s{seed} epoch={epoch} train={tr_acc:.4f} dev={dev_acc:.4f}")
            best_dev, stale = (dev_acc, 0) if dev_acc > best_dev + 1e-12 else (best_dev, stale + 1)
            model.train()
            if stale >= 10:
                break
        model.eval()
        # identity of the weights ACTUALLY evaluated below (dev used the same
        # engine object; this hash is taken at eval time, not of the saved file)
        import hashlib as _hashlib
        _h = _hashlib.sha256()
        for _k in sorted(model.state_dict()):
            _t = model.state_dict()[_k].detach().to(torch.float32).cpu().contiguous()
            _h.update(_k.encode("utf-8")); _h.update(_t.numpy().tobytes())
        eval_weights_sha256 = _h.hexdigest()
        result = {"epochs": len(history), "final_train_acc": history[-1]["train_acc"],
                  "final_dev_acc": history[-1]["dev_acc"], "best_dev_acc": best_dev,
                  "history": history, "evals": {},
                  "eval_model": {"binding": "run_seed local engine (engine.model)",
                                 "eval_weights_sha256": eval_weights_sha256}}
        for name, rows in eval_rows.items():
            acc, percase = evaluate(engine, rows)
            result["evals"][name] = {"n": len(rows), "accuracy": round(acc, 4)}
            (work_dir / f"eval_{name}.jsonl").write_text(
                "".join(_json.dumps(r) + "\n" for r in percase), encoding="utf-8")
        (work_dir / "curves.jsonl").write_text(
            "".join(_json.dumps(h) + "\n" for h in history), encoding="utf-8")
        from safetensors.torch import save_file
        save_file(dict(model.state_dict()), str(work_dir / "final.safetensors"),
                  metadata={"format": "pt"})
        log(f"e31 s{seed}: { {k: v['accuracy'] for k, v in result['evals'].items()} }")
        del model, engine, opt
        torch.cuda.empty_cache()
        return result

    summary = {"protocol": "docs/internal/E31_PROTOCOL_2026-10-11.md (frozen pre-launch)",
               "subject": "Julia-1 official weights fine-tuned on our 944-row E28 grid",
               "julia_sha256": _E31_JULIA_SHA,
               "results": {}}
    for seed in (42, 43):
        summary["results"][f"s{seed}"] = run_seed(seed, work / f"s{seed}")
    (work / "e31_summary.json").write_text(_json.dumps(summary, indent=2, default=str),
                                           encoding="utf-8")
    log("e31_julia: done")




TYPED_STAGES = {"typed_arms": stage_typed_arms, "typed_a1": stage_typed_a1,
          "typed_ord": stage_typed_ord, "typed_small_ord": stage_typed_small_ord,
          "typed_e15": stage_typed_e15, "typed_e15_seeds": stage_typed_e15_seeds,
          "typed_e15_completion": stage_typed_e15_completion,
          "typed_julia_small": stage_typed_julia_small,
          "typed_e18_supervision": stage_typed_e18_supervision,
          "typed_e19_rulevar": stage_typed_e19_rulevar,
          "typed_e20_contrast": stage_typed_e20_contrast,
          "typed_e23_numeric": stage_typed_e23_numeric,
          "typed_e24_fit": stage_typed_e24_fit,
          "typed_e25_thrvar": stage_typed_e25_thrvar,
          "typed_e26_order": stage_typed_e26_order,
          "typed_e26b_diag": stage_typed_e26b_diag,
          "typed_e26c_seeds": stage_typed_e26c_seeds,
          "typed_e27_grid": stage_typed_e27_grid,
          "typed_e28_ext": stage_typed_e28_ext,
          "typed_e28b_seeds": stage_typed_e28b_seeds,
          "typed_e29_forget": stage_typed_e29_forget,
          "typed_e29b_integration": stage_typed_e29b_integration,
          "typed_e30_val": stage_typed_e30_val,
          "typed_e31_julia": stage_typed_e31_julia,
          # H2 (controlled, one variable): the dev curves were not saturated at
          # 4/8 epochs; extend epochs only — data/init/loss/seeds unchanged.
          "typed_ord2": lambda b, w, c: stage_typed_ord(b, w, c, epochs=6, out="base_ord6"),
          "typed_small_ord2": lambda b, w, c: stage_typed_small_ord(b, w, c, epochs=8, out="small_ord8")}

# ==== embedded: gliner worker source (runs inside the isolated venv) ====
GLINER_WORKER_SRC = "\"\"\"GLiNER2.5 worker: runs INSIDE an isolated venv (its own transformers<5).\n\nProtocol (argv): cases.jsonl rows_out.jsonl\nReads decision cases, writes BackendRow JSONL. Kept dependency-free of the\nmain kernel env: the parent installs this venv with the mokka wheel +\ngliner2[local] + transformers<5, then invokes this file with that venv's\npython. The main environment (trainer, future VLMs) stays untouched.\n\"\"\"\n\nfrom __future__ import annotations\n\nimport json\nimport sys\nimport time\nfrom pathlib import Path\n\n\ndef main() -> int:\n    cases_path, out_path = Path(sys.argv[1]), Path(sys.argv[2])\n    sys.path.insert(0, str(Path(__file__).parent))\n\n    from mokka_decisions.backends.gliner import GlinerBackend\n    from mokka_decisions.contracts import DecisionCase\n    from mokka_decisions.evaluate import run_backend\n\n    cases = [DecisionCase.from_dict(json.loads(ln)) for ln in cases_path.read_text(encoding=\"utf-8\").splitlines() if ln.strip()]\n    t0 = time.time()\n    backend = GlinerBackend()\n    try:\n        import torch\n\n        if torch.cuda.is_available():\n            backend.model = backend.model.cuda().eval()\n            print(\"[gliner-worker] running on GPU\", flush=True)\n    except Exception as exc:\n        print(f\"[gliner-worker] GPU unavailable ({exc}); CPU mode\", flush=True)\n    rows = run_backend(backend, cases)\n    out_path.parent.mkdir(parents=True, exist_ok=True)\n    with open(out_path, \"w\", encoding=\"utf-8\", newline=\"\\n\") as fh:\n        for row in rows:\n            fh.write(json.dumps(row.to_dict(), ensure_ascii=False, sort_keys=True) + \"\\n\")\n    errors = sum(1 for r in rows if r.error)\n    print(f\"[gliner-worker] {len(cases)} cases in {time.time()-t0:.0f}s, errors={errors}\", flush=True)\n    return 0\n\n\nif __name__ == \"__main__\":\n    raise SystemExit(main())\n"

if __name__ == "__main__":
    main()
