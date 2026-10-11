"""Assemble the private Kaggle bundle: wheel + configs + prepared data.

Two bundle shapes:

* default: the full job bundle (dataset `gjusev/mokka-decisions-bundle`);
* --checkpoint PATH: a checkpoint-transport bundle (dataset
  `gjusev/mokka-decisions-checkpoint`) carrying weights + optimizer + RNG
  states + metadata, for resume and eval jobs.

Every bundle gets BUNDLE_MANIFEST.json with sha256 of every file.
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


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_wheel(out_dir: Path) -> Path:
    import os

    out_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PIP_CONFIG_FILE="/dev/null")
    subprocess.check_call(
        [sys.executable, "-m", "pip", "wheel", "--no-deps", "-w", str(out_dir), str(ROOT)],
        env=env,
    )
    wheels = sorted(out_dir.glob("mokka_decisions-*.whl"))
    if not wheels:
        raise SystemExit("wheel build produced nothing")
    return wheels[-1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--data", default="data/processed/v1")
    parser.add_argument("--bundle", default="kaggle-bundle")
    parser.add_argument("--checkpoint", default=None, help="build a checkpoint-transport bundle instead")
    parser.add_argument("--mm-dir", default=None, help="include a multimodal dataset dir as mm/")
    parser.add_argument("--typed-dir", default=None, help="include typed-decisions train dir as typed-v1/")
    parser.add_argument("--e19-dir", default=None, help="include e19-rulevar data dir as e19-rulevar/")
    parser.add_argument("--e20-dir", default=None, help="include e20-contrast data dir as e20-contrast/")
    parser.add_argument("--e23-dir", default=None, help="include e23-numeric data dir as e23-numeric/")
    parser.add_argument("--e24-dir", default=None, help="include e24-fit data dir as e24-fit/")
    parser.add_argument("--e25-dir", default=None, help="include e25-thrvar data dir as e25-thrvar/")
    parser.add_argument("--e26-dir", default=None, help="include e26-order data dir as e26-order/")
    parser.add_argument("--e27-dir", default=None, help="include e27-grid data dir as e27-grid/")
    parser.add_argument("--e28-dir", default=None, help="include e28-ext data dir as e28-ext/")
    parser.add_argument("--e29-mix-dir", default=None, help="include e29-mix manifest dir as e29-mix/")
    parser.add_argument("--e30-mix-dir", default=None, help="include e30-mix manifest dir as e30-mix/")
    parser.add_argument("--e35-dir", default=None, help="include e35-extremes data dir as e35-extremes/")
    parser.add_argument("--julia-src", default=None, help="include Julia-1 upstream source (weights+engine) as julia-src/")
    parser.add_argument("--e29-ckpt", default=None, help="include the E28 s43 finalist as e29-mix/e29-ckpts/e28_s43_final.safetensors")
    parser.add_argument("--e26-ckpt-dir", default=None, help="include E26 finalist checkpoints as e26-ckpts/ (s42/s43 treat finals)")
    args = parser.parse_args()

    bundle = Path(args.bundle)
    if bundle.exists():
        shutil.rmtree(bundle)

    if args.checkpoint:
        ckpt = Path(args.checkpoint)
        bundle.mkdir(parents=True)
        shutil.copy2(ckpt, bundle / ckpt.name)
        for sidecar in (ckpt.with_suffix(".json"),):
            if sidecar.exists():
                shutil.copy2(sidecar, bundle / sidecar.name)
        (bundle / "dataset-metadata.json").write_text(
            json.dumps(
                {
                    "title": "mokka-decisions-checkpoint",
                    "id": "gjusev/mokka-decisions-checkpoint",
                    "licenses": [{"name": "CC-BY-4.0"}],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    else:
        wheel = build_wheel(bundle / "dist")
        print(f"wheel: {wheel.name}")
        for cfg in sorted((ROOT / "configs").glob("*.yaml")):
            shutil.copy2(cfg, bundle / cfg.name)
        data = Path(args.data)
        if not (data / "instances").exists():
            raise SystemExit(f"no prepared data under {data}; run scripts/prepare_data.py first")
        shutil.copytree(data, bundle / "data" / "processed" / "v1")
        if args.mm_dir:
            mm = Path(args.mm_dir)
            if not (mm / "manifest.json").exists():
                raise SystemExit(f"no mm manifest under {mm}")
            shutil.copytree(mm, bundle / "mm")
        if args.typed_dir:
            td = Path(args.typed_dir)
            if not (td / "manifest.json").exists():
                raise SystemExit(f"no typed manifest under {td}")
            shutil.copytree(td, bundle / "typed-v1")
        if args.e19_dir:
            ed = Path(args.e19_dir)
            if not (ed / "manifest.json").exists():
                raise SystemExit(f"no e19 manifest under {ed}")
            shutil.copytree(ed, bundle / "e19-rulevar")
        if args.e20_dir:
            ed = Path(args.e20_dir)
            if not (ed / "manifest.json").exists():
                raise SystemExit(f"no e20 manifest under {ed}")
            shutil.copytree(ed, bundle / "e20-contrast")
        if args.e23_dir:
            ed = Path(args.e23_dir)
            if not (ed / "manifest.json").exists():
                raise SystemExit(f"no e23 manifest under {ed}")
            shutil.copytree(ed, bundle / "e23-numeric")
        if args.e24_dir:
            ed = Path(args.e24_dir)
            if not (ed / "manifest.json").exists():
                raise SystemExit(f"no e24 manifest under {ed}")
            shutil.copytree(ed, bundle / "e24-fit")
        if args.e25_dir:
            ed = Path(args.e25_dir)
            if not (ed / "manifest.json").exists():
                raise SystemExit(f"no e25 manifest under {ed}")
            shutil.copytree(ed, bundle / "e25-thrvar")
        if args.e26_dir:
            ed = Path(args.e26_dir)
            if not (ed / "manifest.json").exists():
                raise SystemExit(f"no e26 manifest under {ed}")
            shutil.copytree(ed, bundle / "e26-order")
        if args.e27_dir:
            ed = Path(args.e27_dir)
            if not (ed / "manifest.json").exists():
                raise SystemExit(f"no e27 manifest under {ed}")
            shutil.copytree(ed, bundle / "e27-grid")
        if args.e28_dir:
            ed = Path(args.e28_dir)
            if not (ed / "manifest.json").exists():
                raise SystemExit(f"no e28 manifest under {ed}")
            shutil.copytree(ed, bundle / "e28-ext")
        if args.e29_mix_dir:
            ed = Path(args.e29_mix_dir)
            if not (ed / "manifest.json").exists():
                raise SystemExit(f"no e29-mix manifest under {ed}")
            shutil.copytree(ed, bundle / "e29-mix")
        if args.e30_mix_dir:
            ed = Path(args.e30_mix_dir)
            if not (ed / "manifest.json").exists():
                raise SystemExit(f"no e30-mix manifest under {ed}")
            shutil.copytree(ed, bundle / "e30-mix")

        if args.julia_src:
            import hashlib as _hl
            src = Path(args.julia_src)
            dst = bundle / "julia-src"
            shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__", "holdout-cache", "*.parquet"))
            got = _hl.sha256((dst / "model.safetensors").read_bytes()).hexdigest()
            want = "df853bf7fe424420011f3d0c47a05d7341aa9eefa7fb9f203ea4aada4ad95b72"
            if got != want:
                raise SystemExit(f"julia weights hash mismatch: {got}")
        if args.e35_dir:
            ed = Path(args.e35_dir)
            if not (ed / "manifest.json").exists():
                raise SystemExit(f"no e35 manifest under {ed}")
            shutil.copytree(ed, bundle / "e35-extremes")
        if args.e29_ckpt:
            import hashlib as _hl
            src = Path(args.e29_ckpt)
            dst = bundle / "e29-mix" / "e29-ckpts"
            dst.mkdir(parents=True, exist_ok=True)
            got = _hl.sha256(src.read_bytes()).hexdigest()
            want = "75ea9eed7aac43a1ac1dc7fdeca40329af3335696861db3aae92aec844614829"
            if got != want:
                raise SystemExit(f"e29 candidate checkpoint hash mismatch: {got}")
            shutil.copy2(src, dst / "e28_s43_final.safetensors")
        if args.e26_ckpt_dir:
            import hashlib as _hl
            import json as _js
            src = Path(args.e26_ckpt_dir)
            dst = bundle / "e26-ckpts"
            dst.mkdir(parents=True)
            manifest = {}
            for seed in (42, 43):
                f = src / f"s{seed}_treat" / "final.safetensors"
                if not f.exists():
                    raise SystemExit(f"missing finalist checkpoint: {f}")
                shutil.copy2(f, dst / f"s{seed}_treat_final.safetensors")
                manifest[f"s{seed}_treat_final.safetensors"] = {
                    "sha256": _hl.sha256(f.read_bytes()).hexdigest(), "bytes": f.stat().st_size}
            (dst / "ckpt_manifest.json").write_text(_js.dumps(manifest, indent=2), encoding="utf-8")
        (bundle / "dataset-metadata.json").write_text(
            json.dumps(
                {
                    "title": "mokka-decisions-bundle",
                    "id": "gjusev/mokka-decisions-bundle",
                    "licenses": [{"name": "CC-BY-4.0"}],
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    manifest = {
        "files": {
            str(p.relative_to(bundle)): {"sha256": sha256(p), "bytes": p.stat().st_size}
            for p in sorted(bundle.rglob("*"))
            if p.is_file() and p.name not in ("BUNDLE_MANIFEST.json",)
        },
        "cli": sys.argv,
    }
    (bundle / "BUNDLE_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    total = sum(p.stat().st_size for p in bundle.rglob("*") if p.is_file())
    print(f"bundle: {bundle} ({total / 2**20:.1f} MiB, {len(manifest['files'])} files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
