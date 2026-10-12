"""Upload MK-SH-1 v1.2 to the hub (Gjusev/MK-SH-1): v1.2 card (README.md) +
inference-policy.json. Weights/tokenizer/ARCHITECTURE are already on the hub
unchanged from v1.1 — NOT re-uploaded (576 MB saved); the post-upload
verification re-downloads the full package and proves coherence.

Usage: python scripts/upload_mksh_v12.py
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "runs/hf-export/mksh-1-v1.2"
REPO = "Gjusev/MK-SH-1"
WEIGHTS_SHA = "ea66cdde2271b98efda9dbb5d4437b59aa422cb34daca85d4203508aa56264d5"


def main() -> int:
    policy = json.loads((SRC / "inference-policy.json").read_text(encoding="utf-8"))
    if policy["model"]["weights_sha256"] != WEIGHTS_SHA:
        raise SystemExit("policy weights sha mismatch — abort")
    api = HfApi()
    # pre-flight: what is on the hub right now
    files = {f.rfilename: getattr(f, "size", None) for f in api.list_repo_tree(REPO, recursive=False)}
    print("hub before:", sorted(files))
    if "model.safetensors" not in files:
        raise SystemExit("hub package lacks model.safetensors — abort")

    api.upload_file(path_or_fileobj=str(SRC / "README.md"), path_in_repo="README.md",
                    repo_id=REPO, repo_type="model",
                    commit_message="v1.2: calibration (T=7.653) + abstention policy "
                                   "(tau=0.79, 90% coverage, <=5% selective risk) on "
                                   "v1.1 weights unchanged — card updated, limitation removed")
    api.upload_file(path_or_fileobj=str(SRC / "inference-policy.json"),
                    path_in_repo="inference-policy.json", repo_id=REPO, repo_type="model",
                    commit_message="v1.2: embed inference-policy.json (temperature + "
                                   "abstention + response semantics + provenance)")
    print("uploaded: README.md, inference-policy.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
