"""Create/update the HF Space Gjusev/MK-SH-1-demo from demo/ (canonical
source in this repo; the Space holds a copy — mechanism declared in both
READMEs). CPU basic hardware, Gradio SDK.

Usage: python scripts/push_space.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]
SPACE = "Gjusev/MK-SH-1-demo"
SRC = ROOT / "demo"


def main() -> int:
    api = HfApi()
    try:
        api.repo_info(repo_id=SPACE, repo_type="space")
        print("space exists")
    except Exception:
        api.create_repo(repo_id=SPACE, repo_type="space", space_sdk="gradio",
                        space_hardware="cpu-basic", private=False)
        print("space created (gradio, cpu-basic)")
    api.upload_folder(folder_path=str(SRC), repo_id=SPACE, repo_type="space",
                      commit_message="MK-SH-1 demo: calibrated probabilities + choice + "
                                     "abstention (3 examples: normal / numeric OOD / "
                                     "abstains). Loads model+policy from the hub package.")
    print(f"pushed {SRC} -> https://huggingface.co/spaces/{SPACE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
