"""E29 mix tests: frozen sample rule, manifest integrity, candidate hash."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data/e29-mix/manifest.json"


def test_manifest_rule_reproducible():
    m = json.loads(MANIFEST.read_text(encoding="utf-8"))

    def first_n(path, n):
        rows = [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]
        ranked = sorted(rows, key=lambda r: hashlib.sha256(r["id"].encode()).hexdigest())
        return [r["id"] for r in ranked[:n]]

    assert m["original_train_ids"] == first_n(ROOT / "data/processed/v1/instances/train.jsonl", 944)
    assert m["original_dev_ids"] == first_n(ROOT / "data/processed/v1/eval/dev_eval.jsonl", 300)
    assert len(set(m["original_train_ids"])) == 944
    assert len(set(m["original_dev_ids"])) == 300
    assert not set(m["original_train_ids"]) & set(m["original_dev_ids"])


def test_source_hashes_pinned():
    m = json.loads(MANIFEST.read_text(encoding="utf-8"))
    for rel, sha in m["sources"].items():
        got = hashlib.sha256(_resolve(rel).read_bytes()).hexdigest()
        assert got == sha, rel


def _resolve(rel):
    mapping = {"instances/train.jsonl": "data/processed/v1/instances/train.jsonl",
               "eval/dev_eval.jsonl": "data/processed/v1/eval/dev_eval.jsonl",
               "e28-ext/train_rows.jsonl": "data/e28-ext/train_rows.jsonl",
               "e26-order/dev_rows.jsonl": "data/e26-order/dev_rows.jsonl"}
    return ROOT / mapping[rel]


def test_candidate_hash_literal():
    m = json.loads(MANIFEST.read_text(encoding="utf-8"))
    got = hashlib.sha256((ROOT / "runs/kaggle-typed-e28/s43/final.safetensors").read_bytes()).hexdigest()
    assert got == m["candidate_ckpt_sha256"] == \
        "75ea9eed7aac43a1ac1dc7fdeca40329af3335696861db3aae92aec844614829"


def main():  # pragma: no cover
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")


if __name__ == "__main__":
    raise SystemExit(main())
