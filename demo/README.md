---
title: MK-SH-1 demo
emoji: 🌡️
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 5.50.0
pinned: false
license: apache-2.0
---

# MK-SH-1 demo — calibrated numeric decisions with abstention

Interactive demo of **[Gjusev/MK-SH-1](https://huggingface.co/Gjusev/MK-SH-1)**
(v1.2): a free-form state + a numeric threshold → **calibrated probabilities**,
a **choice**, and — when confidence is below the validated threshold — an
explicit **abstention with reason_code**.

**Canonical source:** [`demo/` in github.com/Gjusev/mokka-decisions](https://github.com/Gjusev/mokka-decisions/tree/main/demo)
(this Space holds a copy, pushed by `scripts/push_space.py` — the repo is the
source of truth). Training/evaluation/calibration record and write-up
material: [`docs/experiments.md`](https://github.com/Gjusev/mokka-decisions/blob/main/docs/experiments.md),
[`docs/model-card.md`](https://github.com/Gjusev/mokka-decisions/blob/main/docs/model-card.md).

Three preloaded examples: a normal confident case, a numeric OOD case at the
edge of the range, and a borderline case where the policy abstains
(`below_validated_threshold`).
