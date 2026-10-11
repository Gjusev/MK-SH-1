# Model card — mokka-decisions OptionScorer v0.1.0

Release: `release/mokka-decisions-v0.1.0` (self-contained; offline-verified in a
fresh process with an empty HF cache and `HF_HUB_OFFLINE=1`).

## Model

- **Architecture**: option scorer over a multilingual encoder. Every option of
  a decision is encoded as the pair `(state, question + " Option: " +
  description)` with **shared** encoder and head weights; a scalar score per
  option; softmax restricted to the valid options. No per-option parameters,
  no label-id embeddings — the model scores option sets it never saw.
- **Encoder (inherited)**: `jhu-clsp/mmBERT-base` (307M params, ModernBERT),
  pretrained by JHU CLSP; **fine-tuned by this project** jointly with the new
  head (1,575 steps, 3 epochs, best epoch by dev).
- **Trained by this project**: scoring head, objective, candidate-set data
  construction, temperature calibration (T = 2.349 on 4,088 held-out cases),
  abstention policy (threshold 0.300 on 4,280 held-out cases).
- **Not claimed**: pretraining from scratch; general decision ability beyond
  the evaluated scenarios; typed decisions (see benchmark note below).

## Training data

BANKING77 (EN) + MASSIVE 1.0 (EN/DE/ES) + CLINC150/OOS (EN) as candidate-set
decisions with a shared `none` option; official test pools untouched; splits
by semantic group before expansion (leakage guards active; 7 exact train/test
duplicates in BANKING77 excluded from the train side). Full provenance:
`docs/dataset-card.md`, manifest hashes in the release.

Recipe: `configs/train.yaml` (16,784 instances, eff. batch 32, LR 2e-5/1e-4,
fp16, seed 42); run report and per-epoch history in `runs/**/run_report.json`.

## Measured results (official test, full budgets, metrics v2)

Candidate sets contain the gold option **by construction** (selection, not
recall). Fixed threshold policy fitted on calibration data only.

| Scenario | n | Forced acc | Macro-F1 | NLL | Brier(multiclass) | ECE | Cov@thr | Risk@thr |
|---|---|---|---|---|---|---|---|---|
| banking candidates (K=6) | 1,000 | **0.986** | 0.985 | 0.049 | 0.023 | 0.010 | 1.000 | 0.014 |
| massive candidates (en/de/es) | 1,500 | **0.969** | 0.937 | 0.095 | 0.049 | 0.011 | 0.999 | 0.031 |
| clinc candidates (K=6) | 800 | **0.978** | 0.975 | 0.067 | 0.034 | 0.005 | 1.000 | 0.023 |
| banking full 77-way | 1,000 | 0.812 | 0.808 | 0.633 | 0.271 | 0.028 | 0.972 | 0.174 |
| clinc OOS (gold none) | 400 | 0.933 | — | 0.252 | 0.116 | 0.036 | 1.000 | 0.068 |
| cross-domain probe (de/es) | 400 | 0.785 | — | 0.861 | 0.389 | 0.150 | 0.968 | 0.215 |

Per language (massive candidates): de 0.967 / en 0.968 / es 0.971 — no weak
language hidden in the average. Option-order permutation: **0 flips / 500**.
Latency (T4, batch=1, K=6): p50 22–26 ms; 78-option decisions: p50 318 ms.

Comparators on identical IDs/options (full table in the evaluation protocol;
rows and per-backend policies in `runs/final-report/`):

- **TF-IDF+LR closed-world**: banking candidates 0.984, full-77 0.874, clinc
  0.989 — wins fixed-taxonomy scenarios; cannot choose `none` (modelling
  choice, not an inherent limit).
- **TF-IDF+LR with trainable reject** (`tfidf_lr_none`): clinc OOS 0.890 and
  cross-domain xprobe **0.963 at 3.8 % selective risk — beats us** (0.785 /
  21.5 %): a linear classifier with a trained reject generalises
  out-of-domain better than this model. Measured negative result, stated
  plainly.
- **Laya zero-shot**: strongest on clinc OOS (0.985 at 82 % coverage) but
  collapses on banking/massive candidates (0.725/0.545) and its confidence
  needs a 0.995 threshold for 17 % risk at 51 % coverage.
- **GLiNER2.5**: `not_run` — two isolated-environment attempts failed on the
  Kaggle image (venv ensurepip / transformers<5 conflicts). Errors recorded;
  never presented as evaluated.

Where this model wins: candidate-set decisions in all three languages with
the best calibration (ECE 0.005–0.011) and the lowest neural latency
(p50 22–26 ms on T4 for K=6), plus exact option-order invariance.

## Typed decisions (Julia-1 benchmark) — honest labelling

- Julia-1 reproduced locally with its official script (isolated venv, CPU):
  **0.7255** (1,451/2,000) vs published 0.7315.
- Our v0.1 zero-shot under the frozen adapter protocol: see
  `docs/typed-decisions-protocol.md` and `runs/typed-decisions/` — a
  **different adaptation budget** (never trained on this format), reported as
  `zero_shot_domain_shift`, never as a like-for-like defeat or victory.
- Adapted same-data comparisons were run in the E24–E34 campaign — see the
  numeric-order section below for the full three-context Julia tables.

## Numeric-order capability (E24–E34 campaign, 2026-10-10/11)

A second lineage (JuliaArch-small: mmBERT-small + marker head, ~140M) was
taught to compare a state value against a question threshold (`value ≥ t`)
without shortcuts, then integrated with original-task replay. Campaign
record: `docs/experiments.md`; release notes:
`docs/RELEASE_NOTES_E24-E29b.md`.

**Validated version** = candidate E28-s43 (sha256 `75ea9eed…`, dev-patience
trained on the 944-row dense grid) + 15 epochs of 944 original-task rows +
944 grid replay (fresh optimizer per 5-epoch phase). Independent validation
on a disjoint sample (E30) met the pre-registered F1″ criterion.

| Metric (validated point) | Value | Reference |
|---|---|---|
| testC — order, known thresholds (n=48) | **0.958–0.979** | E29b/E30 |
| testD — interpolation (n=54) | 0.926 (intact through integration) | E29 |
| testE-interp — fresh 2-decimal OOD (n=116) | **0.922–0.931**, 8/8 pure pairs | E29b/E30 |
| testE-extrap — fresh 2-decimal OOD (n=56) | **0.750–0.786**, 2–4/8 pairs — **the weak number, declared** | E29b |
| Monotonicity VE2 (10 sweeps) | 9/10 | E29b/E30 |
| Original task (routing, 300-row slice) | **0.690–0.733** (from 0.157 specialized) | E29/E30 |

**Julia-1 context (all numbers from the single E32/E34 CPU-float32 harness,
system comparisons — recipes differ, no architecture claims):**

| Context | testC | E-interp | E-extrap | VE2 | Own task |
|---|---|---|---|---|---|
| Julia-1 zero-shot (official weights) | 0.563 | 0.552 | 0.500 | 0/10 | typed-2000 **0.7255** |
| Julia-1 adapted, our grid (E32) | **1.000** | 0.961 | **0.911** | 7/10 | typed-2000 −13.6/−19.0 pp |
| Julia-1 + replay (E34) | **1.000** | 0.940/0.983 | — | — | typed-2000 **−27.1/−23.5 pp**; proxy-task 0.83 |
| Mokka integrated (E30-MIX) | 0.979 | 0.922 | 0.750 | **9/10** | routing **0.733** (recovered) |

Declared reading: adapted Julia transfers numeric order better at the
edges; the integrated Mokka version keeps monotonicity and its task. Replay
gives Julia a proxy task + order simultaneously but degrades its own task
*faster than no replay*. The trade-off refines by which task is at stake;
neither system dominates.

## Calibration & abstention

## Calibration & abstention

Temperature on `cal_temperature` only; threshold on `cal_policy` only. The
5 % selective-risk target is met on calibration data (2.37 %) and holds on
candidate-set test scenarios (1.4–3.1 %); it does **not** hold on the
78-option scenario (17.4 %) or the cross-domain probe (21.5 %) — those are
outside the declared scope. Choosing `none` is a decision; abstaining returns
`choice=null` + reason code.

## Known limitations

- Option descriptions are English across languages (schema-described-once
  scenario); cross-lingual transfer measured, not assumed.
- CLINC option texts are humanized intent names, not authored glosses.
- Cross-domain OOS false acceptance 20.8 % (6.5 % of probe labels audited as
  ambiguous by exact-name collision).
- 77-way full-taxonomy decisions degrade vs fixed classifiers trained on the
  same labels (0.812 vs 0.872).
- Numeric-order capability: E-extrap after integration is the declared weak
  number (0.750, 2/8 pure pairs); monotonicity 9/10 not 10/10; the Julia
  comparisons are **system** comparisons (recipes differ) — no architecture
  claim is made anywhere in this campaign.
- The numeric-order lineage (JuliaArch-small) is not in the v0.1 release
  package; it lives as campaign checkpoints under `runs/` (gitignored).

## Verdict

**`qualified_for_declared_scope`** — declared scope: candidate-set decisions
(2–12 described options, gold present) on banking and assistant intent
domains, EN/DE/ES, with calibrated probabilities and permutation-invariant
scoring; `none`-selection works on in-distribution OOS (93.3 %) but
out-of-domain abstention is **not** within the validated risk budget
(`experimental` for xprobe/77-way scenarios).

## License

Apache-2.0 (code and trained weights of this release). Datasets and the
inherited encoder keep their own licenses (`THIRD_PARTY_NOTICES.md`).
