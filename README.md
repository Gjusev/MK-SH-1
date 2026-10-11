# mokka-decisions

A small open **decision model**: given a state, a question and a list of
described options, it returns calibrated probabilities, a choice and — when
confidence is insufficient — an explicit abstention. This repository carries
the full training/evaluation system plus an eleven-experiment research
campaign (E24–E34, 2026-10-10/11) that taught it a capability it did not have
and audited every step of that claim.

**The surviving differentiator** (E32/E34, same CPU-float32 harness): the
integrated MOKKA version holds a newly-acquired capability (numeric order)
and its original task **simultaneously** — the reference system (Julia-1,
adapted under a mirrored recipe) recovers a proxy task alongside order but
loses its own original task *faster with replay than without* (−27/−24 pp
vs −14/−19 pp). Every advantage sentence below points at the experiment and
commit that measures it; the caveats travel with the claims.

```
state:    "Me cobraron dos veces el mismo pedido."
question: "Which banking support intent does this customer request express?"
options:  [ transaction_charged_twice — "The same transaction was charged twice"
            top_up_reverted — "A top-up was reversed or reverted" ]
→  probabilities over the options (temperature-calibrated)
   choice (or null + reason_code when the policy abstains)
```

It is **not** a chat model and does not generate text for the decision: one
encoder pass per option, a shared scorer head, softmax over the valid options.

## What is ours, what is inherited

| Component | Origin |
|---|---|
| encoder (mmBERT-base/small) | pretrained by JHU CLSP (MIT); **fine-tuned by this project** |
| option-scoring head / marker head, loss, masking | **this project** |
| decision data contract, candidate-set construction, splits | **this project** |
| temperature calibration + abstention policy | **this project** |
| training/eval harness, Kaggle free-tier recipe | **this project** |
| baselines (TF-IDF+LR, Laya, GLiNER2.5); Julia-1 comparator | third-party models, our adapters |

## The campaign E24 → E34 (one line each)

Full entries with all caveats: `docs/experiments.md`. All criteria were
frozen and committed **before** each launch; every closure was adversarially
reviewed by an independent agent/session and every correction integrated.

| # | Question | Result | Key commits |
|---|---|---|---|
| E24 | Can the model *fit* a 48-problem numeric microset? | Yes, 2/2 seeds in <60 steps; generalizes to new values and renamed field; does NOT read a varied threshold | `f65bfb9f` `ca4644f4` |
| E25 | Does varying thresholds in training teach reading them? | Yes — testC 1.000 both seeds, 10/10 threshold-only pairs (later qualified: a decade shortcut fit the same data) | `ab8061ee` `99e8d4b3` |
| E26 | Does breaking the decade shortcut yield real order? | No — no arm passes; decade-heuristic signature dominates | `e175b757` `7e4fbe6b` |
| E26b/c | Stopping rule? Seed variance? | Duration dead (continuation overfits); 6-seed sub-0.90 ceiling replicated | `b4201d5e` `a9b2983d` |
| E27 | Does dense per-value coverage break the ceiling? | **Yes** — testC 0.979/1.000, order transfers to fresh 2-decimal OOD (interp); extrap + monotonicity declared weak | `aabab22a` `a9b2983d` |
| E28 | Extremes + monotone chains close the residuals? | Massive moves (extrap +18/+20 pp, mono 7·9/10) but not in both seeds; interference reading fires, stop-and-rethink honored | `0a9ac914` `9e3be0bc` |
| E28b | Seed check of the edge trade | testC 1.000 both new seeds; jitter at bars, not a systematic trade | `f3e05cae` |
| E29/E29b | Forgetting: does replay protect the capability? | No catastrophe either direction; with replay the interference drops to ~1/4–1/3 of no-replay and monotonicity survives; task 0.157 → 0.690/0.713 | `37d4fd01` `db9baa24` |
| E30 | Independent validation (disjoint sample) | F1″ passes: testC 0.979, E-interp 0.922 (8/8 pairs), task 0.733 — integration not sample-specific | `d5347639` `463aac20` |
| E31 | Julia-1 on our grid — same-data comparison | **Closure invalidated** (evaluator binding bug), see below | `d72b3066` `752b8f3f` |
| E32 | Honest head-to-head, one harness | A-MIXTO: adapted Julia leads numeric transfer at testC=1 row; Mokka wins monotonicity; Julia's own task −13.6/−19.0 pp, Mokka's task +57 pp | `8d757b00` `efdca880` |
| E33 | Valid re-run of E31 with the fixed pipeline | Pipeline confirmed end-to-end (live dev curves); corrective numbers replicated; thr-pairs and VE2 fail the registered bars | `d6b34eae` `874785c3` |
| E34 | Can Julia keep task AND order with replay? | Proxy-task + order: yes (0.83/0.83, testC 1.000, E-interp 0.94/0.98) — its **real** task degrades −27/−24 pp, worse than without replay | `f381940f` `ee583a76` |

## Rigor / self-correction (the arc that defines this repo)

The E31 story is told in full because it is the point, not an embarrassment:

1. **The bug.** The E31 stage fine-tuned Julia-1 on our 944-row grid, but its
   evaluator closed over a stage-scope engine holding the *original* weights —
   dev-based stopping and all test evals measured the wrong model
   (`invalid_model_binding`). The closure claimed "Julia memorizes without
   order; our architecture+recipe is the differentiator."
2. **The detection and public retraction.** Supervisor review found the
   binding defect; the claim was **withdrawn in public commits**
   (`752b8f3f`, `1c2393c1`), the corrective re-evaluation showed Julia-1
   *does* acquire order on our grid, and the frozen artifacts were left
   untouched with an invalidation notice.
3. **The regression that prevents recurrence.** `tests/test_e31_model_binding.py`
   now fails on the pre-fix source: it requires the eval engine to be passed
   explicitly (`evaluate(engine, rows)`), asserts the inference-link guards
   (`engine.model is model`, `engine.forward is engine.model`), and demands an
   **at-eval-time in-memory weights hash** — the identity of the weights that
   actually produced each number.
4. **The valid repetition.** E33 re-ran the protocol with the corrected
   pipeline under pre-registered criteria plus a new V1 "live dev curve"
   validity check, declaring the contamination honestly (results already
   known from the corrective — a pipeline validation, not a discovery).
5. **The honest comparisons.** E32 (one CPU-float32 harness for both systems,
   double-validated against frozen history) and E34 (replay portability) close
   the comparison with every caveat visible: recipes differ, so these are
   **system comparisons, and no architecture claim is possible**.

Criteria documents (frozen before each launch, commit order verifiable):
`docs/E32_CRITERIA…`, `docs/E33_CRITERIA…`, `docs/E34_CRITERIA…`, and the
internal protocol family referenced from `docs/experiments.md`.

## Final comparison table (E32 axis A + E34, one harness, CPU float32)

Recipes are NOT identical (uniform 2e-5 vs grouped LR, 30 vs ~300 epochs,
full-checkpoint vs encoder+head init): these are **complete-system**
comparisons — no architecture claims.

| Metric (final) | Mokka integrated (E30-MIX) | Julia adapted (E32, mean s42/s43) | Julia + replay (E34, s42/s43) |
|---|---|---|---|
| testC (order, known) | 0.979 | **1.000** (1 row ahead) | **1.000 / 1.000** |
| testE-interp (OOD) | 0.922 | 0.961 | 0.940 / 0.983 |
| testE-extrap (OOD) | 0.750 | **0.911** | — |
| Monotonicity (VE2) | **9/10** | 7/10 | — |
| Threshold-only pairs (22) | 13 | **16** | — |
| Its original task | routing 0.733 (recovered from 0.16) | typed-2000 −13.6/−19.0 pp | typed-2000 **−27.1/−23.5 pp**; proxy-task 0.83 |

Read: adapted Julia transfers the numeric capability better at the edges;
the integrated Mokka version keeps monotonicity and recovers its task.
With replay, Julia holds a *proxy* task and order at once, but its *own*
task degrades faster than without replay. The E32 trade-off does not close;
it refines by which task is at stake.

## What survives / what doesn't

**Survives (measured, replicated):**
- Dense per-value coverage with 4-Yes/4-No balance teaches real numeric order
  (shortcuts mathematically excluded by construction; E27, replicated E28b).
- The integration recipe: capability + task coexist under replay mixing,
  validated on a disjoint sample (E29/E29b/E30).
- No catastrophic forgetting in either direction at these budgets (E29).
- The E31→E33 correction arc and its permanent regression.

**Does not survive (declared limits):**
- testE-extrap after integration is the weak number of the family
  (0.75, 2/8 pure pairs at the validated point).
- Full monotonicity on adapted Julia (7/10) vs Mokka (9/10) — neither is 10/10.
- Coexistence of order with *Julia's own* typed distribution (E34).
- Any claim that the **architecture** is the differentiator — recipes differ.
- OOD/threshold-transfer claims from E25 (decade-shortcut qualified) —
  superseded by E26's review.

## Reproduce

```bash
pip install -e .[data,train,baselines]
python -m pytest -q                    # suite: 195 pass (default env)
# .venv-julia (transformers==5.0.0): 194 pass + 1 skip
python scripts/prepare_data.py --config configs/pilot.yaml --out data/processed/v1   # deterministic
python scripts/gen_e25_thrvar.py && python scripts/verify_e25_thrvar.py              # data + independent verifier
python scripts/e27_report.py --out /tmp/r.json                                       # deterministic analyzers
```

Campaign artifacts live under `runs/` (gitignored; every analysis directory
carries `artifact_hashes.json`; reports regenerate byte-exact). Training ran
on Kaggle free tier — see `docs/runbook-kaggle.md` for the quota discipline
(reserve ≥8 h, single active kernel, in-kernel hash guards).

**Status:** campaign E24–E34 closed and published (this README is part of the
closure). GPU quota refreshes 2026-10-17. Version notes:
`docs/RELEASE_NOTES_E24-E29b.md`; internal status/decision logs stay out of
Git by standing instruction.

## Documentation

- `docs/experiments.md` — the full campaign record (one entry per experiment)
- `docs/model-card.md` — trained artifacts, final metrics, limits
- `docs/dataset-card.md`, `docs/evaluation-protocol.md`, `docs/runbook-kaggle.md`
- `docs/typed-decisions-protocol.md` — the typed-decisions contract

## License

Apache-2.0 for code. Datasets and third-party models keep their own licenses
(`THIRD_PARTY_NOTICES.md`).
