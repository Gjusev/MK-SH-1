<table><tr>
<td><img src="assets/logo-mark.png" width="130" alt="MK-SH-1 mark: a whale shark in negative space between two tide bands"></td>
<td><h1>MK-SH-1</h1><p><sub>calibrated decisions with an explicit abstention gate</sub></p></td>
</tr></table>

A small open **decision model**: given a state, a question and a list of
described options, it returns calibrated probabilities, a choice, and when
confidence is too low, an explicit abstention. The repository carries the
training and evaluation system behind it, including a documented research
campaign that taught the model numeric order and audited the resulting claim.
Calibration and abstention were added without retraining.

Public model: **[Gjusev/MK-SH-1](https://huggingface.co/Gjusev/MK-SH-1)**.
v1.2 ships calibrated probabilities (T = 7.653) and a validated abstention
policy (tau = 0.79, 90 % coverage at a 5 % selective-risk budget) on weights
byte-identical to v1.1. Interactive demo:
**[Gjusev/MK-SH-1-demo](https://huggingface.co/spaces/Gjusev/MK-SH-1-demo)**
(source: [`demo/`](demo/)).

```
state:    "Me cobraron dos veces el mismo pedido."
question: "Which banking support intent does this customer request express?"
options:  [ transaction_charged_twice: "The same transaction was charged twice"
            top_up_reverted: "A top-up was reversed or reverted" ]
→  probabilities over the options (temperature-calibrated)
   choice (or null + reason_code when the policy abstains)
```

It is **not** a chat model and does not generate text for the decision. Each
option is scored in one encoder pass; softmax runs over the valid options only.

## The decision contract

One encoder, shared weights, a scalar score per option. The temperature is a
monotone transform: it never changes a choice, only the confidences. The
abstention rule was frozen before the coverage curve was ever plotted (E36).

```mermaid
flowchart LR
    A["state + question + options"] --> B["encoder (mmBERT-small)"]
    B --> C["marker scorer, one MASK per option"]
    C --> D["option scores"]
    D --> E["softmax(logits / T)"]
    E --> F{"max probability >= tau"}
    F -- yes --> G["choice + calibrated probabilities"]
    F -- no --> H["choice = null<br/>reason_code = below_validated_threshold"]
```

## Use cases

<p align="center">
  <a href="https://github.com/Gjusev/MK-SH-1/releases/download/film-1/mksh-presentation.mp4">
    <img src="assets/readme-film-poster.jpg" alt="MK-SH-1 presentation film — watch the 18-second cut" width="100%">
  </a>
</p>

<p align="center"><sub>&#9654; the 18-second presentation cut &mdash; silent, light theme, the decision contract at work. Tap the frame to play.</sub></p>

<p align="center"><sub>Illustrative scores through the shipped formula
(softmax at the fitted temperature, gate at tau): question in, calibrated
probability out, refusal below the gate. Try the
[interactive demo](https://huggingface.co/spaces/Gjusev/MK-SH-1-demo) to run
the checkpoint yourself.</sub></p>

Each case names the measured capability it rests on, and where that
measurement stops.

1. **Threshold routing.** Decisions of the form "is the value at least the
   threshold stated in the question": limits, eligibility checks, price or
   risk cutoffs. Rests on numeric order measured at testC 0.979 on known
   thresholds and E-extrap 0.875 on fresh extreme thresholds (E27, E30, E35).
   Bound: validated on the numeric comparison contract in English
   (motor-temperature style states, one decimal to two); it is not a general
   numeric reasoner, and E-extrap is 7/8 pairs, not 10/10.
2. **Calibrated scoring with a safety valve.** Post-temperature probabilities
   (ECE 0.016 on the dedicated calibration split, E36) plus abstention at
   tau = 0.79: 90 % coverage with 4.86 % selective risk, which buys +2.85 pp
   accuracy on the cases the policy keeps. Suited to decision queues where
   refusals go to human review. Bound: the policy was fitted on the numeric
   surface; the routing task inherits the same global temperature (measured
   descriptively: NLL 2.535 to 0.588, ECE 0.184 to 0.045, no policy fit), and
   the easiest set is slightly over-softened (known-thresholds ECE 0.024 to
   0.042), the declared cost of one scalar.
3. **CPU and edge inference.** 106 ms per question at batch 1 on 8 CPU
   threads, fp32 (E22 v3, both models on the same harness). Bound: that is a
   tie with the Julia-1 reference at batch 1 and 5 to 10 % slower at batch 16;
   the earlier 1.35x speed claim was measured with a flawed v2 harness and is
   retracted below.
4. **A methodological template.** The recipe behind the capability: a dense
   anti-shortcut grid (4 Yes and 4 No per identical value, so any
   question-ignoring rule sits at exactly 0.500 on train), replay integration
   to keep the original task (E29, E30), replay-protected extremes (E35), then
   calibration on a split frozen before measurement (E36). A reproducible way
   to teach one new capability without catastrophic forgetting, demonstrated
   at 140M scale. Bound: one capability, one model family; the recipe is not
   claimed to generalize beyond what the campaign measured.

## Calibration and abstention (E36)

![Calibration before and after the temperature fit across the split and the frozen eval sets](assets/calibration.png)

The temperature was fitted once, on 480 fresh questions, and it is monotone:
it changes no choices, only confidences. The abstention gate at tau = 0.79
keeps 90 % coverage at 4.86 % selective risk; below the gate the model returns
choice = null with reason_code below_validated_threshold.

## Measured comparison

![MK-SH-1 against the field: latency, numeric order, own-task retention and
calibration against Julia-1, Laya and TF-IDF+LR; every bar cites its
experiment](assets/comparison.png)

These are complete-system comparisons, not architecture claims: the recipes
differ. Julia transfers numeric order better at range edges; MK-SH-1 keeps
better monotonicity and recovers its original routing task after integration.
With replay, Julia holds a proxy task plus order while its own original task
degrades faster than without replay. Neither system dominates.

| Measured outcome | MK-SH-1 integrated | Julia-1 adapted | Julia-1 + replay |
|---|---|---|---|
| Order, known thresholds | 0.979 | **1.000** | **1.000 / 1.000** |
| Fresh extreme thresholds | 0.750 | **0.911** | n/a |
| Monotonicity sweeps | **9/10** | 7/10 | n/a |
| Original task after integration | routing 0.733, recovered from 0.16 | typed task -13.6/-19.0 pp | typed task -27.1/-23.5 pp |
| CPU latency, batch 1 | 106 ms | 106 ms | not measured |

The full protocol, per-case artifacts, criteria, corrections and caveats are
in [`docs/experiments.md`](docs/experiments.md). The comparison is deliberately
kept here because it is the shortest honest answer to "what is different?"

## Rigor: the correction that defines this repository

The E31 story is told in full because it is the point, not an embarrassment:

1. **The bug.** The E31 stage fine-tuned Julia-1 on our 944-row grid, but its
   evaluator closed over a stage-scope engine holding the *original* weights.
   Dev-based stopping and all test evals measured the wrong model
   (`invalid_model_binding`). The closure claimed "Julia memorizes without
   order; our architecture and recipe are the differentiator."
2. **Detection and public retraction.** Supervisor review found the binding
   defect. The claim was **withdrawn in public commits** (`3c03322e`,
   `fcebac9f`), the corrective re-evaluation showed Julia-1 *does* acquire
   order on our grid, and the frozen artifacts were left untouched with an
   invalidation notice.
3. **The regression that prevents recurrence.** `tests/test_e31_model_binding.py`
   fails on the pre-fix source: the eval engine must be passed explicitly, the
   inference-link guards are asserted (`engine.model is model`), and every
   eval demands an at-eval-time in-memory weights hash, the identity of the
   weights that actually produced each number.
4. **The valid repetition.** E33 re-ran the protocol with the corrected
   pipeline under pre-registered criteria plus a "live dev curve" validity
   check, declaring the contamination honestly: the results were already
   known from the corrective pass, so it validated the pipeline rather than
   discovering anything.
5. **The honest comparisons.** E32 (one CPU-float32 harness for both systems,
   double-validated against frozen history) and E34 (replay portability) close
   the comparison with every caveat visible. Recipes differ, so these are
   **system comparisons, and no architecture claim is possible**.

```mermaid
flowchart TD
    A["E31 closes: Julia memorizes, no order"] --> B["supervisor review"]
    B --> C["binding bug: evaluator scored the original weights"]
    C --> D["public retraction (3c03322e)"]
    C --> E["corrective re-eval: Julia does acquire order"]
    D --> F["regression: at-eval-time weights hash"]
    E --> G["E33: valid re-run, pipeline proven"]
    F --> G
```

Criteria documents, frozen before each launch with verifiable commit order:
`docs/E32_CRITERIA…` through `docs/E36_CRITERIA…`, plus the internal protocol
family referenced from `docs/experiments.md`.

## What survives, what does not

**Survives, measured and replicated:**

- Dense per-value coverage with 4-Yes/4-No balance teaches real numeric order
  (shortcuts mathematically excluded by construction; E27, replicated E28b).
- The integration recipe: capability and task coexist under replay mixing,
  validated on a disjoint sample (E29, E29b, E30).
- No catastrophic forgetting in either direction at these budgets (E29).
- The E31 to E33 correction arc and its permanent regression.
- Calibration without retraining: one global temperature fitted on a frozen
  split, choice behavior provably unchanged, abstention at a pre-declared
  rule (E36).

**Does not survive, declared limits:**

- Monotonicity is 9/10 at v1.0 and 8/10 at v1.1 (s43 reaches 10/10 but ships
  with lower E-extrap). Neither system is 10/10.
- Coexistence of order with *Julia's own* typed distribution fails under
  replay (E34).
- Any claim that the **architecture** is the differentiator. Recipes differ.
- OOD and threshold-transfer claims from E25 (the decade shortcut,
  retrospectively identified). Superseded by E26's review.
- **The historical efficiency advantage (~1.35x, E22 v2)**: the v3 formal
  measurement shows a tie at batch 1 and Julia 5 to 10 % ahead at batch 16.
  The v2 claim is retracted.
- The single global temperature slightly over-softens the easiest set
  (known-thresholds ECE 0.024 to 0.042). A per-type temperature is the
  obvious next step; one scalar was pre-declared here (E36).

## Reproduce

```bash
pip install -e .[data,train,baselines]
python -m pytest -q                    # suite: 200 pass (default env)
# .venv-julia (transformers==5.0.0): 194 pass + 1 skip
python scripts/prepare_data.py --config configs/pilot.yaml --out data/processed/v1   # deterministic
python scripts/gen_e25_thrvar.py && python scripts/verify_e25_thrvar.py              # data + independent verifier
python scripts/e27_report.py --out /tmp/r.json                                       # deterministic analyzers
python scripts/gen_e36_calib.py && python scripts/verify_e36_calib.py                # calibration split + verifier
python scripts/e36_report.py --out /tmp/e36.json                                     # refuses to overwrite; byte-exact
```

Campaign artifacts live under `runs/` (gitignored; every analysis directory
carries `artifact_hashes.json`; reports regenerate byte-exact). Training ran
on Kaggle free tier. See `docs/runbook-kaggle.md` for the quota discipline.
Version notes: `docs/RELEASE_NOTES_E24-E29b.md`. Internal status and decision
logs stay out of Git by standing instruction.

## Documentation

- `docs/experiments.md`: the full campaign record, one entry per experiment
- `docs/model-card.md`: trained artifacts, final metrics, limits, v1.2
  calibration section
- `docs/dataset-card.md`, `docs/evaluation-protocol.md`, `docs/runbook-kaggle.md`
- `docs/typed-decisions-protocol.md`: the typed-decisions contract
- `demo/`: the interactive Space (canonical source)

## License

Apache-2.0 for code. Datasets and third-party models keep their own licenses
(`THIRD_PARTY_NOTICES.md`).
