<!-- Local mirror of the Hugging Face model card (https://huggingface.co/Gjusev/MK-SH-1). The HF card is canonical; update there first and re-sync this file. -->

<table><tr>
<td><img src="logo-mark.png" width="120" alt="MK-SH-1 mark: a whale shark in negative space between two tide bands"></td>
<td><h1>MK-SH-1</h1><p><sub>a 140M decision model: calibrated probabilities, a choice, and a refusal below the gate</sub></p></td>
</tr></table>

A decision model of about 140M parameters that answers typed questions over
a described state: given a state, a question and a list of options, it
returns calibrated probabilities, a choice, and an explicit abstention when
confidence is insufficient. Its differentiating capability is numeric order:
it compares a value in the state against a threshold stated in the question
(`value >= t`), verified free of shortcuts by construction, taught and
validated in a twelve-experiment campaign with pre-registered criteria, and
held alongside the original task through replay-mixing integration.

The published weights were trained in experiment E35 (sha256 `ea66cdde…`,
criteria frozen before every launch). Calibration and the abstention policy
were added afterwards in experiment E36, fitted on a dedicated held-out
split, without touching a single weight. `inference-policy.json` in this repo
carries the full policy: one global temperature, the gate, and the response
semantics.

Interactive demo: [Gjusev/MK-SH-1-demo](https://huggingface.co/spaces/Gjusev/MK-SH-1-demo)
(Gradio, CPU). Give it a state and a threshold and watch the calibrated
probabilities, the choice, and the abstention with its reason code. Three
interactive use-case interfaces (support triage, lead qualification, stock
alerts) sit above the live lab in the Space.

## What it is for

Each case names the measured capability it rests on, and where that
measurement stops.

1. **Threshold routing.** "Is the value at least the threshold stated in the
   question": limits, eligibility checks, price or risk cutoffs. Rests on
   numeric order measured at testC 0.979 and E-extrap 0.875 on fresh extreme
   thresholds. Bound: validated on the English numeric-comparison contract
   (motor-temperature style states); not a general numeric reasoner.
2. **Calibrated scoring with a safety valve.** ECE 0.016 on the dedicated
   split (E36) plus abstention at tau = 0.79: 90 % coverage with 4.86 %
   selective risk on that split, +2.85 pp accuracy on the cases the policy
   keeps. Suited to decision queues where refusals go to human review.
   Bound: those two numbers are in-sample (tau was chosen on the same rows).
   On held-out sets the 5 % budget holds as a point estimate on the in-range
   numeric sets but not on the fresh OOD extremes (8.2 % risk) or the routing
   task (7.3 %); see the per-set table below.
3. **CPU and edge inference.** 106 ms per question at batch 1 on 8 CPU
   threads, fp32 (E22 v3). Bound: a tie with the Julia-1 reference at batch 1
   and 5 to 10 % behind at batch 16; the earlier 1.35x speed claim is
   retracted.
4. **A methodological template.** Dense anti-shortcut grid, replay
   integration (E29/E30), replay-protected extremes (E35), calibration on a
   frozen split (E36). Bound: one capability, one model family, demonstrated
   at 140M scale; not claimed beyond what the campaign measured.

## Measured capability

| Metric | Value |
|---|---|
| Numeric order, known thresholds (testC, n=48) | 0.979 |
| Interpolation, historical thresholds (testD, n=54) | 0.963 |
| Fresh 2-decimal OOD, interior (testE-interp, n=116) | 0.914, 8/8 pairs |
| Fresh 2-decimal OOD, extremes (testE-extrap, n=56) | 0.875, 7/8 pairs |
| Monotonicity sweeps (VE2, 10 sweeps) | 8/10 |
| Original task (routing, 300-row slice) | 0.800 |

The temperature is monotone: it changes no choices, so accuracy is identical
by construction and only the confidences move. Verified mechanically, case
by case, on every evaluation set. The local CPU harness also reproduced the
frozen E35 Kaggle per-case choices exactly (664/664 across all five sets).

## Calibration & abstention (v1.2, experiment E36)

Fitted on `e36-calib`, a dedicated split of 480 fresh two-decimal numeric
questions (60 values × 8 threshold offsets, 4 Yes/4 No per value). The split
is disjoint from the entire prior numeric universe of the campaign (values
and thresholds of every e24 to e35 train/dev/eval file, 1,106 numbers),
verified at build and re-derived by an independent verifier. It was never
used for training.

**Temperature** (single global scalar, fitted by NLL minimization on the
split only):

| | NLL | ECE (15 bins) |
|---|---|---|
| split, before (T=1) | 0.879 | 0.073 |
| **split, after (T=7.653)** | **0.196** | **0.016** |

Descriptive (T applied, never re-fitted, frozen evals):

| Set | NLL before → after | ECE before → after |
|---|---|---|
| known thresholds (n=48) | 0.225 → 0.062 | 0.024 → 0.042 |
| new thresholds (n=54) | 0.636 → 0.126 | 0.037 → 0.035 |
| monotonicity (n=60) | 0.921 → 0.192 | 0.033 → 0.026 |
| fresh OOD (n=202) | 2.183 → 0.414 | 0.092 → 0.076 |
| routing task (n=300) | 2.535 → 0.588 | 0.184 → 0.045 |

Abstention policy (the rule was frozen before anyone looked at the curve):
tau is the smallest threshold on the grid 0.30:0.005:0.995 whose selective
risk stays at or below 5% on the split, for maximum coverage under the risk
budget (the same rule as this project's v0.1 release). Result: tau = 0.79,
coverage 90.0%, selective accuracy 95.14% (selective risk 4.86%) against
92.29% forced accuracy, so the policy buys +2.85 pp on the cases it keeps.
Abstaining returns
`choice=null` + `reason_code="below_validated_threshold"`. The full
coverage/accuracy/risk curve is in the training repo
(`runs/e36-calib/e36_report.json`, deterministically regenerable).

Those split numbers are in-sample: T was fitted, the grid swept and tau
chosen on the same 480 rows, and the first grid point under 5 % is the one
most likely to sit there by chance. Freezing the rule in advance does not
change that. Measured afterwards on every set at the shipped T and tau
(`scripts/e36_heldout_policy.py`, no refit; risk interval is exact
Clopper-Pearson 95 %):

| Set | n | Coverage | Selective risk | 95 % interval | 5 % budget |
|---|---|---|---|---|---|
| split (fit, in-sample) | 480 | 0.900 | 21/432 = 4.9 % | 3.0 to 7.3 % | point only |
| known thresholds | 48 | 0.979 | 1/47 = 2.1 % | 0.1 to 11.3 % | point only |
| new thresholds | 54 | 0.944 | 1/51 = 2.0 % | 0.1 to 10.5 % | point only |
| monotonicity | 60 | 1.000 | 2/60 = 3.3 % | 0.4 to 11.5 % | point only |
| fresh OOD extremes | 202 | 0.960 | 16/194 = 8.2 % | 4.8 to 13.0 % | **missed** |
| routing task | 300 | 0.683 | 15/205 = 7.3 % | 4.2 to 11.8 % | **missed** |

No set confirms the budget at the upper bound; the small in-range sets are
consistent with it but cannot establish it. On the fresh OOD extremes the
gate keeps 96 % of rows while 8.2 % of the kept ones are wrong, the same
overconfidence the ECE of 0.076 shows. Treat tau = 0.79 as a 5 % budget
validated in-sample on in-range numeric questions only. This check was
raised by an external review on the Hugging Face discussion (2026-10-14).

## Usage

```python
from mokka_decisions.arch_julia import JuliaArchScorer, collate_julia
from mokka_decisions.policy import apply_policy
from safetensors.torch import load_file
from transformers import AutoTokenizer
import json, torch

d = "."  # this repo
policy = json.load(open("inference-policy.json"))
T, tau = policy["temperature"]["value"], policy["abstention"]["threshold"]

model = JuliaArchScorer("jhu-clsp/mmBERT-small")
model.load_state_dict(load_file("model.safetensors"))
model.eval()
tok = AutoTokenizer.from_pretrained(".")

qtype = {"demo": "noul"}
cases = [...]  # DecisionCase objects; see the training repo's contract
batch = collate_julia(tok, cases, qtype, max_length=1024, head_length=256, device="cpu")
with torch.inference_mode():
    logits = model(**{k: batch[k] for k in
        ("input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype")})
probs = (logits / T).softmax(-1)          # calibrated probabilities
response = apply_policy(cases[0],
    {opt.id: p for opt, p in zip(cases[0].options, probs[0].tolist())},
    threshold=tau)                          # choice, or abstain + reason_code
```

## Architecture

`JuliaArchScorer`: see `ARCHITECTURE.md` in this repo and
`src/mokka_decisions/arch_julia.py` in the training repo. Encoder:
mmBERT-small (JHU CLSP, MIT), fine-tuned. Head: 2 pre-norm transformer
layers + type embedding + marker scorer (one MASK per option).

## Training / calibration provenance

Weights: the E35 s42 final checkpoint, from mmBERT-small upstream + fresh seeded
head; dense comparison grid, edge/chain augmentation, dev-patience stopping,
replay-mixing integration (E24 to E30), replay-protected extremes (E35).
Calibration + policy: experiment **E36** (criteria frozen before
measurement, generator committed before the outcome, deterministic
byte-exact analyzer, adversarial conventions of the campaign). Full record:
[github.com/Gjusev/MK-SH-1](https://github.com/Gjusev/MK-SH-1)
(`docs/experiments.md`, `docs/model-card.md`). What this model is good for,
with the measured capability and the bound behind each case:
[Use cases](https://github.com/Gjusev/MK-SH-1#use-cases). The E31 correction
story (a wrong closure retracted in public commits, a valid re-run, and a
permanent regression that prevents recurrence) is in the repo's README under
"Rigor": honesty is treated as a feature of the evaluation system, not a
footnote.

## Limitations (declared)

- Single global temperature: it slightly over-softens the easiest
  distribution (known-thresholds set: ECE 0.024 to 0.042) while improving
  every fresh, OOD and task distribution by a lot (routing ECE 0.184 to
  0.045). A per-type temperature, as Julia-1 ships, is the obvious next
  step; one scalar was the pre-declared choice here.
- The policy was fitted on the numeric surface only (the model's public
  decision contract); routing probabilities inherit the global temperature
  (measured descriptively above).
- E-extrap 0.875 is not 10/10; monotonicity 8/10 (seed s43 reaches 10/10
  with lower E-extrap).
- No architecture claims vs Julia-1: complete-system comparisons only.

## Attribution

- Encoder: [mmBERT-small](https://huggingface.co/jhu-clsp/mmBERT-small) by
  JHU CLSP, MIT license, fine-tuned by this project.
- Comparator: Julia-1 (public release), used under its license for
  evaluation only.
- Training, evaluation and calibration system: this project (Apache-2.0).

## License

Apache-2.0 (code and trained weights of this export). The inherited encoder
keeps its MIT license.
