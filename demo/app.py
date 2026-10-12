"""MK-SH-1 interactive demo (Space Gjusev/MK-SH-1-demo, Gradio, CPU basic).

Canonical source: demo/ in github.com/Gjusev/mokka-decisions (pushed to this
Space by scripts/push_space.py — the Space holds a copy; the repo is the
source of truth).

Loads the model FROM THE HUB (Gjusev/MK-SH-1 v1.2): weights + tokenizer +
embedded inference-policy.json. One encoder pass, softmax(logits / T), and
the abstention policy: choice = argmax when max calibrated probability >=
tau, else choice=null + reason_code.
"""

from __future__ import annotations

import json

import gradio as gr
import torch
from huggingface_hub import snapshot_download

from mokka_decisions.arch_julia import JuliaArchScorer, collate_julia
from mokka_decisions.contracts import DecisionCase, Option
from mokka_decisions.policy import apply_policy
from safetensors.torch import load_file
from transformers import AutoTokenizer

MODEL_ID = "Gjusev/MK-SH-1"
REPO_URL = "https://github.com/Gjusev/mokka-decisions"
MODEL_URL = "https://huggingface.co/Gjusev/MK-SH-1"

torch.set_num_threads(2)
PKG = snapshot_download(MODEL_ID)
POLICY = json.load(open(f"{PKG}/inference-policy.json", encoding="utf-8"))
T = POLICY["temperature"]["value"]
TAU = POLICY["abstention"]["threshold"]

_tok = AutoTokenizer.from_pretrained(PKG)
_model = JuliaArchScorer("jhu-clsp/mmBERT-small")
_model.load_state_dict(load_file(f"{PKG}/model.safetensors"), strict=True)
_model.eval()

OPTIONS = (Option(id="false", description="No"), Option(id="true", description="Yes"))
NAME = {"false": "No", "true": "Yes"}
FIELD = "motor_temp_c"


def decide(state_text: str, threshold: float):
    state_text = state_text.strip()
    try:
        parsed = json.loads(state_text)
    except json.JSONDecodeError as e:
        raise gr.Error(f"State must be valid JSON: {e}")
    question = f"Is the motor temperature reading at least {threshold:g} degrees Celsius?"
    case = DecisionCase(id="demo", group_id="demo", source="demo", language="en",
                         domain=f"num_{FIELD}", state=state_text, question=question,
                         options=OPTIONS, target_option=None, split=None)
    qtypes = {"demo": "noul"}
    batch = collate_julia(_tok, [case], qtypes, 1024, 256, "cpu")
    with torch.inference_mode():
        logits = _model(**{k: batch[k] for k in
                           ("input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype")})
    probs = (logits[0, :2].float() / T).softmax(-1).tolist()
    p = {"false": probs[0], "true": probs[1]}
    resp = apply_policy(case, p, threshold=TAU)

    conf = max(p.values())
    if resp.abstain:
        decision = (f"## ⚠️ ABSTAINS\n\n**choice = null** — reason_code: "
                    f"`{resp.reason_code}`\n\nMax calibrated probability "
                    f"**{conf:.3f}** < τ **{TAU}** → the policy refuses to answer.")
    else:
        agree = ""
        if isinstance(parsed.get(FIELD), (int, float)):
            gold = "true" if parsed[FIELD] >= threshold else "false"
            ok = resp.choice == gold
            agree = ("\n\nRule check: value "
                     f"`{parsed[FIELD]}` {'≥' if gold == 'true' else '<'} threshold "
                     f"`{threshold:g}` → correct answer **{NAME[gold]}** — model "
                     f"{'✅ agrees' if ok else '❌ disagrees'}.")
        decision = (f"## ✅ Choice: **{NAME[resp.choice]}**\n\nCalibrated confidence "
                    f"**{conf:.3f}** ≥ τ **{TAU}**.{agree}")
    rule_note = ""
    if isinstance(parsed.get(FIELD), (int, float)) and resp.abstain:
        gold = "true" if parsed[FIELD] >= threshold else "false"
        rule_note = (f"\n\nRule check: correct answer is **{NAME[gold]}** — the model's "
                     "forced guess would have been risky, which is exactly what the "
                     "policy catches.")
    return ({"No": p["false"], "Yes": p["true"]},
            decision + rule_note,
            f"Question: *{question}*")


EXAMPLES = [
    ['{"equipment_id": "cu-0006", "motor_temp_c": 30.43, "site": "site-alpha"}', 37.43],
    ['{"equipment_id": "cu-0283", "motor_temp_c": 104.29, "site": "site-bravo"}', 103.29],
    ['{"equipment_id": "cu-0029", "motor_temp_c": 36.29, "site": "site-foxtrot"}', 37.29],
]

desc = (
    f"**[{MODEL_ID}]({MODEL_URL})** — a ~140M decision model with **numeric order**: it "
    "compares a value in the state against a threshold in the question. Probabilities are "
    f"**calibrated** (T = {T}, fitted on a dedicated held-out split, E36) and the policy "
    f"**abstains** with an explicit reason code when confidence < τ = {TAU} (90% coverage at "
    "≤5% selective risk). Weights, tokenizer and the policy all load from the hub package. "
    f"Full campaign record: [mokka-decisions]({REPO_URL})."
)

demo = gr.Interface(
    fn=decide,
    inputs=[
        gr.Textbox(label="State (free-form JSON; the value lives in motor_temp_c)",
                   value=EXAMPLES[0][0], lines=3),
        gr.Number(label="Threshold t (question: is the reading at least t °C?)", value=37.43),
    ],
    outputs=[
        gr.Label(label="Calibrated probabilities"),
        gr.Markdown(label="Decision"),
        gr.Markdown(label="Rendered question"),
    ],
    title="MK-SH-1 — calibrated numeric decisions with abstention",
    description=desc,
    examples=EXAMPLES,
    examples_per_page=3,
    allow_flagging="never",
)

demo.launch()
