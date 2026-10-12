"""MK-SH-1 demo (Space Gjusev/MK-SH-1-demo, Gradio, CPU basic).

Canonical source: demo/ in github.com/Gjusev/MK-SH-1 (the Space holds a copy,
pushed by scripts/push_space.py).

Everything below the "decision" function is presentation; the decision path
(parse state, render question, one encoder pass, softmax over logits / T,
apply the frozen abstention policy, rule check) is the shipped v1.2 behavior
and must not change with the skin.
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
REPO_URL = "https://github.com/Gjusev/MK-SH-1"
MODEL_URL = "https://huggingface.co/Gjusev/MK-SH-1"

torch.set_num_threads(2)  # cpu-basic gives two vCPUs; more threads just contend
PKG = snapshot_download(MODEL_ID)
POLICY = json.load(open(f"{PKG}/inference-policy.json", encoding="utf-8"))
T = POLICY["temperature"]["value"]     # single global scalar fitted on e36-calib (E36)
TAU = POLICY["abstention"]["threshold"]  # frozen rule: abstain when max calibrated prob < tau

_tok = AutoTokenizer.from_pretrained(PKG)
_model = JuliaArchScorer("jhu-clsp/mmBERT-small")
_model.load_state_dict(load_file(f"{PKG}/model.safetensors"), strict=True)
_model.eval()

OPTIONS = (Option(id="false", description="No"), Option(id="true", description="Yes"))
NAME = {"false": "No", "true": "Yes"}
FIELD = "motor_temp_c"

# the three shipped examples, chosen from the frozen E36 logits with margins
# wide enough that the shown behavior is stable across CPU thread counts
EXAMPLES = [
    ['{"equipment_id": "cu-0006", "motor_temp_c": 30.43, "site": "site-alpha"}', 37.43],
    ['{"equipment_id": "cu-0283", "motor_temp_c": 104.29, "site": "site-bravo"}', 103.29],
    ['{"equipment_id": "cu-0029", "motor_temp_c": 36.29, "site": "site-foxtrot"}', 37.29],
]


def decide(state_text: str, threshold: float):
    state_text = state_text.strip()
    try:
        parsed = json.loads(state_text)
    except json.JSONDecodeError as e:
        raise gr.Error(f"State must be valid JSON ({e}). Copy the preloaded state and edit the numbers.")
    question = f"Is the motor temperature reading at least {threshold:g} degrees Celsius?"
    case = DecisionCase(id="demo", group_id="demo", source="demo", language="en",
                         domain=f"num_{FIELD}", state=state_text, question=question,
                         options=OPTIONS, target_option=None, split=None)
    qtypes = {"demo": "noul"}
    batch = collate_julia(_tok, [case], qtypes, 1024, 256, "cpu")
    with torch.inference_mode():
        logits = _model(**{k: batch[k] for k in
                           ("input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype")})
    # temperature is monotone: same choice as raw logits, calibrated confidences
    probs = (logits[0, :2].float() / T).softmax(-1).tolist()
    p = {"false": probs[0], "true": probs[1]}
    resp = apply_policy(case, p, threshold=TAU)

    value = parsed.get(FIELD) if isinstance(parsed, dict) else None
    rule = rule_line(value, threshold, resp)
    return (render_probs(p), render_verdict(p, resp, rule), render_question(question))


def rule_line(value, threshold, resp):
    """The question is rule-computable, so the demo can show the correct answer."""
    if not isinstance(value, (int, float)):
        return None
    gold = "true" if value >= threshold else "false"
    if resp.abstain:
        forced = "right" if resp.candidate == gold else "wrong"
        return (f"Rule check: the correct answer is {NAME[gold]}. The model's forced guess "
                f"({NAME[resp.candidate]}) would have been {forced}.")
    ok = "agrees" if resp.choice == gold else "disagrees"
    return (f"Rule check: value {value:g} {'at least' if gold == 'true' else 'below'} "
            f"threshold {threshold:g}, correct answer {NAME[gold]}. The model {ok}.")


def render_probs(p):
    rows = "".join(
        f'<div class="prob-row"><span class="opt">{NAME[k]}</span>'
        f'<span class="num">{p[k]:.3f}</span>'
        f'<span class="bar"><i style="width:{p[k] * 100:.1f}%"></i></span></div>'
        for k in ("false", "true"))
    return f'<div class="probs">{rows}</div>'


def render_verdict(p, resp, rule):
    conf = max(p.values())
    if resp.abstain:
        head = "Abstains: choice is null"
        meta = (f'Reason code <span class="code">below_validated_threshold</span>. '
                f'Max calibrated probability <span class="num">{conf:.3f}</span> is below '
                f'the threshold <span class="num">{TAU:g}</span>, so the policy refuses to answer.')
        kind = "refused"
    else:
        head = f"Choice: {NAME[resp.choice]}"
        meta = (f'Calibrated confidence <span class="num">{conf:.3f}</span> at or above '
                f'the threshold <span class="num">{TAU:g}</span>.')
        kind = "accepted"
    rule_html = f'<div class="rule">{rule}</div>' if rule else ""
    return f'<div class="verdict {kind}"><div class="v-head">{head}</div>' \
           f'<div class="v-meta">{meta}</div>{rule_html}</div>'


def render_question(question):
    import re
    q = re.sub(r"(\d+(?:\.\d+)?)", r'<span class="num">\1</span>', question)
    return f'<div class="q-asked">Question asked: {q}</div>'


HEAD = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Geist:wght@400;500;600&family=Geist+Mono:wght@400;500&display=swap" rel="stylesheet">
"""

DARK_BOOT = """() => {
  try { localStorage.setItem("gradio_theme", "dark"); } catch (_) {}
  document.documentElement.classList.add("dark");
  if (document.body) document.body.classList.add("dark");
}"""

CSS = """
:root {
  color-scheme: dark;
  --body-background-fill: #07101C; --background-fill-primary: #0C1827;
  --background-fill-secondary: #102238; --panel-background-fill: #0C1827;
  --block-background-fill: #0C1827; --input-background-fill: #091421;
  --border-color-primary: #24415F; --border-color-accent: #5F97D3;
  --block-border-color: #24415F; --panel-border-color: #24415F;
  --body-text-color: #EAF2FF; --block-title-text-color: #C8D8EA;
  --color-accent: #8BBBF2; --color-accent-soft: #0D263D; --link-text-color: #C7E1FF;
  --button-primary-background-fill: #8BBBF2; --button-primary-text-color: #07101C;
}
html, body { background: #07101C !important; }
.gradio-container { max-width: 1080px; margin: 0 auto; padding: 56px 24px 72px !important;
  background: #07101C; font-family: 'Geist', system-ui, sans-serif; color: #EAF2FF; }
.gradio-container .footer, .footer { display: none !important; }
#hero { border-left: 2px solid #7AA9E6; padding-left: 20px; }
#hero .brand { display: flex; align-items: flex-start; gap: 20px; margin-bottom: 14px; }
#hero .brand img { border-radius: 10px; display: block; box-shadow: 0 16px 48px rgba(34, 78, 128, 0.24); }
#hero h1 { font-family: 'Geist', system-ui, sans-serif; font-weight: 600; letter-spacing: -0.045em;
  font-size: 48px; line-height: 0.98; margin: 0 0 10px; color: #F5F9FF; }
#hero .sub { font-size: 16px; line-height: 1.55; color: #B6C8DF; max-width: 560px; margin: 0; }
#hero .meta { font-family: 'Geist Mono', monospace; font-size: 13px; line-height: 1.55; color: #A7BED8; margin: 0; }
.num, .code { font-family: 'Geist Mono', monospace; font-size: 0.94em; }
.code { background: transparent; color: inherit; }

.panel { background: #0C1827; border: 1px solid #24415F; border-radius: 12px;
  padding: 28px !important; transition: border-color 240ms ease, box-shadow 240ms ease; }
.panel:hover { border-color: #4274A8; box-shadow: 0 18px 46px rgba(0, 0, 0, 0.20); }
.kicker { display: block; font-family: 'Geist Mono', monospace; font-size: 12px; color: #82B8FF; margin: 0 0 10px; letter-spacing: 0.02em; }

.gradio-container input[type="text"], .gradio-container input[type="number"],
.gradio-container textarea { border-radius: 8px !important; border-color: #284966 !important;
  background: #091421 !important; color: #F5F9FF !important; caret-color: #B9DCFF !important;
  font-family: 'Geist Mono', monospace !important; font-size: 14px !important; }
.gradio-container label, .gradio-container .label-wrap { color: #C8D8EA !important; }
.gradio-container button.primary { background: #8BBBF2 !important; color: #07101C !important;
  border-radius: 7px !important; border: 1px solid #B9DCFF !important; font-weight: 650 !important;
  min-height: 48px !important; font-size: 15px !important;
  box-shadow: none !important; transition: transform 120ms ease, background 120ms ease; }
.gradio-container button.primary:hover { background: #B9DCFF !important; transform: none; }
.gradio-container button.primary:active { transform: scale(0.99); }
.gradio-container button.primary:focus-visible { outline: 3px solid #EAF2FF !important; outline-offset: 3px; }
.gradio-container button.primary:disabled { background: #4D6883 !important; color: #D6E4F3 !important; opacity: 1 !important; }
.gradio-container :focus { --ring-color: #82B8FF !important; }

.probs { display: flex; flex-direction: column; gap: 10px; margin-top: 4px; }
.prob-row { display: grid; grid-template-columns: 44px 64px 1fr; align-items: center; gap: 14px; }
.prob-row .opt { font-size: 14px; }
.prob-row .num { font-size: 15px; }
.bar { height: 4px; background: #1C334B; border-radius: 2px; overflow: hidden; }
.bar i { display: block; height: 100%; background: #82B8FF; }

.verdict { border-radius: 8px; padding: 16px 18px; margin-top: 8px; }
.verdict.accepted { background: #0D263D; color: #B9DCFF; border: 1px solid #35638F; }
.verdict.refused { background: #291822; color: #F2BAC7; border: 1px solid #734153; }
.v-head { font-size: 16px; font-weight: 600; margin-bottom: 6px; }
.v-meta { font-size: 14px; line-height: 1.5; }
.verdict .rule { margin-top: 10px; padding-top: 10px; border-top: 1px solid rgba(0,0,0,0.08);
  font-size: 13px; line-height: 1.5; opacity: 0.85; }
.q-asked { font-size: 14px; line-height: 1.55; color: #B4C8DE; margin-top: 18px; }

.vignette-scroll { width: 100%; overflow-x: auto; overflow-y: hidden; border: 1px solid #24415F;
  border-radius: 12px; background: #07101C; scrollbar-color: #5F97D3 #0C1827; }
.vignette { display: block; width: 1008px; min-width: 1008px; height: 700px; margin: 0 auto;
  border: none; background: #07101C; }
.usecase-intro { color: #B4C8DE; font-size: 15px; line-height: 1.6; margin: -2px 0 14px; max-width: 720px; }
.swipehint { display: none; color: #B4C8DE; font-family: 'Geist Mono', monospace;
  font-size: 12.5px; line-height: 1.5; margin: 8px 2px 0; }

.examples { margin-top: 40px; }
.examples .table { border-radius: 12px; border: 1px solid #24415F; background: #0C1827; }
.examples button.secondary, .examples .table-row { background: transparent !important; box-shadow: none !important; }
.examples table { border-collapse: collapse; width: 100%; }
.examples tr { border-bottom: 1px solid #1F3853 !important; transition: background 160ms ease; }
.examples tr:last-child { border-bottom: none !important; }
.examples tr:hover { background: #102238; }
.examples td { padding: 14px 16px !important; font-family: 'Geist Mono', monospace; font-size: 12.5px; }

#foot { margin-top: 48px; padding-top: 20px; border-top: 1px solid #1F3853;
  font-size: 13.5px; line-height: 1.55; color: #A7BED8; }
#foot a { color: #B9DCFF; text-decoration: underline; text-underline-offset: 3px; margin-right: 18px; }
#foot .note { margin-top: 8px; font-size: 12.5px; }

.rise { opacity: 0; transform: translateY(12px);
  animation: rise 600ms cubic-bezier(0.16,1,0.3,1) forwards; }
.d1 { animation-delay: 80ms; } .d2 { animation-delay: 160ms; } .d3 { animation-delay: 240ms; }
@keyframes rise { to { opacity: 1; transform: none; } }
@media (prefers-reduced-motion: reduce) { .rise { animation: none; opacity: 1; transform: none; } }

@media (max-width: 720px) {
  .gradio-container { padding: 28px 14px 48px !important; }
  #hero { padding-left: 14px; }
  #hero .brand { gap: 12px; }
  #hero .brand img { width: 88px; height: 44px; }
  #hero h1 { font-size: 38px; }
  #hero .sub { font-size: 15px; }
  .panel { padding: 20px 16px !important; }
  .vignette { margin: 0; }
  .swipehint { display: block; }
  .examples { margin-top: 28px; }
  #foot { margin-top: 36px; }
}

@media (prefers-contrast: more) {
  .usecase-intro, .q-asked, #foot, #hero .meta { color: #D6E4F3; }
  .panel, .vignette-scroll { border-color: #7CA9D6; }
}

/* MAREA nocturna: persist the same composed dark surface on every device. */
.dark {
  --body-background-fill: #07101C; --background-fill-primary: #0C1827;
  --background-fill-secondary: #102238; --panel-background-fill: #0C1827;
  --block-background-fill: #0C1827; --input-background-fill: #091421;
  --border-color-primary: #24415F; --border-color-accent: #4274A8;
  --block-border-color: #24415F; --panel-border-color: #24415F;
  --body-text-color: #EAF2FF; --block-title-text-color: #B6C8DF;
  --color-accent: #78AEEB; --color-accent-soft: #0D263D; --link-text-color: #B9DCFF;
  --button-primary-background-fill: #78AEEB; --button-primary-text-color: #07101C;
  color-scheme: dark;
}
.dark .gradio-container, .dark { background: #07101C !important; color: #EAF2FF; }
"""


def build():
    hero = """
    <header id="hero" class="rise">
      <div class="brand">
        <img src="/gradio_api/file=assets/logo-mark.png" alt="MK-SH-1 mark"
             width="112" height="56">
        <div>
          <h1>MK-SH-1</h1>
          <p class="sub">A decision model that reads a value from the state and answers a
          threshold question. It returns calibrated probabilities, a choice, and a refusal
          when its confidence is too low.</p>
        </div>
      </div>
      <p class="meta">v1.2 &middot; 140M parameters &middot; T = 7.653133 &middot;
      tau = 0.79 &middot; loads from Gjusev/MK-SH-1</p>
    </header>
    """
    foot = f"""
    <footer id="foot" class="rise d3">
      <a href="{REPO_URL}">Repository</a>
      <a href="{MODEL_URL}">Model card</a>
      <a href="{REPO_URL}/blob/main/docs/experiments.md">Experiment record (E24&ndash;E36)</a>
      <div class="note">Probabilities are temperature-scaled and the abstention rule was
      frozen before measurement (experiment E36). The rule check uses the fact that this
      question is computable from the state.</div>
    </footer>
    """
    gr.set_static_paths(paths=["assets", "usecases"])
    # The demo uses one intentional MAREA nocturna surface, including when the
    # host/browser preference would otherwise switch Gradio back to light mode.
    # Blocks expects a callable JS expression; raw statements abort hydration.
    with gr.Blocks(css=CSS, title="MK-SH-1 demo", js=DARK_BOOT, head=HEAD) as demo:
        gr.HTML(hero)
        with gr.Column(elem_classes=["rise", "d1"]):
            gr.HTML('<span class="kicker">Use cases</span>'
                    '<p class="usecase-intro">Three interactive interface studies for queues that '
                    'benefit from an explicit review lane. They illustrate the decision contract; '
                    'the live checkpoint runs in the lab immediately below.</p>')
            with gr.Tabs():
                for tab_label, src in (("Support triage", "usecases/triage-soporte.html"),
                                       ("Lead qualification", "usecases/lead-qual.html"),
                                       ("Stock alerts", "usecases/stock-alerts.html")):
                    with gr.Tab(tab_label):
                        gr.HTML(f'<div class="vignette-scroll" tabindex="0" '
                                'aria-label="Interactive use case; scroll horizontally on small screens">'
                                f'<iframe class="vignette" src="/gradio_api/file={src}" '
                                'title="MK-SH-1 use case vignette" loading="lazy" scrolling="no"></iframe>'
                                '</div><p class="swipehint">Swipe horizontally to inspect every lane at full, readable size.</p>')
        with gr.Column(elem_classes=["panel", "rise", "d1"]):
            gr.HTML('<span class="kicker">Live decision lab</span>')
            state = gr.Textbox(
                label="State, as JSON (the value lives in motor_temp_c)",
                value=EXAMPLES[0][0], lines=3)
            threshold = gr.Number(
                label="Threshold t, in degrees (question: is the reading at least t?)",
                value=37.43)
            run = gr.Button("Run decision", variant="primary")
            out_probs = gr.HTML('<div class="q-asked">Calibrated probabilities appear here '
                                'after you run a decision.</div>')
            out_verdict = gr.HTML('<div class="q-asked">The choice, or the refusal with its '
                                  'reason code, appears here.</div>')
            out_question = gr.HTML("")
            run.click(decide, inputs=[state, threshold],
                      outputs=[out_probs, out_verdict, out_question], api_name="decide")
        with gr.Column(elem_classes=["rise", "d2"]):
            gr.HTML('<span class="kicker">Run a real checkpoint example</span>')
            gr.Examples(examples=EXAMPLES, inputs=[state, threshold],
                        fn=decide, outputs=[out_probs, out_verdict, out_question],
                        cache_examples=False,
                        examples_per_page=3)
        gr.HTML(foot)
    return demo


demo = build()
demo.launch()
