import json
import os
import random
import re
from tqdm import tqdm
from vllm import LLM, SamplingParams

# ================= CONFIG ==================
INPUT_FILE = "stage1_drafts.jsonl"
OUTPUT_FILE = "siv_emotional_strict.jsonl"
PROGRESS_FILE = "stage2_progress.txt"

MODEL_NAME = "Qwen/Qwen2.5-14B-Instruct-AWQ"

BATCH_SIZE = 16
GPU_UTIL = 0.95
MAX_TOKENS = 1024
# ==========================================


# ================= LOCKED REGISTRIES ==================

CANONICAL_STATES = [
    "anxiety_rumination","panic_arousal","depressive_withdrawal","trauma_activation",
    "attachment_insecurity","emotional_dysregulation","grief_processing",
    "shame_identity_threat","burnout_exhaustion","interpersonal_conflict",
    "existential_distress","avoidance_numbing"
]

INTERVENTIONS = [
    "Validation","Empathic Reflection","Metacognitive Observation",
    "Cognitive Restructuring","Somatic Regulation","Behavioral Activation",
    "Motivational Interviewing","Problem Solving","Narrative Reframing",
    "Attachment Repair Framing","Psychoeducation","Safety Stabilization",
    "Resource Orientation","Boundary Modeling","Values Clarification"
]

DELIVERY_STYLES = [
    "Warm & Maternal","Scientific & Grounded","Direct & Coach-like",
    "Quiet & Contemplative","Playful & Metaphorical"
]

STYLE_TARGETS = {
    "Warm & Maternal": 0.35,
    "Scientific & Grounded": 0.20,
    "Direct & Coach-like": 0.15,
    "Quiet & Contemplative": 0.15,
    "Playful & Metaphorical": 0.10
}

RESPONSE_INTENTS = [
    "pure_validation","validate_then_reflect",
    "validate_then_reframe","ground_then_redirect","safety_then_referral"
]

RISK_INTENTS = ["NONE","DISTRESS","CRISIS"]
CONFIDENCE_LEVELS = ["low","medium","high"]

CLINICAL_GOALS = [
    "contain_distress","increase_awareness","shift_cognition",
    "stabilize_arousal","support_expression","build_agency",
    "strengthen_attachment_security"
]


# ===== ROTATION TRACKERS =====
last_interventions = []
last_observed_states = []
style_counts = {k: 0 for k in DELIVERY_STYLES}
total_generated = 0


# ================= HARD SCHEMA PROMPT ==================

SYSTEM_PROMPT = f"""
You are a Clinical Supervisor generating SIV therapy training data.

CLINICAL STRUCTURE RULE (NON-NEGOTIABLE):

Intervention MUST determine how the assistant reply is written.

Validation:
- Emotional acknowledgement only
- No advice
- No reframing
- No action suggestions

Empathic Reflection:
- Mirror emotional pattern
- Name internal experience
- No solutions

Metacognitive Observation:
- Highlight thinking loops
- Point out rumination / patterns
- No behavioral instructions

Somatic Regulation:
- Include grounding / breath / body awareness

Cognitive Restructuring:
- Identify thought distortion
- Offer reframe

Behavioral Activation:
- Suggest small, gentle action

Problem Solving:
- Structured next steps allowed

Motivational Interviewing:
- Explore ambivalence
- Elicit internal motivation

Narrative Reframing:
- Reinterpret meaning of experience

Attachment Repair Framing:
- Safety + relational reassurance

Safety Stabilization:
- Immediate grounding + safety

Do NOT mix multiple interventions in one response.
One row = one clinical move.

STRICT RULES:

1. Produce ONE SIV training row in valid JSON.
2. Emotional depth REQUIRED but must remain clinically grounded.
3. DO NOT be poetic by default.
4. DO NOT use archetypes ("inner child", "soul", etc).
5. Assistant response: 85–120 words max.

THERAPY REALISM:
- Begin with emotional stabilization.
- Avoid early advice.

LANGUAGE:
- English only.

INTERVENTION ROTATION:
- Do not repeat same intervention twice consecutively.

OBSERVED STATE ENTROPY:
- Avoid reusing same observed_state consecutively.
- Must be phenomenological, short, signal-like.

DELIVERY STYLE BALANCE:
- Follow natural distribution across dataset.

CRITICAL:

You MUST choose ONLY from these registries:

canonical_state ∈ {CANONICAL_STATES}
intervention ∈ {INTERVENTIONS}
delivery_style ∈ {DELIVERY_STYLES}
response_intent ∈ {RESPONSE_INTENTS}
risk_intent ∈ {RISK_INTENTS}
confidence_level ∈ {CONFIDENCE_LEVELS}
clinical_goal ∈ {CLINICAL_GOALS}

observed_state must be:
- short
- phenomenological
- signal-like
- NOT narrative

OUTPUT STRUCTURE:

{{
  "siv_metadata": {{
    "observed_state": "...",
    "canonical_state": "...",
    "intervention": "...",
    "delivery_style": "...",
    "response_intent": "...",
    "risk_intent": "...",
    "confidence_level": "...",
    "clinical_goal": "..."
  }},
  "conversations": [
    {{"from":"user","value":"..."}},
    {{"from":"assistant","value":"..."}}
  ],
  "source": {{
    "origin": "stage2_emotional_rewrite",
    "dataset_version": "siv_v2"
  }}
}}

JSON ONLY.
NO MARKDOWN.
"""


# ================= UTILITIES ==================

def load_inputs():
    if not os.path.exists(INPUT_FILE): return []
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        return [json.loads(x) for x in f]

def load_progress():
    if not os.path.exists(PROGRESS_FILE): return 0
    return int(open(PROGRESS_FILE).read().strip())

def save_progress(i):
    with open(PROGRESS_FILE, "w") as f:
        f.write(str(i))


def pick_style():
    global total_generated
    if total_generated < 50:
        return random.choice(DELIVERY_STYLES)

    deficits = {
        k: STYLE_TARGETS[k] - (style_counts[k] / max(total_generated,1))
        for k in DELIVERY_STYLES
    }
    return max(deficits, key=deficits.get)


def rotation_ok(intervention, observed_state):

    if len(last_interventions) >= 1:
        if last_interventions[-1] == intervention:
            return False

    if len(last_observed_states) >= 1:
        if last_observed_states[-1] == observed_state:
            return False

    return True


def build_prompt(sample):

    style = pick_style()

    user_content = f"""
RAW DRAFT:
User: {sample["draft_user"]}
Assistant: {sample["draft_assistant"]}

TASK:
Transform into a clinical therapy interaction.
Maintain realism.

Use delivery_style: {style}

Output JSON only.
"""

    return f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n{user_content}<|im_end|>\n<|im_start|>assistant\n"


def extract_json(text):
    text = text.strip()
    text = re.sub(r"```json\s*", "", text)
    text = re.sub(r"```", "", text)
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1:
        return text[start:end+1]
    return None


def is_valid_row(obj):
    try:
        meta = obj["siv_metadata"]

        if meta["canonical_state"] not in CANONICAL_STATES: return False
        if meta["intervention"] not in INTERVENTIONS: return False
        if meta["delivery_style"] not in DELIVERY_STYLES: return False
        if meta["response_intent"] not in RESPONSE_INTENTS: return False
        if meta["risk_intent"] not in RISK_INTENTS: return False
        if meta["confidence_level"] not in CONFIDENCE_LEVELS: return False
        if meta["clinical_goal"] not in CLINICAL_GOALS: return False

        if not rotation_ok(meta["intervention"], meta["observed_state"]):
            return False

        style_counts[meta["delivery_style"]] += 1
        last_interventions.append(meta["intervention"])
        last_observed_states.append(meta["observed_state"])

        return True
    except:
        return False


# ================= MAIN ==================

def main():

    global total_generated

    data = load_inputs()
    start = load_progress()

    print(f"🚀 Starting STRICT SIV generation from index {start}")

    llm = LLM(
        model=MODEL_NAME,
        gpu_memory_utilization=GPU_UTIL,
        quantization="awq_marlin",
        enforce_eager=True,
        max_num_seqs=64
    )

    sampling = SamplingParams(
        temperature=0.7,
        max_tokens=MAX_TOKENS,
        stop=["<|im_end|>"]
    )

    with open(OUTPUT_FILE, "a", encoding="utf-8") as fout:
        for i in tqdm(range(start, len(data), BATCH_SIZE)):

            batch = data[i:i+BATCH_SIZE]
            prompts = [build_prompt(x) for x in batch]

            outputs = llm.generate(prompts, sampling)

            for out in outputs:
                try:
                    text = extract_json(out.outputs[0].text)
                    if not text:
                        continue

                    obj = json.loads(text)

                    if is_valid_row(obj):
                        fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
                        total_generated += 1
                except:
                    continue

            fout.flush()
            save_progress(i + len(batch))

    print("✅ STRICT SIV emotional dataset generation complete")


if __name__ == "__main__":
    main()
