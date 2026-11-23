# Synthetic Instruction Validation (SIV) Clinical Dialogue Engine

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Dataset](https://img.shields.io/badge/Dataset-Canonical%20SIV%20Schema-brightgreen.svg)](#canonical-siv-schema)

A two-stage high-throughput synthetic dialogue pipeline creating clinically grounded instruction-tuning datasets for psychological LLMs. Enforces 12 canonical emotional states and 15 locked clinical interventions while ensuring zero diagnostic overreach or medication advice.

---

## Canonical SIV Schema

Each dialogue is represented as a structured JSONL record:

```json
{
  "state": "anxiety_rumination",
  "intervention": "Metacognitive Observation",
  "value": "Quiet & Contemplative",
  "user_prompt": "I can't stop thinking about what my manager said yesterday. I keep replaying it in my head over and over.",
  "assistant_response": "It sounds like your mind is caught in an exhausting loop, holding on to every word. Notice how much energy it takes to keep replaying that moment. What happens if we step back for a moment and just observe that repetitive thought as something your mind is producing, rather than an absolute truth?"
}
```

### The 12 Canonical Emotional States
1. `anxiety_rumination`
2. `panic_arousal`
3. `depressive_withdrawal`
4. `trauma_activation`
5. `attachment_insecurity`
6. `emotional_dysregulation`
7. `grief_processing`
8. `shame_identity_threat`
9. `burnout_exhaustion`
10. `interpersonal_conflict`
11. `existential_distress`
12. `avoidance_numbing`

### The 15 Locked Clinical Interventions
1. `Validation`
2. `Empathic Reflection`
3. `Metacognitive Observation`
4. `Cognitive Restructuring`
5. `Somatic Regulation`
6. `Behavioral Activation`
7. `Motivational Interviewing`
8. `Problem Solving`
9. `Narrative Reframing`
10. `Attachment Repair Framing`
11. `Psychoeducation`
12. `Safety Stabilization`
13. `Resource Orientation`
14. `Boundary Modeling`
15. `Values Clarification`

---

## Pipeline Workflow

```
 [ Clinical Text Chunks / Video Drafts ]
                   │
                   ▼
   ┌──────────────────────────────────────────────┐
   │ Stage 1: Draft Generator (vLLM / Qwen 7B)    │
   │ ├─ Extracts raw human emotional struggle     │
   │ └─ Suppresses diagnostic labels & brand terms│
   └──────────────────────┬───────────────────────┘
                          │
                          ▼
   ┌──────────────────────────────────────────────┐
   │ Stage 2: SIV Generator (Qwen 14B / DeepSeek) │
   │ ├─ Maps to 12 canonical emotional states     │
   │ └─ Formulates 15 locked clinical strategies  │
   └──────────────────────┬───────────────────────┘
                          │
                          ▼
   ┌──────────────────────────────────────────────┐
   │ Stage 3: SHA-256 Deduplication & Cleaning    │
   │ ├─ Normalized user_prompt + assistant_resp   │
   │ └─ Strict schema validation & zero leakage   │
   └──────────────────────────────────────────────┘
```

---

## Installation & Testing

```bash
pip install -r requirements.txt
pytest tests/
```

### Running Deduplication on Custom Datasets

```bash
python src/siv_dedup_normalizer.py input_dialogues.jsonl output_deduped.jsonl
```

---

## Dataset Release Notice

* A representative **100-row sample** is packaged with this repository at `samples/siv_clinical_dialogues_sample.jsonl`.
* The full merged and deduplicated **44,901-row clinical dialogue dataset** is hosted externally in `_huggingface_datasets/siv_clinical_dialogues_merged.jsonl`.

---

## License

Apache-2.0. Cleanroom extraction and packaging release.
