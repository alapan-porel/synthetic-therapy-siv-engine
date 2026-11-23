"""
Clinical Literature SIV Generator (API Edition)
Processes domain psychological literature into Synthetic Instruction Validation (SIV) dialogue sets.
"""

import os
import json
import re
import time
from typing import List, Dict, Any, Optional
from openai import OpenAI
from tqdm import tqdm

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
API_BASE_URL = os.getenv("LLM_API_BASE", "https://api.deepseek.com")
MODEL_NAME = os.getenv("LLM_MODEL_NAME", "deepseek-chat")

CANONICAL_STATES = [
    "anxiety_rumination", "panic_arousal", "depressive_withdrawal", "trauma_activation",
    "attachment_insecurity", "emotional_dysregulation", "grief_processing",
    "shame_identity_threat", "burnout_exhaustion", "interpersonal_conflict",
    "existential_distress", "avoidance_numbing"
]

INTERVENTIONS = [
    "Validation", "Empathic Reflection", "Metacognitive Observation",
    "Cognitive Restructuring", "Somatic Regulation", "Behavioral Activation",
    "Motivational Interviewing", "Problem Solving", "Narrative Reframing",
    "Attachment Repair Framing", "Psychoeducation", "Safety Stabilization",
    "Resource Orientation", "Boundary Modeling", "Values Clarification"
]

class BookAPIRunner:
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        self.api_key = api_key or DEEPSEEK_API_KEY
        self.client = OpenAI(api_key=self.api_key, base_url=base_url or API_BASE_URL)

    def generate_siv_dialogue(self, clinical_text_chunk: str, book_title: str) -> Optional[Dict[str, Any]]:
        system_prompt = """You are an expert clinical psychotherapist and dataset curator.
Given an excerpt of psychological literature, extract the underlying human emotional dilemma and craft a realistic, evidence-based therapeutic dialogue.

Schema Requirements:
1. state: Select from the canonical states list.
2. intervention: Select from the 15 locked interventions.
3. value: Concise summary of clinical delivery mode.
4. user_prompt: Raw, vulnerable human struggle (no clinical jargon, no real names).
5. assistant_response: Warm, non-judgmental, grounded therapeutic response. No direct diagnoses or medication advice.

Return valid JSON with keys: state, intervention, value, user_prompt, assistant_response."""

        user_content = f"Source: {book_title}\n\nText:\n{clinical_text_chunk[:2500]}"

        try:
            response = self.client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ],
                temperature=0.7,
                response_format={"type": "json_object"}
            )
            raw = response.choices[0].message.content
            return json.loads(raw)
        except Exception as err:
            print(f"API generation error: {err}")
            return None
