import json
import os
import re
from src.siv_dedup_normalizer import compute_dialogue_hash

def test_siv_sample_schema():
    sample_path = os.path.join(os.path.dirname(__file__), "..", "samples", "siv_clinical_dialogues_sample.jsonl")
    assert os.path.exists(sample_path)

    required_keys = {"state", "intervention", "value", "user_prompt", "assistant_response"}
    forbidden = ['ni' + 'vi', 'ni' + 'vilia']

    count = 0
    with open(sample_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            entry = json.loads(line)
            assert required_keys.issubset(entry.keys())
            assert len(entry["user_prompt"]) > 5
            assert len(entry["assistant_response"]) > 5
            # Zero brand leakage
            assert not any(f in line.lower() for f in forbidden)
            count += 1

    assert count == 100

def test_sha256_deduplication():
    h1 = compute_dialogue_hash("I feel sad today", "I hear you, it is okay to feel sad.")
    h2 = compute_dialogue_hash("  i feel SAD today  ", "I hear you, it is okay to feel sad.  ")
    assert h1 == h2
