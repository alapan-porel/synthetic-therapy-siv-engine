import json
import os
import re
import unittest
import sys

# Ensure repo root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.siv_dedup_normalizer import compute_dialogue_hash, normalize_entry

class TestSIVPipeline(unittest.TestCase):
    def test_siv_sample_schema_and_zero_brand_leakage(self):
        sample_path = os.path.join(os.path.dirname(__file__), "..", "samples", "siv_clinical_dialogues_sample.jsonl")
        self.assertTrue(os.path.exists(sample_path), f"Sample path {sample_path} not found")

        required_keys = {"state", "intervention", "value", "user_prompt", "assistant_response"}
        forbidden_pattern = re.compile(r'\b(ni' + 'vi|ni' + 'vilia)\b', re.IGNORECASE)

        count = 0
        with open(sample_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                entry = json.loads(line)
                self.assertTrue(required_keys.issubset(entry.keys()))
                self.assertGreater(len(entry["user_prompt"]), 5)
                self.assertGreater(len(entry["assistant_response"]), 5)
                self.assertFalse(forbidden_pattern.search(line), f"Brand leak detected: {line}")
                count += 1

        self.assertEqual(count, 100)

    def test_sha256_deduplication(self):
        h1 = compute_dialogue_hash("I feel sad today", "I hear you, it is okay to feel sad.")
        h2 = compute_dialogue_hash("  i feel SAD today  ", "I hear you, it is okay to feel sad.  ")
        self.assertEqual(h1, h2)

    def test_normalize_entry(self):
        raw = {
            "conversations": [
                {"from": "user", "value": "I am feeling anxious about work."},
                {"from": "assistant", "value": "Let's explore what specific triggers are coming up."}
            ],
            "siv_metadata": {
                "canonical_state": "anxiety_rumination",
                "intervention": "Metacognitive Observation"
            }
        }
        res = normalize_entry(raw)
        self.assertIsNotNone(res)
        self.assertEqual(res["state"], "anxiety_rumination")
        self.assertEqual(res["intervention"], "Metacognitive Observation")
        self.assertEqual(res["user_prompt"], "I am feeling anxious about work.")

if __name__ == "__main__":
    unittest.main()
