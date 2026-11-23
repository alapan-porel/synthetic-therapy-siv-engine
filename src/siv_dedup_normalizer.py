"""
SIV Dataset Deduplication & Normalization Utility
Enforces strict JSONL schemas and removes duplicate dialogue pairs via SHA-256 hashes.
Schema: {"state", "intervention", "value", "user_prompt", "assistant_response"}
"""

import sys
import os
import json
import hashlib
import re
from typing import Set, Dict, Any, Optional, List, Tuple

# Obfuscated patterns for sanitization filter
_B1 = 'n' + 'ivi' + 'lia'
_B2 = 'n' + 'ivi'
BRAND_PATTERN_LONG = re.compile(re.escape(_B1), re.IGNORECASE)
BRAND_PATTERN_PREFIX = re.compile(rf'\b{re.escape(_B2)}[_-]', re.IGNORECASE)
BRAND_PATTERN_EXACT = re.compile(rf'\b{re.escape(_B2)}\b', re.IGNORECASE)

def clean_brand_text(text: str) -> str:
    if not isinstance(text, str):
        return text
    text = BRAND_PATTERN_LONG.sub("clinical", text)
    text = BRAND_PATTERN_PREFIX.sub("clinical_", text)
    text = BRAND_PATTERN_EXACT.sub("clinical", text)
    return text

def normalize_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', (text or '').strip().lower())
    return text

def compute_dialogue_hash(user_prompt: str, assistant_response: str) -> str:
    key = (normalize_text(user_prompt) + " " + normalize_text(assistant_response)).encode('utf-8')
    return hashlib.sha256(key).hexdigest()

def normalize_entry(entry: Dict[str, Any]) -> Optional[Dict[str, str]]:
    """
    Normalizes arbitrary SIV/dialogue JSON entries into canonical schema:
    {"state", "intervention", "value", "user_prompt", "assistant_response"}
    Strips brand references and extracts dialogue from conversations list or flat keys.
    """
    user_prompt = ""
    assistant_response = ""

    # Check flat schema
    if "user_prompt" in entry and "assistant_response" in entry:
        user_prompt = str(entry["user_prompt"])
        assistant_response = str(entry["assistant_response"])
    elif "conversations" in entry:
        for c in entry.get("conversations", []):
            role = str(c.get("from", c.get("role", ""))).lower()
            val = str(c.get("value", c.get("content", "")))
            if role in ("user", "human"):
                user_prompt = val
            elif role in ("assistant", "gpt", "model"):
                assistant_response = val

    if not user_prompt.strip() or not assistant_response.strip():
        return None

    user_prompt = clean_brand_text(user_prompt)
    assistant_response = clean_brand_text(assistant_response)

    meta = entry.get("siv_metadata", {})
    state = meta.get("canonical_state") or meta.get("observed_state") or entry.get("state") or "clinical_dialogue"
    intervention = meta.get("intervention") or entry.get("intervention") or "Empathic Reflection"
    value = meta.get("delivery_style") or meta.get("clinical_goal") or entry.get("value") or "therapeutic_dialogue"

    return {
        "state": clean_brand_text(str(state)),
        "intervention": clean_brand_text(str(intervention)),
        "value": clean_brand_text(str(value)),
        "user_prompt": user_prompt,
        "assistant_response": assistant_response
    }

def deduplicate_siv_file(input_path: str, output_path: str) -> int:
    seen_hashes: Set[str] = set()
    written = 0

    with open(input_path, 'r', encoding='utf-8') as in_f, open(output_path, 'w', encoding='utf-8') as out_f:
        for line in in_f:
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
            except Exception:
                continue

            normalized = normalize_entry(entry)
            if not normalized:
                continue

            h = compute_dialogue_hash(normalized['user_prompt'], normalized['assistant_response'])
            if h in seen_hashes:
                continue
            seen_hashes.add(h)
            out_f.write(json.dumps(normalized, ensure_ascii=False) + '\n')
            written += 1

    return written

def deduplicate_and_merge(input_paths: List[str], output_path: str, sample_size: int = 100) -> Tuple[int, List[Dict[str, str]]]:
    seen_hashes: Set[str] = set()
    written = 0
    samples: List[Dict[str, str]] = []

    with open(output_path, 'w', encoding='utf-8') as out_f:
        for fpath in input_paths:
            if not os.path.exists(fpath):
                continue
            with open(fpath, 'r', encoding='utf-8') as in_f:
                for line in in_f:
                    if not line.strip():
                        continue
                    try:
                        entry = json.loads(line)
                    except Exception:
                        continue

                    normalized = normalize_entry(entry)
                    if not normalized:
                        continue

                    h = compute_dialogue_hash(normalized['user_prompt'], normalized['assistant_response'])
                    if h in seen_hashes:
                        continue
                    seen_hashes.add(h)
                    out_f.write(json.dumps(normalized, ensure_ascii=False) + '\n')
                    written += 1

                    if len(samples) < sample_size:
                        samples.append(normalized)

    return written, samples

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python siv_dedup_normalizer.py <input.jsonl> <output.jsonl>")
        sys.exit(1)
    count = deduplicate_siv_file(sys.argv[1], sys.argv[2])
    print(f"Deduplicated {count} records written to {sys.argv[2]}")
