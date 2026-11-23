"""
SIV Dataset Deduplication & Normalization Utility
Enforces strict JSONL schemas and removes duplicate dialogue pairs via SHA-256 hashes.
"""

import sys
import json
import hashlib
import re
from typing import Set

def normalize_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text.strip().lower())
    return text

def compute_dialogue_hash(user_prompt: str, assistant_response: str) -> str:
    key = (normalize_text(user_prompt) + " " + normalize_text(assistant_response)).encode('utf-8')
    return hashlib.sha256(key).hexdigest()

def deduplicate_siv_file(input_path: str, output_path: str) -> int:
    seen_hashes: Set[str] = set()
    written = 0

    with open(input_path, 'r', encoding='utf-8') as in_f, open(output_path, 'w', encoding='utf-8') as out_f:
        for line in in_f:
            if not line.strip():
                continue
            entry = json.loads(line)
            u = entry.get('user_prompt', '')
            a = entry.get('assistant_response', '')
            if not u or not a:
                continue

            h = compute_dialogue_hash(u, a)
            if h in seen_hashes:
                continue
            seen_hashes.add(h)
            out_f.write(json.dumps(entry, ensure_ascii=False) + '\n')
            written += 1

    return written

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python siv_dedup_normalizer.py <input.jsonl> <output.jsonl>")
        sys.exit(1)
    count = deduplicate_siv_file(sys.argv[1], sys.argv[2])
    print(f"Deduplicated {count} records written to {sys.argv[2]}")
