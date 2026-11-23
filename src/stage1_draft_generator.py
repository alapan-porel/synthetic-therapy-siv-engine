import json
import os
from tqdm import tqdm
from vllm import LLM, SamplingParams
import orjson

INPUT_FILE = "clinical_targets.jsonl"               # or clinical_targets_high_value.jsonl
OUTPUT_FILE = "stage1_drafts.jsonl"
BATCH_SIZE = 16                                 # 4090 sweet spot: 12–20
MAX_TOKENS = 300
TEMPERATURE = 0.7

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct-AWQ"

PROMPT_TEMPLATE = """
You are generating draft therapy training data.

Create:
1. A realistic emotional user concern
2. A supportive assistant reply

Rules:
- No diagnosis
- No medication advice
- No real names
- No institutions, schools, or brand references
- No program or admissions advice
- Focus on emotional experience and human struggle
- Assistant should sound warm, human, and non-clinical

Metadata inspiration only:
Title: {title}
Description: {description}
Credentials: {credentials}
Specialization: {specialization}

Return STRICT JSON:

{{
  "draft_user": "...",
  "draft_assistant": "..."
}}
"""

def build_prompt(item):
    return PROMPT_TEMPLATE.format(
        title=item.get("title", ""),
        description=item.get("description", ""),
        credentials=item.get("meta", {}).get("credentials", ""),
        specialization=", ".join(item.get("meta", {}).get("specialization", []))
    )

def main():
    llm = LLM(
        model=MODEL_NAME,
        gpu_memory_utilization=0.80,
        disable_log_stats=True
    )

    sampling_params = SamplingParams(
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS
    )

    processed = 0
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            processed = sum(1 for _ in f)

    print(f"Resuming from record {processed}")

    with open(INPUT_FILE, "r", encoding="utf-8") as fin, \
         open(OUTPUT_FILE, "a", encoding="utf-8") as fout:

        batch = []
        for i, line in enumerate(fin):
            if i < processed:
                continue

            obj = json.loads(line)
            batch.append(obj)

            if len(batch) == BATCH_SIZE:
                run_batch(batch, llm, sampling_params, fout)
                batch = []

        if batch:
            run_batch(batch, llm, sampling_params, fout)

def run_batch(batch, llm, sampling_params, fout):
    prompts = [build_prompt(x) for x in batch]
    outputs = llm.generate(prompts, sampling_params)

    for src, out in zip(batch, outputs):
        text = out.outputs[0].text.strip()

        try:
            parsed = orjson.loads(text)
            record = {
                "draft_user": parsed["draft_user"],
                "draft_assistant": parsed["draft_assistant"],
                "source": {
                    "origin": "metadata_synthesis",
                    "video_id": src.get("video_id"),
                    "title": src.get("title")
                }
            }
            fout.write(orjson.dumps(record).decode() + "\n")
        except Exception:
            # Skip bad generations silently
            pass

if __name__ == "__main__":
    main()
