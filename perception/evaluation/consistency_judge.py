import os
import re
import json
import tqdm
import argparse
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI

# Judge Prompt
SYSTEM_PROMPT = """You are a rigorous consistency auditor. Your task is to determine if a model's 'internal monologue' (<think> tag) is logically consistent with its 'final response' (<answer> tag).

Rules for Inconsistency:
1. If the <think> block concludes that the answer should be a specific option (e.g., 'C' or '145') but the <answer> block says a different option or value (e.g., 'B' or '140'), it is INCONSISTENT.
2. If the <think>    block performs a calculation reaching one numerical value but the <answer> block provides a different one, it is INCONSISTENT.
3. Pay close attention to mapping: if choices are (A) 10, (B) 20, (C) 30, and <think> says "the answer is 30" but <answer> says "A", it is INCONSISTENT.

Rules for Consistency:
1. If both blocks point to the same final choice or value, it is CONSISTENT.
2. Minor formatting differences or additional commentary in <answer> that doesn't contradict <think> are CONSISTENT.

Your output MUST following this format:
Status: [consistent or inconsistent]
Explanation: [A brief explanation of why it is consistent or inconsistent, citing specific parts of the think and answer blocks if necessary]
"""

USER_PROMPT_TEMPLATE = """Question: {question}
Model Response: {response}

Is the model consistent? (consistent/inconsistent)"""

def parse_args():
    parser = argparse.ArgumentParser(description="Consistency Judge for LLM outputs")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the mathvista.json file")
    parser.add_argument("--output_file", type=str, default="consistency_results.json", help="Path to save results")
    parser.add_argument("--model_name", type=str, default="gpt-4o", help="Judge model name")
    parser.add_argument("--api_key", type=str, default="sk-Eek0pnQdHdSPIvwfYMvDoCWoI0H6QoakugWZfCViQeWJlIsD", help="API key")
    parser.add_argument("--api_url", type=str, default="https://yunwu.ai/v1", help="API base URL")
    parser.add_argument("--parallel_num", type=int, default=20, help="Number of parallel requests")
    return parser.parse_args()

def call_api(client, model, messages, max_tokens=512, temperature=0.0):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"API call failed: {e}")
        return "ERROR"

def main():
    args = parse_args()
    client = OpenAI(api_key=args.api_key, base_url=args.api_url)

    if not os.path.exists(args.input_file):
        print(f"File not found: {args.input_file}")
        return

    with open(args.input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Convert to list for easier processing
    items = []
    if isinstance(data, dict):
        for pid, item in data.items():
            item['id'] = pid
            items.append(item)
    else:
        items = data

    print(f"Loaded {len(items)} samples.")

    def process_item(item):
        question = item.get('query', '')
        choices = item.get('choices', 'None')
        response = item.get('response', '')
        
        user_prompt = USER_PROMPT_TEMPLATE.format(
            question=question,
            choices=choices,
            response=response
        )
        
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ]
        
        raw_result = call_api(client, args.model_name, messages)
        
        # Parse status and explanation
        status = "unknown"
        explanation = raw_result
        
        status_match = re.search(r"Status:\s*(consistent|inconsistent)", raw_result, re.IGNORECASE)
        if status_match:
            status = status_match.group(1).lower()
        elif "inconsistent" in raw_result.lower():
            status = "inconsistent"
        elif "consistent" in raw_result.lower():
            status = "consistent"
            
        explanation_match = re.search(r"Explanation:\s*(.*)", raw_result, re.IGNORECASE | re.DOTALL)
        if explanation_match:
            explanation = explanation_match.group(1).strip()
            
        return {
            "id": item.get('id', 'unknown'),
            "consistency": status,
            "explanation": explanation,
            "raw_judge_output": raw_result
        }

    print(f"Judging consistency for {len(items)} samples using {args.parallel_num} threads...")
    
    results = []
    with ThreadPoolExecutor(max_workers=args.parallel_num) as executor:
        for res in tqdm.tqdm(executor.map(process_item, items), total=len(items)):
            results.append(res)

    # Count results
    inconsistent_count = sum(1 for r in results if r['consistency'] == 'inconsistent')
    consistent_count = sum(1 for r in results if r['consistency'] == 'consistent')
    unknown_count = sum(1 for r in results if r['consistency'] == 'unknown')
    error_count = sum(1 for r in results if r['consistency'] == 'error') or sum(1 for r in results if "ERROR" in r['raw_judge_output'])

    print("-" * 30)
    print(f"Judging finished.")
    print(f"Consistent: {consistent_count}")
    print(f"Inconsistent: {inconsistent_count}")
    print(f"Unknown/Error: {unknown_count + error_count}")
    print(f"Inconsistency rate: {inconsistent_count / len(items):.2%}")
    print("-" * 30)

    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    print(f"Results saved to {args.output_file}")

if __name__ == "__main__":
    main()
