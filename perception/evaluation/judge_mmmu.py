import os
import re
import json
import tqdm
import argparse
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI


def extraction_examples():
    example_1 = """
Hint: Please answer the question and provide the correct option letter, e.g., A, B, C, D, at the end.\n
Question: Which option best matches the figure?\n
Choices: A. Circle B. Triangle C. Square D. Pentagon\n
Model response: After checking the figure, the correct answer is C. Square.\n
Extracted answer: <answer>C</answer>
"""

    example_2 = """
Hint: Please answer the question and provide the final option letters separated by commas, e.g., A, C, at the end.\n
Question: Which options are correct?\n
Choices: A. 1 B. 2 C. 3 D. 4\n
Model response: The correct choices are A and C.\n
Extracted answer: <answer>A, C</answer>
"""

    return [example_1, example_2]


task_description = """
Please read the following example.
Then extract the answer from the model response and type it at the end of the prompt.\n
"""

system_message = (
    "You are a helpful assistant. Your task is to judge if the **Response** is correct based on the **Ground Truth Answer**. "
    "Reply 'true' if the **Response** is correct, otherwise 'false'. Do not need to explain, just 'true' or 'false'."
)

choice_template = (
    "**Choices**: {choices}\n"
    "**Response**: {response}\n"
    "**Ground Truth Answer**: {gt_answer}\n"
)


def parse_args():
    parser = argparse.ArgumentParser(description="MMMU API Judge ArgParser")
    parser.add_argument("--model_name", "--model-name", type=str, default="gpt-4o", help="Model name for API")
    parser.add_argument("--api_key", type=str, required=True, help="API key")
    parser.add_argument("--api_url", type=str, default="https://api.openai.com/v1", help="API base URL")
    parser.add_argument("--eval_basedir", "--eval-basedir", type=str, default="",
                        help="Base directory to scan for json files")
    parser.add_argument("--eval_file", "--eval-file", type=str, default="",
                        help="Directly specify a single json file to judge")
    parser.add_argument("--parallel_num", "--parallel-num", type=int, default=10, help="Number of parallel requests")
    return parser.parse_args()


def call_api(client, model, messages, max_tokens=128, temperature=0.0):
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


def construct_messages_for_extraction(all_data: list):
    extraction_data_indices = []
    messages = []
    for idx, dataitem in enumerate(all_data):
        response = dataitem["response"]
        answer_catcher = re.compile(r"<answer>(.+?)</answer>", flags=re.MULTILINE | re.DOTALL)
        answer_match = re.search(answer_catcher, response)
        if answer_match is not None:
            dataitem["extracted_answer"] = answer_match.group(0)
            continue

        sys_msg = task_description + "\n" + "\n".join(extraction_examples())
        user_query = dataitem["question"] + "\n\n" + "Model response: " + response + "\n\n" + "Extracted answer: "
        messages.append([
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": user_query},
        ])
        extraction_data_indices.append(idx)
    return messages, extraction_data_indices


def construct_messages_for_judge(dataitem: dict):
    def make_choice_string(choices: list):
        choice_string = []
        for idx, choice in enumerate(choices):
            choice_string.append(f"{chr(ord('A') + idx)}. {choice}")
        return "\n".join(choice_string)

    extracted_answer = dataitem.get("extracted_answer", dataitem["response"])
    answer_catcher = re.compile(r"<answer>(.+?)</answer>", flags=re.MULTILINE | re.DOTALL)
    answer_match = re.search(answer_catcher, extracted_answer)
    response = answer_match.group(1) if answer_match is not None else extracted_answer

    user_query = choice_template.format(
        response=response,
        choices=make_choice_string(dataitem["parsed_options"]),
        gt_answer=dataitem["answer"],
    )
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_query},
    ]
    return messages


def read_mmmu_eval_data(filepaths: list[str]):
    all_data = []
    for filepath in tqdm.tqdm(filepaths, desc="Loading Data files"):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            items = data.values() if isinstance(data, dict) else data
            for item in items:
                data_id = item.pop("id", "unknown")
                all_data.append({
                    "id": filepath + "###" + str(data_id),
                    **item,
                })
        except Exception as e:
            print(f"Error reading {filepath}: {e}")
    return all_data


def main(args):
    client = OpenAI(api_key=args.api_key, base_url=args.api_url)

    filepaths = []
    if args.eval_file:
        if os.path.exists(args.eval_file):
            filepaths.append(args.eval_file)
        else:
            print(f"File not found: {args.eval_file}")
            return
    elif args.eval_basedir:
        for root, dirs, files in os.walk(args.eval_basedir):
            for filename in files:
                if filename.endswith(".json") and not filename.endswith("judge_results_v4.json"):
                    filepaths.append(os.path.join(root, filename))
    else:
        print("Please provide either --eval_basedir or --eval_file")
        return

    def _filter_func(path):
        dirname = os.path.dirname(path)
        base_name = os.path.splitext(os.path.basename(path))[0]
        if os.path.exists(os.path.join(dirname, f"{base_name}-judge_v4.json")):
            return False
        return True

    if not args.eval_file:
        filepaths = list(filter(_filter_func, filepaths))

    if not filepaths:
        print("No new files to judge.")
        return

    data_for_judge = read_mmmu_eval_data(filepaths=filepaths)
    if not data_for_judge:
        print("No data found in files.")
        return

    extraction_messages, extraction_indices = construct_messages_for_extraction(data_for_judge)
    if len(extraction_messages) > 0:
        print(f"Extracting answers for {len(extraction_messages)} samples using {args.parallel_num} threads...")
        with ThreadPoolExecutor(max_workers=args.parallel_num) as executor:
            extraction_results = list(tqdm.tqdm(
                executor.map(lambda msg: call_api(client, args.model_name, msg), extraction_messages),
                total=len(extraction_messages),
                desc="Extraction"
            ))
        for index, result in zip(extraction_indices, extraction_results):
            data_for_judge[index]["extracted_answer"] = result

    judge_messages = [construct_messages_for_judge(item) for item in data_for_judge]
    print(f"Judging {len(judge_messages)} samples using {args.parallel_num} threads...")
    with ThreadPoolExecutor(max_workers=args.parallel_num) as executor:
        judge_results = list(tqdm.tqdm(
            executor.map(lambda msg: call_api(client, args.model_name, msg), judge_messages),
            total=len(judge_messages),
            desc="Judging"
        ))

    results_to_files = {}
    for idx, response in enumerate(judge_results):
        raw_data = data_for_judge[idx]
        try:
            source_filepath, pid = raw_data["id"].split("###")
        except Exception:
            continue

        dirname = os.path.dirname(source_filepath)
        filename = os.path.basename(source_filepath)
        base_name = os.path.splitext(filename)[0]
        output_filepath = os.path.join(dirname, f"{base_name}-judge_v4.json")

        results_to_files[output_filepath] = results_to_files.get(output_filepath, []) + [
            {
                "id": pid,
                "extracted_answer": raw_data.get("extracted_answer", raw_data["response"]),
                "judge_result": response
            }
        ]

    for filepath, results in results_to_files.items():
        print(f"Saving results to {filepath}")
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    args = parse_args()
    main(args)
