# -*- coding: utf-8 -*-
import os
import re
import json
import tqdm
import glob
import argparse
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI


def extraction_examples():
    example_1 = """
1.
Model response: 'Rounded to two decimal places, the perimeter of the sector is approximately:\n\n(-2, 1)'
Extracted Answer: <answer>(-2, 1)</answer>
"""  # noqa

    example_2 = """
2.
Model response: 'at those points.\n\nTherefore, the correct option that represents the meaning of the intersection points of the graphs is:\n\nD. They give the solutions to the equation $f(t)=g(t)$.",'
Extracted Answer: <answer>D</answer>
"""  # noqa

    example_3 = """
3.
Model response: ' at 1 (there's a closed circle at y = 1), the range in interval notation is \\((-4, 1]\\).\n\nFinal values:\nDomain: \\((-3, 3]\\)\nRange: \\((-4, 1]\\)'
Extracted Answer: <answer>Domain: \\((-3, 3]\\)\nRange: \\((-4, 1]\\)</answer>
"""  # noqa

    example_4 = """
4.
Model response: 'As it stands, I cannot provide the correct option letter because there isn't enough information to solve for 'y'.'
Extracted Answer: <answer>null</answer>
"""  # noqa

    example_5 = """
5.
Model response: 'Given that AB = 17.6 meters, we can now substitute into the equation:\n\nd = 17.6 / cos(38\u00b0)\n\nTherefore, to one decimal place, the distance d between Ned and Bart is approximately 22.3 meters.'
Extracted answer: <answer>22.3</answer>
"""  # noqa

    example_6 = """
6.
Model response:  have all the coefficients for the quadratic function:\n\\( f(x) = ax^2 + bx + c \\)\n\\( f(x) = -1x^2 - 2x + 1 \\)\n\nTherefore, the equation for the graphed function \\( f \\) is:\n\\( f(x) = -x^2 - 2x + 1 \\)"'
Extracted answer: <answer>f(x) = -x^2 - 2x + 1</answer>
"""  # noqa

    return [example_1, example_2, example_3, example_4, example_5, example_6]


task_description = """
I am providing you a response from a model to a math problem, termed 'Model Response'. You should extract the answer from the response as 'Extracted Answer'. Directly output the extracted answer with no explanation.\n\n
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

openend_template = (
    "**Response**: {response}\n"
    "**Ground Truth Answer**: {gt_answer}\n"
)


def parse_args():
    parser = argparse.ArgumentParser(description="MathVerse API Judge ArgParser")
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
        response = dataitem['response']
        answer_catcher = re.compile(r"<answer>(.+?)</answer>", flags=re.MULTILINE | re.DOTALL)
        answer_match = re.search(answer_catcher, response)
        if answer_match is not None:
            dataitem['extracted_answer'] = answer_match.group(0)
            continue

        sys_msg = task_description + "\n" + "\n\n".join(extraction_examples())
        user_query = "Model response: " + response + "\n\n" + "Extracted answer: "
        messages.append([
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": user_query},
        ])
        extraction_data_indices.append(idx)
    return messages, extraction_data_indices


def construct_messages_for_judge(dataitem: dict):
    extracted_answer = dataitem.get('extracted_answer', dataitem['response'])
    answer_catcher = re.compile(r"<answer>(.+?)</answer>", flags=re.MULTILINE | re.DOTALL)
    answer_match = re.search(answer_catcher, extracted_answer)
    response = answer_match.group(1) if answer_match is not None else extracted_answer

    if dataitem.get('question_type') == "multi-choice":
        choices_catcher = re.compile(r"(?:Choices|Choice):(.+)", flags=re.MULTILINE | re.DOTALL)
        choices_match = re.search(choices_catcher, dataitem.get('question_for_eval', ''))
        choices_str = choices_match.group(1) if choices_match else "N/A"
        user_query = choice_template.format(
            response=response,
            choices=choices_str,
            gt_answer=dataitem['answer'],
        )
    else:  # free-form
        user_query = openend_template.format(
            response=response,
            gt_answer=dataitem['answer'],
        )
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_query},
    ]
    return messages


def read_mathverse_eval_data(filepaths: list[str]):
    all_data = []
    for filepath in tqdm.tqdm(filepaths, desc="Loading Data files"):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            # MathVerse usually has a list structure
            items = data.values() if isinstance(data, dict) else data
            for item in items:
                data_id = item.pop('sample_index', None) or item.pop('id', 'unknown')
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

    data_for_judge = read_mathverse_eval_data(filepaths=filepaths)
    if not data_for_judge:
        print("No data found in files.")
        return

    # Phase 1: Answer Extraction
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
            data_for_judge[index]['extracted_answer'] = result

    # Phase 2: Correctness Judging
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
            source_filepath, pid = raw_data['id'].split('###')
        except:
            continue

        dirname = os.path.dirname(source_filepath)
        filename = os.path.basename(source_filepath)
        base_name = os.path.splitext(filename)[0]
        output_filepath = os.path.join(dirname, f"{base_name}-judge_v4.json")

        results_to_files[output_filepath] = results_to_files.get(output_filepath, []) + [
            {
                "id": pid,
                "extracted_answer": raw_data.get('extracted_answer', raw_data['response']),
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