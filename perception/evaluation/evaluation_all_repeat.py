"""
Unified benchmark evaluation runner, including LLM-as-judge stability evaluation.

Quick start
-----------
Run this file from ``perception/evaluation`` so that the local ``judge_*.py``
modules can be imported. The following evaluates a checkpoint once and then
uses the specified judge once for each selected benchmark:

.. code-block:: bash

    cd perception/evaluation
    python evaluation_all_repeat.py \
      --checkpoint ../../models/qwen2-vl-7b \
      --tensor-parallel-size 2 \
      --benchmarks mathvista,mathverse,mathvision,wemath,mmmu_pro_10,mmmu_pro_vision,logicvista,hallusionbench,freak \
      --out-dir ./eval_results/qwen2_vl_7b \
      --judge-model-name <judge-model-name> \
      --api-key "$JUDGE_API_KEY" \
      --api-url <judge-api-base-url>

Judge stability mode
--------------------
For a rebuttal-style LLM-as-judge stability measurement, add ``--repeat N``
(normally ``N=5``). The model is generated **once** with deterministic
sampling (``temperature=0``, ``top_k=1``); the exact same saved responses are
then sent to the judge N independent times. This isolates judge randomness and
does *not* measure variation caused by sampling different model responses.

.. code-block:: bash

    cd perception/evaluation
    python evaluation_all_repeat.py \
      --checkpoint ../../models/qwen2-vl-7b \
      --tensor-parallel-size 2 \
      --benchmarks hallusionbench,freak \
      --out-dir ./eval_results/qwen2_vl_7b_judge_repeat5 \
      --judge-model-name <judge-model-name> \
      --api-key "$JUDGE_API_KEY" \
      --api-url <judge-api-base-url> \
      --judge-parallel-num 10 \
      --repeat 5

Outputs
-------
``--out-dir`` contains one generated-response file per benchmark, one judge
file per repeat (with a ``_run0`` ... ``_run<N-1>`` suffix in stability mode),
and ``stability_summary.json``. The summary reports the N judge accuracies,
their sample mean, sample standard deviation, and a 95% t confidence interval.
Without ``--repeat`` (or with ``--repeat 1``), the script instead writes
``evaluation_all_summary.json`` after a normal single evaluation run.

Scope and limitations
---------------------
This is the legacy judge-based runner. Its supported benchmarks are the names
listed by ``--help``; it does not include the deterministic MMStar, POPE, or
V*Bench evaluators added to ``evaluation_all.py``. For those benchmarks, use
``evaluation_all.py`` and estimate evaluation uncertainty with item-level
bootstrap rather than this judge-repeat mode.

Why this script exists
----------------------
The existing evaluation scripts under `perception/evaluation/` run one benchmark per process.
That means each benchmark re-initializes vLLM, which is expensive. This script keeps the original
single-benchmark scripts unchanged, but adds a unified entrypoint that:

1. Initializes one shared `LLM` instance and one shared `Qwen2_5_VLProcessor`.
2. Runs multiple benchmarks sequentially in one process.
3. Keeps result saving independent for each benchmark, matching the standalone scripts.
4. Immediately runs the corresponding judge for that benchmark after generation finishes.
5. Uses a benchmark registry so future benchmarks can be added by registering a few functions.

Important design principles
---------------------------
- Do not modify or replace the standalone benchmark scripts.
- Reuse their benchmark-specific field mappings and their judge logic.
- Only centralize orchestration, shared model initialization, and common execution utilities.
- Save outputs benchmark-by-benchmark, then judge benchmark-by-benchmark.
- If one benchmark fails, the script can either stop immediately or continue to the next one,
  depending on the CLI flag.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import random
import time

import numpy as np
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Sequence

import tqdm
from codetiming import Timer
from datasets import Image, load_dataset
from openai import OpenAI
from transformers import AutoProcessor
from vllm import LLM, SamplingParams

import judge_freak
import judge_hallusionbench
import judge_logicvista
import judge_mathverse
import judge_mathvision
import judge_mathvista
import judge_mmmu
import judge_wemath


SYSTEM_PROMPT = (
    "You FIRST think about the reasoning process as an internal monologue and then provide the final answer. "
    "The reasoning process MUST BE enclosed within <think> </think> tags, and the answer process MUST BE enclosed within <answer> </answer> tags. "
    "The final answer MUST BE put in \\boxed{} in <answer> </answer> tags."
)

HALLUSION_SYSTEM_PROMPT = (
    "You FIRST think about the reasoning process as an internal monologue and then provide the final answer. "
    "The reasoning process MUST BE enclosed within <think> </think> tags, and the answer process MUST BE enclosed within <answer> </answer> tags. "
    "The final answer MUST be either Yes or No, and MUST BE put in \\boxed{} in <answer> </answer> tags."
)

MATHVISION_WEMATH_SYSTEM_PROMPT = (
    "You FIRST think about the reasoning process as an internal monologue and then provide the final answer. "
    "The reasoning process MUST BE enclosed within <think> </think> tags, and the answer process MUST BE enclosed within <answer> </answer> tags. "
)


def _parse_options(raw_options: Any) -> List[str]:
    if isinstance(raw_options, list):
        return [str(item) for item in raw_options]
    if isinstance(raw_options, str):
        try:
            parsed = ast.literal_eval(raw_options)
            if isinstance(parsed, list):
                return [str(item) for item in parsed]
        except Exception:
            pass
    return [str(raw_options)]


def _format_mcq_choices(choices: Sequence[str]) -> str:
    return "\n".join(f"{chr(ord('A') + idx)}. {choice}" for idx, choice in enumerate(choices))


def _build_mathvista_messages(data_item: dict) -> List[dict]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": data_item["decoded_image"]},
                {"type": "text", "text": data_item["query"] + " " + SYSTEM_PROMPT},
            ],
        }
    ]


def _sanitize_mathvista_record(data_item: dict) -> dict:
    record = dict(data_item)
    record.pop("decoded_image", None)
    return record


def _save_mathvista_outputs(records: List[dict], output_path: str) -> None:
    output_dict = {}
    for record in records:
        output_dict[record["pid"]] = record
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_dict, f, indent=4, ensure_ascii=False)


def _build_mathverse_messages(data_item: dict) -> List[dict]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": data_item["image"]},
                {"type": "text", "text": data_item["question"] + " " + SYSTEM_PROMPT},
            ],
        }
    ]


def _sanitize_mathverse_record(data_item: dict) -> dict:
    record = dict(data_item)
    record.pop("image", None)
    return record


def _build_mathvision_messages(data_item: dict) -> List[dict]:
    options = data_item.get("options", [])
    if options and len(options) > 1:
        query_text = data_item["question"] + "\nChoices:\n" + _format_mcq_choices(options)
    else:
        query_text = data_item["question"]

    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": data_item["decoded_image"]},
                {"type": "text", "text": query_text + " " + MATHVISION_WEMATH_SYSTEM_PROMPT},
            ],
        }
    ]


def _sanitize_mathvision_record(data_item: dict) -> dict:
    record = dict(data_item)
    record.pop("decoded_image", None)
    return record


def _build_mmmu_messages(data_item: dict) -> List[dict]:
    parsed_options = _parse_options(data_item["options"])
    question = data_item["question"] + "\nChoices:\n" + _format_mcq_choices(parsed_options)

    content = []
    for idx in range(1, 8):
        image = data_item.get(f"image_{idx}")
        if image is not None:
            content.append({"type": "image", "image": image})
    content.append({"type": "text", "text": question + " " + SYSTEM_PROMPT})

    return [{"role": "user", "content": content}]


def _sanitize_mmmu_record(data_item: dict) -> dict:
    record = dict(data_item)
    record["parsed_options"] = _parse_options(data_item["options"])
    for idx in range(1, 8):
        record.pop(f"image_{idx}", None)
    return record


def _build_mmmu_vision_messages(data_item: dict) -> List[dict]:
    parsed_options = _parse_options(data_item["options"])
    text_prompt = "Choices:\n" + _format_mcq_choices(parsed_options)
    content = [
        {"type": "image", "image": data_item["image"]},
        {"type": "text", "text": text_prompt + " " + SYSTEM_PROMPT}
    ]
    return [{"role": "user", "content": content}]


def _sanitize_mmmu_vision_record(data_item: dict) -> dict:
    record = dict(data_item)
    record["parsed_options"] = _parse_options(data_item["options"])
    record.pop("image", None)
    return record


def _build_logicvista_messages(data_item: dict) -> List[dict]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": data_item["image"]},
                {"type": "text", "text": data_item["question"] + " " + SYSTEM_PROMPT},
            ],
        }
    ]


def _sanitize_logicvista_record(data_item: dict) -> dict:
    record = dict(data_item)
    record.pop("image", None)
    return record


def _build_hallusionbench_messages(data_item: dict) -> List[dict]:
    prompt_text = data_item["question"] + " Answer the question with Yes or No only. " + HALLUSION_SYSTEM_PROMPT
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": data_item["image"]},
                {"type": "text", "text": prompt_text},
            ],
        }
    ]


def _sanitize_hallusionbench_record(data_item: dict) -> dict:
    record = dict(data_item)
    record.pop("image", None)
    return record


def _build_freak_messages(data_item: dict) -> List[dict]:
    question = data_item["question"]
    if data_item.get("type") == "mcq" and data_item.get("options") is not None:
        question = question + "\nChoices:\n" + _format_mcq_choices(data_item["options"])
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": data_item["image"]},
                {"type": "text", "text": question + " " + SYSTEM_PROMPT},
            ],
        }
    ]


def _sanitize_freak_record(data_item: dict) -> dict:
    record = dict(data_item)
    record.pop("image", None)
    return record


def _build_wemath_messages(data_item: dict) -> List[dict]:
    query = data_item["question"] + "\nChoices:\n" + data_item["option"]
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": data_item["image_path"]},
                {"type": "text", "text": query + " " + MATHVISION_WEMATH_SYSTEM_PROMPT},
            ],
        }
    ]


def _sanitize_wemath_record(data_item: dict) -> dict:
    record = dict(data_item)
    record.pop("image_path", None)
    return record


def _default_save_outputs(records: List[dict], output_path: str) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)


def _mathvista_judge_output_path(result_filepath: str) -> str:
    dirname = os.path.dirname(result_filepath)
    filename = os.path.basename(result_filepath)
    dataset_name = filename.split("-")[0]
    return os.path.join(dirname, f"{dataset_name}-qwen_judge_results_v4.json")


def _default_judge_output_path(result_filepath: str) -> str:
    dirname = os.path.dirname(result_filepath)
    base_name = os.path.splitext(os.path.basename(result_filepath))[0]
    return os.path.join(dirname, f"{base_name}-judge_v4.json")


def _qwen_judge_output_path(result_filepath: str) -> str:
    dirname = os.path.dirname(result_filepath)
    base_name = os.path.splitext(os.path.basename(result_filepath))[0]
    return os.path.join(dirname, f"{base_name}-qwen_judge_results_v4.json")


def _mathvision_judge_output_path(result_filepath: str) -> str:
    dirname = os.path.dirname(result_filepath)
    filename = os.path.basename(result_filepath)
    return os.path.join(dirname, f"{filename}-qwen_judge_results_v4.json")


@dataclass(frozen=True)
class BenchmarkSpec:
    name: str
    hf_root: str
    hf_split: str
    result_filename: str
    prepare_messages: Callable[[dict], List[dict]]
    sanitize_record: Callable[[dict], dict]
    save_outputs: Callable[[List[dict], str], None]
    judge_module: Any
    judge_read_fn: Callable[[List[str]], List[dict]]
    judge_output_path_fn: Callable[[str], str]
    hf_config: str | None = None
    cast_image_columns: tuple[str, ...] = ("image",)
    max_images_per_prompt: int = 1


BENCHMARK_SPECS: Dict[str, BenchmarkSpec] = {
    "mathvista": BenchmarkSpec(
        name="mathvista",
        hf_root="AI4Math/MathVista",
        hf_split="testmini",
        result_filename="mathvista.json",
        prepare_messages=_build_mathvista_messages,
        sanitize_record=_sanitize_mathvista_record,
        save_outputs=_save_mathvista_outputs,
        judge_module=judge_mathvista,
        judge_read_fn=judge_mathvista.read_mathvista_eval_data,
        judge_output_path_fn=_mathvista_judge_output_path,
        cast_image_columns=(),
        max_images_per_prompt=1,
    ),
    "mathverse": BenchmarkSpec(
        name="mathverse",
        hf_root="AI4Math/MathVerse",
        hf_split="testmini",
        result_filename="mathverse_test.json",
        prepare_messages=_build_mathverse_messages,
        sanitize_record=_sanitize_mathverse_record,
        save_outputs=_default_save_outputs,
        judge_module=judge_mathverse,
        judge_read_fn=judge_mathverse.read_mathverse_eval_data,
        judge_output_path_fn=_default_judge_output_path,
        hf_config="testmini",
        cast_image_columns=("image",),
        max_images_per_prompt=1,
    ),
    "mathvision": BenchmarkSpec(
        name="mathvision",
        hf_root="MathLLMs/MathVision",
        hf_split="test",
        result_filename="mathvision_test.json",
        prepare_messages=_build_mathvision_messages,
        sanitize_record=_sanitize_mathvision_record,
        save_outputs=_default_save_outputs,
        judge_module=judge_mathvision,
        judge_read_fn=judge_mathvision.read_mathvision_eval_data,
        judge_output_path_fn=_mathvision_judge_output_path,
        cast_image_columns=("decoded_image",),
        max_images_per_prompt=1,
    ),
    "mmmu_pro_10": BenchmarkSpec(
        name="mmmu_pro_10",
        hf_root="MMMU/MMMU_Pro",
        hf_split="test",
        result_filename="mmmu_pro_10.json",
        prepare_messages=_build_mmmu_messages,
        sanitize_record=_sanitize_mmmu_record,
        save_outputs=_default_save_outputs,
        judge_module=judge_mmmu,
        judge_read_fn=judge_mmmu.read_mmmu_eval_data,
        judge_output_path_fn=_default_judge_output_path,
        hf_config="standard (10 options)",
        cast_image_columns=(),
        max_images_per_prompt=7,
    ),
    "mmmu_pro_vision": BenchmarkSpec(
        name="mmmu_pro_vision",
        hf_root="MMMU/MMMU_Pro",
        hf_split="test",
        result_filename="mmmu_pro_vision.json",
        prepare_messages=_build_mmmu_vision_messages,
        sanitize_record=_sanitize_mmmu_vision_record,
        save_outputs=_default_save_outputs,
        judge_module=judge_mmmu,
        judge_read_fn=judge_mmmu.read_mmmu_eval_data,
        judge_output_path_fn=_default_judge_output_path,
        hf_config="vision",
        cast_image_columns=("image",),
        max_images_per_prompt=1,
    ),
    "logicvista": BenchmarkSpec(
        name="logicvista",
        hf_root="lscpku/LogicVista",
        hf_split="test",
        result_filename="logicvista_test.json",
        prepare_messages=_build_logicvista_messages,
        sanitize_record=_sanitize_logicvista_record,
        save_outputs=_default_save_outputs,
        judge_module=judge_logicvista,
        judge_read_fn=judge_logicvista.read_logicvista_eval_data,
        judge_output_path_fn=_default_judge_output_path,
        cast_image_columns=("image",),
        max_images_per_prompt=1,
    ),
    "hallusionbench": BenchmarkSpec(
        name="hallusionbench",
        hf_root="lmms-lab/HallusionBench",
        hf_split="image",
        result_filename="hallusionbench_test.json",
        prepare_messages=_build_hallusionbench_messages,
        sanitize_record=_sanitize_hallusionbench_record,
        save_outputs=_default_save_outputs,
        judge_module=judge_hallusionbench,
        judge_read_fn=judge_hallusionbench.read_hallusionbench_eval_data,
        judge_output_path_fn=_default_judge_output_path,
        cast_image_columns=("image",),
        max_images_per_prompt=1,
    ),
    "wemath": BenchmarkSpec(
        name="wemath",
        hf_root="We-Math/We-Math",
        hf_split="testmini",
        result_filename="wemath_test.json",
        prepare_messages=_build_wemath_messages,
        sanitize_record=_sanitize_wemath_record,
        save_outputs=_default_save_outputs,
        judge_module=judge_wemath,
        judge_read_fn=judge_wemath.read_wemath_eval_data,
        judge_output_path_fn=_qwen_judge_output_path,
        cast_image_columns=(),
        max_images_per_prompt=1,
    ),
    "freak": BenchmarkSpec(
        name="freak",
        hf_root=os.getenv("FREAK_DATASET_ID", ""),
        hf_split="test",
        result_filename="freak_test.json",
        prepare_messages=_build_freak_messages,
        sanitize_record=_sanitize_freak_record,
        save_outputs=_default_save_outputs,
        judge_module=judge_freak,
        judge_read_fn=judge_freak.read_freak_eval_data,
        judge_output_path_fn=_default_judge_output_path,
        cast_image_columns=("image",),
        max_images_per_prompt=1,
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run all perception benchmarks with one shared vLLM instance.")
    parser.add_argument(
        "--benchmarks",
        type=str,
        default="mathvista,mathverse,mathvision,wemath,mmmu_pro_10,mmmu_pro_vision,logicvista,hallusionbench,freak",
        help="Comma-separated benchmark names. Supported: " + ", ".join(BENCHMARK_SPECS.keys()),
    )
    parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint for vLLM inference.")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--out-dir", type=str, default="results_all")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--judge-model-name", type=str, default="gpt-4o")
    parser.add_argument("--api-key", type=str, default=os.getenv("CORRECTION_PROJECT_API_KEY"))
    parser.add_argument("--api-url", type=str, default=os.getenv("CORRECTION_PROJECT_API_URL", "https://api.openai.com/v1"))
    parser.add_argument("--judge-parallel-num", type=int, default=10)
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="If set, continue to the next benchmark even if the current one fails.",
    )
    parser.add_argument(
        "--repeat", type=int, default=1, metavar="N",
        help="Run the judge pipeline N times on the same model outputs to measure LLM-as-judge stability. "
             "Generation is performed only once (deterministic at T=0). Default: 1 (normal single run).",
    )
    return parser.parse_args()


def load_benchmark_dataset(spec: BenchmarkSpec):
    if spec.hf_config is None:
        dataset = load_dataset(spec.hf_root)[spec.hf_split]
    else:
        dataset = load_dataset(spec.hf_root, spec.hf_config)[spec.hf_split]

    for column_name in spec.cast_image_columns:
        dataset = dataset.cast_column(column_name, Image(decode=True))
    return dataset


def prepare_generation_inputs(
    dataset,
    spec: BenchmarkSpec,
    processor,
) -> tuple[List[dict], List[dict]]:
    inputs: List[dict] = []
    sanitized_records: List[dict] = []

    for _, data_item in tqdm.tqdm(enumerate(dataset), desc=f"Preparing {spec.name}", total=len(dataset)):
        messages = spec.prepare_messages(data_item)
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        # 通用视觉提取：直接从 message 中拿取 PIL Image 或者路径，屏蔽掉特定模型的预处理逻辑
        image_data = []
        for msg in messages:
            if "content" in msg and isinstance(msg["content"], list):
                for item in msg["content"]:
                    if isinstance(item, dict) and item.get("type") == "image":
                        image_data.append(item["image"])
        
        # 若需要支持超过一张图片，vLLM 要求传入 list; 如果只要一张，就传本身或者单个元素的 list 都可以
        mm_data = {}
        if len(image_data) > 1:
            mm_data["image"] = image_data
        elif len(image_data) == 1:
            # 对于单图情况，原生地传单个 PIL.Image
            mm_data["image"] = image_data[0]

        inputs.append(
            {
                "prompt": prompt,
                "multi_modal_data": mm_data if mm_data else None,
            }
        )
        sanitized_records.append(spec.sanitize_record(data_item))

    return inputs, sanitized_records


def generate_benchmark_outputs(
    llm: LLM,
    sampling_params: SamplingParams,
    inputs: List[dict],
    sanitized_records: List[dict],
    spec: BenchmarkSpec,
) -> List[dict]:
    with Timer(
        name=f"{spec.name} vLLM Perf Timer",
        text="{name}: Cosume {minutes:.2f} minutes for generation.",
    ) as timer:
        model_outputs = llm.generate(inputs, sampling_params=sampling_params)
    print(f"{timer.name}: {timer.last}")

    outputs: List[dict] = []
    for record, model_output in zip(sanitized_records, model_outputs):
        outputs.append(
            {
                **record,
                "response": model_output.outputs[0].text,
            }
        )
    return outputs


def save_benchmark_outputs(spec: BenchmarkSpec, outputs: List[dict], result_filepath: str) -> None:
    os.makedirs(os.path.dirname(result_filepath), exist_ok=True)
    spec.save_outputs(outputs, result_filepath)
    print(f"[{spec.name}] Results saved to {result_filepath}")


def _run_parallel_api_calls(
    module: Any,
    client: OpenAI,
    model_name: str,
    messages_list: List[List[dict]],
    parallel_num: int,
    desc: str,
) -> List[str]:
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=parallel_num) as executor:
        results = list(
            tqdm.tqdm(
                executor.map(lambda msg: module.call_api(client, model_name, msg), messages_list),
                total=len(messages_list),
                desc=desc,
            )
        )
    return results


def run_judge_pipeline(
    spec: BenchmarkSpec,
    result_filepath: str,
    client: OpenAI,
    judge_model_name: str,
    judge_parallel_num: int,
    run_index: int | None = None,
) -> dict:
    """run_index: if set, appends _run{i} to the judge output filename (for repeat mode)."""

    print(f"[{spec.name}] Loading saved results for judge: {result_filepath}")
    data_for_judge = spec.judge_read_fn([result_filepath])
    if not data_for_judge:
        raise RuntimeError(f"[{spec.name}] No data found for judge from {result_filepath}")

    extraction_messages, extraction_indices = spec.judge_module.construct_messages_for_extraction(data_for_judge)
    if len(extraction_messages) > 0:
        print(f"[{spec.name}] Extracting answers for {len(extraction_messages)} samples using {judge_parallel_num} threads...")
        extraction_results = _run_parallel_api_calls(
            module=spec.judge_module,
            client=client,
            model_name=judge_model_name,
            messages_list=extraction_messages,
            parallel_num=judge_parallel_num,
            desc=f"{spec.name} Extraction",
        )
        for index, result in zip(extraction_indices, extraction_results):
            data_for_judge[index]["extracted_answer"] = result

    judge_messages = [spec.judge_module.construct_messages_for_judge(item) for item in data_for_judge]
    print(f"[{spec.name}] Judging {len(judge_messages)} samples using {judge_parallel_num} threads...")
    judge_results = _run_parallel_api_calls(
        module=spec.judge_module,
        client=client,
        model_name=judge_model_name,
        messages_list=judge_messages,
        parallel_num=judge_parallel_num,
        desc=f"{spec.name} Judging",
    )

    base_judge_path = spec.judge_output_path_fn(result_filepath)
    if run_index is not None:
        base, ext = os.path.splitext(base_judge_path)
        judge_output_filepath = f"{base}_run{run_index}{ext}"
    else:
        judge_output_filepath = base_judge_path
    judge_outputs = []
    for idx, response in enumerate(judge_results):
        raw_data = data_for_judge[idx]
        try:
            _, pid = raw_data["id"].split("###")
        except Exception as exc:
            raise RuntimeError(f"[{spec.name}] Invalid judge item id format: {raw_data.get('id')}") from exc

        judge_outputs.append(
            {
                "id": pid,
                "extracted_answer": raw_data.get("extracted_answer", raw_data["response"]),
                "judge_result": response,
            }
        )

    with open(judge_output_filepath, "w", encoding="utf-8") as f:
        json.dump(judge_outputs, f, ensure_ascii=False, indent=4)
    print(f"[{spec.name}] Judge results saved to {judge_output_filepath}")

    num_correct = sum(1 for item in judge_outputs if item["judge_result"].strip().lower() == "true")
    num_total = len(judge_outputs)
    accuracy = (num_correct / num_total) if num_total > 0 else 0.0
    print(f"[{spec.name}] Judge accuracy: {num_correct}/{num_total} = {accuracy:.4f}")
    return {
        "judge_output_filepath": judge_output_filepath,
        "num_correct": num_correct,
        "num_total": num_total,
        "accuracy": accuracy,
    }


def build_shared_llm(checkpoint: str, tensor_parallel_size: int) -> LLM:
    return LLM(
        model=checkpoint,
        trust_remote_code=True,
        tensor_parallel_size=tensor_parallel_size,
        limit_mm_per_prompt={"image": 7},
        gpu_memory_utilization=0.85,
        enable_prefix_caching=True,
        max_num_seqs=512,
        max_model_len=20000,
    )


def build_sampling_params() -> SamplingParams:
    return SamplingParams(
        temperature=0.0,
        top_k=1,
        n=1,
        max_tokens=4096,
        skip_special_tokens=False,
    )


def resolve_benchmark_specs(benchmark_names: str) -> List[BenchmarkSpec]:
    names = [name.strip().lower() for name in benchmark_names.split(",") if name.strip()]
    unknown = [name for name in names if name not in BENCHMARK_SPECS]
    if unknown:
        raise ValueError(f"Unknown benchmark(s): {unknown}. Supported: {sorted(BENCHMARK_SPECS.keys())}")
    return [BENCHMARK_SPECS[name] for name in names]


def run_single_benchmark(
    spec: BenchmarkSpec,
    llm: LLM,
    processor,
    sampling_params: SamplingParams,
    client: OpenAI,
    args: argparse.Namespace,
) -> dict:
    print("=" * 100)
    print(f"Starting benchmark: {spec.name}")
    start_time = time.time()

    dataset = load_benchmark_dataset(spec)
    print(f"[{spec.name}] Loaded dataset split `{spec.hf_split}` with {len(dataset)} samples.")

    inputs, sanitized_records = prepare_generation_inputs(dataset, spec, processor)
    outputs = generate_benchmark_outputs(llm, sampling_params, inputs, sanitized_records, spec)

    result_filepath = os.path.join(args.out_dir, spec.result_filename)
    save_benchmark_outputs(spec, outputs, result_filepath)
    judge_summary = run_judge_pipeline(
        spec=spec,
        result_filepath=result_filepath,
        client=client,
        judge_model_name=args.judge_model_name,
        judge_parallel_num=args.judge_parallel_num,
    )

    elapsed = time.time() - start_time
    print(f"[{spec.name}] Completed in {elapsed / 60:.2f} minutes.")
    return {
        "benchmark": spec.name,
        "status": "success",
        "result_filepath": result_filepath,
        "judge_filepath": judge_summary["judge_output_filepath"],
        "num_samples": len(outputs),
        "judge_num_correct": judge_summary["num_correct"],
        "judge_num_total": judge_summary["num_total"],
        "judge_accuracy": judge_summary["accuracy"],
        "elapsed_seconds": elapsed,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Stability helpers (for --repeat mode)
# ─────────────────────────────────────────────────────────────────────────────
def compute_stability_stats(benchmark_name: str, accuracies: list[float], num_samples: int) -> dict:
    """Mean, std, and 95% t-CI across repeat judge runs."""
    n = len(accuracies)
    mean = float(np.mean(accuracies))
    std  = float(np.std(accuracies, ddof=1)) if n > 1 else 0.0
    ci_lo = ci_hi = mean
    if n >= 2:
        try:
            from scipy import stats as _stats
            ci_lo, ci_hi = _stats.t.interval(0.95, df=n - 1, loc=mean, scale=std / np.sqrt(n))
            ci_lo, ci_hi = float(ci_lo), float(ci_hi)
        except ImportError:
            # fallback: 2-sigma interval
            ci_lo = mean - 2 * std / np.sqrt(n)
            ci_hi = mean + 2 * std / np.sqrt(n)
    return {
        "benchmark":    benchmark_name,
        "n_repeats":    n,
        "num_samples":  num_samples,
        "accuracies":   [round(a, 4) for a in accuracies],
        "mean":         round(mean, 4),
        "std":          round(std, 4),
        "ci_95_lower":  round(ci_lo, 4),
        "ci_95_upper":  round(ci_hi, 4),
    }


def run_single_benchmark_stability(
    spec: BenchmarkSpec,
    llm,
    processor,
    sampling_params,
    client: OpenAI,
    args: argparse.Namespace,
    n_repeats: int,
) -> dict:
    """Generate once, judge n_repeats times, return stability statistics."""
    print("=" * 100)
    print(f"[{spec.name}] Starting stability run (generation ×1, judge ×{n_repeats})")
    start_time = time.time()

    dataset = load_benchmark_dataset(spec)
    print(f"[{spec.name}] Loaded {len(dataset)} samples.")
    inputs, sanitized_records = prepare_generation_inputs(dataset, spec, processor)
    outputs = generate_benchmark_outputs(llm, sampling_params, inputs, sanitized_records, spec)
    result_filepath = os.path.join(args.out_dir, spec.result_filename)
    save_benchmark_outputs(spec, outputs, result_filepath)

    accuracies: list[float] = []
    for i in range(n_repeats):
        print(f"  [{spec.name}] Judge run {i + 1}/{n_repeats} ...")
        judge_summary = run_judge_pipeline(
            spec=spec,
            result_filepath=result_filepath,
            client=client,
            judge_model_name=args.judge_model_name,
            judge_parallel_num=args.judge_parallel_num,
            run_index=i,
        )
        acc = judge_summary["accuracy"]
        accuracies.append(acc)
        print(f"  [{spec.name}] Run {i + 1}: accuracy = {acc:.4f}")

    stats = compute_stability_stats(spec.name, accuracies, len(outputs))
    elapsed = time.time() - start_time
    stats["elapsed_seconds"] = round(elapsed, 1)
    print(f"  [{spec.name}] Stability summary: mean={stats['mean']:.4f}  "
          f"std={stats['std']:.4f}  "
          f"95% CI=[{stats['ci_95_lower']:.4f}, {stats['ci_95_upper']:.4f}]")
    return stats


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    benchmark_specs = resolve_benchmark_specs(args.benchmarks)
    if any(spec.name == "freak" for spec in benchmark_specs) and not os.getenv("FREAK_DATASET_ID"):
        raise ValueError("Set FREAK_DATASET_ID to the Hugging Face dataset id before evaluating FREAK.")
    print("Selected benchmarks:", [spec.name for spec in benchmark_specs])

    llm = build_shared_llm(args.checkpoint, args.tensor_parallel_size)
    processor = AutoProcessor.from_pretrained(args.checkpoint, trust_remote_code=True)
    sampling_params = build_sampling_params()
    client = OpenAI(api_key=args.api_key, base_url=args.api_url)

    # ── Stability mode (--repeat N) ────────────────────────────────────────────
    if args.repeat > 1:
        print(f"\n{'=' * 100}")
        print(f"STABILITY MODE: repeat judge ×{args.repeat} for each benchmark")
        print(f"{'=' * 100}\n")
        stability_summaries = []
        for spec in benchmark_specs:
            try:
                result = run_single_benchmark_stability(
                    spec=spec, llm=llm, processor=processor,
                    sampling_params=sampling_params, client=client,
                    args=args, n_repeats=args.repeat,
                )
                stability_summaries.append({"status": "success", **result})
            except Exception as exc:
                stability_summaries.append({"benchmark": spec.name, "status": "failed", "error": repr(exc)})
                print(f"[{spec.name}] FAILED: {exc}")
                if not args.continue_on_error:
                    break

        stability_path = os.path.join(args.out_dir, "stability_summary.json")
        with open(stability_path, "w", encoding="utf-8") as f:
            json.dump(stability_summaries, f, ensure_ascii=False, indent=2)

        print("\n" + "=" * 100)
        print("STABILITY RESULTS")
        print("=" * 100)
        header = f"{'Benchmark':<22} {'N':>4} {'Mean':>8} {'Std':>8} {'95% CI':>22}"
        print(header)
        print("-" * len(header))
        for s in stability_summaries:
            if s.get("status") == "success":
                ci_str = f"[{s['ci_95_lower']:.4f}, {s['ci_95_upper']:.4f}]"
                print(f"{s['benchmark']:<22} {s['n_repeats']:>4} {s['mean']:>8.4f} {s['std']:>8.4f} {ci_str:>22}")
            else:
                print(f"{s['benchmark']:<22}  FAILED: {s.get('error', '')}")
        print(f"\nStability summary saved to: {stability_path}")
        return

    # ── Normal single-run mode ─────────────────────────────────────────────────
    summaries = []
    for spec in benchmark_specs:
        try:
            summary = run_single_benchmark(
                spec=spec, llm=llm, processor=processor,
                sampling_params=sampling_params, client=client, args=args,
            )
            summaries.append(summary)
        except Exception as exc:
            error_summary = {"benchmark": spec.name, "status": "failed", "error": repr(exc)}
            summaries.append(error_summary)
            print(f"[{spec.name}] FAILED: {exc}")
            if not args.continue_on_error:
                print("Stopping because `--continue-on-error` is not enabled.")
                break

    summary_path = os.path.join(args.out_dir, "evaluation_all_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summaries, f, ensure_ascii=False, indent=2)
    print("=" * 100)
    print(f"Saved run summary to {summary_path}")
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
