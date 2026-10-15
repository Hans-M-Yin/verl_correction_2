"""
evaluation_all_for_internvl.py
─────────────────────────────────────────────────────────────────────────────
InternVL evaluation runner using LMDeploy (TurboMind backend).

Key differences from evaluation_all.py (vLLM):
  - Uses `lmdeploy.pipeline` instead of `vllm.LLM`.
  - No manual apply_chat_template; LMDeploy handles the chat template and
    image token insertion internally.
  - Images are passed as PIL Image objects directly in the OpenAI-style
    message content — no base64 encoding, no mm_data dict needed.
  - Batch inference: pass a list of message dicts to pipe() at once.

Usage:
  cd perception/evaluation
  python evaluation_all_for_internvl.py \
      --benchmarks mathvista,mathverse \
      --checkpoint OpenGVLab/InternVL2_5-8B \
      --tensor-parallel-size 2 \
      --out-dir results_internvl \
      --api-key $OPENAI_API_KEY
"""

from __future__ import annotations

import argparse
import ast
import base64
import json
import os
import random
import time
from io import BytesIO
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Sequence

import tqdm
from codetiming import Timer
from datasets import Image, load_dataset
from openai import OpenAI

# LMDeploy imports
from lmdeploy import GenerationConfig, TurbomindEngineConfig, pipeline
from lmdeploy.vl import load_image as lmdeploy_load_image

import judge_freak
import judge_hallusionbench
import judge_logicvista
import judge_mathverse
import judge_mathvision
import judge_mathvista
import judge_mmmu
import judge_wemath

# ─────────────────────────────────────────────────────────────────────────────
# System prompts (identical to evaluation_all.py)
# ─────────────────────────────────────────────────────────────────────────────
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


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
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


# ─────────────────────────────────────────────────────────────────────────────
# Message builders
# LMDeploy's image_url.url MUST be a string (URL, file path, or data URI).
# PIL Image objects must be converted to base64 data URIs first.
# ─────────────────────────────────────────────────────────────────────────────
def _pil_to_data_uri(image) -> str:
    """Convert a PIL Image (or pass-through strings/paths) to a base64 data URI."""
    from PIL import Image as PILImage
    if isinstance(image, str):
        # Already a URL, file path, or data URI — pass through as-is
        return image
    if isinstance(image, PILImage.Image):
        buf = BytesIO()
        fmt = image.format or "PNG"
        image.save(buf, format=fmt)
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        mime = {"JPEG": "jpeg", "JPG": "jpeg", "PNG": "png",
                "GIF": "gif", "WEBP": "webp"}.get(fmt.upper(), "png")
        return f"data:image/{mime};base64,{b64}"
    # Fallback: bytes dict from HuggingFace datasets
    if isinstance(image, dict) and "bytes" in image:
        b64 = base64.b64encode(image["bytes"]).decode("utf-8")
        return f"data:image/png;base64,{b64}"
    raise ValueError(f"Unsupported image type for LMDeploy: {type(image)}")


def _img_block(image) -> dict:
    """Wrap an image (PIL, path, URL, or bytes dict) into an LMDeploy image_url block."""
    return {"type": "image_url", "image_url": {"url": _pil_to_data_uri(image)}}


def _build_mathvista_messages(data_item: dict) -> List[dict]:
    return [{"role": "user", "content": [
        _img_block(data_item["decoded_image"]),
        {"type": "text", "text": data_item["query"] + " " + SYSTEM_PROMPT},
    ]}]


def _build_mathverse_messages(data_item: dict) -> List[dict]:
    return [{"role": "user", "content": [
        _img_block(data_item["image"]),
        {"type": "text", "text": data_item["question"] + " " + SYSTEM_PROMPT},
    ]}]


def _build_mathvision_messages(data_item: dict) -> List[dict]:
    options = data_item.get("options", [])
    query_text = data_item["question"] + "\nChoices:\n" + _format_mcq_choices(options) if options else data_item["question"]
    return [{"role": "user", "content": [
        _img_block(data_item["decoded_image"]),
        {"type": "text", "text": query_text + " " + MATHVISION_WEMATH_SYSTEM_PROMPT},
    ]}]


def _build_mmmu_messages(data_item: dict) -> List[dict]:
    parsed_options = _parse_options(data_item["options"])
    question = data_item["question"] + "\nChoices:\n" + _format_mcq_choices(parsed_options)
    content = []
    for idx in range(1, 8):
        image = data_item.get(f"image_{idx}")
        if image is not None:
            content.append(_img_block(image))
    content.append({"type": "text", "text": question + " " + SYSTEM_PROMPT})
    return [{"role": "user", "content": content}]


def _build_mmmu_vision_messages(data_item: dict) -> List[dict]:
    parsed_options = _parse_options(data_item["options"])
    text_prompt = "Choices:\n" + _format_mcq_choices(parsed_options)
    return [{"role": "user", "content": [
        _img_block(data_item["image"]),
        {"type": "text", "text": text_prompt + " " + SYSTEM_PROMPT},
    ]}]


def _build_logicvista_messages(data_item: dict) -> List[dict]:
    return [{"role": "user", "content": [
        _img_block(data_item["image"]),
        {"type": "text", "text": data_item["question"] + " " + SYSTEM_PROMPT},
    ]}]


def _build_hallusionbench_messages(data_item: dict) -> List[dict]:
    prompt_text = data_item["question"] + " Answer the question with Yes or No only. " + HALLUSION_SYSTEM_PROMPT
    return [{"role": "user", "content": [
        _img_block(data_item["image"]),
        {"type": "text", "text": prompt_text},
    ]}]


def _build_freak_messages(data_item: dict) -> List[dict]:
    question = data_item["question"]
    if data_item.get("type") == "mcq" and data_item.get("options"):
        question += "\nChoices:\n" + _format_mcq_choices(data_item["options"])
    return [{"role": "user", "content": [
        _img_block(data_item["image"]),
        {"type": "text", "text": question + " " + SYSTEM_PROMPT},
    ]}]


def _build_wemath_messages(data_item: dict) -> List[dict]:
    query = data_item["question"] + "\nChoices:\n" + data_item["option"]
    return [{"role": "user", "content": [
        _img_block(data_item["image_path"]),
        {"type": "text", "text": query + " " + MATHVISION_WEMATH_SYSTEM_PROMPT},
    ]}]


# ─────────────────────────────────────────────────────────────────────────────
# Sanitization & saving (identical to evaluation_all.py)
# ─────────────────────────────────────────────────────────────────────────────
def _default_sanitize(data_item: dict, img_col: str = "image") -> dict:
    record = dict(data_item)
    record.pop(img_col, None)
    return record


def _save_mathvista_outputs(records: List[dict], output_path: str) -> None:
    output_dict = {record["pid"]: record for record in records}
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_dict, f, indent=4, ensure_ascii=False)


def _default_save_outputs(records: List[dict], output_path: str) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)


def _default_judge_output_path(result_filepath: str) -> str:
    dirname = os.path.dirname(result_filepath)
    base_name = os.path.splitext(os.path.basename(result_filepath))[0]
    return os.path.join(dirname, f"{base_name}-judge_v4.json")


def _mathvista_judge_output_path(result_filepath: str) -> str:
    dirname = os.path.dirname(result_filepath)
    filename = os.path.basename(result_filepath)
    dataset_name = filename.split("-")[0]
    return os.path.join(dirname, f"{dataset_name}-qwen_judge_results_v4.json")


def _qwen_judge_output_path(result_filepath: str) -> str:
    dirname = os.path.dirname(result_filepath)
    base_name = os.path.splitext(os.path.basename(result_filepath))[0]
    return os.path.join(dirname, f"{base_name}-qwen_judge_results_v4.json")


def _mathvision_judge_output_path(result_filepath: str) -> str:
    dirname = os.path.dirname(result_filepath)
    filename = os.path.basename(result_filepath)
    return os.path.join(dirname, f"{filename}-qwen_judge_results_v4.json")


# ─────────────────────────────────────────────────────────────────────────────
# BenchmarkSpec & registry
# ─────────────────────────────────────────────────────────────────────────────
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


BENCHMARK_SPECS: Dict[str, BenchmarkSpec] = {
    "mathvista": BenchmarkSpec(
        "mathvista", "AI4Math/MathVista", "testmini", "mathvista.json",
        _build_mathvista_messages,
        lambda x: _default_sanitize(x, "decoded_image"),
        _save_mathvista_outputs,
        judge_mathvista, judge_mathvista.read_mathvista_eval_data,
        _mathvista_judge_output_path,
        cast_image_columns=(),
    ),
    "mathverse": BenchmarkSpec(
        "mathverse", "AI4Math/MathVerse", "testmini", "mathverse_test.json",
        _build_mathverse_messages, _default_sanitize, _default_save_outputs,
        judge_mathverse, judge_mathverse.read_mathverse_eval_data,
        _default_judge_output_path,
        hf_config="testmini", cast_image_columns=("image",),
    ),
    "mathvision": BenchmarkSpec(
        "mathvision", "MathLLMs/MathVision", "test", "mathvision_test.json",
        _build_mathvision_messages,
        lambda x: _default_sanitize(x, "decoded_image"),
        _default_save_outputs,
        judge_mathvision, judge_mathvision.read_mathvision_eval_data,
        _mathvision_judge_output_path,
        cast_image_columns=("decoded_image",),
    ),
    "mmmu_pro_10": BenchmarkSpec(
        "mmmu_pro_10", "MMMU/MMMU_Pro", "test", "mmmu_pro_10.json",
        _build_mmmu_messages,
        lambda x: {k: v for k, v in x.items() if not k.startswith("image_")} | {"parsed_options": _parse_options(x["options"])},
        _default_save_outputs,
        judge_mmmu, judge_mmmu.read_mmmu_eval_data,
        _default_judge_output_path,
        hf_config="standard (10 options)", cast_image_columns=(),
    ),
    "mmmu_pro_vision": BenchmarkSpec(
        "mmmu_pro_vision", "MMMU/MMMU_Pro", "test", "mmmu_pro_vision.json",
        _build_mmmu_vision_messages,
        lambda x: {k: v for k, v in _default_sanitize(x).items()} | {"parsed_options": _parse_options(x["options"])},
        _default_save_outputs,
        judge_mmmu, judge_mmmu.read_mmmu_eval_data,
        _default_judge_output_path,
        hf_config="vision", cast_image_columns=("image",),
    ),
    "logicvista": BenchmarkSpec(
        "logicvista", "lscpku/LogicVista", "test", "logicvista_test.json",
        _build_logicvista_messages, _default_sanitize, _default_save_outputs,
        judge_logicvista, judge_logicvista.read_logicvista_eval_data,
        _default_judge_output_path,
        cast_image_columns=("image",),
    ),
    "hallusionbench": BenchmarkSpec(
        "hallusionbench", "lmms-lab/HallusionBench", "image", "hallusionbench_test.json",
        _build_hallusionbench_messages, _default_sanitize, _default_save_outputs,
        judge_hallusionbench, judge_hallusionbench.read_hallusionbench_eval_data,
        _default_judge_output_path,
        cast_image_columns=("image",),
    ),
    "wemath": BenchmarkSpec(
        "wemath", "We-Math/We-Math", "testmini", "wemath_test.json",
        _build_wemath_messages,
        lambda x: _default_sanitize(x, "image_path"),
        _default_save_outputs,
        judge_wemath, judge_wemath.read_wemath_eval_data,
        _qwen_judge_output_path,
        cast_image_columns=(),
    ),
    "freak": BenchmarkSpec(
        "freak", os.getenv("FREAK_DATASET_ID", ""), "test", "freak_test.json",
        _build_freak_messages, _default_sanitize, _default_save_outputs,
        judge_freak, judge_freak.read_freak_eval_data,
        _default_judge_output_path,
        cast_image_columns=("image",),
    ),
}


# ─────────────────────────────────────────────────────────────────────────────
# Dataset loading
# ─────────────────────────────────────────────────────────────────────────────
def load_benchmark_dataset(spec: BenchmarkSpec):
    if spec.hf_config is None:
        dataset = load_dataset(spec.hf_root)[spec.hf_split]
    else:
        dataset = load_dataset(spec.hf_root, spec.hf_config)[spec.hf_split]
    for column_name in spec.cast_image_columns:
        dataset = dataset.cast_column(column_name, Image(decode=True))
    return dataset


# ─────────────────────────────────────────────────────────────────────────────
# LMDeploy inference (replaces vLLM generate)
# ─────────────────────────────────────────────────────────────────────────────
def generate_benchmark_outputs(
    pipe,
    gen_config: GenerationConfig,
    dataset,
    spec: BenchmarkSpec,
) -> List[dict]:
    """
    Build all prompts, call LMDeploy pipeline in one batch, return output records.

    LMDeploy's pipe() accepts a list of OpenAI-style message lists and handles
    chat template application and image token insertion internally.
    """
    all_messages: List[List[dict]] = []
    sanitized_records: List[dict] = []

    for data_item in tqdm.tqdm(dataset, desc=f"Preparing {spec.name}", total=len(dataset)):
        messages = spec.prepare_messages(data_item)
        all_messages.append(messages)
        sanitized_records.append(spec.sanitize_record(data_item))

    print(f"[{spec.name}] Running LMDeploy inference on {len(all_messages)} samples...")
    with Timer(
        name=f"{spec.name} LMDeploy Perf Timer",
        text="{name}: {minutes:.2f} minutes for generation.",
    ):
        # pipe() accepts a list → processes as a batch
        responses = pipe(all_messages, gen_config=gen_config)

    outputs: List[dict] = []
    for record, response in zip(sanitized_records, responses):
        outputs.append({**record, "response": response.text})
    return outputs


# ─────────────────────────────────────────────────────────────────────────────
# Judge pipeline (identical logic to evaluation_all.py)
# ─────────────────────────────────────────────────────────────────────────────
def _run_parallel_api_calls(module, client, model_name, messages_list, parallel_num, desc):
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=parallel_num) as executor:
        results = list(tqdm.tqdm(
            executor.map(lambda msg: module.call_api(client, model_name, msg), messages_list),
            total=len(messages_list),
            desc=desc,
        ))
    return results


def run_judge_pipeline(spec, result_filepath, client, judge_model_name, judge_parallel_num):
    print(f"[{spec.name}] Loading saved results for judge: {result_filepath}")
    data_for_judge = spec.judge_read_fn([result_filepath])
    if not data_for_judge:
        raise RuntimeError(f"[{spec.name}] No data found for judge from {result_filepath}")

    extraction_messages, extraction_indices = spec.judge_module.construct_messages_for_extraction(data_for_judge)
    if extraction_messages:
        extraction_results = _run_parallel_api_calls(
            spec.judge_module, client, judge_model_name, extraction_messages,
            judge_parallel_num, f"{spec.name} Extraction",
        )
        for index, result in zip(extraction_indices, extraction_results):
            data_for_judge[index]["extracted_answer"] = result

    judge_messages = [spec.judge_module.construct_messages_for_judge(item) for item in data_for_judge]
    print(f"[{spec.name}] Judging {len(judge_messages)} samples...")
    judge_results = _run_parallel_api_calls(
        spec.judge_module, client, judge_model_name, judge_messages,
        judge_parallel_num, f"{spec.name} Judging",
    )

    judge_output_filepath = spec.judge_output_path_fn(result_filepath)
    judge_outputs = []
    for idx, response in enumerate(judge_results):
        raw_data = data_for_judge[idx]
        try:
            _, pid = raw_data["id"].split("###")
        except Exception as exc:
            raise RuntimeError(f"[{spec.name}] Invalid id format: {raw_data.get('id')}") from exc
        judge_outputs.append({
            "id": pid,
            "extracted_answer": raw_data.get("extracted_answer", raw_data["response"]),
            "judge_result": response,
        })

    with open(judge_output_filepath, "w", encoding="utf-8") as f:
        json.dump(judge_outputs, f, ensure_ascii=False, indent=4)
    print(f"[{spec.name}] Judge results saved to {judge_output_filepath}")

    num_correct = sum(1 for item in judge_outputs if item["judge_result"].strip().lower() == "true")
    num_total = len(judge_outputs)
    accuracy = (num_correct / num_total) if num_total > 0 else 0.0
    print(f"[{spec.name}] Accuracy: {num_correct}/{num_total} = {accuracy:.4f}")
    return {"judge_output_filepath": judge_output_filepath,
            "num_correct": num_correct, "num_total": num_total, "accuracy": accuracy}


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run all perception benchmarks on InternVL via LMDeploy."
    )
    parser.add_argument(
        "--benchmarks", type=str,
        default="mathvista,mathverse,mathvision,wemath,mmmu_pro_10,mmmu_pro_vision,logicvista,hallusionbench,freak",
    )
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path or HuggingFace ID for the InternVL checkpoint.")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--session-len", type=int, default=32768,
                        help="Max context window length (tokens).")
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--out-dir", type=str, default="results_internvl")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--judge-model-name", type=str, default="gpt-4o")
    parser.add_argument("--api-key", type=str, default=os.getenv("CORRECTION_PROJECT_API_KEY"),
                        help="API key for judge (defaults to --api-key).")
    parser.add_argument("--api-url", type=str, default=os.getenv("CORRECTION_PROJECT_API_URL", "https://api.openai.com/v1"),
                        help="Base URL for judge API (defaults to --api-url).")
    parser.add_argument("--judge-parallel-num", type=int, default=10)
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    benchmark_names = [n.strip().lower() for n in args.benchmarks.split(",") if n.strip()]
    if "freak" in benchmark_names and not os.getenv("FREAK_DATASET_ID"):
        raise ValueError("Set FREAK_DATASET_ID to the Hugging Face dataset id before evaluating FREAK.")
    unknown = [n for n in benchmark_names if n not in BENCHMARK_SPECS]
    if unknown:
        raise ValueError(f"Unknown benchmarks: {unknown}. Supported: {sorted(BENCHMARK_SPECS.keys())}")
    benchmark_specs = [BENCHMARK_SPECS[n] for n in benchmark_names]
    print("Selected benchmarks:", [s.name for s in benchmark_specs])

    # ── Initialize LMDeploy pipeline ─────────────────────────────────────────
    print(f"Loading model with LMDeploy: {args.checkpoint}")
    backend_config = TurbomindEngineConfig(
        tp=args.tensor_parallel_size,
        session_len=args.session_len,
    )
    pipe = pipeline(args.checkpoint, backend_config=backend_config)
    gen_config = GenerationConfig(
        temperature=0.0,
        top_k=1,
        max_new_tokens=args.max_tokens,
    )

    client = OpenAI(api_key=args.api_key, base_url=args.api_url)

    summaries = []
    for spec in benchmark_specs:
        print("=" * 100)
        print(f"Starting benchmark: {spec.name}")
        start_time = time.time()
        try:
            dataset = load_benchmark_dataset(spec)
            print(f"[{spec.name}] Loaded {len(dataset)} samples.")

            outputs = generate_benchmark_outputs(pipe, gen_config, dataset, spec)

            result_filepath = os.path.join(args.out_dir, spec.result_filename)
            os.makedirs(os.path.dirname(result_filepath), exist_ok=True)
            spec.save_outputs(outputs, result_filepath)
            print(f"[{spec.name}] Results saved to {result_filepath}")

            judge_summary = run_judge_pipeline(
                spec, result_filepath, client, args.judge_model_name, args.judge_parallel_num,
            )
            elapsed = time.time() - start_time
            print(f"[{spec.name}] Completed in {elapsed / 60:.2f} minutes.")
            summaries.append({
                "benchmark": spec.name, "status": "success",
                "result_filepath": result_filepath,
                "judge_filepath": judge_summary["judge_output_filepath"],
                "num_samples": len(outputs),
                "judge_num_correct": judge_summary["num_correct"],
                "judge_num_total": judge_summary["num_total"],
                "judge_accuracy": judge_summary["accuracy"],
                "elapsed_seconds": elapsed,
            })
        except Exception as exc:
            summaries.append({"benchmark": spec.name, "status": "failed", "error": repr(exc)})
            print(f"[{spec.name}] FAILED: {exc}")
            if not args.continue_on_error:
                print("Stopping because --continue-on-error is not set.")
                break

    summary_path = os.path.join(args.out_dir, "evaluation_internvl_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summaries, f, ensure_ascii=False, indent=2)
    print("=" * 100)
    print(f"Summary saved to {summary_path}")
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
