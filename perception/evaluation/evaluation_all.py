"""
Unified benchmark evaluation runner.

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
import glob
import json
import os
import random
import re
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence

import tqdm
from codetiming import Timer
from datasets import Image, load_dataset
from openai import OpenAI
from qwen_vl_utils import process_vision_info
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


def _build_mmstar_messages(data_item: dict) -> List[dict]:
    """MMStar stores its choices directly in ``question``."""
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": data_item["image"]},
                {
                    "type": "text",
                    "text": data_item["question"] + " " + SYSTEM_PROMPT,
                },
            ],
        }
    ]


def _sanitize_mmstar_record(data_item: dict) -> dict:
    record = dict(data_item)
    record.pop("image", None)
    return record


def _build_pope_messages(data_item: dict) -> List[dict]:
    # Official POPE uses ``text``; lmms-lab's faithful HF formatting uses
    # ``question``.  Support both without changing the evaluation protocol.
    question = str(data_item.get("text", data_item.get("question", ""))).strip()
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": data_item["image"]},
                {
                    "type": "text",
                    "text": question + " Answer Yes or No only. " + HALLUSION_SYSTEM_PROMPT,
                },
            ],
        }
    ]


def _sanitize_pope_record(data_item: dict) -> dict:
    record = dict(data_item)
    record.pop("image", None)
    return record


def _build_vstar_messages(data_item: dict) -> List[dict]:
    question = data_item["question"]
    if data_item.get("options"):
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


def _sanitize_vstar_record(data_item: dict) -> dict:
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


def _load_pope_dataset(args: argparse.Namespace) -> List[dict]:
    """Load the three official MSCOCO POPE splits from a local POPE checkout.

    The official repository distributes annotation jsonl files separately from
    COCO images.  Keeping both locations explicit prevents accidental use of a
    third-party reformat with a different protocol.
    """
    requested_splits = _parse_pope_splits(args.pope_splits)
    if not args.pope_annotation_dir and not args.pope_image_dir:
        # This is the lmms-eval formatting of the original POPE benchmark. It
        # includes images, so it needs neither a separate COCO download nor a
        # preprocessing step.
        dataset = load_dataset("lmms-lab/POPE", "default")["test"]
        dataset = dataset.cast_column("image", Image(decode=True))
        return [
            {
                **dict(item),
                "text": item["question"],
                "label": str(item["answer"]).strip().lower(),
                "pope_split": item["category"],
            }
            for item in dataset
            if str(item["category"]).strip().lower() in requested_splits
        ]
    if not args.pope_annotation_dir or not args.pope_image_dir:
        raise ValueError(
            "Set both --pope-annotation-dir and --pope-image-dir, or omit both to load lmms-lab/POPE directly."
        )

    records: List[dict] = []
    for split_name in ("random", "popular", "adversarial"):
        if split_name not in requested_splits:
            continue
        candidates = [
            os.path.join(args.pope_annotation_dir, f"coco_pope_{split_name}.json"),
            os.path.join(args.pope_annotation_dir, f"coco_pope_{split_name}.jsonl"),
            os.path.join(args.pope_annotation_dir, f"pope_{split_name}.json"),
            os.path.join(args.pope_annotation_dir, f"pope_{split_name}.jsonl"),
        ]
        annotation_path = next((path for path in candidates if os.path.isfile(path)), None)
        if annotation_path is None:
            raise FileNotFoundError(
                f"Could not find the official POPE {split_name} annotation in {args.pope_annotation_dir}."
            )
        with open(annotation_path, "r", encoding="utf-8") as f:
            for line_number, line in enumerate(f, start=1):
                if not line.strip():
                    continue
                item = json.loads(line)
                image_name = item.get("image")
                if not image_name:
                    raise ValueError(f"Missing `image` in {annotation_path}:{line_number}")
                image_path = image_name if os.path.isabs(image_name) else os.path.join(args.pope_image_dir, image_name)
                if not os.path.isfile(image_path):
                    raise FileNotFoundError(f"POPE image not found: {image_path}")
                label = str(item.get("label", "")).strip().lower()
                if label not in {"yes", "no"}:
                    raise ValueError(f"Unexpected POPE label {label!r} in {annotation_path}:{line_number}")
                records.append({
                    **item,
                    "id": item.get("question_id", f"{split_name}-{line_number}"),
                    "image": image_path,
                    "label": label,
                    "pope_split": split_name,
                })
    return records


def _parse_pope_splits(raw_splits: str) -> set[str]:
    valid_splits = {"random", "popular", "adversarial"}
    requested_splits = {item.strip().lower() for item in raw_splits.split(",") if item.strip()}
    if not requested_splits:
        raise ValueError("--pope-splits must contain at least one split.")
    unknown_splits = requested_splits - valid_splits
    if unknown_splits:
        raise ValueError(f"Unknown POPE split(s): {sorted(unknown_splits)}. Supported: {sorted(valid_splits)}")
    return requested_splits


def _load_vstar_dataset(args: argparse.Namespace) -> List[dict]:
    """Load V*Bench from the official repository layout.

    Each annotation JSON lives beside its image; option 0 is the official
    correct option.  The loader intentionally discovers every subtask folder
    rather than relying on private filenames from a particular checkout.
    """
    if not args.vstar_data_dir:
        raise ValueError("V*Bench requires --vstar-data-dir pointing to the official benchmark folder.")
    root_dir = os.path.abspath(args.vstar_data_dir)
    manifest_path = os.path.join(root_dir, "test_questions.jsonl")
    records: List[dict] = []

    # The current HF release exposes a top-level 191-item manifest. Prefer it
    # because it retains official answer labels and defines the official V*
    # test set (direct_attributes and relative_position); auxiliary folders
    # such as OCR and GPT4V-hard are intentionally not mixed into its score.
    if os.path.isfile(manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as f:
            for line_number, line in enumerate(f, start=1):
                if not line.strip():
                    continue
                item = json.loads(line)
                # craigwu/vstar_bench uses ``text``; older manifests use
                # ``question`` or ``query``.  The supplied text already
                # contains the official choices and direct-letter instruction.
                question = item.get("question", item.get("query", item.get("text")))
                if not question:
                    raise ValueError(f"Missing question/query in {manifest_path}:{line_number}")
                image_path = _resolve_vstar_image(root_dir, item.get("images", item.get("image")), item)
                answer = _normalize_vstar_answer(item.get("answer", item.get("label")))
                if answer is None:
                    raise ValueError(f"Missing or invalid answer in {manifest_path}:{line_number}")
                category = str(item.get("category") or _infer_vstar_category(image_path, root_dir))
                records.append({
                    **item,
                    "id": str(item.get("id", item.get("question_id", line_number))),
                    "question": question,
                    "image": image_path,
                    "answer": answer,
                    "category": category,
                })
    else:
        # Older official layout: one annotation JSON beside each image, where
        # option 0 is the correct option by convention.
        annotation_paths = sorted(glob.glob(os.path.join(root_dir, "**", "*.json"), recursive=True))
        for annotation_path in annotation_paths:
            if "/.cache/" in annotation_path:
                continue
            with open(annotation_path, "r", encoding="utf-8") as f:
                item = json.load(f)
            if not isinstance(item, dict) or "question" not in item or "options" not in item:
                continue
            image_path = _resolve_vstar_image(root_dir, None, item, annotation_path=annotation_path)
            options = item["options"]
            if not isinstance(options, list) or len(options) < 2:
                raise ValueError(f"Invalid options in V*Bench annotation {annotation_path}")
            records.append({
                **item,
                "id": os.path.relpath(annotation_path, root_dir),
                "image": image_path,
                "answer": "A",  # Official legacy V*Bench convention.
                "category": _infer_vstar_category(annotation_path, root_dir),
            })
    if not records:
        raise ValueError(
            f"No valid V*Bench samples found under {root_dir}. "
            "The directory currently appears to contain only download metadata; "
            "complete the Hugging Face snapshot download so images and annotations are present."
        )
    return records


def _normalize_vstar_answer(raw_answer: Any) -> str | None:
    if raw_answer is None:
        return None
    match = re.search(r"\b([A-Z])\b", str(raw_answer).upper())
    return match.group(1) if match else None


def _infer_vstar_category(path: str, root_dir: str) -> str:
    official_categories = {"direct_attributes", "relative_position", "OCR", "GPT4V-hard"}
    relative_parts = os.path.relpath(path, root_dir).split(os.sep)
    return next((part for part in relative_parts if part in official_categories), os.path.basename(os.path.dirname(path)))


def _resolve_vstar_image(
    root_dir: str,
    image_value: Any,
    item: dict,
    annotation_path: str | None = None,
) -> str:
    """Resolve image references in both V*Bench release formats."""
    image_references = image_value if isinstance(image_value, list) else [image_value]
    image_references = [reference for reference in image_references if reference]
    if annotation_path is not None:
        stem = os.path.splitext(annotation_path)[0]
        image_references.extend(stem + extension for extension in (".jpg", ".JPG", ".jpeg", ".png", ".webp"))

    category = item.get("category")
    for reference in image_references:
        reference = str(reference)
        candidates = [
            reference,
            os.path.join(root_dir, reference),
            os.path.join(root_dir, os.path.basename(reference)),
        ]
        if category:
            candidates.append(os.path.join(root_dir, str(category), os.path.basename(reference)))
        # Manifest paths from the original evaluation setup can include an
        # unrelated prefix such as ../tasks/vstar/processed/<category>/file.
        for official_category in ("direct_attributes", "relative_position", "OCR", "GPT4V-hard"):
            candidates.append(os.path.join(root_dir, official_category, os.path.basename(reference)))
        for candidate in candidates:
            if os.path.isfile(candidate):
                return candidate
    raise FileNotFoundError(
        f"Could not resolve V*Bench image {image_references!r} under {root_dir}."
    )


def _extract_final_answer(response: str) -> str:
    """Prefer the final answer field so letters in chain-of-thought are ignored."""
    boxed = re.findall(r"\\boxed\s*\{\s*([^}]*)\s*\}", response, flags=re.IGNORECASE | re.DOTALL)
    if boxed:
        return boxed[-1].strip()
    answer_tags = re.findall(r"<answer>\s*(.*?)\s*</answer>", response, flags=re.IGNORECASE | re.DOTALL)
    if answer_tags:
        return answer_tags[-1].strip()
    return response.strip()


def _extract_choice(response: str) -> str | None:
    final_answer = _extract_final_answer(response)
    matches = re.findall(r"\b([A-Z])\b", final_answer.upper())
    return matches[-1] if matches else None


def _extract_yes_no(response: str) -> str | None:
    final_answer = _extract_final_answer(response).lower()
    matches = re.findall(r"\b(yes|no)\b", final_answer)
    return matches[-1] if matches else None


def _write_local_evaluation(result_filepath: str, benchmark: str, metrics: dict, per_sample: List[dict]) -> dict:
    output_path = os.path.splitext(result_filepath)[0] + "-metrics.json"
    payload = {"benchmark": benchmark, "metrics": metrics, "per_sample": per_sample}
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return {"evaluation_filepath": output_path, **metrics}


def _evaluate_mcq_outputs(outputs: List[dict], benchmark: str, category_key: str = "category") -> dict:
    per_sample = []
    category_counts: dict[str, list[int]] = {}
    for item in outputs:
        prediction = _extract_choice(item["response"])
        target = str(item["answer"]).strip().upper()
        correct = prediction == target
        category = str(item.get(category_key, "overall"))
        category_counts.setdefault(category, [0, 0])
        category_counts[category][0] += int(correct)
        category_counts[category][1] += 1
        per_sample.append({"id": item.get("id", item.get("index")), "prediction": prediction, "target": target, "correct": correct, "category": category})
    total = len(per_sample)
    correct_total = sum(int(item["correct"]) for item in per_sample)
    metrics = {
        "num_correct": correct_total,
        "num_total": total,
        "accuracy": correct_total / total if total else 0.0,
        "by_category": {key: count[0] / count[1] for key, count in sorted(category_counts.items())},
    }
    return {"metrics": metrics, "per_sample": per_sample}


def _evaluate_mmstar_outputs(outputs: List[dict]) -> dict:
    return _evaluate_mcq_outputs(outputs, "mmstar", category_key="category")


def _evaluate_vstar_outputs(outputs: List[dict]) -> dict:
    return _evaluate_mcq_outputs(outputs, "vstar", category_key="category")


def _pope_metrics_from_counts(counts: dict[str, int]) -> dict:
    tp, fp, tn, fn = (counts[key] for key in ("tp", "fp", "tn", "fn"))
    total = tp + fp + tn + fn
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "num_correct": tp + tn,
        "num_total": total,
        "accuracy": (tp + tn) / total if total else 0.0,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "yes_ratio": (tp + fp) / total if total else 0.0,
        "confusion_matrix": dict(counts),
    }


def _evaluate_pope_outputs(outputs: List[dict]) -> dict:
    counts = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
    per_sample = []
    by_split_counts: dict[str, dict[str, int]] = {}
    for item in outputs:
        prediction = _extract_yes_no(item["response"])
        target = str(item["label"]).lower()
        correct = prediction == target
        if target == "yes":
            if prediction == "yes":
                outcome = "tp"
            else:
                outcome = "fn"
        else:
            if prediction == "yes":
                outcome = "fp"
            else:
                outcome = "tn"
        split = item["pope_split"]
        counts[outcome] += 1
        by_split_counts.setdefault(split, {"tp": 0, "fp": 0, "tn": 0, "fn": 0})
        by_split_counts[split][outcome] += 1
        per_sample.append({"id": item.get("id"), "prediction": prediction, "target": target, "correct": correct, "split": split})
    metrics = _pope_metrics_from_counts(counts)
    by_split = {key: _pope_metrics_from_counts(split_counts) for key, split_counts in sorted(by_split_counts.items())}
    metrics["by_split"] = by_split
    metrics["macro_average"] = {
        metric_name: sum(split_metrics[metric_name] for split_metrics in by_split.values()) / len(by_split)
        if by_split else 0.0
        for metric_name in ("accuracy", "precision", "recall", "f1", "yes_ratio")
    }
    return {"metrics": metrics, "per_sample": per_sample}


@dataclass(frozen=True)
class BenchmarkSpec:
    name: str
    hf_root: str
    hf_split: str
    result_filename: str
    prepare_messages: Callable[[dict], List[dict]]
    sanitize_record: Callable[[dict], dict]
    save_outputs: Callable[[List[dict], str], None]
    judge_module: Any | None
    judge_read_fn: Callable[[List[str]], List[dict]] | None
    judge_output_path_fn: Callable[[str], str] | None
    hf_config: str | None = None
    cast_image_columns: tuple[str, ...] = ("image",)
    max_images_per_prompt: int = 1
    dataset_loader: Optional[Callable[[argparse.Namespace], Any]] = None
    local_evaluator: Optional[Callable[[List[dict]], dict]] = None


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
    "mmstar": BenchmarkSpec(
        name="mmstar",
        hf_root="Lin-Chen/MMStar",
        hf_split="val",
        result_filename="mmstar_val.json",
        prepare_messages=_build_mmstar_messages,
        sanitize_record=_sanitize_mmstar_record,
        save_outputs=_default_save_outputs,
        judge_module=None,
        judge_read_fn=None,
        judge_output_path_fn=None,
        cast_image_columns=("image",),
        max_images_per_prompt=1,
        local_evaluator=_evaluate_mmstar_outputs,
    ),
    "pope": BenchmarkSpec(
        name="pope",
        hf_root="lmms-lab/POPE",
        hf_split="test",
        result_filename="pope.json",
        prepare_messages=_build_pope_messages,
        sanitize_record=_sanitize_pope_record,
        save_outputs=_default_save_outputs,
        judge_module=None,
        judge_read_fn=None,
        judge_output_path_fn=None,
        cast_image_columns=(),
        max_images_per_prompt=1,
        dataset_loader=_load_pope_dataset,
        local_evaluator=_evaluate_pope_outputs,
    ),
    "vstar": BenchmarkSpec(
        name="vstar",
        hf_root="",
        hf_split="",
        result_filename="vstar.json",
        prepare_messages=_build_vstar_messages,
        sanitize_record=_sanitize_vstar_record,
        save_outputs=_default_save_outputs,
        judge_module=None,
        judge_read_fn=None,
        judge_output_path_fn=None,
        cast_image_columns=(),
        max_images_per_prompt=1,
        dataset_loader=_load_vstar_dataset,
        local_evaluator=_evaluate_vstar_outputs,
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
    parser.add_argument("--judge-temperature", type=float, default=0.0,
                        help="Temperature for answer extraction and LLM judging (default keeps legacy behavior).")
    parser.add_argument(
        "--pope-annotation-dir",
        type=str,
        default=None,
        help="Optional official local annotation directory. Omit with --pope-image-dir to load lmms-lab/POPE directly.",
    )
    parser.add_argument(
        "--pope-image-dir",
        type=str,
        default=None,
        help="Optional matching official MSCOCO image directory; must be used together with --pope-annotation-dir.",
    )
    parser.add_argument(
        "--pope-splits",
        type=str,
        default="random,popular,adversarial",
        help="Comma-separated POPE splits to run: random,popular,adversarial (default: all three).",
    )
    parser.add_argument(
        "--vstar-data-dir",
        type=str,
        default=None,
        help="Official V*Bench directory; annotation JSON files must be next to their paired images.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="If set, continue to the next benchmark even if the current one fails.",
    )
    return parser.parse_args()


def load_benchmark_dataset(spec: BenchmarkSpec, args: argparse.Namespace):
    if spec.dataset_loader is not None:
        return spec.dataset_loader(args)
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
    processor: Any,
) -> tuple[List[dict], List[dict]]:
    inputs: List[dict] = []
    sanitized_records: List[dict] = []

    for _, data_item in tqdm.tqdm(enumerate(dataset), desc=f"Preparing {spec.name}", total=len(dataset)):
        messages = spec.prepare_messages(data_item)
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_data, _ = process_vision_info(messages)

        inputs.append(
            {
                "prompt": prompt,
                "multi_modal_data": {
                    "image": image_data,
                },
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
    temperature: float = 0.0,
) -> List[str]:
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=parallel_num) as executor:
        results = list(
            tqdm.tqdm(
                executor.map(
                    lambda msg: module.call_api(client, model_name, msg, temperature=temperature),
                    messages_list,
                ),
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
    judge_temperature: float = 0.0,
) -> dict:
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
            temperature=judge_temperature,
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
        temperature=judge_temperature,
    )

    judge_output_filepath = spec.judge_output_path_fn(result_filepath)
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
    processor: Any,
    sampling_params: SamplingParams,
    client: OpenAI,
    args: argparse.Namespace,
) -> dict:
    print("=" * 100)
    print(f"Starting benchmark: {spec.name}")
    start_time = time.time()

    dataset = load_benchmark_dataset(spec, args)
    print(f"[{spec.name}] Loaded dataset split `{spec.hf_split}` with {len(dataset)} samples.")

    inputs, sanitized_records = prepare_generation_inputs(dataset, spec, processor)
    outputs = generate_benchmark_outputs(llm, sampling_params, inputs, sanitized_records, spec)

    result_filepath = os.path.join(args.out_dir, spec.result_filename)
    save_benchmark_outputs(spec, outputs, result_filepath)
    if spec.local_evaluator is not None:
        local_result = spec.local_evaluator(outputs)
        judge_summary = _write_local_evaluation(
            result_filepath=result_filepath,
            benchmark=spec.name,
            metrics=local_result["metrics"],
            per_sample=local_result["per_sample"],
        )
        print(f"[{spec.name}] Deterministic metrics: {json.dumps(local_result['metrics'], ensure_ascii=False)}")
    else:
        judge_summary = run_judge_pipeline(
            spec=spec,
            result_filepath=result_filepath,
            client=client,
            judge_model_name=args.judge_model_name,
            judge_parallel_num=args.judge_parallel_num,
            judge_temperature=args.judge_temperature,
        )

    elapsed = time.time() - start_time
    print(f"[{spec.name}] Completed in {elapsed / 60:.2f} minutes.")
    return {
        "benchmark": spec.name,
        "status": "success",
        "result_filepath": result_filepath,
        "evaluation_filepath": judge_summary.get("evaluation_filepath", judge_summary.get("judge_output_filepath")),
        "num_samples": len(outputs),
        "num_correct": judge_summary["num_correct"],
        "num_total": judge_summary["num_total"],
        "accuracy": judge_summary["accuracy"],
        "elapsed_seconds": elapsed,
    }


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

    summaries = []
    for spec in benchmark_specs:
        try:
            summary = run_single_benchmark(
                spec=spec,
                llm=llm,
                processor=processor,
                sampling_params=sampling_params,
                client=client,
                args=args,
            )
            summaries.append(summary)
        except Exception as exc:
            error_summary = {
                "benchmark": spec.name,
                "status": "failed",
                "error": repr(exc),
            }
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
