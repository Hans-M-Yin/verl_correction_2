"""Measure inference-time sensitivity to the Figure 7 trigger wording.

For each trigger in one language pool, this script runs exactly one continuation
for the same fixed set of type-1 (injected-error) holdout examples.  The trigger
is appended to the existing erroneous assistant prefix exactly as in training:
``caption_modified + ' ' + trigger``, followed by removal of ``<|im_end|>``.

This is an inference-time wording-sensitivity probe, not a replacement for a
training-time trigger-pool ablation.  Use a held-out ``test.parquet`` for paper
claims, never the training parquet.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import statistics
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import tqdm
from transformers import AutoProcessor
from vllm import LLM, SamplingParams

from evaluate_augmented_reflection import (
    Sample,
    _make_sample,
    judge_records,
    render_prompt,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRIGGER_FILE = REPO_ROOT / "perception_2" / "utils" / "triggers.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-file", required=True, help="Held-out parquet containing type-1 augmented rows.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--trigger-file", default=str(DEFAULT_TRIGGER_FILE))
    parser.add_argument("--language", choices=["english", "chinese"], default="english",
                        help="Evaluate one Figure 7 language pool at a time on matching-language samples.")
    parser.add_argument("--num-samples", type=int, default=500,
                        help="Number of fixed type-1 samples; use 0 for all eligible samples.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--include-no-trigger", action="store_true",
                        help="Also run the same samples with no Figure 7 trigger.")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--max-num-seqs", type=int, default=256)
    parser.add_argument("--max-model-len", type=int, default=20000)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Defaults to greedy decoding to isolate wording from sampling noise.")
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--judge-model", default="gpt-5.4-mini")
    parser.add_argument("--api-key", default=os.getenv("JUDGE_API_KEY", os.getenv("CORRECTION_PROJECT_API_KEY")), help="Defaults to $JUDGE_API_KEY or $CORRECTION_PROJECT_API_KEY.")
    parser.add_argument("--api-url", default=os.getenv("JUDGE_API_URL", os.getenv("CORRECTION_PROJECT_API_URL", "https://api.openai.com/v1")))
    parser.add_argument("--judge-parallel-num", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=120.0)
    return parser.parse_args()


def _has_chinese(text: str) -> bool:
    return any("\u4e00" <= char <= "\u9fff" for char in text)


def load_augmented_samples(data_file: str, language: str, count: int | None, seed: int) -> list[Sample]:
    table = pq.read_table(data_file, columns=["prompt", "images", "extra_info", "reward_model"])
    candidates: list[tuple[int, dict[str, Any]]] = []
    for index, row in enumerate(table.to_pylist()):
        extra = row.get("extra_info") or {}
        messages = row.get("prompt") or []
        if extra.get("type") != 1 or not messages or messages[-1].get("role") != "assistant":
            continue
        content = messages[-1].get("content")
        if not isinstance(content, str):
            continue
        is_chinese = _has_chinese(content)
        if (language == "chinese") != is_chinese:
            continue
        candidates.append((index, row))

    random.Random(seed).shuffle(candidates)
    if count is not None:
        candidates = candidates[:count]
    samples = [sample for index, row in candidates if (sample := _make_sample(index, row, True)) is not None]
    if not samples:
        raise RuntimeError(f"No valid {language} type-1 samples found in {data_file}.")
    return samples


def generate_one_condition(
    llm: LLM,
    processor: AutoProcessor,
    samples: list[Sample],
    trigger: str | None,
    trigger_id: str,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    inputs = [
        {"prompt": render_prompt(processor, sample, trigger=trigger), "multi_modal_data": {"image": sample.image}}
        for sample in tqdm.tqdm(samples, desc=f"Preparing {trigger_id}")
    ]
    sampling = SamplingParams(
        n=1,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        skip_special_tokens=False,
    )
    outputs = llm.generate(inputs, sampling)
    return [
        {
            "id": sample.sample_id,
            "source_row_index": sample.source_row_index,
            "is_augmented": True,
            "trigger_id": trigger_id,
            "trigger": trigger,
            "question": sample.question,
            "answer": sample.answer,
            "injected_caption": sample.injected_caption,
            "modify_info": sample.modify_info,
            "num_modify": sample.num_modify,
            "response": output.outputs[0].text,
        }
        for sample, output in zip(samples, outputs, strict=True)
    ]


def summarize_trigger(records: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [record for record in records if record.get("judge", {}).get("has_reflection") is not None]
    reflected = [record for record in valid if record["judge"]["has_reflection"]]
    successful = [
        record for record in valid
        if record["judge"]["fully_corrects_all"] and record["judge"]["is_final_answer_correct"]
    ]
    total_modifications = sum(record["num_modify"] for record in valid)
    corrected_modifications = sum(record["judge"]["corrected_modification_count"] for record in valid)

    def ratio(numerator: int, denominator: int) -> float | None:
        return round(numerator / denominator, 4) if denominator else None

    return {
        "n_samples": len(records),
        "valid_judged": len(valid),
        "n_reflections": len(reflected),
        "reflection_rate": ratio(len(reflected), len(valid)),
        "n_successful_corrections": len(successful),
        "successful_correction_rate": ratio(len(successful), len(valid)),
        "successful_correction_given_reflection": ratio(len(successful), len(reflected)),
        "final_answer_accuracy": ratio(
            sum(record["judge"]["is_final_answer_correct"] for record in valid), len(valid)
        ),
        "final_answer_accuracy_among_reflections": ratio(
            sum(record["judge"]["is_final_answer_correct"] for record in reflected), len(reflected)
        ),
        "corrected_modifications": corrected_modifications,
        "total_modifications": total_modifications,
        "per_modification_correction_rate": ratio(corrected_modifications, total_modifications),
    }


def summarize_across_triggers(conditions: dict[str, dict[str, Any]]) -> dict[str, dict[str, float | int]]:
    """Report wording sensitivity as mean/std/min/max across Figure 7 entries."""
    metric_names = [
        "reflection_rate",
        "successful_correction_rate",
        "successful_correction_given_reflection",
        "final_answer_accuracy",
        "per_modification_correction_rate",
    ]
    trigger_conditions = [value for key, value in conditions.items() if key != "no_trigger"]
    result: dict[str, dict[str, float | int]] = {}
    for metric in metric_names:
        values = [value[metric] for value in trigger_conditions if value.get(metric) is not None]
        if not values:
            continue
        result[metric] = {
            "n_triggers": len(values),
            "mean": round(statistics.mean(values), 4),
            "std": round(statistics.pstdev(values), 4),
            "variance": round(statistics.pvariance(values), 6),
            "min": round(min(values), 4),
            "max": round(max(values), 4),
        }
    return result


def main() -> None:
    args = parse_args()
    if not args.api_key:
        raise ValueError("Pass --api-key or set JUDGE_API_KEY for the LLM judge.")
    if Path(args.data_file).name == "train.parquet":
        print("WARNING: train.parquet is diagnostic only; use a held-out augmented test parquet for paper claims.")

    trigger_data = json.loads(Path(args.trigger_file).read_text(encoding="utf-8"))
    triggers = trigger_data.get(args.language)
    if not isinstance(triggers, list) or not all(isinstance(item, str) and item.strip() for item in triggers):
        raise ValueError(f"No valid {args.language!r} trigger list in {args.trigger_file}.")
    requested_count = None if args.num_samples == 0 else args.num_samples
    samples = load_augmented_samples(args.data_file, args.language, requested_count, args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = [
        {"id": sample.sample_id, "source_row_index": sample.source_row_index, "num_modify": sample.num_modify}
        for sample in samples
    ]
    manifest_path = out_dir / "sample_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Selected {len(samples)} fixed {args.language} augmented samples.")

    conditions: list[tuple[str, str | None]] = [
        (f"{args.language}_{index:02d}", trigger) for index, trigger in enumerate(triggers, start=1)
    ]
    if args.include_no_trigger:
        conditions.insert(0, ("no_trigger", None))

    processor = AutoProcessor.from_pretrained(args.checkpoint, trust_remote_code=True)
    llm = LLM(
        model=args.checkpoint,
        trust_remote_code=True,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        limit_mm_per_prompt={"image": 1},
        enable_prefix_caching=True,
        max_num_seqs=args.max_num_seqs,
        max_model_len=args.max_model_len,
    )
    summary: dict[str, Any] = {
        "checkpoint": args.checkpoint,
        "data_file": args.data_file,
        "trigger_file": args.trigger_file,
        "language": args.language,
        "num_samples": len(samples),
        "seed": args.seed,
        "sampling": {"n": 1, "temperature": args.temperature, "top_p": args.top_p, "max_tokens": args.max_tokens},
        "sample_manifest": str(manifest_path),
        "conditions": {},
    }
    try:
        for trigger_id, trigger in conditions:
            print(f"\n[{trigger_id}] {trigger if trigger is not None else 'NO TRIGGER'}")
            records = generate_one_condition(llm, processor, samples, trigger, trigger_id, args)
            generated_path = out_dir / f"{trigger_id}_generations.json"
            generated_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
            judged = judge_records(records, samples, args)
            judged_path = out_dir / f"{trigger_id}_judged.json"
            judged_path.write_text(json.dumps(judged, ensure_ascii=False, indent=2), encoding="utf-8")
            stats = summarize_trigger(judged)
            summary["conditions"][trigger_id] = {
                "trigger": trigger,
                "generated_file": str(generated_path),
                "judged_file": str(judged_path),
                **stats,
            }
            (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
            print(json.dumps(stats, ensure_ascii=False))
        summary["across_figure7_triggers"] = summarize_across_triggers(summary["conditions"])
        (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    finally:
        del llm, processor
        gc.collect()
    print(f"Trigger-sensitivity summary: {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
