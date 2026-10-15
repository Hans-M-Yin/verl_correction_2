"""Evaluate reflection as injected-error detection and correction.

The input parquet contains two controlled groups:

* ``extra_info.type == 1``: an erroneous visual description is inserted as an
  unfinished assistant prefix; the model should revisit it.
* ``extra_info.type == 0``: a normal image-question prompt with no injected
  erroneous assistant prefix.

For a fixed random sample from each group, every checkpoint is evaluated on
the exact same inputs. A vision-capable LLM judge produces reflection and
correction labels, from which this script reports reflection precision/recall,
successful-correction precision/recall, and answer accuracy among reflections.

The precision values are with respect to the *injected-error detection task*.
Reflection on a normal sample is not intrinsically undesirable: a model may
legitimately self-correct its own reasoning. It is counted as a false positive
only for this controlled question of whether reflection tracks injected errors.
"""

from __future__ import annotations

import argparse
import base64
import copy
import gc
import io
import json
import os
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import pyarrow.parquet as pq
import tqdm
from openai import OpenAI
from PIL import Image
from transformers import AutoProcessor
from vllm import LLM, SamplingParams


JUDGE_SYSTEM_PROMPT = """You are a strict evaluator of explicit self-reflection in visual reasoning.

You will receive an image, question, reference answer, and a model response.
There are two controlled sample types.

1. INJECTED-ERROR sample: an earlier assistant prefix intentionally contains
   incorrect visual claims. Decide whether the continuation explicitly
   revisits, rejects, or revises that earlier content. For every injected
   modification, determine whether the continuation states the supplied
   correct visual evidence.
2. NORMAL sample: no erroneous assistant prefix was inserted. Decide whether
   the response itself contains genuine explicit self-correction: an earlier
   belief/calculation/visual interpretation is reconsidered or revised.

Generic phrases such as "let me check", ordinary step-by-step reasoning, and
merely giving the correct answer do NOT count as reflection. Do not infer an
unstated correction.

For an INJECTED-ERROR sample, ``fully_corrects_all`` is true only when every
injected modification is explicitly corrected. For a NORMAL sample it must be
false and ``corrected_modification_count`` must be 0. Judge final-answer
correctness independently.

Return only a JSON object with exactly these fields:
{
  "has_reflection": true,
  "fully_corrects_all": false,
  "corrected_modification_count": 1,
  "is_final_answer_correct": true,
  "evidence": "short quote or N/A",
  "reason": "brief explanation"
}"""


@dataclass
class Sample:
    sample_id: str
    source_row_index: int
    is_augmented: bool
    messages: list[dict[str, Any]]
    image: Image.Image
    question: str
    answer: str
    injected_caption: Optional[str]
    modify_info: list[list[str]]
    num_modify: int


def parse_models(values: list[str]) -> list[tuple[str, str]]:
    models: list[tuple[str, str]] = []
    seen: set[str] = set()
    for value in values:
        if "=" not in value:
            raise ValueError(f"Invalid --model {value!r}; use LABEL=CHECKPOINT.")
        label, checkpoint = (part.strip() for part in value.split("=", 1))
        if not label or not checkpoint or label in seen:
            raise ValueError("Each --model needs a non-empty, unique LABEL=CHECKPOINT pair.")
        models.append((label, checkpoint))
        seen.add(label)
    return models


def image_from_record(record: dict[str, Any]) -> Image.Image:
    images = record.get("images") or []
    if not images:
        raise ValueError("Sample has no image.")
    image = images[0]
    if isinstance(image, dict) and image.get("bytes") is not None:
        return Image.open(io.BytesIO(image["bytes"])).convert("RGB")
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    raise ValueError(f"Unsupported image value: {type(image)!r}")


def _sample_records(records: list[tuple[int, dict]], count: Optional[int], seed: int) -> list[tuple[int, dict]]:
    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)
    return shuffled if count is None else shuffled[:count]


def _make_sample(source_row_index: int, row: dict, is_augmented: bool) -> Optional[Sample]:
    extra = row.get("extra_info") or {}
    messages = row.get("prompt") or []
    if not messages:
        return None
    answer = str(extra.get("answer", (row.get("reward_model") or {}).get("ground_truth", "")))
    question = str(extra.get("question", ""))
    source_idx = extra.get("source_idx", extra.get("idx", source_row_index))
    sample_kind = "augmented" if is_augmented else "normal"
    try:
        image = image_from_record(row)
    except ValueError:
        return None

    if is_augmented:
        caption = extra.get("caption_modified")
        modify_info = extra.get("modify_info")
        if not caption or not modify_info or messages[-1].get("role") != "assistant":
            return None
        pairs = [[str(value) for value in pair] for pair in modify_info]
        return Sample(
            sample_id=f"{sample_kind}:{source_row_index}:{source_idx}",
            source_row_index=source_row_index,
            is_augmented=True,
            messages=messages,
            image=image,
            question=question,
            answer=answer,
            injected_caption=str(caption),
            modify_info=pairs,
            num_modify=int(extra.get("num_modify", len(pairs))),
        )

    # Normal examples must end with a user turn. Otherwise a type-0 row with
    # an unexpected assistant prefix would no longer be a clean negative case.
    if messages[-1].get("role") != "user":
        return None
    return Sample(
        sample_id=f"{sample_kind}:{source_row_index}:{source_idx}",
        source_row_index=source_row_index,
        is_augmented=False,
        messages=messages,
        image=image,
        question=question,
        answer=answer,
        injected_caption=None,
        modify_info=[],
        num_modify=0,
    )


def load_samples(
    data_file: str,
    num_augmented_samples: Optional[int],
    num_normal_samples: Optional[int],
    seed: int,
) -> list[Sample]:
    """Draw reproducible samples separately from type-1 and type-0 rows."""
    table = pq.read_table(data_file, columns=["prompt", "images", "extra_info", "reward_model"])
    indexed_rows = list(enumerate(table.to_pylist()))
    augmented_rows = [(idx, row) for idx, row in indexed_rows if (row.get("extra_info") or {}).get("type") == 1]
    normal_rows = [(idx, row) for idx, row in indexed_rows if (row.get("extra_info") or {}).get("type") == 0]
    if not augmented_rows:
        raise RuntimeError("No augmented rows found: expected extra_info.type == 1.")
    if not normal_rows:
        raise RuntimeError("No normal rows found: expected extra_info.type == 0.")

    selected_augmented = _sample_records(augmented_rows, num_augmented_samples, seed)
    selected_normal = _sample_records(normal_rows, num_normal_samples, seed)
    augmented = [sample for idx, row in selected_augmented if (sample := _make_sample(idx, row, True)) is not None]
    normal = [sample for idx, row in selected_normal if (sample := _make_sample(idx, row, False)) is not None]
    if not augmented or not normal:
        raise RuntimeError("No valid samples remained after loading images/prompts for one of the two groups.")

    # Interleave groups deterministically so neither group systematically gets
    # a privileged position in a vLLM batch.
    samples = augmented + normal
    random.Random(seed).shuffle(samples)
    return samples


def build_vllm_messages(messages: list[dict[str, Any]], image: Image.Image) -> list[dict[str, Any]]:
    """Replace the parquet's textual ``<image>`` marker with a vLLM image item."""
    converted: list[dict[str, Any]] = []
    inserted_image = False
    for message in messages:
        content = message.get("content", "")
        if isinstance(content, str) and "<image>" in content:
            if inserted_image:
                raise ValueError("This evaluator supports exactly one image per sample.")
            converted.append({
                **message,
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": content.replace("<image>", "", 1).lstrip()},
                ],
            })
            inserted_image = True
        else:
            converted.append(dict(message))
    if not inserted_image:
        raise ValueError("Expected an <image> marker in the prompt.")
    return converted


def render_prompt(processor, sample: Sample, trigger: Optional[str] = None) -> str:
    """Render the unfinished assistant prefix; optionally append a training trigger.

    ``trigger`` follows ``perception_2.utils.sample_trigger`` exactly: it is
    appended to the final assistant prefix before the closing ``<|im_end|>``
    is removed.  It is therefore model history, not a new user instruction.
    """
    messages = sample.messages
    if trigger is not None:
        if not sample.is_augmented or messages[-1].get("role") != "assistant":
            raise ValueError("A trigger can only be appended to an augmented assistant prefix.")
        messages = copy.deepcopy(messages)
        content = messages[-1].get("content")
        if not isinstance(content, str):
            raise ValueError("This evaluator currently expects string assistant content for triggers.")
        messages[-1]["content"] = content + " " + trigger
    prompt = processor.apply_chat_template(
        build_vllm_messages(messages, sample.image), tokenize=False, add_generation_prompt=True
    )
    if not sample.is_augmented:
        return prompt
    # Mirror AgentLoopBase.apply_chat_template: continue the injected assistant
    # prefix rather than starting a fresh assistant message after it.
    end = prompt.rfind("<|im_end|>")
    if end < 0:
        raise RuntimeError("Expected Qwen <|im_end|> assistant terminator was not found.")
    return prompt[:end]


def generate_for_model(label: str, checkpoint: str, samples: list[Sample], args: argparse.Namespace) -> list[dict]:
    print(f"\n[{label}] Loading {checkpoint}")
    processor = AutoProcessor.from_pretrained(checkpoint, trust_remote_code=True)
    inputs = [
        {"prompt": render_prompt(processor, sample), "multi_modal_data": {"image": sample.image}}
        for sample in tqdm.tqdm(samples, desc=f"[{label}] Preparing prompts")
    ]
    llm = LLM(
        model=checkpoint,
        trust_remote_code=True,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        limit_mm_per_prompt={"image": 1},
        enable_prefix_caching=True,
        max_num_seqs=args.max_num_seqs,
        max_model_len=args.max_model_len,
    )
    outputs = llm.generate(
        inputs, SamplingParams(temperature=0.0, top_k=1, max_tokens=args.max_tokens, skip_special_tokens=False)
    )
    records = []
    for sample, output in zip(samples, outputs, strict=True):
        records.append({
            "id": sample.sample_id,
            "source_row_index": sample.source_row_index,
            "is_augmented": sample.is_augmented,
            "question": sample.question,
            "answer": sample.answer,
            "injected_caption": sample.injected_caption,
            "modify_info": sample.modify_info,
            "num_modify": sample.num_modify,
            "response": output.outputs[0].text,
        })
    del llm, processor
    gc.collect()
    return records


def _judge_error(exc: Exception | str) -> dict[str, Any]:
    return {
        "has_reflection": None,
        "fully_corrects_all": None,
        "corrected_modification_count": None,
        "is_final_answer_correct": None,
        "evidence": "N/A",
        "reason": "",
        "judge_error": repr(exc),
    }


def judge_one(client: OpenAI, model: str, sample: Sample, response: str, timeout: float) -> dict[str, Any]:
    if sample.is_augmented:
        modifications = "\n".join(
            f"{idx}. WRONG injected claim: {pair[0]}\n   CORRECT visual evidence: {pair[1]}"
            for idx, pair in enumerate(sample.modify_info, start=1)
        )
        context = (
            "SAMPLE TYPE: INJECTED-ERROR\n"
            f"INJECTED ASSISTANT DESCRIPTION:\n{sample.injected_caption}\n\n"
            f"MODIFICATIONS TO CHECK:\n{modifications}\n\n"
        )
    else:
        context = (
            "SAMPLE TYPE: NORMAL\n"
            "No erroneous assistant description was injected before the response.\n\n"
        )
    user_text = (
        f"QUESTION:\n{sample.question}\n\nREFERENCE ANSWER:\n{sample.answer}\n\n"
        f"{context}MODEL RESPONSE / CONTINUATION:\n{response}"
    )
    buffer = io.BytesIO()
    sample.image.save(buffer, format="PNG")
    image_b64 = base64.b64encode(buffer.getvalue()).decode("ascii")
    try:
        completion = client.chat.completions.create(
            model=model,
            temperature=0.0,
            max_tokens=384,
            timeout=timeout,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
                    {"type": "text", "text": user_text},
                ]},
            ],
        )
        parsed = json.loads(completion.choices[0].message.content)
        has_reflection = bool(parsed.get("has_reflection", False))
        if sample.is_augmented:
            count = max(0, min(int(parsed.get("corrected_modification_count", 0)), sample.num_modify))
            fully_corrects_all = bool(parsed.get("fully_corrects_all", False)) and count == sample.num_modify
        else:
            count, fully_corrects_all = 0, False
        return {
            "has_reflection": has_reflection,
            "fully_corrects_all": fully_corrects_all,
            "corrected_modification_count": count,
            "is_final_answer_correct": bool(parsed.get("is_final_answer_correct", False)),
            "evidence": str(parsed.get("evidence", "N/A")),
            "reason": str(parsed.get("reason", "")),
            "judge_error": None,
        }
    except Exception as exc:
        return _judge_error(exc)


def judge_records(records: list[dict], samples: list[Sample], args: argparse.Namespace) -> list[dict]:
    sample_by_id = {sample.sample_id: sample for sample in samples}
    client = OpenAI(api_key=args.api_key, base_url=args.api_url)
    judged: list[dict] = []
    with ThreadPoolExecutor(max_workers=args.judge_parallel_num) as executor:
        futures = {
            executor.submit(judge_one, client, args.judge_model, sample_by_id[record["id"]], record["response"], args.timeout): record
            for record in records
        }
        for future in tqdm.tqdm(as_completed(futures), total=len(futures), desc="LLM judging reflection"):
            record = futures[future]
            try:
                record["judge"] = future.result()
            except Exception as exc:
                record["judge"] = _judge_error(exc)
            judged.append(record)
    return judged


def _ratio(numerator: int, denominator: int) -> Optional[float]:
    return round(numerator / denominator, 4) if denominator else None


def summarize(records: list[dict]) -> dict[str, Any]:
    valid = [record for record in records if record.get("judge", {}).get("has_reflection") is not None]
    augmented = [record for record in valid if record["is_augmented"]]
    normal = [record for record in valid if not record["is_augmented"]]
    reflected = [record for record in valid if record["judge"]["has_reflection"]]
    augmented_reflected = [record for record in augmented if record["judge"]["has_reflection"]]
    normal_reflected = [record for record in normal if record["judge"]["has_reflection"]]
    successful = [
        record for record in augmented
        if record["judge"]["fully_corrects_all"] and record["judge"]["is_final_answer_correct"]
    ]
    total_modifications = sum(record["num_modify"] for record in augmented)
    corrected_modifications = sum(record["judge"]["corrected_modification_count"] for record in augmented)
    answer_correct_among_reflections = sum(record["judge"]["is_final_answer_correct"] for record in reflected)

    return {
        "total_samples": len(records),
        "valid_judged": len(valid),
        "n_augmented": len(augmented),
        "n_normal": len(normal),
        "n_reflections": len(reflected),
        "n_augmented_reflections": len(augmented_reflected),
        "n_normal_reflections": len(normal_reflected),
        "reflection_recall": _ratio(len(augmented_reflected), len(augmented)),
        "reflection_precision": _ratio(len(augmented_reflected), len(reflected)),
        "normal_reflection_rate": _ratio(len(normal_reflected), len(normal)),
        "n_successful_corrections": len(successful),
        "successful_correction_recall": _ratio(len(successful), len(augmented)),
        "successful_correction_precision": _ratio(len(successful), len(reflected)),
        "answer_accuracy_among_reflections": _ratio(answer_correct_among_reflections, len(reflected)),
        "augmented_final_answer_accuracy": _ratio(
            sum(record["judge"]["is_final_answer_correct"] for record in augmented), len(augmented)
        ),
        "normal_final_answer_accuracy": _ratio(
            sum(record["judge"]["is_final_answer_correct"] for record in normal), len(normal)
        ),
        "corrected_modifications": corrected_modifications,
        "total_modifications": total_modifications,
        "per_modification_correction_rate": _ratio(corrected_modifications, total_modifications),
        "confusion_matrix_injected_error_vs_reflection": {
            "tp_augmented_reflection": len(augmented_reflected),
            "fp_normal_reflection": len(normal_reflected),
            "fn_augmented_no_reflection": len(augmented) - len(augmented_reflected),
            "tn_normal_no_reflection": len(normal) - len(normal_reflected),
        },
    }


def sample_manifest(samples: list[Sample]) -> list[dict]:
    return [
        {
            "id": sample.sample_id,
            "source_row_index": sample.source_row_index,
            "is_augmented": sample.is_augmented,
            "num_modify": sample.num_modify,
        }
        for sample in samples
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-file", required=True, help="Parquet with prompt/images/extra_info fields.")
    parser.add_argument("--model", action="append", required=True, metavar="LABEL=CHECKPOINT")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--max-samples", type=int, default=300, metavar="N",
                        help="Number of type==1 augmented samples; use 0 for all.")
    parser.add_argument("--num-normal-samples", type=int, default=300, metavar="N",
                        help="Number of type==0 normal samples; use 0 for all.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--max-num-seqs", type=int, default=256)
    parser.add_argument("--max-model-len", type=int, default=20000)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--judge-model", default="gpt-5.4-mini")
    parser.add_argument("--api-key", default=os.getenv("JUDGE_API_KEY"), help="Defaults to $JUDGE_API_KEY.")
    parser.add_argument("--api-url", default=os.getenv("JUDGE_API_URL", "https://yunwu.ai/v1"))
    parser.add_argument("--judge-parallel-num", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=120.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.api_key:
        raise ValueError("Pass --api-key or set JUDGE_API_KEY for the LLM judge.")
    models = parse_models(args.model)
    args.max_samples = None if args.max_samples == 0 else args.max_samples
    args.num_normal_samples = None if args.num_normal_samples == 0 else args.num_normal_samples
    if os.path.basename(args.data_file) == "train.parquet":
        print("WARNING: train.parquet is a diagnostic; use an unseen augmented holdout for paper claims.")
    samples = load_samples(args.data_file, args.max_samples, args.num_normal_samples, args.seed)
    n_augmented = sum(sample.is_augmented for sample in samples)
    print(f"Selected {n_augmented} augmented (type==1) and {len(samples) - n_augmented} normal (type==0) samples.")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "sample_manifest.json"
    manifest_path.write_text(json.dumps(sample_manifest(samples), ensure_ascii=False, indent=2), encoding="utf-8")
    summary: dict[str, Any] = {
        "data_file": args.data_file,
        "seed": args.seed,
        "requested_augmented_samples": args.max_samples,
        "requested_normal_samples": args.num_normal_samples,
        "sample_manifest": str(manifest_path),
        "models": {},
    }
    for label, checkpoint in models:
        generated_path = out_dir / f"{label}_generations.json"
        judged_path = out_dir / f"{label}_judged.json"
        records = generate_for_model(label, checkpoint, samples, args)
        generated_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
        judged = judge_records(records, samples, args)
        judged_path.write_text(json.dumps(judged, ensure_ascii=False, indent=2), encoding="utf-8")
        stats = summarize(judged)
        summary["models"][label] = {
            "checkpoint": checkpoint,
            "generated_file": str(generated_path),
            "judged_file": str(judged_path),
            **stats,
        }
        (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[{label}] {json.dumps(stats, ensure_ascii=False)}")
    print(f"Comparison summary: {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
