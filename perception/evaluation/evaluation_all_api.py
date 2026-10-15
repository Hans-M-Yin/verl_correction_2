"""
evaluation_all_api.py
─────────────────────────────────────────────────────────────────────────────
API-based version of evaluation_all.py for closed-source models (GPT-4o,
Claude, Gemini, etc.).

Differences from evaluation_all.py:
  - No vLLM / local model loading.  Inference goes through an OpenAI-
    compatible chat-completions endpoint.
  - Images are base64-encoded and sent as data URIs in the message content.
  - Inference requests are parallelized with a configurable thread pool.
  - All other logic (benchmark specs, judge pipeline, saving) is identical.

Usage:
  cd perception/evaluation
  python evaluation_all_api.py \
      --benchmarks mathvista,mathverse \
      --model gpt-4o \
      --api-key $OPENAI_API_KEY \
      --api-url https://api.openai.com/v1 \
      --out-dir results_api \
      --judge-model-name gpt-4o \
      --infer-workers 16
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import random
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from typing import Any, List

import tqdm
from openai import OpenAI
from PIL import Image as PILImage

# ── Reuse everything from the local-vLLM version ────────────────────────────
from evaluation_all import (
    BENCHMARK_SPECS,
    BenchmarkSpec,
    load_benchmark_dataset,
    resolve_benchmark_specs,
    run_judge_pipeline,
    save_benchmark_outputs,
)


# ─────────────────────────────────────────────────────────────────────────────
# Image → base64 helpers
# ─────────────────────────────────────────────────────────────────────────────

def _pil_to_base64(img: PILImage.Image, fmt: str = "PNG") -> str:
    buf = BytesIO()
    img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _to_data_uri(image_source) -> str:
    """Convert various image representations to a base64 data URI."""
    if isinstance(image_source, PILImage.Image):
        b64 = _pil_to_base64(image_source)
        return f"data:image/png;base64,{b64}"
    if isinstance(image_source, str):
        # Could be a file path or URL
        if image_source.startswith(("http://", "https://", "data:")):
            return image_source
        # Local file path
        with open(image_source, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        ext = os.path.splitext(image_source)[1].lower().lstrip(".")
        mime = {"jpg": "jpeg", "jpeg": "jpeg", "png": "png", "gif": "gif", "webp": "webp"}.get(ext, "png")
        return f"data:image/{mime};base64,{b64}"
    if isinstance(image_source, dict) and "bytes" in image_source:
        b64 = base64.b64encode(image_source["bytes"]).decode("utf-8")
        return f"data:image/png;base64,{b64}"
    raise ValueError(f"Unsupported image type: {type(image_source)}")


# ─────────────────────────────────────────────────────────────────────────────
# Convert benchmark messages → OpenAI chat API format (with base64 images)
# ─────────────────────────────────────────────────────────────────────────────

def _convert_messages_for_api(messages: List[dict]) -> List[dict]:
    """Convert processor-style messages (with PIL images) to OpenAI API format."""
    api_messages = []
    for msg in messages:
        role = msg["role"]
        content = msg.get("content", "")
        if isinstance(content, str):
            api_messages.append({"role": role, "content": content})
            continue
        # content is a list of dicts with type=image / type=text
        api_content = []
        for block in content:
            if block.get("type") == "text":
                api_content.append({"type": "text", "text": block["text"]})
            elif block.get("type") == "image":
                data_uri = _to_data_uri(block["image"])
                api_content.append({
                    "type": "image_url",
                    "image_url": {"url": data_uri},
                })
        api_messages.append({"role": role, "content": api_content})
    return api_messages


# ─────────────────────────────────────────────────────────────────────────────
# Parallel inference via API
# ─────────────────────────────────────────────────────────────────────────────

def _single_api_call(
    client: OpenAI,
    model: str,
    api_messages: List[dict],
    max_tokens: int,
    temperature: float,
    max_retries: int = 3,
) -> str:
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=api_messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return resp.choices[0].message.content
        except Exception as e:
            if attempt < max_retries - 1:
                wait = 2 ** attempt
                print(f"  [retry {attempt+1}/{max_retries}] {e}, waiting {wait}s...")
                time.sleep(wait)
            else:
                print(f"  [FAILED after {max_retries} retries] {e}")
                traceback.print_exc()
                return f"[ERROR] {e}"


def generate_benchmark_outputs_api(
    client: OpenAI,
    model: str,
    dataset,
    spec: BenchmarkSpec,
    max_tokens: int = 4096,
    temperature: float = 0.0,
    workers: int = 16,
) -> List[dict]:
    """Prepare messages, call API in parallel, and return output records."""

    # 1. Build all messages and sanitized records
    all_api_messages: List[List[dict]] = []
    sanitized_records: List[dict] = []
    for data_item in tqdm.tqdm(dataset, desc=f"Preparing {spec.name}", total=len(dataset)):
        messages = spec.prepare_messages(data_item)
        api_messages = _convert_messages_for_api(messages)
        all_api_messages.append(api_messages)
        sanitized_records.append(spec.sanitize_record(data_item))

    # 2. Parallel API calls
    print(f"[{spec.name}] Sending {len(all_api_messages)} requests with {workers} workers...")
    results: List[str | None] = [None] * len(all_api_messages)
    start = time.time()

    with ThreadPoolExecutor(max_workers=workers) as pool:
        future_to_idx = {
            pool.submit(_single_api_call, client, model, msgs, max_tokens, temperature): i
            for i, msgs in enumerate(all_api_messages)
        }
        for future in tqdm.tqdm(
            as_completed(future_to_idx),
            total=len(future_to_idx),
            desc=f"{spec.name} inference",
        ):
            idx = future_to_idx[future]
            results[idx] = future.result()

    elapsed = time.time() - start
    print(f"[{spec.name}] Inference done in {elapsed/60:.2f} minutes.")

    # 3. Assemble outputs
    outputs: List[dict] = []
    for record, response_text in zip(sanitized_records, results):
        outputs.append({**record, "response": response_text or ""})
    return outputs


# ─────────────────────────────────────────────────────────────────────────────
# CLI & main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run all perception benchmarks via a remote OpenAI-compatible API."
    )
    parser.add_argument(
        "--benchmarks",
        type=str,
        default="mathvista,mathverse,mathvision,wemath,mmmu_pro_10,mmmu_pro_vision,logicvista,hallusionbench,freak",
        help="Comma-separated benchmark names. Supported: " + ", ".join(BENCHMARK_SPECS.keys()),
    )
    parser.add_argument("--model", type=str, required=True,
                        help="Model name for the inference API (e.g. gpt-4o, claude-sonnet-4-20250514).")
    parser.add_argument("--api-key", type=str, default="sk-Eek0pnQdHdSPIvwfYMvDoCWoI0H6QoakugWZfCViQeWJlIsD")
    parser.add_argument("--api-url", type=str, default="https://yunwu.ai/v1",
                        help="Base URL for inference API.")
    parser.add_argument("--out-dir", type=str, default="results_api")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=4096, help="Max tokens for inference.")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--infer-workers", type=int, default=16,
                        help="Number of parallel workers for inference API calls.")
    # Judge arguments (same as evaluation_all.py)
    parser.add_argument("--judge-model-name", type=str, default="gpt-4o")
    parser.add_argument("--judge-api-key", type=str, default="sk-Eek0pnQdHdSPIvwfYMvDoCWoI0H6QoakugWZfCViQeWJlIsD",
                        help="API key for judge (defaults to --api-key).")
    parser.add_argument("--judge-api-url", type=str, default="https://yunwu.ai/v1",
                        help="Base URL for judge API (defaults to --api-url).")
    parser.add_argument("--judge-parallel-num", type=int, default=10)
    parser.add_argument("--continue-on-error", action="store_true",
                        help="Continue to the next benchmark even if the current one fails.")
    return parser.parse_args()


def run_single_benchmark_api(
    spec: BenchmarkSpec,
    infer_client: OpenAI,
    judge_client: OpenAI,
    args: argparse.Namespace,
) -> dict:
    print("=" * 100)
    print(f"Starting benchmark: {spec.name}")
    start_time = time.time()

    dataset = load_benchmark_dataset(spec)
    print(f"[{spec.name}] Loaded dataset split `{spec.hf_split}` with {len(dataset)} samples.")

    outputs = generate_benchmark_outputs_api(
        client=infer_client,
        model=args.model,
        dataset=dataset,
        spec=spec,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        workers=args.infer_workers,
    )

    result_filepath = os.path.join(args.out_dir, spec.result_filename)
    save_benchmark_outputs(spec, outputs, result_filepath)

    judge_summary = run_judge_pipeline(
        spec=spec,
        result_filepath=result_filepath,
        client=judge_client,
        judge_model_name=args.judge_model_name,
        judge_parallel_num=args.judge_parallel_num,
    )

    elapsed = time.time() - start_time
    print(f"[{spec.name}] Completed in {elapsed / 60:.2f} minutes.")
    return {
        "benchmark": spec.name,
        "status": "success",
        "model": args.model,
        "result_filepath": result_filepath,
        "judge_filepath": judge_summary["judge_output_filepath"],
        "num_samples": len(outputs),
        "judge_num_correct": judge_summary["num_correct"],
        "judge_num_total": judge_summary["num_total"],
        "judge_accuracy": judge_summary["accuracy"],
        "elapsed_seconds": elapsed,
    }


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    benchmark_specs = resolve_benchmark_specs(args.benchmarks)
    print("Selected benchmarks:", [spec.name for spec in benchmark_specs])
    print(f"Inference model: {args.model}")
    print(f"Inference workers: {args.infer_workers}")

    infer_client = OpenAI(api_key=args.api_key, base_url=args.api_url)
    judge_client = OpenAI(
        api_key=args.judge_api_key or args.api_key,
        base_url=args.judge_api_url or args.api_url,
    )

    summaries = []
    for spec in benchmark_specs:
        try:
            summary = run_single_benchmark_api(
                spec=spec,
                infer_client=infer_client,
                judge_client=judge_client,
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
            traceback.print_exc()
            if not args.continue_on_error:
                print("Stopping because `--continue-on-error` is not enabled.")
                break

    summary_path = os.path.join(args.out_dir, "evaluation_api_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summaries, f, ensure_ascii=False, indent=2)
    print("=" * 100)
    print(f"Saved run summary to {summary_path}")
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
