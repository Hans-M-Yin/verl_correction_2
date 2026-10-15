"""
Analyze reflection patterns in VQA model outputs with an API judge.

Goal:
  1. Find samples where the model shows explicit reflection / self-correction.
  2. Keep only samples whose final answer is correct.
  3. Classify reflection as:
       - logical: the model fixed a reasoning / arithmetic / deduction mistake
       - visual:  the model first misread the image, then re-extracted correct visual info
  4. Detect harmful revisions: an early correct visual, mathematical, or
     logical claim is explicitly revised into an incorrect claim.

Supported result formats:
  - MathVista: a dict keyed by ``pid``.
  - MathVerse: a list whose records contain ``sample_index``.

Example:
  python judge_useful.py \
      --eval-json /path/to/mathvista.json \
      --benchmark mathvista \
      --api-key $OPENAI_API_KEY \
      --api-url https://api.openai.com/v1 \
      --model gpt-4o \
      --output reflection_analysis_mathvista.json \
      --parallel 8

  python judge_useful.py \
      --eval-json /path/to/mathverse_test.json \
      --benchmark mathverse \
      --api-key $OPENAI_API_KEY \
      --api-url https://api.openai.com/v1 \
      --model gpt-4o \
      --output reflection_analysis_mathverse.json \
      --parallel 8
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from typing import Any

import tqdm


BENCHMARK_CONFIGS: dict[str, dict[str, Any]] = {
    "mathvista": {
        "hf_root": "AI4Math/MathVista",
        "hf_split": "testmini",
        "hf_config": None,
        "id_col": "pid",
        "image_col": "decoded_image",
        "question_col": "query",
        "json_is_dict": True,
        "json_id_field": "pid",
        "cast_image": True,
    },
    "mathverse": {
        # Must match evaluation_all.py's MathVerse ``testmini`` protocol.
        "hf_root": "AI4Math/MathVerse",
        "hf_split": "testmini",
        "hf_config": "testmini",
        "id_col": "sample_index",
        "image_col": "image",
        "question_col": "question",
        "json_is_dict": False,
        "json_id_field": "sample_index",
        "cast_image": True,
    },
}


SYSTEM_PROMPT = """\
You are a strict judge for reflection behavior in visual question answering.

You will be given:
- An image
- The question
- The model's full response
- The ground-truth answer

Your tasks:
1. Decide whether the model's FINAL answer is correct.
2. Decide whether the response contains genuine reflection / self-correction.
3. If there is reflection, classify it:
   - "logical": the model corrected reasoning, arithmetic, deduction, or constraint handling.
   - "visual": the model first misread / misdescribed image content, then revisited the image and corrected the visual evidence.
   - "none": no real reflection happened.
   - "unclear": there may be correction language, but the source of correction is ambiguous.
4. Track the direction of any explicit revision of the SAME claim. The claim
   may concern visual evidence, arithmetic, geometry, symbolic manipulation,
   or logical deduction:
   - "correct_to_wrong": an early explicit claim is correct given the image,
     question, and reference answer, then the response explicitly replaces it
     with an incorrect claim.
   - "wrong_to_correct": an earlier claim is explicitly revised into a
     correct claim.
   - "correct_to_correct": both the early and revised claims are correct.
   - "wrong_to_wrong": an incorrect claim is revised into another incorrect claim.
   - "no_revision": no explicit revision of a claim occurs.
   - "unclear": the relevant claims or their relation are ambiguous.
   Also label revision_scope as "visual", "logical", "mixed", "none", or
   "unclear" based on the revised claim.

Important rules:
- Reflection requires an explicit revision, reconsideration, backtracking, or correction of an earlier belief.
- Mere step-by-step reasoning is NOT reflection.
- A response that is consistently correct from start to finish is NOT reflection.
- For "visual", the earlier mistake must be about image content itself: count, text, label, shape, color, chart value, object identity, position, etc.
- For "logical", the visual perception may stay stable while the reasoning changes.
- If the response is correct but has no real reflection, output reflection_type="none".
- Label "correct_to_wrong" ONLY when the early correct claim and later wrong
  replacement are both explicit and refer to the same claim. Do NOT use it
  merely because the final answer is wrong or a new, unrelated mistake appears.

Return ONLY a JSON object with exactly these fields:
{
  "is_correct": true,
  "has_reflection": false,
  "reflection_type": "none",
  "reflection_evidence": "short quote or N/A",
  "earlier_mistake": "short quote or N/A",
  "corrected_to": "short quote or N/A",
  "revision_direction": "no_revision",
  "revision_scope": "none",
  "early_claim": "short quote or N/A",
  "late_claim": "short quote or N/A",
  "reason": "1-3 sentence explanation"
}
"""


USER_TEMPLATE = """\
QUESTION:
{question}

GROUND_TRUTH_ANSWER:
{ground_truth}

MODEL_RESPONSE:
{response}
"""


def pil_to_base64(img) -> str:
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def load_eval_items(eval_json_path: str, cfg: dict[str, Any]) -> list[dict[str, Any]]:
    with open(eval_json_path, encoding="utf-8") as f:
        raw = json.load(f)

    items = list(raw.values()) if cfg["json_is_dict"] and isinstance(raw, dict) else raw
    id_field = cfg["json_id_field"]

    result: list[dict[str, Any]] = []
    for item in items:
        result.append({
            "id": str(item.get(id_field, item.get("pid", item.get("id", "?")))),
            "question": str(item.get("question", item.get("query", ""))),
            "query": str(item.get("query", item.get("question", ""))),
            "response": str(item.get("response", "")),
            "answer": str(item.get("answer", "")),
            "image": item.get("image"),
            "raw_item": item,
        })
    return result


def load_local_image_index(items: list[dict[str, Any]], eval_json_path: str, image_root: str | None) -> dict[str, Any]:
    from PIL import Image

    eval_dir = os.path.dirname(os.path.abspath(eval_json_path))
    index: dict[str, Any] = {}
    for item in items:
        rel_path = item.get("image")
        if not rel_path:
            continue

        candidates = []
        if image_root:
            candidates.append(os.path.join(image_root, rel_path))
            candidates.append(os.path.join(image_root, os.path.basename(rel_path)))
        candidates.append(os.path.join(eval_dir, rel_path))
        candidates.append(os.path.abspath(rel_path))

        for path in candidates:
            if path and os.path.exists(path):
                try:
                    index[item["id"]] = Image.open(path).convert("RGB")
                    break
                except Exception:
                    pass
    return index


def load_hf_image_index(cfg: dict[str, Any]) -> dict[str, Any]:
    from datasets import Image as HFImage, load_dataset

    print(f"Loading HuggingFace dataset: {cfg['hf_root']} / {cfg['hf_split']} ...")
    kwargs: dict[str, Any] = {}
    if cfg["hf_config"]:
        kwargs["name"] = cfg["hf_config"]

    ds = load_dataset(cfg["hf_root"], split=cfg["hf_split"], **kwargs)
    if cfg["cast_image"] and cfg["image_col"] in ds.column_names:
        ds = ds.cast_column(cfg["image_col"], HFImage(decode=True))

    index: dict[str, Any] = {}
    for row in tqdm.tqdm(ds, desc="Indexing images", leave=False):
        index[str(row[cfg["id_col"]])] = row[cfg["image_col"]]
    return index


def get_image_index(
    items: list[dict[str, Any]],
    eval_json_path: str,
    image_root: str | None,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    local_index = load_local_image_index(items, eval_json_path, image_root)
    if len(local_index) == len(items):
        print(f"Loaded all {len(local_index)} images from local paths.")
        return local_index

    print(f"Loaded {len(local_index)} / {len(items)} images from local paths.")
    hf_index = load_hf_image_index(cfg)
    hf_index.update(local_index)
    return hf_index


def detect_reflection(
    client,
    model_name: str,
    question: str,
    ground_truth: str,
    response: str,
    pil_image,
    timeout: float | None = None,
    max_retries: int = 3,
) -> dict[str, Any]:
    b64 = pil_to_base64(pil_image)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{b64}"},
                },
                {
                    "type": "text",
                    "text": USER_TEMPLATE.format(
                        question=question[:2500],
                        ground_truth=ground_truth[:500],
                        response=response[:8000],
                    ),
                },
            ],
        },
    ]

    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0.0,
                max_tokens=512,
                timeout=timeout,
                response_format={"type": "json_object"},
            )
            raw = resp.choices[0].message.content.strip()
            parsed = json.loads(raw)
            reflection_type = str(parsed.get("reflection_type", "none")).strip().lower()
            if reflection_type not in {"logical", "visual", "none", "unclear"}:
                reflection_type = "unclear"
            revision_direction = str(parsed.get("revision_direction", "no_revision")).strip().lower()
            valid_directions = {
                "correct_to_wrong", "wrong_to_correct", "correct_to_correct",
                "wrong_to_wrong", "no_revision", "unclear",
            }
            if revision_direction not in valid_directions:
                revision_direction = "unclear"
            revision_scope = str(parsed.get("revision_scope", "none")).strip().lower()
            if revision_scope not in {"visual", "logical", "mixed", "none", "unclear"}:
                revision_scope = "unclear"
            has_reflection = bool(parsed.get("has_reflection", False))
            if reflection_type == "none":
                has_reflection = False

            return {
                "is_correct": bool(parsed.get("is_correct", False)),
                "has_reflection": has_reflection,
                "reflection_type": reflection_type,
                "reflection_evidence": str(parsed.get("reflection_evidence", "N/A")),
                "earlier_mistake": str(parsed.get("earlier_mistake", "N/A")),
                "corrected_to": str(parsed.get("corrected_to", "N/A")),
                "revision_direction": revision_direction,
                "revision_scope": revision_scope,
                "early_claim": str(parsed.get("early_claim", "N/A")),
                "late_claim": str(parsed.get("late_claim", "N/A")),
                "reason": str(parsed.get("reason", "")),
                "raw_judge_output": raw,
            }
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                return {
                    "is_correct": None,
                    "has_reflection": None,
                    "reflection_type": "error",
                    "reflection_evidence": "N/A",
                    "earlier_mistake": "N/A",
                    "corrected_to": "N/A",
                    "revision_direction": "error",
                    "revision_scope": "error",
                    "early_claim": "N/A",
                    "late_claim": "N/A",
                    "reason": f"API_ERROR: {e}",
                    "raw_judge_output": "",
                }

    return {
        "is_correct": None,
        "has_reflection": None,
        "reflection_type": "error",
        "reflection_evidence": "N/A",
        "earlier_mistake": "N/A",
        "corrected_to": "N/A",
        "revision_direction": "error",
        "revision_scope": "error",
        "early_claim": "N/A",
        "late_claim": "N/A",
        "reason": "API_ERROR: max retries exceeded",
        "raw_judge_output": "",
    }


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(results)
    valid = [r for r in results if r.get("has_reflection") is not None and r.get("is_correct") is not None]
    selected = [r for r in valid if r["is_correct"] and r["has_reflection"]]

    def _count(reflection_type: str) -> int:
        return sum(1 for r in selected if r.get("reflection_type") == reflection_type)

    explicit_revisions = [
        r for r in valid
        if r.get("revision_direction") in {
            "correct_to_wrong", "wrong_to_correct", "correct_to_correct", "wrong_to_wrong"
        }
    ]
    harmful_revisions = [
        r for r in valid if r.get("revision_direction") == "correct_to_wrong"
    ]
    reflected = [r for r in valid if r.get("has_reflection")]
    harmful_visual = [r for r in harmful_revisions if r.get("revision_scope") == "visual"]
    harmful_logical = [r for r in harmful_revisions if r.get("revision_scope") == "logical"]
    harmful_mixed = [r for r in harmful_revisions if r.get("revision_scope") == "mixed"]

    return {
        "total_samples": total,
        "valid_judged": len(valid),
        "n_correct": sum(1 for r in valid if r["is_correct"]),
        "n_has_reflection": sum(1 for r in valid if r["has_reflection"]),
        "n_selected_correct_with_reflection": len(selected),
        "n_logical_reflection": _count("logical"),
        "n_visual_reflection": _count("visual"),
        "n_unclear_reflection": _count("unclear"),
        "selected_rate_among_valid": round(len(selected) / len(valid), 4) if valid else None,
        "n_explicit_revisions": len(explicit_revisions),
        "n_harmful_revisions_correct_to_wrong": len(harmful_revisions),
        "n_harmful_visual_revisions": len(harmful_visual),
        "n_harmful_logical_revisions": len(harmful_logical),
        "n_harmful_mixed_revisions": len(harmful_mixed),
        "harmful_revision_rate_among_valid": round(len(harmful_revisions) / len(valid), 4) if valid else None,
        "harmful_revision_rate_given_revision": (
            round(len(harmful_revisions) / len(explicit_revisions), 4) if explicit_revisions else None
        ),
        "harmful_revision_rate_given_reflection": (
            round(len(harmful_revisions) / len(reflected), 4) if reflected else None
        ),
    }


def save_results(output_path: str, existing: dict[str, dict[str, Any]], new_results: list[dict[str, Any]], quiet: bool = False) -> None:
    all_results = list(existing.values()) + new_results
    stats = summarize(all_results)
    selected = [r for r in all_results if r.get("is_correct") and r.get("has_reflection")]
    logical = [r for r in selected if r.get("reflection_type") == "logical"]
    visual = [r for r in selected if r.get("reflection_type") == "visual"]
    harmful_revisions = [
        r for r in all_results if r.get("revision_direction") == "correct_to_wrong"
    ]

    payload = {
        "stats": stats,
        "selected_samples": selected,
        "selected_logical": logical,
        "selected_visual": visual,
        "harmful_revision_samples": harmful_revisions,
        "per_sample": all_results,
    }

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    if not quiet:
        print("\n" + "=" * 68)
        print("  REFLECTION ANALYSIS RESULTS")
        print("=" * 68)
        print(f"  Total samples                    : {stats['total_samples']}")
        print(f"  Valid judged                     : {stats['valid_judged']}")
        print(f"  Correct final answers            : {stats['n_correct']}")
        print(f"  Has reflection                   : {stats['n_has_reflection']}")
        print(f"  Selected (correct + reflection)  : {stats['n_selected_correct_with_reflection']}")
        print(f"  Logical reflection               : {stats['n_logical_reflection']}")
        print(f"  Visual reflection                : {stats['n_visual_reflection']}")
        print(f"  Unclear reflection               : {stats['n_unclear_reflection']}")
        print(f"  Explicit claim revisions         : {stats['n_explicit_revisions']}")
        print(f"  Harmful revisions (C→W)          : {stats['n_harmful_revisions_correct_to_wrong']}")
        harmful_rate = stats["harmful_revision_rate_among_valid"]
        harmful_rate_str = f"{harmful_rate:.1%}" if harmful_rate is not None else "N/A"
        print(f"  Harmful-revision rate / valid    : {harmful_rate_str}")
        rate = stats["selected_rate_among_valid"]
        rate_str = f"{rate:.1%}" if rate is not None else "N/A"
        print(f"  Selected rate among valid        : {rate_str}")
        print("=" * 68)
        print(f"\nResults saved to: {output_path}")


def analyze(args: argparse.Namespace) -> None:
    cfg = BENCHMARK_CONFIGS[args.benchmark]

    print(f"\nLoading eval JSON: {args.eval_json}")
    items = load_eval_items(args.eval_json, cfg)
    if args.max_samples is not None:
        items = items[:args.max_samples]
    print(f"  -> {len(items)} samples")

    image_index = get_image_index(items, args.eval_json, args.image_root, cfg)

    existing_results: dict[str, dict[str, Any]] = {}
    if os.path.exists(args.output) and not args.rerun_existing:
        try:
            with open(args.output, encoding="utf-8") as f:
                saved = json.load(f)
            for r in saved.get("per_sample", []):
                existing_results[r["id"]] = r
            print(f"Resuming: {len(existing_results)} samples already processed.")
        except Exception:
            pass

    if args.azure_endpoint:
        from openai import AzureOpenAI

        default_headers = {"X-TT-LOGID": args.azure_logid} if args.azure_logid else None
        client = AzureOpenAI(
            api_key=args.api_key,
            api_version=args.azure_api_version,
            azure_endpoint=args.azure_endpoint,
            default_headers=default_headers,
        )
        print(
            f"Using Azure judge endpoint: {args.azure_endpoint} "
            f"(api_version={args.azure_api_version}, model={args.model})"
        )
    else:
        from openai import OpenAI

        client = OpenAI(api_key=args.api_key, base_url=args.api_url)

    todo = [it for it in items if it["id"] not in existing_results]
    print(f"\nJudging reflection in {len(todo)} samples (skipping {len(existing_results)} already done) ...")

    def process_one(item: dict[str, Any]) -> dict[str, Any]:
        sample_id = item["id"]
        pil_img = image_index.get(sample_id)
        if pil_img is None:
            return {
                "id": sample_id,
                "is_correct": None,
                "has_reflection": None,
                "reflection_type": "error",
                "reflection_evidence": "N/A",
                "earlier_mistake": "N/A",
                "corrected_to": "N/A",
                "revision_direction": "error",
                "revision_scope": "error",
                "early_claim": "N/A",
                "late_claim": "N/A",
                "reason": "IMAGE_NOT_FOUND",
                "question": item["question"],
                "query": item["query"],
                "response": item["response"],
                "answer": item["answer"],
            }

        result = detect_reflection(
            client=client,
            model_name=args.model,
            question=item["query"] or item["question"],
            ground_truth=item["answer"],
            response=item["response"],
            pil_image=pil_img,
            timeout=args.timeout,
        )
        return {
            "id": sample_id,
            "is_correct": result["is_correct"],
            "has_reflection": result["has_reflection"],
            "reflection_type": result["reflection_type"],
            "reflection_evidence": result["reflection_evidence"],
            "earlier_mistake": result["earlier_mistake"],
            "corrected_to": result["corrected_to"],
            "revision_direction": result["revision_direction"],
            "revision_scope": result["revision_scope"],
            "early_claim": result["early_claim"],
            "late_claim": result["late_claim"],
            "reason": result["reason"],
            "raw_judge_output": result["raw_judge_output"],
            "question": item["question"],
            "query": item["query"],
            "response": item["response"],
            "answer": item["answer"],
        }

    new_results: list[dict[str, Any]] = []

    with ThreadPoolExecutor(max_workers=args.parallel) as executor:
        futures = {executor.submit(process_one, it): it for it in todo}
        pbar = tqdm.tqdm(total=len(futures), desc="Analyzing")
        for future in as_completed(futures):
            it = futures[future]
            try:
                new_results.append(future.result())
            except Exception as e:
                new_results.append({
                    "id": it["id"],
                    "is_correct": None,
                    "has_reflection": None,
                    "reflection_type": "error",
                    "reflection_evidence": "N/A",
                    "earlier_mistake": "N/A",
                    "corrected_to": "N/A",
                    "revision_direction": "error",
                    "revision_scope": "error",
                    "early_claim": "N/A",
                    "late_claim": "N/A",
                    "reason": f"ERROR: {e}",
                    "raw_judge_output": "",
                    "question": it["question"],
                    "query": it["query"],
                    "response": it["response"],
                    "answer": it["answer"],
                })

            pbar.update(1)
            all_so_far = list(existing_results.values()) + new_results
            valid = [r for r in all_so_far if r.get("has_reflection") is not None and r.get("is_correct") is not None]
            selected = [r for r in valid if r["is_correct"] and r["has_reflection"]]
            logical = sum(1 for r in selected if r.get("reflection_type") == "logical")
            visual = sum(1 for r in selected if r.get("reflection_type") == "visual")
            pbar.set_postfix_str(
                f"valid={len(valid)} | selected={len(selected)} | logical={logical} | visual={visual}",
                refresh=True,
            )

            if len(new_results) % 20 == 0:
                save_results(args.output, existing_results, new_results, quiet=True)

        pbar.close()

    save_results(args.output, existing_results, new_results, quiet=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect reflection behavior and classify it as logical or visual."
    )
    parser.add_argument("--eval-json", required=True, metavar="FILE", help="Path to eval JSON.")
    parser.add_argument(
        "--benchmark",
        default="mathvista",
        choices=sorted(BENCHMARK_CONFIGS.keys()),
        help="Benchmark type for image loading.",
    )
    parser.add_argument("--image-root", default=None, metavar="DIR", help="Optional local image root.")
    parser.add_argument("--api-key", default=os.getenv("CORRECTION_PROJECT_API_KEY"), metavar="KEY", help="API key for judge model.")
    parser.add_argument(
        "--api-url",
        default=os.getenv("CORRECTION_PROJECT_API_URL", "https://api.openai.com/v1"),
        metavar="URL",
        help="API base URL.",
    )
    parser.add_argument(
        "--azure-endpoint",
        default=None,
        metavar="URL",
        help="When set, use AzureOpenAI at this endpoint instead of the OpenAI-compatible --api-url.",
    )
    parser.add_argument(
        "--azure-api-version",
        default="2024-02-01",
        metavar="VERSION",
        help="Azure OpenAI API version (used only with --azure-endpoint).",
    )
    parser.add_argument(
        "--azure-logid",
        default=os.environ.get("AZURE_LOGID", ""),
        metavar="LOGID",
        help="Optional X-TT-LOGID header for the Azure gateway.",
    )
    parser.add_argument("--model", default="gpt-5.4-mini", metavar="NAME", help="Judge model name.")
    parser.add_argument("--output", default="reflection_analysis.json", metavar="FILE", help="Output JSON path.")
    parser.add_argument("--parallel", type=int, default=8, metavar="N", help="Parallel requests.")
    parser.add_argument("--timeout", type=float, default=120.0, metavar="SECONDS", help="API timeout.")
    parser.add_argument("--max-samples", type=int, default=None, metavar="N", help="Only analyze first N samples.")
    parser.add_argument(
        "--rerun-existing",
        action="store_true",
        help="Rejudge samples already present in --output (use after changing judge labels).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.benchmark not in BENCHMARK_CONFIGS:
        print(f"[ERROR] Unknown benchmark: {args.benchmark}", file=sys.stderr)
        sys.exit(1)
    analyze(args)


if __name__ == "__main__":
    main()
