"""Repeat Qwen-judge evaluation over already saved ``evaluation_all.py`` outputs.

This script deliberately performs **no model inference**.  ``evaluation_all.py``
uses deterministic greedy decoding, so its saved generation files are the
canonical predictions for a checkpoint.  For every checkpoint-name/benchmark,
this runner copies that saved generation file into a repeat directory and sends
it to an OpenAI-compatible judge (normally a locally deployed Qwen3-VL-32B)
``--repeat`` times.  It then reports mean, sample standard deviation, and
sample variance of the judge accuracy.

The checkpoint-list is UTF-8 text with one *directory name under
--eval-results-dir* per line.  Blank lines and lines starting with ``#`` are
ignored.  For example, if saved results live in
``perception/evaluation/eval_results/foo/mathvista.json``, the list line is
simply ``foo``.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
from pathlib import Path
from typing import Any

from openai import OpenAI

import evaluation_all as evaluation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint-list", required=True,
                        help="TXT file: one existing eval_results checkpoint-directory name per line.")
    parser.add_argument("--eval-results-dir", default="perception/evaluation/eval_results",
                        help="Parent directory containing existing evaluation_all.py result folders.")
    parser.add_argument("--benchmarks", default="mathvista,mathverse,mathvision,wemath",
                        help="Same comma-separated judge-based benchmark names accepted by evaluation_all.py.")
    parser.add_argument("--repeat", type=int, default=3,
                        help="Independent Qwen-judge passes for each saved generation file.")
    parser.add_argument("--out-dir", required=True,
                        help="New directory for repeated judge outputs and the aggregate summary.")
    parser.add_argument("--judge-model-name", default="Qwen3-VL-32B")
    parser.add_argument("--judge-temperature", type=float, default=0.7,
                        help="Qwen judge temperature for both extraction and correctness calls.")
    parser.add_argument("--api-key", default=os.getenv("JUDGE_API_KEY", "EMPTY"))
    parser.add_argument("--api-url", default=os.getenv("JUDGE_API_URL", os.getenv("CORRECTION_PROJECT_API_URL")),
                        help="OpenAI-compatible Qwen3-VL-32B judge base URL, e.g. http://HOST:PORT/v1.")
    parser.add_argument("--judge-parallel-num", type=int, default=8)
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args()


def load_checkpoint_names(path: str) -> list[str]:
    names: list[str] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        value = line.strip()
        if value and not value.startswith("#"):
            if Path(value).name != value:
                raise ValueError(
                    f"Checkpoint list entries must be directory names, not paths: {value!r}. "
                    "Use --eval-results-dir to choose the parent directory."
                )
            names.append(value)
    if not names:
        raise ValueError(f"No checkpoint names found in {path}.")
    return names


def aggregate_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    accuracies = [run["accuracy"] for run in runs if run.get("status") == "success"]
    if not accuracies:
        return {"status": "failed", "runs": runs}
    return {
        "status": "success",
        "runs": runs,
        "n_successful_runs": len(accuracies),
        "accuracies": accuracies,
        "mean": round(statistics.mean(accuracies), 6),
        "std": round(statistics.stdev(accuracies), 6) if len(accuracies) > 1 else 0.0,
        "variance": round(statistics.variance(accuracies), 8) if len(accuracies) > 1 else 0.0,
        "min": round(min(accuracies), 6),
        "max": round(max(accuracies), 6),
    }


def judge_saved_benchmark(
    spec: Any,
    source_checkpoint_dir: Path,
    output_checkpoint_dir: Path,
    client: OpenAI,
    args: argparse.Namespace,
) -> dict[str, Any]:
    if spec.local_evaluator is not None:
        raise ValueError(
            f"{spec.name} has a deterministic local evaluator, not an LLM judge. "
            "It does not need judge-repeat error bars."
        )
    source_file = source_checkpoint_dir / spec.result_filename
    if not source_file.is_file():
        raise FileNotFoundError(f"Saved generation file not found: {source_file}")

    print(f"Starting {spec.name}: reuse {source_file.name}, judge {args.repeat} times")
    runs: list[dict[str, Any]] = []
    for repeat_index in range(args.repeat):
        repeat_file = output_checkpoint_dir / spec.name / f"repeat_{repeat_index + 1:02d}" / spec.result_filename
        repeat_file.parent.mkdir(parents=True, exist_ok=True)
        try:
            # A unique copy prevents run_judge_pipeline from overwriting prior
            # repeat judge files, while guaranteeing identical model outputs.
            shutil.copy2(source_file, repeat_file)
            judge_summary = evaluation.run_judge_pipeline(
                spec=spec,
                result_filepath=str(repeat_file),
                client=client,
                judge_model_name=args.judge_model_name,
                judge_parallel_num=args.judge_parallel_num,
                judge_temperature=args.judge_temperature,
            )
            runs.append({"repeat": repeat_index + 1, "status": "success", **judge_summary})
        except Exception as exc:
            runs.append({"repeat": repeat_index + 1, "status": "failed", "error": repr(exc)})
            print(f"[{spec.name}] judge repeat {repeat_index + 1} failed: {exc}")

    aggregate = aggregate_runs(runs)
    aggregate.update({
        "benchmark": spec.name,
        "source_generation_file": str(source_file),
    })
    if aggregate["status"] == "success":
        print(f"[{spec.name}] mean={aggregate['mean']:.4f}, std={aggregate['std']:.4f}, "
              f"runs={aggregate['accuracies']}")
    return aggregate


def main() -> None:
    args = parse_args()
    if args.repeat < 1:
        raise ValueError("--repeat must be >= 1.")
    checkpoint_names = load_checkpoint_names(args.checkpoint_list)
    specs = evaluation.resolve_benchmark_specs(args.benchmarks)
    eval_results_dir = Path(args.eval_results_dir)
    root = Path(args.out_dir)
    root.mkdir(parents=True, exist_ok=True)
    client = OpenAI(api_key=args.api_key, base_url=args.api_url)

    summary: dict[str, Any] = {
        "checkpoint_list": args.checkpoint_list,
        "eval_results_dir": str(eval_results_dir),
        "benchmarks": [spec.name for spec in specs],
        "repeat": args.repeat,
        "mode": "judge_only_reusing_saved_generations",
        "judge": {
            "model": args.judge_model_name,
            "api_url": args.api_url,
            "parallel": args.judge_parallel_num,
            "temperature": args.judge_temperature,
        },
        "checkpoints": [],
    }
    for checkpoint_name in checkpoint_names:
        source_checkpoint_dir = eval_results_dir / checkpoint_name
        output_checkpoint_dir = root / checkpoint_name
        output_checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_summary: dict[str, Any] = {
            "checkpoint_name": checkpoint_name,
            "source_dir": str(source_checkpoint_dir),
            "out_dir": str(output_checkpoint_dir),
            "benchmarks": {},
        }
        print(f"\n{'#' * 100}\nCheckpoint results: {source_checkpoint_dir}")
        for spec in specs:
            try:
                checkpoint_summary["benchmarks"][spec.name] = judge_saved_benchmark(
                    spec, source_checkpoint_dir, output_checkpoint_dir, client, args
                )
            except Exception as exc:
                checkpoint_summary["benchmarks"][spec.name] = {"status": "failed", "error": repr(exc)}
                print(f"[{spec.name}] FAILED: {exc}")
                if not args.continue_on_error:
                    summary["checkpoints"].append(checkpoint_summary)
                    (root / "checkpoint_repeat_summary.json").write_text(
                        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
                    )
                    raise
        summary["checkpoints"].append(checkpoint_summary)
        (root / "checkpoint_repeat_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    print("\n" + "=" * 100)
    for checkpoint in summary["checkpoints"]:
        print(f"Checkpoint: {checkpoint['checkpoint_name']}")
        for benchmark, result in checkpoint["benchmarks"].items():
            if result.get("status") == "success":
                print(f"  {benchmark}: {result['mean']:.4f} ± {result['std']:.4f} "
                      f"(variance={result['variance']:.8f}, n={result['n_successful_runs']})")
            else:
                print(f"  {benchmark}: FAILED")
    print(f"Saved summary: {root / 'checkpoint_repeat_summary.json'}")


if __name__ == "__main__":
    main()
