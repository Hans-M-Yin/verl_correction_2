"""Summarise the visual-centric benchmarks added to ``evaluation_all.py``.

Columns are MMStar accuracy, POPE Random/Popular/Adversarial (F1 by default),
and V*Bench accuracy. A model may have evaluated only a subset of POPE splits:
missing splits are displayed as ``–`` rather than causing the whole row to be
discarded.

Usage
-----
    # Each immediate subdirectory of this root becomes one model/checkpoint row.
    python summary3.py ./eval_results

    # Compare selected result directories with paper-ready labels.
    python summary3.py --eval_dirs \
        ./eval_results/qwen2.5-vl-7b \
        ./eval_results/merged_checkpoints/0423_modify_correction_low_kl_30k_augmented_SPRIOUS_REAL_global_step_360 \
        --row_labels 'Qwen2.5-VL-7B' 'MIRA' --detail

    # Use POPE accuracy rather than F1 for the three POPE columns.
    python summary3.py ./eval_results --pope-metric accuracy
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Optional


_METRIC_FILES = {
    "mmstar": "mmstar_val-metrics.json",
    "pope": "pope-metrics.json",
    "vstar": "vstar-metrics.json",
}
_POPE_SPLITS = ("random", "popular", "adversarial")


def _read_metrics(path: str) -> Optional[dict[str, Any]]:
    if not os.path.isfile(path):
        return None
    try:
        with open(path, encoding="utf-8") as file:
            payload = json.load(file)
        metrics = payload.get("metrics")
        if not isinstance(metrics, dict):
            raise ValueError("missing top-level `metrics` object")
        return metrics
    except Exception as exc:
        print(f"  [WARN] Cannot read {path}: {exc}")
        return None


def _as_float(value: Any) -> Optional[float]:
    return float(value) if isinstance(value, (float, int)) else None


def read_eval_dir(eval_dir: str, pope_metric: str) -> dict[str, Any]:
    mmstar = _read_metrics(os.path.join(eval_dir, _METRIC_FILES["mmstar"]))
    pope = _read_metrics(os.path.join(eval_dir, _METRIC_FILES["pope"]))
    vstar = _read_metrics(os.path.join(eval_dir, _METRIC_FILES["vstar"]))

    row: dict[str, Any] = {
        "mmstar": _as_float(mmstar.get("accuracy")) if mmstar else None,
        "vstar": _as_float(vstar.get("accuracy")) if vstar else None,
        "mmstar_metrics": mmstar,
        "pope_metrics": pope,
        "vstar_metrics": vstar,
    }
    by_split = pope.get("by_split", {}) if pope else {}
    for split in _POPE_SPLITS:
        split_metrics = by_split.get(split, {}) if isinstance(by_split, dict) else {}
        row[f"pope_{split}"] = _as_float(split_metrics.get(pope_metric))
    return row


def _fmt(value: Optional[float]) -> str:
    return "–" if value is None else f"{value * 100:.1f}"


def print_table(rows: list[dict[str, Any]], pope_metric: str) -> None:
    pope_label = "F1" if pope_metric == "f1" else "Acc"
    columns = [
        ("Model / Checkpoint", "label"),
        ("MMStar Acc", "mmstar"),
        (f"POPE-R {pope_label}", "pope_random"),
        (f"POPE-P {pope_label}", "pope_popular"),
        (f"POPE-A {pope_label}", "pope_adversarial"),
        ("V*Bench Acc", "vstar"),
    ]
    widths = {
        heading: max(len(heading), *(len(row["label"]) if key == "label" else len(_fmt(row[key])) for row in rows))
        for heading, key in columns
    }
    header = " | ".join(heading.ljust(widths[heading]) for heading, _ in columns)
    separator = " | ".join("-" * widths[heading] for heading, _ in columns)
    print("| " + header + " |")
    print("| " + separator + " |")
    for row in rows:
        values = []
        for heading, key in columns:
            value = row["label"] if key == "label" else _fmt(row[key])
            values.append(value.ljust(widths[heading]))
        print("| " + " | ".join(values) + " |")


def _format_breakdown(metrics: Optional[dict[str, Any]], field: str) -> str:
    if not metrics:
        return "NOT FOUND"
    by_category = metrics.get(field, {})
    if not isinstance(by_category, dict) or not by_category:
        return "N/A"
    return ", ".join(f"{key}={value * 100:.1f}" for key, value in sorted(by_category.items()) if isinstance(value, (int, float)))


def print_detail(label: str, row: dict[str, Any]) -> None:
    print(f"\n=== Detail: {label} ===")
    mmstar = row["mmstar_metrics"]
    vstar = row["vstar_metrics"]
    pope = row["pope_metrics"]
    if mmstar:
        print(f"  MMStar: {mmstar.get('num_correct')}/{mmstar.get('num_total')} | "
              f"categories: {_format_breakdown(mmstar, 'by_category')}")
    else:
        print("  MMStar: NOT FOUND")
    if pope:
        by_split = pope.get("by_split", {})
        for split in _POPE_SPLITS:
            metrics = by_split.get(split) if isinstance(by_split, dict) else None
            if metrics:
                print(f"  POPE-{split}: {metrics.get('num_correct')}/{metrics.get('num_total')} | "
                      f"Acc={_fmt(_as_float(metrics.get('accuracy')))} | "
                      f"P={_fmt(_as_float(metrics.get('precision')))} | "
                      f"R={_fmt(_as_float(metrics.get('recall')))} | "
                      f"F1={_fmt(_as_float(metrics.get('f1')))} | "
                      f"Yes={_fmt(_as_float(metrics.get('yes_ratio')))}")
            else:
                print(f"  POPE-{split}: NOT RUN")
    else:
        print("  POPE: NOT FOUND")
    if vstar:
        print(f"  V*Bench: {vstar.get('num_correct')}/{vstar.get('num_total')} | "
              f"categories: {_format_breakdown(vstar, 'by_category')}")
    else:
        print("  V*Bench: NOT FOUND")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarise MMStar, POPE, and V*Bench results.")
    parser.add_argument("eval_root", nargs="?", help="Root directory whose immediate subdirectories are rows.")
    parser.add_argument("--eval_dirs", nargs="+", metavar="DIR", help="Explicit evaluation-result directories.")
    parser.add_argument("--row_labels", nargs="*", metavar="LABEL", help="Optional labels matching --eval_dirs.")
    parser.add_argument("--pope-metric", choices=("f1", "accuracy"), default="f1",
                        help="Metric shown in the three POPE table columns (default: f1).")
    parser.add_argument("--detail", action="store_true", help="Print category and POPE metric breakdowns.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if bool(args.eval_root) == bool(args.eval_dirs):
        raise ValueError("Provide exactly one of ROOT_DIR or --eval_dirs.")
    if args.eval_root:
        if not os.path.isdir(args.eval_root):
            raise FileNotFoundError(args.eval_root)
        pairs = [
            (os.path.join(args.eval_root, name), name)
            for name in sorted(os.listdir(args.eval_root))
            if os.path.isdir(os.path.join(args.eval_root, name))
        ]
    else:
        labels = args.row_labels or [os.path.basename(os.path.abspath(path)) for path in args.eval_dirs]
        if len(labels) != len(args.eval_dirs):
            raise ValueError("--row_labels must have the same number of entries as --eval_dirs.")
        pairs = list(zip(args.eval_dirs, labels))

    rows = []
    for eval_dir, label in pairs:
        if not os.path.isdir(eval_dir):
            print(f"  [WARN] Skipping nonexistent directory: {eval_dir}")
            continue
        row = {"label": label, **read_eval_dir(eval_dir, args.pope_metric)}
        rows.append(row)
        if args.detail:
            print_detail(label, row)
    print(f"\n### Visual-centric Benchmark Summary (%, POPE columns: {args.pope_metric})\n")
    if rows:
        print_table(rows, args.pope_metric)
    else:
        print("(no evaluation directories found)")


if __name__ == "__main__":
    main()
