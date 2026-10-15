"""
perception/evaluation/summary2.py
-----------------------------------
Reads judge result JSON files and prints a Markdown table.

Covers: mmmu_pro_overall (= avg of 10-options & vision), logicvista,
        hallusionbench, freak.

Judge file name conventions (mirrors evaluation_all.py):
  mmmu_pro_10     → <dir>/mmmu_pro_10-judge_v4.json
  mmmu_pro_vision → <dir>/mmmu_pro_vision-judge_v4.json
  logicvista      → <dir>/logicvista_test-judge_v4.json
  hallusionbench  → <dir>/hallusionbench_test-judge_v4.json
  freak           → <dir>/freak_test-judge_v4.json

Usage
-----
  # Scan every sub-directory of a root folder (one row each):
  python summary2.py /path/to/results_root

  # Explicit directories:
  python summary2.py --eval_dirs ckpt_step200 ckpt_step400
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, Optional


# ---------------------------------------------------------------------------
# Judge file mapping – must mirror evaluation_all.py naming
# ---------------------------------------------------------------------------

_JUDGE_FILES: Dict[str, str] = {
    "mmmu_pro_10":     "mmmu_pro_10-judge_v4.json",
    "mmmu_pro_vision": "mmmu_pro_vision-judge_v4.json",
    "logicvista":      "logicvista_test-judge_v4.json",
    "hallusionbench":  "hallusionbench_test-judge_v4.json",
    "freak":           "freak_test-judge_v4.json",
}

# mmmu_pro_overall is derived, not a raw file
_DISPLAY_COLUMNS = ["mmmu_pro_overall", "logicvista", "hallusionbench", "freak"]


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def _read_judge_accuracy(filepath: str) -> Optional[float]:
    """Return fraction of 'true' judgements, or None if file is missing."""
    if not os.path.isfile(filepath):
        return None
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            records = json.load(f)
        if not records:
            return None
        items = list(records.values()) if isinstance(records, dict) else records
        num_correct = sum(
            1 for item in items
            if str(item.get("judge_result", "")).strip().lower() == "true"
        )
        return num_correct / len(items)
    except Exception as exc:
        print(f"  [WARN] Could not read {filepath}: {exc}")
        return None


def _eval_dir_results(eval_dir: str) -> Dict[str, Optional[float]]:
    raw: Dict[str, Optional[float]] = {}
    for bench, filename in _JUDGE_FILES.items():
        raw[bench] = _read_judge_accuracy(os.path.join(eval_dir, filename))
    return raw


def _compute_display_row(raw: Dict[str, Optional[float]]) -> Dict[str, Optional[float]]:
    """Derive mmmu_pro_overall and pass through the rest."""
    row: Dict[str, Optional[float]] = dict(raw)
    s10 = raw.get("mmmu_pro_10")
    sv  = raw.get("mmmu_pro_vision")
    if s10 is not None and sv is not None:
        row["mmmu_pro_overall"] = (s10 + sv) / 2.0
    elif s10 is not None:
        row["mmmu_pro_overall"] = s10
    elif sv is not None:
        row["mmmu_pro_overall"] = sv
    else:
        row["mmmu_pro_overall"] = None
    return row


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def _fmt(value: Optional[float]) -> str:
    return "–" if value is None else f"{value * 100:.1f}"


def _print_markdown_table(rows: list[dict]) -> None:
    headers = ["Model / Checkpoint"] + _DISPLAY_COLUMNS
    col_widths = {h: len(h) for h in headers}
    for row in rows:
        col_widths["Model / Checkpoint"] = max(
            col_widths["Model / Checkpoint"], len(row["label"])
        )
        for col in _DISPLAY_COLUMNS:
            col_widths[col] = max(col_widths[col], len(_fmt(row.get(col))))

    def _pad(text: str, key: str) -> str:
        return text.ljust(col_widths[key])

    header_row = " | ".join(_pad(h, h) for h in headers)
    sep_row    = " | ".join("-" * col_widths[h] for h in headers)
    print(f"| {header_row} |")
    print(f"| {sep_row} |")
    for row in rows:
        cells = [_pad(row["label"], "Model / Checkpoint")]
        for col in _DISPLAY_COLUMNS:
            cells.append(_pad(_fmt(row.get(col)), col))
        print("| " + " | ".join(cells) + " |")


def _print_detail(eval_dir: str, raw: Dict[str, Optional[float]]) -> None:
    print(f"\n=== Detail: {eval_dir} ===")
    for bench, filename in _JUDGE_FILES.items():
        val = raw.get(bench)
        tag = f"{val * 100:.2f}%" if val is not None else "NOT FOUND"
        print(f"  {bench:<20} {tag}  ({filename})")
    overall = _compute_display_row(raw).get("mmmu_pro_overall")
    if overall is not None:
        print(f"  {'mmmu_pro_overall':<20} {overall * 100:.2f}%  (avg of 10-options + vision)")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarise extra benchmarks: MMMU-Pro Overall, LogicVista, HallusionBench, FREAK."
    )
    parser.add_argument(
        "eval_root",
        nargs="?",
        default=None,
        metavar="ROOT_DIR",
        help="Parent directory; each immediate sub-directory becomes one table row.",
    )
    parser.add_argument(
        "--eval_dirs",
        nargs="+",
        default=None,
        metavar="DIR",
        help="Explicitly list eval directories (alternative to ROOT_DIR).",
    )
    parser.add_argument(
        "--row_labels",
        nargs="*",
        default=None,
        metavar="LABEL",
        help="Optional display labels matching --eval_dirs order.",
    )
    parser.add_argument(
        "--detail",
        action="store_true",
        help="Print per-file breakdown for each directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.eval_root and args.eval_dirs:
        raise ValueError("Specify either a positional ROOT_DIR or --eval_dirs, not both.")

    if args.eval_root:
        root = args.eval_root
        if not os.path.isdir(root):
            raise FileNotFoundError(f"Root directory not found: {root}")
        subdirs = sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))
        if not subdirs:
            print(f"[WARN] No sub-directories found in {root}")
            return
        eval_pairs = [(os.path.join(root, d), d) for d in subdirs]
    elif args.eval_dirs:
        labels = args.row_labels or [os.path.basename(os.path.abspath(d)) for d in args.eval_dirs]
        if len(labels) != len(args.eval_dirs):
            raise ValueError("--row_labels count must match --eval_dirs count.")
        eval_pairs = list(zip(args.eval_dirs, labels))
    else:
        raise ValueError("Provide either a positional ROOT_DIR or --eval_dirs.")

    table_rows = []
    for eval_dir, label in eval_pairs:
        if not os.path.isdir(eval_dir):
            print(f"[WARN] Directory not found, skipping: {eval_dir}")
            continue
        raw = _eval_dir_results(eval_dir)
        display = _compute_display_row(raw)
        if args.detail:
            _print_detail(eval_dir, raw)
        row = {"label": label, **display}
        table_rows.append(row)

    print("\n### Extended Benchmark Summary (accuracy in %)\n")
    if table_rows:
        _print_markdown_table(table_rows)
    else:
        print("(no results found)")


if __name__ == "__main__":
    main()
