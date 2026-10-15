#!/usr/bin/env bash
# Judge harmful claim revisions on saved MIRA and Vanilla-GRPO outputs.
#
# Each benchmark is judged from saved inference only: no vLLM model is loaded.
# A harmful revision is an explicit correct visual, mathematical, or logical
# claim in early reasoning that the model later revises into an incorrect one.
#
# Optional environment overrides:
#   JUDGE_API_KEY="$AZURE_OPENAI_API_KEY" AZURE_LOGID="$YOUR_AZURE_LOG_ID" \
#   JUDGE_PARALLEL=2 \
#   bash perception/evaluation/run_overcorrection_analysis.sh
#
# Set RERUN_EXISTING=1 to rejudge all samples even if the per-benchmark JSON
# already exists.  By default the underlying judge script resumes safely.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

EVAL_ROOT="${EVAL_ROOT:-perception/evaluation/eval_results}"
MIRA_DIR="${MIRA_DIR:-0423_modify_correction_low_kl_30k_augmented_SPRIOUS_REAL_global_step_360}"
GRPO_DIR="${GRPO_DIR:-0423_run_vanilla_GRPO_30k_global_step_224}"
JUDGE_MODEL="${JUDGE_MODEL:-gpt-5.2-2025-12-11}"
JUDGE_AZURE_ENDPOINT="${JUDGE_AZURE_ENDPOINT:-}"
JUDGE_AZURE_API_VERSION="${JUDGE_AZURE_API_VERSION:-2024-02-01}"
JUDGE_API_KEY="${JUDGE_API_KEY:-${AZURE_OPENAI_API_KEY:-}}"
AZURE_LOGID="${AZURE_LOGID:-}"
JUDGE_PARALLEL="${JUDGE_PARALLEL:-2}"
RERUN_EXISTING="${RERUN_EXISTING:-0}"

if [[ -z "$JUDGE_API_KEY" ]]; then
    echo "Set JUDGE_API_KEY or AZURE_OPENAI_API_KEY before running this script." >&2
    exit 1
fi
if [[ -z "$JUDGE_AZURE_ENDPOINT" ]]; then
    echo "Set JUDGE_AZURE_ENDPOINT before running this script." >&2
    exit 1
fi

labels=("mira" "vanilla_grpo")
dirs=("$MIRA_DIR" "$GRPO_DIR")
benchmarks=("mathvista" "mathverse")

for model_index in "${!labels[@]}"; do
    label="${labels[$model_index]}"
    eval_dir="$EVAL_ROOT/${dirs[$model_index]}"
    if [[ ! -d "$eval_dir" ]]; then
        echo "Missing evaluation directory for $label: $eval_dir" >&2
        exit 1
    fi

    for benchmark in "${benchmarks[@]}"; do
        if [[ "$benchmark" == "mathvista" ]]; then
            eval_json="$eval_dir/mathvista.json"
        else
            eval_json="$eval_dir/mathverse_test.json"
        fi
        output="$eval_dir/reflection_analysis_${benchmark}_harmful_revision.json"

        if [[ ! -f "$eval_json" ]]; then
            echo "Missing saved inference for $label / $benchmark: $eval_json" >&2
            exit 1
        fi

        echo ""
        echo "=== $label | $benchmark ==="
        cmd=(
            python3 perception/evaluation/judge_useful.py
            --eval-json "$eval_json"
            --benchmark "$benchmark"
            --output "$output"
            --model "$JUDGE_MODEL"
            --api-key "$JUDGE_API_KEY"
            --azure-endpoint "$JUDGE_AZURE_ENDPOINT"
            --azure-api-version "$JUDGE_AZURE_API_VERSION"
            --azure-logid "$AZURE_LOGID"
            --parallel "$JUDGE_PARALLEL"
        )
        if [[ "$RERUN_EXISTING" == "1" ]]; then
            cmd+=(--rerun-existing)
        fi
        "${cmd[@]}"
    done
done

SUMMARY_OUTPUT="$EVAL_ROOT/harmful_revision_summary_mira_vs_vanilla_grpo.json"
python3 - "$EVAL_ROOT" "$MIRA_DIR" "$GRPO_DIR" "$SUMMARY_OUTPUT" <<'PY'
import json
import sys
from pathlib import Path

eval_root, mira_dir, grpo_dir, output_path = map(Path, sys.argv[1:])
models = {"MIRA": mira_dir, "Vanilla GRPO": grpo_dir}
benchmarks = ("mathvista", "mathverse")
rows = []

for label, model_dir in models.items():
    for benchmark in benchmarks:
        path = eval_root / model_dir / f"reflection_analysis_{benchmark}_harmful_revision.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        stats = payload["stats"]
        rows.append({
            "model": label,
            "benchmark": benchmark,
            "valid_judged": stats["valid_judged"],
            "explicit_revisions": stats["n_explicit_revisions"],
            "correct_to_wrong": stats["n_harmful_revisions_correct_to_wrong"],
            "harmful_visual": stats["n_harmful_visual_revisions"],
            "harmful_logical": stats["n_harmful_logical_revisions"],
            "harmful_mixed": stats["n_harmful_mixed_revisions"],
            "rate_among_valid": stats["harmful_revision_rate_among_valid"],
            "rate_given_revision": stats["harmful_revision_rate_given_revision"],
            "rate_given_reflection": stats["harmful_revision_rate_given_reflection"],
            "per_sample_file": str(path),
        })

output = {"definition": "Explicit early correct visual, mathematical, or logical claim later revised into an incorrect claim.", "rows": rows}
output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")

print("\n=== Harmful claim-revision summary ===")
print("Model         Benchmark  C→W / valid  Explicit revisions  Visual/Logical  Rate / valid  Rate / revision")
for row in rows:
    rate_valid = "N/A" if row["rate_among_valid"] is None else f"{100 * row['rate_among_valid']:.2f}%"
    rate_revision = "N/A" if row["rate_given_revision"] is None else f"{100 * row['rate_given_revision']:.2f}%"
    print(
        f"{row['model']:<13} {row['benchmark']:<10} "
        f"{row['correct_to_wrong']}/{row['valid_judged']:<11} "
        f"{row['explicit_revisions']:<19} {row['harmful_visual']}/{row['harmful_logical']:<14} {rate_valid:<13} {rate_revision}"
    )
print(f"\nSummary saved to: {output_path}")
PY
