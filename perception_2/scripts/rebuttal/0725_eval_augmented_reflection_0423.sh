#!/usr/bin/env bash
# Evaluate explicit visual self-correction on the augmented data used by
# 0724_qwen2_vl_7b.sh. Run this script from any directory.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

# Match the two-GPU convention used by 0724_visual_bench.sh. Override before
# launching if these devices are occupied, e.g. CUDA_VISIBLE_DEVICES=2,3 bash ...
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6,7}"

# The judge service is intentionally configured through the environment: do
# not commit an API key to this repository.
JUDGE_API_KEY="EMPTY"
# : "${JUDGE_API_KEY:?Set JUDGE_API_KEY before running this script.}"
JUDGE_API_URL="${JUDGE_API_URL:-http://localhost:6658/v1}"
JUDGE_MODEL="${JUDGE_MODEL:-Qwen3-VL-32B}"
NUM_SAMPLES="${NUM_SAMPLES:-1000}"
NUM_NORMAL_SAMPLES="${NUM_NORMAL_SAMPLES:-1000}"
SEED="${SEED:-42}"
JUDGE_PARALLEL_NUM="${JUDGE_PARALLEL_NUM:-2}"

# This is the exact augmented train parquet used by 0724_qwen2_vl_7b.sh.
DATA_FILE="_preprocessed_dataset/0416_agumented/train.parquet"
CHECKPOINT="../models/qwen2.5-vl-7b"
OUT_DIR="perception/evaluation/augmented_reflection_results/qwen2.5-vl-7b_aug_${NUM_SAMPLES}_normal_${NUM_NORMAL_SAMPLES}"

CUDA_VISIBLE_DEVICES=6,7 python perception/evaluation/evaluate_augmented_reflection.py \
  --data-file "$DATA_FILE" \
  --model "MIRA=$CHECKPOINT" \
  --out-dir "$OUT_DIR" \
  --max-samples "$NUM_SAMPLES" \
  --num-normal-samples "$NUM_NORMAL_SAMPLES" \
  --seed "$SEED" \
  --tensor-parallel-size 2 \
  --judge-model "$JUDGE_MODEL" \
  --api-key "$JUDGE_API_KEY" \
  --api-url "$JUDGE_API_URL" \
  --judge-parallel-num "$JUDGE_PARALLEL_NUM"
