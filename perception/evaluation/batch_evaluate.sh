#!/bin/bash

# ==========================================================
# 批量评测与打分脚本（增强版）
# 用法:
#   1. 跑全部:
#      bash batch_evaluate.sh
#
#   2. 跑单个模型:
#      bash batch_evaluate.sh ./merged_checkpoints ./eval_results model_A
#
#   3. 跑多个模型:
#      bash batch_evaluate.sh ./merged_checkpoints ./eval_results "model_A model_B"
# ==========================================================

# 1. 基础配置
TARGET_ROOT="${1:-./merged_checkpoints}"    # 模型目录
OUT_DIR_ROOT="${2:-./eval_results}"         # 输出目录
SELECTED_MODELS="$3"                        # 用户指定模型（可选）

# API 配置
export CORRECTION_PROJECT_API_KEY="${CORRECTION_PROJECT_API_KEY:-sk-Eek0pnQdHdSPIvwfYMvDoCWoI0H6QoakugWZfCViQeWJlIsD}"
export CORRECTION_PROJECT_API_URL="${CORRECTION_PROJECT_API_URL:-https://yunwu.ai/v1}"

# 2. 检查模型目录
if [ ! -d "$TARGET_ROOT" ]; then
    echo "错误: 目标路径 '$TARGET_ROOT' 不存在！"
    exit 1
fi

mkdir -p "$OUT_DIR_ROOT"

echo "--------------------------------------------------------"
echo "开始批量运行评测和打分程序..."
echo "模型根目录: $TARGET_ROOT"
echo "评测输出根目录: $OUT_DIR_ROOT"
echo "--------------------------------------------------------"

# 3. 确定要处理的模型列表
if [ -z "$SELECTED_MODELS" ]; then
    MODEL_LIST=$(ls "$TARGET_ROOT")
else
    MODEL_LIST=$SELECTED_MODELS
fi

# 4. 遍历模型
for MODEL_NAME in $MODEL_LIST; do
    MODEL_PATH="$TARGET_ROOT/$MODEL_NAME"

    # 跳过不存在的目录
    if [ ! -d "$MODEL_PATH" ]; then
        echo ">>> 跳过: $MODEL_NAME (目录不存在)"
        continue
    fi

    MODEL_OUT_DIR="$OUT_DIR_ROOT/$MODEL_NAME"
    mkdir -p "$MODEL_OUT_DIR"

    echo ">>> 处理模型: $MODEL_NAME"
    echo "    路径: $MODEL_PATH"

    # --- 第一步: Evaluation ---
    echo "    [Step 1/2] 运行基础评测..."

    python evaluation_mathvista.py --checkpoint "$MODEL_PATH" --out-dir "$MODEL_OUT_DIR"
    python evaluation_mathverse.py --checkpoint "$MODEL_PATH" --out-dir "$MODEL_OUT_DIR"
    python evaluation_mathvision.py --checkpoint "$MODEL_PATH" --out-dir "$MODEL_OUT_DIR"
    python evaluation_wemath.py --checkpoint "$MODEL_PATH" --out-dir "$MODEL_OUT_DIR"

    # --- 第二步: Judge ---
    echo "    [Step 2/2] 运行模型打分 (Judge)..."

    python judge_mathvista.py \
        --api_key "$CORRECTION_PROJECT_API_KEY" \
        --api_url "$CORRECTION_PROJECT_API_URL" \
        --eval_basedir "$MODEL_OUT_DIR" \
        --eval_file "mathvista.json"

    python judge_mathverse.py \
        --api_key "$CORRECTION_PROJECT_API_KEY" \
        --api_url "$CORRECTION_PROJECT_API_URL" \
        --eval_basedir "$MODEL_OUT_DIR" \
        --eval_file "mathverse_test.json"

    python judge_mathvision.py \
        --api_key "$CORRECTION_PROJECT_API_KEY" \
        --api_url "$CORRECTION_PROJECT_API_URL" \
        --eval_basedir "$MODEL_OUT_DIR" \
        --eval_file "mathvision_test.json"

    # 可选 wemath judge
    if [ -f "judge_wemath.py" ]; then
        python judge_wemath.py \
            --api_key "$CORRECTION_PROJECT_API_KEY" \
            --api_url "$CORRECTION_PROJECT_API_URL" \
            --eval_basedir "$MODEL_OUT_DIR" \
            --eval_file "wemath_test.json"
    else
        echo "    [Notice] 跳过 judge_wemath.py (文件不存在)"
    fi

    echo "<<< 完成模型: $MODEL_NAME"
    echo "--------------------------------------------------------"
done

echo "批量任务全部完成。"
echo "所有结果保存在: $OUT_DIR_ROOT"