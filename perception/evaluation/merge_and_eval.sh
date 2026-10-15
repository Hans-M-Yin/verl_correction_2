#!/bin/bash

# ==========================================================
# Checkpoint 合并 + 自动评估脚本
# ==========================================================

# 1. 基础配置
CHECKPOINTS_BASE="./checkpoints"
PROJECT_NAME="${1:-verl_correction}"
TARGET_ROOT="${2:-./merged_checkpoints}"
SELECTED_EXPS="$3"

# 路径展开
CHECKPOINTS_BASE="${CHECKPOINTS_BASE/#\~/$HOME}"
TARGET_ROOT="${TARGET_ROOT/#\~/$HOME}"

PROJECT_PATH="$CHECKPOINTS_BASE/$PROJECT_NAME"

if [ ! -d "$PROJECT_PATH" ]; then
    echo "错误: 项目路径 '$PROJECT_PATH' 不存在！"
    exit 1
fi

mkdir -p "$TARGET_ROOT"

echo "--------------------------------------------------------"
echo "正在扫描项目: $PROJECT_NAME"
echo "源地址: $PROJECT_PATH"
echo "输出目录: $TARGET_ROOT"
echo "--------------------------------------------------------"

# 2. 确定 EXP
if [ -z "$SELECTED_EXPS" ]; then
    EXP_LIST=$(ls "$PROJECT_PATH")
else
    EXP_LIST=$SELECTED_EXPS
fi

# 3. 遍历 EXP
for EXP_NAME in $EXP_LIST; do
    EXP_PATH="$PROJECT_PATH/$EXP_NAME"

    if [ ! -d "$EXP_PATH" ]; then
        echo ">>> 跳过: $EXP_NAME (目录不存在)"
        continue
    fi

    echo "发现实验: $EXP_NAME"

    # 4. 遍历 STEP
    for STEP_PATH in "$EXP_PATH"/*; do
        if [ -d "$STEP_PATH" ]; then
            STEP_NAME=$(basename "$STEP_PATH")

            TARGET_DIR_NAME="${EXP_NAME}_${STEP_NAME}"
            FULL_TARGET_PATH="$TARGET_ROOT/$TARGET_DIR_NAME"

            echo "  [合并] $STEP_NAME -> $TARGET_DIR_NAME"

            python -m verl.model_merger merge \
                --backend fsdp \
                --local_dir "${STEP_PATH}/actor" \
                --target_dir "$FULL_TARGET_PATH"

            if [ $? -eq 0 ]; then
                echo "  [合并成功]"

                # ==============================
                # ⭐ 立即进行评估
                # ==============================
                echo "  [开始评估] $TARGET_DIR_NAME"

                python perception/evaluation/evaluation_all.py \
                    --checkpoint "$FULL_TARGET_PATH" \
                    --benchmarks "mathvista,mathverse,mathvision,wemath" \
                    --continue-on-error \
                    --out-dir "./perception/evaluation/eval_results/$TARGET_DIR_NAME" \
                    --api-key "sk-Eek0pnQdHdSPIvwfYMvDoCWoI0H6QoakugWZfCViQeWJlIsD" \
                    --api-url "https://yunwu.ai/v1"
                if [ $? -eq 0 ]; then
                    echo "  [评估完成]"
                else
                    echo "  [评估失败，但继续]"
                fi

            else
                echo "  [合并失败] 跳过评估..."
            fi

            echo "  ---"
        fi
    done
done

echo "--------------------------------------------------------"
echo "全部处理完成。"