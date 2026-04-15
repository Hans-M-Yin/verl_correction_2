#!/bin/bash

# ==========================================================
# Checkpoint 合并脚本（支持指定 EXP）
#
# 用法:
# 1. 默认处理全部
#    bash merge_checkpoints.sh
#
# 2. 指定项目
#    bash merge_checkpoints.sh verl_correction
#
# 3. 指定保存目录
#    bash merge_checkpoints.sh verl_correction ./merged_checkpoints
#
# 4. 指定 EXP（单个或多个）
#    bash merge_checkpoints.sh verl_correction ./merged_checkpoints "exp1 exp2"
# ==========================================================

# 1. 基础配置
CHECKPOINTS_BASE="./checkpoints"
PROJECT_NAME="${1:-verl_correction}"
TARGET_ROOT="${2:-./merged_checkpoints}"
SELECTED_EXPS="$3"   # ⭐ 新增：用户指定 EXP

# 处理路径中的 ~
CHECKPOINTS_BASE="${CHECKPOINTS_BASE/#\~/$HOME}"
TARGET_ROOT="${TARGET_ROOT/#\~/$HOME}"

# 2. 检查项目路径
PROJECT_PATH="$CHECKPOINTS_BASE/$PROJECT_NAME"
if [ ! -d "$PROJECT_PATH" ]; then
    echo "错误: 项目路径 '$PROJECT_PATH' 不存在！"
    exit 1
fi

mkdir -p "$TARGET_ROOT"

echo "--------------------------------------------------------"
echo "正在扫描项目: $PROJECT_NAME"
echo "源地址: $PROJECT_PATH"
echo "--------------------------------------------------------"

# 3. 确定要处理的 EXP 列表
if [ -z "$SELECTED_EXPS" ]; then
    EXP_LIST=$(ls "$PROJECT_PATH")
else
    EXP_LIST=$SELECTED_EXPS
fi

# 4. 遍历 EXP
for EXP_NAME in $EXP_LIST; do
    EXP_PATH="$PROJECT_PATH/$EXP_NAME"

    if [ ! -d "$EXP_PATH" ]; then
        echo ">>> 跳过: $EXP_NAME (目录不存在)"
        continue
    fi

    echo "发现实验: $EXP_NAME"

    # 5. 遍历 STEP
    for STEP_PATH in "$EXP_PATH"/*; do
        if [ -d "$STEP_PATH" ]; then
            STEP_NAME=$(basename "$STEP_PATH")

            TARGET_DIR_NAME="${EXP_NAME}_${STEP_NAME}"
            FULL_TARGET_PATH="$TARGET_ROOT/$TARGET_DIR_NAME"

            echo "  [处理中] $STEP_NAME -> $TARGET_DIR_NAME"

            python -m verl.model_merger merge \
                --backend fsdp \
                --local_dir "${STEP_PATH}/actor" \
                --target_dir "$FULL_TARGET_PATH"

            if [ $? -eq 0 ]; then
                echo "  [成功]"
            else
                echo "  [失败] 跳过..."
            fi
            echo "  ---"
        fi
    done
done

echo "--------------------------------------------------------"
echo "处理完成。"