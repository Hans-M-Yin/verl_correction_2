## 1. 赋予执行权限
#chmod +x batch_evaluate.sh
#
## 2. 设置 API 环境变量并执行
#export CORRECTION_PROJECT_API_KEY="你的API_KEY"
#export CORRECTION_PROJECT_API_URL="你的API_URL"
#
## 3. 运行脚本 (第一个参数是模型根目录，第二个是输出根目录)
#bash batch_evaluate.sh ./merged_checkpoints ./eval_results


#!/bin/bash

# 1. 基础配置
TARGET_ROOT="${1:-./merged_checkpoints}"    # 模型目录
OUT_DIR_ROOT="${2:-./eval_results}"         # 总输出目录

# API 配置 (可以直接在此修改，或执行前 export)
export CORRECTION_PROJECT_API_KEY="${CORRECTION_PROJECT_API_KEY:-sk-Eek0pnQdHdSPIvwfYMvDoCWoI0H6QoakugWZfCViQeWJlIsD}"
export CORRECTION_PROJECT_API_URL="${CORRECTION_PROJECT_API_URL:-https://yunwu.ai/v1}"

# 2. 检查路径
if [ ! -d "$TARGET_ROOT" ]; then
    echo "错误: 目标路径 '$TARGET_ROOT' 不存在！"
    exit 1
fi

mkdir -p "$OUT_DIR_ROOT"

# 3. 遍历模型文件夹
for MODEL_PATH in "$TARGET_ROOT"/*; do
    if [ -d "$MODEL_PATH" ]; then
        MODEL_NAME=$(basename "$MODEL_PATH")
        MODEL_OUT_DIR="$OUT_DIR_ROOT/$MODEL_NAME"
        mkdir -p "$MODEL_OUT_DIR"

        echo ">>> 正在处理模型: $MODEL_NAME"

        # --- 评测阶段 ---
        python evaluation_mathvista.py --checkpoint "$MODEL_PATH" --out-dir "$MODEL_OUT_DIR"
        python evaluation_mathverse.py --checkpoint "$MODEL_PATH" --out-dir "$MODEL_OUT_DIR"
        python evaluation_mathvision.py --checkpoint "$MODEL_PATH" --out-dir "$MODEL_OUT_DIR"
        python evaluation_wemath.py --checkpoint "$MODEL_PATH" --out-dir "$MODEL_OUT_DIR"

        # --- 打分阶段 ---
        python judge_mathvista.py --api_key "$CORRECTION_PROJECT_API_KEY" --api_url "$CORRECTION_PROJECT_API_URL" --eval_basedir "$MODEL_OUT_DIR" --eval_file "mathvista.json"
        python judge_mathverse.py --api_key "$CORRECTION_PROJECT_API_KEY" --api_url "$CORRECTION_PROJECT_API_URL" --eval_basedir "$MODEL_OUT_DIR" --eval_file "mathverse_test.json"
        python judge_mathvision.py --api_key "$CORRECTION_PROJECT_API_KEY" --api_url "$CORRECTION_PROJECT_API_URL" --eval_basedir "$MODEL_OUT_DIR" --eval_file "mathvision_test.json"

        if [ -f "judge_wemath.py" ]; then
            python judge_wemath.py --api_key "$CORRECTION_PROJECT_API_KEY" --api_url "$CORRECTION_PROJECT_API_URL" --eval_basedir "$MODEL_OUT_DIR" --eval_file "wemath_test.json"
        fi

        echo "<<< 模型 $MODEL_NAME 处理完成"
        echo "--------------------------------------------------------"
    fi
done
