MODEL_PATH="/data/yzh/qwen3_vl_4b_thinking"
PORT=18903
SERVED_NAME="qwen3-vl-4b-thinking"
TENSOR_PARALLEL_SIZE=2
MAX_MODEL_LEN=20000
GPU_MEMORY_UTILIZATION=0.8
MAX_NUM_SEQS=128
echo "AAA Starting vLLM server..."

# Original transformers in verl_env:4.54.0
CUDA_VISIBLE_DEVICES=4,5 vllm serve $MODEL_PATH \
    --port $PORT \
    --gpu-memory-utilization $GPU_MEMORY_UTILIZATION \
    --max-model-len $MAX_MODEL_LEN \
    --tensor-parallel-size $TENSOR_PARALLEL_SIZE \
    --served-model-name $SERVED_NAME \
    --trust-remote-code \
    --disable-log-requests \
    --max-num-seqs $MAX_NUM_SEQS \
