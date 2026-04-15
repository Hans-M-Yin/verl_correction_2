MODEL_PATH="/share5/yzh/qwen2.5-vl-32b"
PORT=18903
SERVED_NAME="qwen2.5-vl-32b"
TENSOR_PARALLEL_SIZE=8
MAX_MODEL_LEN=32768
GPU_MEMORY_UTILIZATION=0.8
echo "AAA Starting vLLM server..."
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 vllm serve $MODEL_PATH \
    --port $PORT \
    --gpu-memory-utilization $GPU_MEMORY_UTILIZATION \
    --max-model-len $MAX_MODEL_LEN \
    --tensor-parallel-size $TENSOR_PARALLEL_SIZE \
    --served-model-name $SERVED_NAME \
    --trust-remote-code \
    --disable-log-requests