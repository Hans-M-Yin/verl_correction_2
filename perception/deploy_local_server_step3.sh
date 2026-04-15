MODEL_PATH="/share3/yzh/qwen2.5-vl-7b"
PORT=18903
SERVED_NAME="qwen2.5-vl-7b"
TENSOR_PARALLEL_SIZE=1
MAX_MODEL_LEN=30000
GPU_MEMORY_UTILIZATION=0.9
MAX_NUM_SEQS=256
echo "AAA Starting vLLM server..."
CUDA_VISIBLE_DEVICES=3 vllm serve $MODEL_PATH \
    --port $PORT \
    --gpu-memory-utilization $GPU_MEMORY_UTILIZATION \
    --max-model-len $MAX_MODEL_LEN \
    --tensor-parallel-size $TENSOR_PARALLEL_SIZE \
    --served-model-name $SERVED_NAME \
    --trust-remote-code \
    --disable-log-requests \
    --seed 42 \
    --max-num-seqs $MAX_NUM_SEQS \
