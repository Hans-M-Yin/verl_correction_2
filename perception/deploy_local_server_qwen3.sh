MODEL_PATH="/data/yzh/qwen3-vl-30b-a3b-instruct"
PORT=18903
SERVED_NAME="qwen3-vl-30b-a3b"
TENSOR_PARALLEL_SIZE=4
MAX_MODEL_LEN=16000
GPU_MEMORY_UTILIZATION=0.8
MAX_NUM_SEQS=256
echo "AAA Starting vLLM server..."
CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve $MODEL_PATH \
    --port $PORT \
    --gpu-memory-utilization $GPU_MEMORY_UTILIZATION \
    --max-model-len $MAX_MODEL_LEN \
    --tensor-parallel-size $TENSOR_PARALLEL_SIZE \
    --served-model-name $SERVED_NAME \
    --trust-remote-code \
    --disable-log-requests \
    --max-num-seqs $MAX_NUM_SEQS \
