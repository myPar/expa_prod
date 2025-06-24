#!/bin/bash

for arg in "$@"; do
  case $arg in
    hf_token=*)
      hf_token="${arg#*=}"
      ;;
    hf_hub_offline=*)
      hf_hub_offline="${arg#*=}"
      ;;
    model_name=*)
      model_name="${arg#*=}"
      ;;
    openai_api_key=*)
      openai_api_key="${arg#*=}"
      ;;
    model_load_timeout=*)
      model_load_timeout="${arg#*=}"
      ;;
    vllm_port=*)
      proxy_port="${arg#*=}"
      ;;      
    gpu_count=*)
      gpu_count="${arg#*=}"
      ;;
    hf_home=*)
      hf_home="${arg#*=}"
      ;;  
    vllm_cache*)
      vllm_cache="${arg#*=}"
      ;;        
    *)
      echo "Unknown argument: $arg"
      ;;
  esac
done

# Set necessary env variables from the input args
export HF_TOKEN="${hf_token}"
export HF_HUB_OFFLINE="${hf_hub_offline:-0}"
export MODEL_NAME="${model_name:-RedHatAI/DeepSeek-R1-Distill-Qwen-32B-quantized.w4a16}"
export OPENAI_API_KEY="${openai_api_key:-token-abc123}"
export MODEL_LOAD_TIMEOUT="${model_load_timeout:-600}"  # seconds to wait for vLLM to be ready default - 10m
export GPU_COUNT="${gpu_count:-1}"

export MODEL_SERVER_PORT="${vllm_port:-8001}"

export HF_HOME="${hf_home:-$HOME/.cache/huggingface}"
export VLLM_CACHE_ROOT="${vllm_cache:-$HOME/.cache/vllm}"

echo "--- argument's values ---"
echo "hf_token=$HF_TOKEN"
echo "hf_hub_offline=$HF_HUB_OFFLINE"
echo "model_name=$MODEL_NAME"
echo "openai_api_key=$OPENAI_API_KEY"
echo "model_load_timeout=$MODEL_LOAD_TIMEOUT"
echo "gpu_count=$GPU_COUNT"
echo "vllm server port=$MODEL_SERVER_PORT"
echo "vllm cache dir=$VLLM_CACHE_ROOT"
echo "hf home dir=$HF_HOME"
echo "-------------------------"

# Start vLLM server
echo "Starting vLLM server with model: $MODEL_NAME"
vllm serve "$MODEL_NAME" \
    --enable-reasoning \
    --reasoning-parser deepseek_r1 \
    --api-key="$OPENAI_API_KEY" \
    --tensor-parallel-size="$GPU_COUNT" \
    --port $MODEL_SERVER_PORT &

VLLM_PID=$!
echo "vLLM PID: $VLLM_PID"
echo $VLLM_PID > vllm.pid

# Wait for vLLM server to be ready
echo "Waiting for vLLM server to start..."
for ((i=1; i<=MODEL_LOAD_TIMEOUT; i++)); do
    if curl -s -H "Authorization: Bearer $OPENAI_API_KEY" http://localhost:$MODEL_SERVER_PORT/v1/models | grep -q '"object"'; then
        echo "✅ vLLM server is up!"
        break
    fi
    echo "⏳ vLLM server not ready yet... retrying ($i)"
    sleep 1
done
