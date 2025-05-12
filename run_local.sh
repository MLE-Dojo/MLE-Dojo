#!/bin/bash

# Read HF_TOKEN from .env file
if [ -f .env ]; then
    source .env
else
    echo "Error: .env file not found"
    exit 1
fi

# Configuration variables
VLLM_GPU=${VLLM_GPU:-0} # Default to 0 if not set
# Default to "1". Can be overridden with space-separated list, e.g., EXP_GPU="1 2"
EXP_GPU=${EXP_GPU:-"1"}
LOCAL_PORT=${LOCAL_PORT:-8314} # Default to 8314 if not set
MODEL=${MODEL:-"meta-llama/Llama-3.1-8B-Instruct"} # Default model
VLLM_HEALTH_CHECK_URL="http://localhost:${LOCAL_PORT}/health"
VLLM_READINESS_TIMEOUT=180 # Timeout in seconds to wait for vLLM

# Check if VLLM_GPU overlaps with any GPU in EXP_GPU
echo "Checking for GPU overlap: VLLM_GPU=$VLLM_GPU, EXP_GPU=\"$EXP_GPU\""
for gpu_index in $EXP_GPU; do
    if [ "$VLLM_GPU" -eq "$gpu_index" ]; then
        echo "Error: VLLM_GPU ($VLLM_GPU) overlaps with one of the EXP_GPU indices ($EXP_GPU)"
        exit 1
    fi
done
echo "No GPU overlap detected."

# Function to clean up the vLLM container
cleanup() {
    echo "Cleaning up vLLM container $container_id..."
    if [ ! -z "$container_id" ]; then
        docker stop "$container_id" > /dev/null
        docker rm "$container_id" > /dev/null
        echo "vLLM container $container_id stopped and removed."
    else
        echo "No container ID found to clean up."
    fi
}

# Set trap to ensure cleanup happens on script exit or interruption
# EXIT: Normal exit
# INT: Interrupt (Ctrl+C)
# TERM: Termination signal
trap cleanup EXIT INT TERM

# Start vLLM in the background and capture the container ID
echo "Starting vLLM server on GPU $VLLM_GPU with model $MODEL..."
container_id=$(docker run -d --runtime nvidia --gpus device=$VLLM_GPU \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --env "HUGGING_FACE_HUB_TOKEN=$HF_TOKEN" \
    -p $LOCAL_PORT:8000 \
    --ipc=host \
    vllm/vllm-openai:latest \
    --model $MODEL 2>&1) # Capture output, including potential errors

# Check if docker run was successful
if [ $? -ne 0 ]; then
    echo "Error starting vLLM container:"
    echo "$container_id" # Print error message from docker run
    # Explicitly nullify container_id since startup failed, prevents cleanup attempt
    container_id=""
    exit 1
fi

# Validate container_id (basic check)
if [ -z "$container_id" ] || [ ${#container_id} -ne 64 ]; then
     echo "Error: Failed to get valid container ID. Docker run output:"
     echo "$container_id"
     container_id="" # Ensure cleanup doesn't run with invalid ID
     exit 1
fi

echo "vLLM container started with ID: $container_id"
echo "Waiting for vLLM server to become ready at $VLLM_HEALTH_CHECK_URL (timeout: ${VLLM_READINESS_TIMEOUT}s)..."

# Wait for vLLM server to be ready
start_time=$(date +%s)
while true; do
    # Use curl to check the health endpoint
    # -s: silent, -f: fail silently (exit code > 0 on HTTP errors), -o /dev/null: discard output
    if curl -sf -o /dev/null "$VLLM_HEALTH_CHECK_URL"; then
        echo "vLLM server is ready."
        break
    fi

    # Check for timeout
    current_time=$(date +%s)
    elapsed_time=$((current_time - start_time))
    if [ "$elapsed_time" -ge "$VLLM_READINESS_TIMEOUT" ]; then
        echo "Error: vLLM server did not become ready within $VLLM_READINESS_TIMEOUT seconds."
        echo "Check vLLM container logs: docker logs $container_id"
        exit 1 # Trap will handle cleanup
    fi

    # Wait before retrying
    sleep 5
done


# Run the experiment script
# Pass the potentially multi-value EXP_GPU variable directly to --gpu-indices
echo "Starting run.py on GPU(s) $EXP_GPU..."
python run.py \
    --gpu-indices "$EXP_GPU" \
    --max-tasks-per-gpu 1 \
    --competitions random-acts-of-pizza \
    --docker-image mle-dojo \
    --task-runtime-seconds 43200 \
    --kaggle-data ./data/prepared \
    --log-dir ./results/logs \
    --output-dir ./results/output

run_py_exit_code=$?

if [ $run_py_exit_code -ne 0 ]; then
    echo "run.py exited with error code $run_py_exit_code."
    # The trap will still execute cleanup
    exit $run_py_exit_code
else
    echo "run.py finished successfully."
    # The trap will execute cleanup upon normal exit
fi

# Note: The 'trap cleanup EXIT INT TERM' ensures cleanup runs automatically here
# No explicit call to cleanup() is needed unless the trap is removed.
