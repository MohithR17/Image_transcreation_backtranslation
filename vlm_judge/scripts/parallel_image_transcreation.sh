#!/bin/bash
# Parallel evaluation for Image Transcreation task
# Evaluates multiple countries in parallel using available GPUs
#
# Usage:
#   ./parallel_image_transcreation.sh <i2i_model> [num_parallel] [vlm_model]
#
# Examples:
#   ./parallel_image_transcreation.sh flux2-klein 2
#   ./parallel_image_transcreation.sh flux2-klein 4 "Qwen/Qwen2-VL-7B-Instruct"

set -e

# Set HuggingFace cache directory
export HF_HOME=/data/hf_cache/mohithr
export HF_HUB_CACHE=/data/hf_cache/mohithr/hub
export TRANSFORMERS_CACHE=/data/hf_cache/mohithr/hub
export HUGGINGFACE_HUB_CACHE=/data/hf_cache/mohithr/hub

MODEL=${1:-"flux2-klein"}
NUM_PARALLEL=${2:-2}  # Default to 2 parallel processes
VLM_MODEL=${3:-""}    # Optional VLM override

COUNTRIES=(
    "brazil"
    "india"
    "japan"
    "nigeria"
    "portugal"
    "turkey"
    "the-united-states"
)

echo "=========================================="
echo "Parallel Image Transcreation Evaluation"
echo "=========================================="
echo "I2I Model: ${MODEL}"
if [ -n "$VLM_MODEL" ]; then
    echo "VLM: ${VLM_MODEL} (override)"
else
    echo "VLM: (from config - Qwen2-VL-7B)"
fi
echo "Countries: ${#COUNTRIES[@]}"
echo "Parallel processes: ${NUM_PARALLEL}"
echo "HF_HOME: ${HF_HOME}"
echo "HF_HUB_CACHE: ${HF_HUB_CACHE}"
echo "=========================================="
echo ""

# Determine VLM name for results directory
if [ -n "$VLM_MODEL" ]; then
    VLM_NAME=$(echo "$VLM_MODEL" | sed 's|.*/||' | sed 's|-Instruct||')
else
    VLM_NAME="Qwen2-VL-7B"
fi

# Create results directory
RESULTS_DIR="results/image_transcreation/${VLM_NAME}/${MODEL}"
mkdir -p "${RESULTS_DIR}"

echo "Results directory: ${RESULTS_DIR}/"
echo ""

# Function to evaluate a single country on a specific GPU
evaluate_country() {
    local country=$1
    local gpu_id=$2
    local model=$3
    local vlm=$4
    local output_dir=$5
    
    echo "[GPU ${gpu_id}] >>> Evaluating ${country}..."
    
    # Navigate to vlm_judge directory
    cd "$(dirname "$0")/.."
    
    # Build command with GPU selection (HF cache already set via export at top of script)
    cmd="CUDA_VISIBLE_DEVICES=${gpu_id} python run_evaluation.py --config configs/image_transcreation.yaml --model \"${model}\" --country \"${country}\" --output \"${output_dir}/${country}.json\""
    
    # Add VLM override if specified
    if [ -n "$vlm" ]; then
        cmd="$cmd --vlm \"${vlm}\""
    fi
    
    # Execute
    eval $cmd
    
    echo "[GPU ${gpu_id}] ✓ Completed ${country}"
}

export -f evaluate_country

# Get the vlm_judge directory path
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VLM_JUDGE_DIR="$(dirname "$SCRIPT_DIR")"

# Create logs directory for job tracking (relative to vlm_judge dir)
LOGS_DIR="${VLM_JUDGE_DIR}/logs/parallel_${MODEL}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${LOGS_DIR}"

# Track background jobs
pids=()
gpu_counter=0

echo "Logs directory: ${LOGS_DIR}/"
echo "Starting parallel evaluation..."
echo ""

# Launch evaluations in parallel
for country in "${COUNTRIES[@]}"; do
    # Assign GPU (round-robin)
    gpu_id=$((gpu_counter % NUM_PARALLEL))
    
    # Launch in background
    evaluate_country "$country" "$gpu_id" "$MODEL" "$VLM_MODEL" "$RESULTS_DIR" > "${LOGS_DIR}/${country}.log" 2>&1 &
    pids+=($!)
    
    echo "Launched: ${country} on GPU ${gpu_id} (PID: ${pids[-1]})"
    
    gpu_counter=$((gpu_counter + 1))
    
    # If we've reached NUM_PARALLEL jobs, wait for one to complete
    if [ ${#pids[@]} -ge $NUM_PARALLEL ]; then
        # Wait for any job to complete
        wait -n
        # Find which job completed
        for i in "${!pids[@]}"; do
            if ! kill -0 "${pids[$i]}" 2>/dev/null; then
                unset 'pids[$i]'
            fi
        done
        pids=("${pids[@]}") # Reindex array
    fi
done

echo ""
echo "Waiting for remaining jobs to complete..."

# Wait for all remaining jobs
for pid in "${pids[@]}"; do
    wait "$pid"
done

echo ""
echo "=========================================="
echo "Parallel evaluation complete!"
echo "=========================================="
echo "Results directory: ${RESULTS_DIR}/"
echo ""
echo "Result files:"
for country in "${COUNTRIES[@]}"; do
    if [ -f "${RESULTS_DIR}/${country}.json" ]; then
        echo "  ✓ ${country}.json"
    else
        echo "  ✗ ${country}.json (FAILED)"
    fi
done
echo ""

# Show logs for failed jobs
echo "Logs saved in:"
echo "  ${LOGS_DIR}/"
echo ""
echo "To view logs:"
echo "  tail -f ${LOGS_DIR}/*.log          # Follow all logs"
echo "  tail -f ${LOGS_DIR}/brazil.log     # Follow specific country"
echo "  cat ${LOGS_DIR}/brazil.log         # View completed log"
echo ""
