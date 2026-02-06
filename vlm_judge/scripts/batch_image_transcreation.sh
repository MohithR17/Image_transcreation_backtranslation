#!/bin/bash
# Batch evaluation for Image Transcreation task
# Evaluates a specific I2I model across all countries
#
# Usage:
#   ./batch_image_transcreation.sh <i2i_model> [vlm_model]
#
# Examples:
#   ./batch_image_transcreation.sh flux2-klein
#   ./batch_image_transcreation.sh flux2-klein "Qwen/Qwen2-VL-2B-Instruct"

set -e

MODEL=${1:-"flux2-klein"}
VLM_MODEL=${2:-""}  # Optional VLM override

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
echo "Image Transcreation Batch Evaluation"
echo "=========================================="
echo "I2I Model: ${MODEL}"
if [ -n "$VLM_MODEL" ]; then
    echo "VLM: ${VLM_MODEL} (override)"
else
    echo "VLM: (from config)"
fi
echo "Countries: ${#COUNTRIES[@]}"
echo "=========================================="
echo ""

# Determine VLM name for results directory
if [ -n "$VLM_MODEL" ]; then
    # Extract VLM name from model string (e.g., "Qwen/Qwen2-VL-7B-Instruct" -> "Qwen2-VL-7B")
    VLM_NAME=$(echo "$VLM_MODEL" | sed 's|.*/||' | sed 's|-Instruct||')
else
    # Use default from config
    VLM_NAME="default"
fi

# Create results directory structure: results/image_transcreation/{vlm}/{i2i_model}/
RESULTS_DIR="results/image_transcreation/${VLM_NAME}/${MODEL}"
mkdir -p "${RESULTS_DIR}"

echo "Results directory: ${RESULTS_DIR}/"
echo ""

# Evaluate each country
for country in "${COUNTRIES[@]}"; do
    echo ">>> Evaluating ${country}..."
    
    # Build command
    cmd="python run_evaluation.py --config configs/image_transcreation.yaml --model \"${MODEL}\" --country \"${country}\" --output \"${RESULTS_DIR}/${country}.json\""
    
    # Add VLM override if specified
    if [ -n "$VLM_MODEL" ]; then
        cmd="$cmd --vlm \"${VLM_MODEL}\""
    fi
    
    # Execute
    eval $cmd
    
    echo "✓ Completed ${country}"
    echo ""
done

echo "=========================================="
echo "Batch evaluation complete!"
echo "=========================================="
echo "Results directory: ${RESULTS_DIR}/"
echo ""
echo "Result files:"
for country in "${COUNTRIES[@]}"; do
    echo "  - ${country}.json"
done
echo ""
