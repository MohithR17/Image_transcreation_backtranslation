#!/bin/bash
# Batch evaluate all T2I models for CUBE_1k task

set -e

# Get VLM model (default or from argument)
VLM="${1:-Qwen/Qwen2-VL-2B-Instruct}"
echo "Using VLM: $VLM"

# Extract VLM name for directory structure
VLM_NAME=$(echo "$VLM" | cut -d'/' -f2 | sed 's/-Instruct//')
echo "VLM name: $VLM_NAME"

# T2I models to evaluate (must match directory names in eval/CUBE_1k/outputs/)
T2I_MODELS=(
    "flux-dev"
    "qwen-image-2512"
)

echo "=================================================="
echo "Batch T2I CUBE Evaluation"
echo "=================================================="
echo "T2I Models: ${T2I_MODELS[*]}"
echo "VLM Judge: $VLM"
echo "=================================================="
echo ""

# Run evaluation for each model
for model in "${T2I_MODELS[@]}"; do
    echo "Starting evaluation for T2I model: $model"
    
    python run_evaluation.py \
        --config configs/t2i_cube.yaml \
        --model "$model" \
        --vlm "$VLM" \
        --output "results/t2i_cube/${VLM_NAME}/${model}/evaluation.json"
    
    echo "✓ Completed: $model"
    echo ""
done

echo "=================================================="
echo "All T2I model evaluations complete!"
echo "Results saved to: results/t2i_cube/${VLM_NAME}/"
echo "=================================================="
