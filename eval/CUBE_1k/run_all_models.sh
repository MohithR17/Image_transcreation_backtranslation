#!/bin/bash

# Run T2I models on CUBE_1k dataset
# This script runs all available T2I models on the CUBE dataset
# Usage: 
#   ./run_all_models.sh                    # Run all default models
#   ./run_all_models.sh flux-dev           # Run single model
#   ./run_all_models.sh flux-dev sdxl      # Run multiple specific models

echo "=========================================="
echo "T2I CUBE Evaluation - Running All Models"
echo "=========================================="
echo ""

# Change to the CUBE_1k directory where T2I_cube.py is located
cd "$(dirname "$0")"

# Set PyTorch CUDA memory allocation settings
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Default models to run if no arguments provided
default_models=("flux-dev" "qwen-image-2512")

# If arguments provided, use them as models. Otherwise use defaults.
if [ $# -eq 0 ]; then
    models=("${default_models[@]}")
    echo "No models specified. Running all default models: ${models[*]}"
else
    models=("$@")
    echo "Running specified models: ${models[*]}"
fi

echo ""

# Process each model
for model in "${models[@]}"; do
    echo "=========================================="
    echo "Running model: $model"
    echo "=========================================="
    
    # Run with CUDA memory optimization
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python T2I_cube.py \
        --model "$model" \
        --cube_data data/cube_1k.json \
        --output_dir outputs \
        --seed 42
    
    echo ""
    echo "Completed: $model"
    echo ""
done

echo "=========================================="
echo "All models completed!"
echo "=========================================="
echo "Results saved in: outputs/"
