#!/bin/bash
#SBATCH --job-name=vlm_img_trans
#SBATCH --partition=general
#SBATCH --qos=normal
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=2-00:00:00
#SBATCH --output=/home/mohithr/independent_study/Image_transcreation_backtranslation/vlm_judge/logs/batch_img_trans_%j.out
#SBATCH --error=/home/mohithr/independent_study/Image_transcreation_backtranslation/vlm_judge/logs/batch_img_trans_%j.err

# Batch Image Transcreation Evaluation - SLURM Job
# Evaluates a specific I2I model across all countries using VLM judge
#
# Usage:
#   sbatch batch_image_transcreation_slurm.sh <i2i_model> [vlm_model]
#
# Examples:
#   sbatch batch_image_transcreation_slurm.sh flux2-klein
#   sbatch batch_image_transcreation_slurm.sh flux2-klein "Qwen/Qwen2-VL-2B-Instruct"

set -e

echo "========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Partition: $SLURM_JOB_PARTITION"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $SLURM_GPUS"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: $SLURM_MEM_PER_NODE MB"
echo "========================================="
echo ""

# Print start time
echo "Job started at: $(date)"
echo ""

# Activate conda environment
source ~/.bashrc
conda activate image-transcreation

# Print environment info
echo "Python version:"
python --version
echo ""
echo "PyTorch version:"
python -c "import torch; print(torch.__version__); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')"
echo ""

# Set Hugging Face cache
export HF_HOME=/data/hf_cache/mohithr
export TRANSFORMERS_CACHE=/data/hf_cache/mohithr

# Change to vlm_judge directory
cd /home/mohithr/independent_study/Image_transcreation_backtranslation/vlm_judge

# Create logs directory if it doesn't exist
mkdir -p logs

# Get arguments
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

# Capture exit code
EXIT_CODE=$?

echo ""
echo "========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Batch evaluation completed successfully!"
    echo "Results directory: ${RESULTS_DIR}/"
    echo ""
    echo "Result files:"
    for country in "${COUNTRIES[@]}"; do
        echo "  - ${country}.json"
    done
else
    echo "✗ Batch evaluation failed with exit code: $EXIT_CODE"
fi
echo "Job finished at: $(date)"
echo "========================================="

exit $EXIT_CODE
