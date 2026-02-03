# Image Transcreation - Modular Multi-Model Evaluation

## Overview

A modular system for cultural image transcreation using different image editing models. The code separates **data logic** from **model-specific logic** for easy extensibility.

## Project Structure

```
Image_transcreation_backtranslation/
├── models/                           # Shared model implementations (root level)
│   └── I2I/                          # Image-to-Image editing models
│       ├── __init__.py
│       ├── README.md                 # Model documentation
│       ├── instructpix2pix.py        # InstructPix2Pix model
│       ├── sdxl-instructpix2pix.py   # SDXL InstructPix2Pix
│       ├── cosxl-edit.py             # CosXL Edit model
│       ├── magicbrush.py             # MagicBrush model
│       ├── qwen-image-edit.py        # Qwen Image Edit (VLM-based)
│       └── flux2-klein.py            # FLUX.2 Klein 4B
│
├── eval/                             # Evaluation tasks
│   └── I2I_Image_transcreation/      # Image transcreation evaluation
│       ├── I2I_trancreation.py       # Main evaluation script
│       ├── run_all_countries.sh      # Batch processing script
│       ├── README.md                 # This file
│       ├── requirements.txt          # Python dependencies
│       ├── configs/                  # Data configurations
│       │   ├── generate_configs.sh
│       │   └── part1/
│       │       ├── brazil.yaml
│       │       ├── japan.yaml
│       │       └── ...
│       ├── data/                     # Source images
│       │   └── part1/
│       │       ├── brazil.json
│       │       ├── india.json
│       │       └── ...
│       └── outputs/                  # Generated images
│           └── part1/
│               ├── instructpix2pix/
│               ├── qwen-image-edit/
│               └── ...
│
└── llm_judge/                        # LLM-as-judge evaluation
    ├── qwen_vl_judge.py              # Generic VLM evaluation framework
    └── evaluate_from_metadata.py     # Evaluate from metadata CSVs
```

**Key Design Decisions:**
- **Shared Models**: Models are at root level (`/models/I2I/`) so they can be reused across multiple evaluation tasks
- **Evaluation Isolation**: Each eval task has its own directory with configs, data, and outputs
- **Model Import**: Evaluations import models via: `from models.I2I import <model_name>`

## Quick Start

### 1. Generate Configs for All Countries

```bash
cd configs
bash generate_configs.sh
```

This creates configs in `configs/part1/`:
- `brazil.yaml`, `india.yaml`, `japan.yaml`, etc.

### 2. Run with Different Models

All models are automatically imported from `/models/I2I/`:

```bash
# InstructPix2Pix (fast, good quality)
python I2I_trancreation.py \
    --config configs/part1/japan.yaml \
    --model instructpix2pix

# Qwen-Image-Edit (VLM-based, excellent quality)
python I2I_trancreation.py \
    --config configs/part1/japan.yaml \
    --model qwen-image-edit

# FLUX.2 Klein (fast distilled model, excellent quality)
python I2I_trancreation.py \
    --config configs/part1/japan.yaml \
    --model flux2-klein

# SDXL InstructPix2Pix (slow, best quality)
python I2I_trancreation.py \
    --config configs/part1/japan.yaml \
    --model sdxl-instructpix2pix
```

**Or use the batch script for all countries:**

```bash
# Run all countries with a specific model
./run_all_countries.sh qwen-image-edit

# Run specific countries only
./run_all_countries.sh flux2-klein --countries japan india brazil

# Skip certain countries
./run_all_countries.sh instructpix2pix --skip usa canada

# Continue on errors
./run_all_countries.sh sdxl-instructpix2pix --continue-on-error
```

### Output Directory Structure

Outputs are automatically organized by model and target country:
```
outputs/part1/
├── instructpix2pix/
│   ├── japan/
│   │   ├── metadata.csv
│   │   ├── brazil_img1.jpg
│   │   └── india_img2.jpg
│   └── brazil/
│       └── ...
├── qwen-image-edit/
│   ├── japan/
│   └── brazil/
├── flux2-klein/
│   └── japan/
└── sdxl-instructpix2pix/
    └── ...
```

**Format**: `./outputs/part1/<model_name>/<target_country>/`

**Metadata CSV includes:**
- `src_image_path`: Original image URL/path
- `src_country`: Source country of the image
- `src_category`: Category (food, architecture, etc.)
- `tgt_image_path`: Path to generated image
- `prompt`: Instruction used for editing
- `status`: success/cuda_oom/error/download_failed

## Config File Structure

Configs contain **only data-related parameters**:

```yaml
# Where to get images FROM
source_countries: ['brazil', 'india', 'japan', ...]

# Path to data
source_data_path: ./data/part1

# Where to adapt TO
target_country: Japan

# What to tell the model
prompt: make the image culturally relevant to Japan

# Generation parameters
seed: 0
image_guidance: 1.5
num_inference_steps: 100
text_guidance: 7.5
debug: False
```

**Note**: No `output_dir` or `model` in config! These are determined automatically.

## Supported Models

All models are located in `/models/I2I/` at the project root:

| Model | File | HuggingFace ID | Quality | Speed | VRAM |
|-------|------|----------------|---------|-------|------|
| InstructPix2Pix | `instructpix2pix.py` | `timbrooks/instruct-pix2pix` | Good | Fast | ~8GB |
| SDXL-InstructPix2Pix | `sdxl-instructpix2pix.py` | `diffusers/sdxl-instructpix2pix-768` | Best | Slow | ~16GB |
| CosXL-Edit | `cosxl-edit.py` | `stabilityai/cosxl` | Best | Slow | ~16GB |
| MagicBrush | `magicbrush.py` | Custom LoRA | Better | Fast | ~8GB |
| Qwen-Image-Edit | `qwen-image-edit.py` | `Qwen/Qwen-Image-Edit` | Excellent | Medium | ~12-16GB |
| FLUX.2 Klein | `flux2-klein.py` | `black-forest-labs/FLUX.2-klein-4B` | Excellent | Fast | ~8-10GB |

**Model Selection Guide:**
- **Fast prototyping**: InstructPix2Pix, FLUX.2 Klein
- **Best quality**: SDXL-InstructPix2Pix, CosXL-Edit, Qwen-Image-Edit
- **VLM-based editing**: Qwen-Image-Edit (understands complex instructions)
- **Low VRAM**: InstructPix2Pix, MagicBrush, FLUX.2 Klein

See `/models/I2I/README.md` for detailed model documentation.

## Example Workflows

### Run All Countries with Batch Script

```bash
# Run all countries with qwen-image-edit
./run_all_countries.sh qwen-image-edit

# Run only specific countries
./run_all_countries.sh flux2-klein --countries japan india brazil

# Skip certain countries
./run_all_countries.sh instructpix2pix --skip usa canada

# Continue even if some countries fail
./run_all_countries.sh sdxl-instructpix2pix --continue-on-error
```

### Test One Country with Multiple Models

```bash
# Run Japan config with 3 different models
for model in instructpix2pix qwen-image-edit flux2-klein
do
    python I2I_trancreation.py \
        --config configs/part1/japan.yaml \
        --model "$model"
done
```

Results will be in:
- `outputs/part1/instructpix2pix/japan/`
- `outputs/part1/qwen-image-edit/japan/`
- `outputs/part1/flux2-klein/japan/`

### Run All Countries with One Model (Manual)

```bash
# Run qwen-image-edit on all countries manually
for config in configs/part1/*.yaml
do
    python I2I_trancreation.py \
        --config "$config" \
        --model qwen-image-edit
done
```

### Quick Test with Debug Mode

```bash
# Edit any config file and set: debug: True
# This processes only 20 images for quick testing

python I2I_trancreation.py \
    --config configs/part1/japan.yaml \
    --model instructpix2pix
```

## Environment Setup

### 1. Install Dependencies

```bash
# Create conda environment
conda create -n image-transcreation python=3.10
conda activate image-transcreation

# Install from requirements
cd eval/I2I_Image_transcreation
pip install -r requirements.txt
```

### 2. Set HuggingFace Cache (Optional)

```bash
# Use shared cache directory (recommended for clusters)
export HF_HOME=/data/hf_cache

# Or add to ~/.bashrc for persistence
echo 'export HF_HOME=/data/hf_cache' >> ~/.bashrc
```

### 3. Verify Model Access

```bash
# Test model imports
python -c "from models.I2I import instructpix2pix; print('✓ Models accessible')"
```

## Data Format

Your data should be in `./data/part1/`:

```
data/part1/
├── brazil.json
├── india.json
├── japan.json
└── ...
```

Each JSON file:
```json
{
  "food": {
    "img1": "path/to/image1.jpg",
    "img2": "path/to/image2.jpg"
  },
  "architecture": {
    "img3": "path/to/image3.jpg"
  }
}
```

## Parameters

### Config Parameters

- **source_countries**: List of countries to take source images from
- **source_data_path**: Directory containing country JSON files
- **target_country**: Country to adapt images to
- **prompt**: Instruction for the editing model
- **seed**: Random seed for reproducibility
- **image_guidance**: Preservation strength (0.5-3.0, default 1.5)
- **num_inference_steps**: Quality vs speed (20-100, default 100)
- **text_guidance**: Prompt adherence (5.0-15.0, default 7.5)
- **debug**: Process only 20 images (True/False)

### Command Line Arguments

- **--config**: Path to config YAML file
- **--model**: Model to use (overrides default, optional)

## Output Files

Each run creates:
- **Edited images**: `<source_country>_<original_filename>.jpg`
- **metadata.csv**: Processing log with columns:
  - `src_image_path`: Original image URL/path
  - `src_country`: Source country
  - `src_category`: Image category (food, architecture, etc.)
  - `tgt_image_path`: Generated image path
  - `prompt`: Instruction used
  - `status`: success/cuda_oom/error/download_failed

### Using Metadata for Evaluation

The metadata CSV can be used with the LLM-as-judge framework:

```bash
# Evaluate generated images with Qwen-VL
cd ../../llm_judge
python evaluate_from_metadata.py \
    --metadata ../eval/I2I_Image_transcreation/outputs/part1/qwen-image-edit/japan/metadata.csv \
    --template cultural_appropriateness \
    --model Qwen/Qwen2-VL-7B-Instruct
```

See `/llm_judge/README.md` for evaluation details.

## Troubleshooting

### CUDA Out of Memory
- Images are automatically resized to 1024px max
- Use smaller model: InstructPix2Pix instead of SDXL
- Script automatically skips OOM images and continues

### Model Not Found
```bash
pip install --upgrade diffusers transformers accelerate
```

### Slow Generation
- Reduce `num_inference_steps` in config (try 50 or 30)
- Enable `debug: True` for testing
- Use InstructPix2Pix instead of SDXL models
```

### 3. Compare Multiple Models

```bash
# Test same config with 3 different models
for model in "timbrooks/instruct-pix2pix" "osunlp/MagicBrush" "diffusers/sdxl-instructpix2pix-768"
do
    python eval/I2I_trancreation.py \
        --config configs/part1/e2e-instruct/japan.yaml \
        --model "$model"
done
```

## Output Structure

```
outputs/
└── part1/
    └── model_name/
        └── country/
            ├── metadata.csv              # Processing log
            ├── brazil_image1.jpg         # Generated images
            ├── india_image2.jpg
            └── ...
```

### Metadata CSV Format

```csv
src_image_path,src_country,tgt_image_path,prompt,status
./data/part1/brazil/img1.jpg,brazil,./outputs/.../brazil_img1.jpg,make the image...,success
./data/part1/india/img2.jpg,india,./outputs/.../india_img2.jpg,make the image...,success
```

## Troubleshooting

### CUDA Out of Memory
- Images are automatically resized to 1024px max
- Use smaller model: InstructPix2Pix or FLUX.2 Klein instead of SDXL
- Script automatically skips OOM images and continues
- Enable CPU offload in model config (already enabled for most models)

### Model Not Found or Import Error
```bash
# Update diffusers to latest version
pip install --upgrade diffusers transformers accelerate

# Verify model imports work
python -c "from models.I2I import instructpix2pix"
```

### Model Loading Errors
- **Qwen-Image-Edit**: Requires `diffusers>=0.30.0`
- **FLUX.2 Klein**: Requires `diffusers>=0.30.0`
- Check `/models/I2I/README.md` for model-specific requirements

### Slow Generation
- Reduce `num_inference_steps` in config (try 50 or 30)
- Enable `debug: True` for testing (processes only 20 images)
- Use faster models: InstructPix2Pix, FLUX.2 Klein, MagicBrush
- For FLUX.2 Klein: Already optimized to 4 steps

### HuggingFace Cache Issues
```bash
# Set cache directory
export HF_HOME=/data/hf_cache

# Or use default (may fill up home directory)
# ~/.cache/huggingface
```

## Adding New Models

To add a new model to the shared model library:

1. **Create model file** in `/models/I2I/`:
```python
# /models/I2I/my_new_model.py
import torch
from diffusers import YourPipeline

_pipeline = None

def load_pipeline(device="cuda"):
    global _pipeline
    if _pipeline is None:
        _pipeline = YourPipeline.from_pretrained("model/name")
        _pipeline.to(device)
    return _pipeline

def edit_image(image, prompt, config):
    pipeline = load_pipeline()
    # Your editing logic here
    result = pipeline(image=image, prompt=prompt, **config)
    return result.images[0]
```

2. **Update** `/models/I2I/__init__.py` if needed

3. **Document** in `/models/I2I/README.md`

4. **Use** in evaluation:
```bash
python I2I_trancreation.py --config configs/part1/japan.yaml --model my-new-model
```

The evaluation script will automatically import `from models.I2I.my_new_model import edit_image`.

## Related Components

### Model Library
- **Location**: `/models/I2I/`
- **Documentation**: `/models/I2I/README.md`
- **Models**: 6+ image editing models (InstructPix2Pix, Qwen-Image-Edit, FLUX.2 Klein, etc.)

### LLM-as-Judge Evaluation
- **Location**: `/llm_judge/`
- **Purpose**: Evaluate generated images with VLMs (Qwen2-VL, etc.)
- **Usage**: Reads metadata.csv files and scores images
- **Templates**: Cultural appropriateness, image quality, instruction following, etc.

### Text-to-Image Generation
- **Location**: `/ArchAla/T2I_generation.py`
- **Purpose**: Generate images from scratch (not editing)
- **Models**: SDXL, SD3, FLUX Schnell/Dev, Playground v2.5

## Citation

If you use this code, please cite:

```bibtex
@article{transcreation2024,
  title={Image Transcreation: Cultural Adaptation through Image Editing},
  year={2024}
}
```

## License

MIT License
