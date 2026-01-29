# Image Transcreation - Modular Multi-Model Evaluation

## Overview

A modular system for cultural image transcreation using different image editing models. The code separates **data logic** from **model-specific logic** for easy extensibility.

## Architecture

```
I2I_Image_transcreation/
├── I2I_trancreation.py          # Main script (data logic only)
├── models/                       # Model implementations
│   ├── instructpix2pix.py       # InstructPix2Pix
│   ├── magicbrush.py            # MagicBrush
│   └── sdxl-instructpix2pix.py  # SDXL InstructPix2Pix
└── configs/                      # Data configurations
    ├── generate_configs.sh
    └── part1/
        ├── brazil.yaml
        ├── japan.yaml
        └── ...
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed design.

## Quick Start

### 1. Generate Configs for All Countries

```bash
cd configs
bash generate_configs.sh
```

This creates configs in `configs/part1/`:
- `brazil.yaml`, `india.yaml`, `japan.yaml`, etc.

### 2. Run with Different Models

The same config works with any model:

```bash
# InstructPix2Pix (fast, good quality)
python I2I_trancreation.py \
    --config configs/part1/japan.yaml \
    --model instructpix2pix

# MagicBrush (fast, better quality)
python I2I_trancreation.py \
    --config configs/part1/japan.yaml \
    --model magicbrush

# SDXL InstructPix2Pix (slow, best quality)
python I2I_trancreation.py \
    --config configs/part1/japan.yaml \
    --model sdxl-instructpix2pix
```

### Output Directory Structure

Outputs are automatically organized as:
```
outputs/part1/
├── instruct-pix2pix/
│   ├── japan/
│   │   ├── metadata.csv
│   │   ├── brazil_img1.jpg
│   │   └── india_img2.jpg
│   └── brazil/
│       └── ...
├── MagicBrush/
│   ├── japan/
│   └── brazil/
└── sdxl-instructpix2pix-768/
    └── ...
```

**Format**: `./outputs/part1/<model_name>/<target_country>/`

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

| Model | HuggingFace ID | Quality | Speed |
|-------|---------------|---------|-------|
| InstructPix2Pix | `timbrooks/instruct-pix2pix` | Good | Fast |
| MagicBrush | `osunlp/MagicBrush` | Better | Fast |
| SDXL-InstructPix2Pix | `diffusers/sdxl-instructpix2pix-768` | Best | Slow |
| CosXL-Edit | `stabilityai/cosxl-edit` | Best | Slow |

## Example Workflows

### Test One Country with Multiple Models

```bash
# Run Japan config with 3 different models
for model in "timbrooks/instruct-pix2pix" "osunlp/MagicBrush" "diffusers/sdxl-instructpix2pix-768"
do
    python ../../I2I_trancreation.py \
        --config configs/part1/japan.yaml \
        --model "$model"
done
```

Results will be in:
- `outputs/part1/instruct-pix2pix/japan/`
- `outputs/part1/MagicBrush/japan/`
- `outputs/part1/sdxl-instructpix2pix-768/japan/`

### Run All Countries with One Model

```bash
# Run MagicBrush on all countries
for config in configs/part1/*.yaml
do
    python ../../I2I_trancreation.py \
        --config "$config" \
        --model osunlp/MagicBrush
done
```

### Quick Test with Debug Mode

```bash
# Edit any config file and set: debug: True
# This processes only 20 images for quick testing

python ../../I2I_trancreation.py \
    --config configs/part1/japan.yaml \
    --model timbrooks/instruct-pix2pix
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
  - `src_image_path`: Original image path
  - `src_country`: Source country
  - `tgt_image_path`: Generated image path
  - `prompt`: Instruction used
  - `status`: success/cuda_oom/error/download_failed

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

1. **Reduce image size**: Images are auto-resized to 1024px max
2. **Use smaller model**: Try InstructPix2Pix instead of SDXL
3. **Enable memory optimizations** (already enabled in script):
   - Attention slicing
   - VAE slicing

### Model Loading Errors

```bash
# Install latest diffusers
pip install --upgrade diffusers transformers accelerate

# For SDXL models
pip install --upgrade diffusers[torch]
```

### Slow Generation

- Reduce `num_inference_steps` (try 50 or 30)
- Use smaller models (InstructPix2Pix vs SDXL)
- Enable `debug: true` for testing

## Adding Custom Models

To add a new model, update the `load_model()` function in `I2I_trancreation.py`:

```python
elif "your-model-name" in model_name.lower():
    pipe = YourModelPipeline.from_pretrained(
        model_name,
        torch_dtype=torch.float16
    )
```

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
