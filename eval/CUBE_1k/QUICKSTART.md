# T2I CUBE Evaluation - Quick Start Guide

## Overview

This setup allows you to run Text-to-Image (T2I) models on the CUBE_1k dataset, which contains 1000 cultural concept prompts across different countries and domains.

## What Was Created

### 1. T2I Models (`models/T2I/`)
- **flux-dev.py**: High-quality FLUX.1 Dev model (50 steps, state-of-the-art)
- **qwen-image-2512.py**: Qwen vision-language T2I model (multilingual, cultural support)
- **flux-schnell.py**: Fast FLUX.1 Schnell model (4 steps) - optional
- **sdxl.py**: Stable Diffusion XL model (50 steps) - optional
- **__init__.py**: Package initialization
- **README.md**: Model documentation

### 2. Evaluation Script (`eval/CUBE_1k/T2I_cube.py`)
Main script that:
- Loads CUBE_1k dataset
- Runs specified T2I model on all prompts
- Saves generated images
- Creates metadata.json with name, country, domain, prompt for each image

### 3. CUBE_1k Documentation
- **eval/CUBE_1k/README.md**: Complete usage guide
- **eval/CUBE_1k/run_all_models.sh**: Script to run all models
- **eval/CUBE_1k/config_example.yaml**: Example configuration
- **eval/CUBE_1k/requirements.txt**: Python dependencies

## Quick Start

### 1. Install Dependencies

```bash
cd eval/CUBE_1k
pip install -r requirements.txt
```

### 2. Run a Model

```bash
### 2. Run a Model

```bash
# From the CUBE_1k directory
cd eval/CUBE_1k

# Run with default model (flux-dev)
python T2I_cube.py

# Run with Qwen model
python T2I_cube.py --model qwen-image-2512

# Debug mode (only 20 samples)
python T2I_cube.py --debug
```
```

### 3. Run All Models

```bash
cd eval/CUBE_1k
./run_all_models.sh
```

## Output Structure

```
eval/CUBE_1k/
├── data/
│   └── cube_1k.json              # Input dataset (1000 prompts)
├── outputs/
│   ├── flux-dev/
│   │   ├── brazil_cuisine_carne_de_panela.png
│   │   ├── brazil_cuisine_bobo_de_camarao.png
│   │   ├── ...
│   │   └── metadata.json          # Contains name, country, domain, prompt
│   └── qwen-image-2512/
│       ├── brazil_cuisine_carne_de_panela.png
│       └── metadata.json
```

## Metadata Format

Each `metadata.json` contains:

```json
[
  {
    "name": "carne de panela",
    "country": "Brazil",
    "domain": "cuisine",
    "prompt": "A high resolution image of carne de panela from Brazilian cuisine, realistic",
    "image_path": "outputs/flux-dev/brazil_cuisine_carne_de_panela.png",
    "status": "success"
  }
]
```

## Command Line Options

```bash
python T2I_cube.py \
  --model flux-dev \                  # Model: flux-dev, qwen-image-2512
  --cube_data data/cube_1k.json \
  --output_dir outputs \
  --num_inference_steps 50 \          # Default varies by model
  --guidance_scale 3.5 \              # Default varies by model
  --seed 42 \
  --height 1024 \
  --width 1024 \
  --debug \                           # Process only 20 samples
  --max_samples 100                   # Limit to 100 samples
```

## Model Comparison

| Model | Steps | Quality | Speed | VRAM | Guidance Scale |
|-------|-------|---------|-------|------|----------------|
| flux-dev | 50 | State-of-the-art | Slow | High | 3.5 |
| qwen-image-2512 | 50 | Very High | Slow | High | 7.5 |
| flux-schnell | 4 | High | Very Fast | High | 0.0 |
| sdxl | 50 | Very High | Slow | High | 7.5 |

## Features

✅ **Resume Support**: Skips already generated images  
✅ **Incremental Saving**: Metadata saved after each image  
✅ **Error Handling**: CUDA OOM and other errors logged  
✅ **Progress Tracking**: Detailed logs for each image  
✅ **Reproducible**: Seed control for consistent results  

## Similar to I2I_transcreation.py

This script follows the same pattern as `eval/I2I_Image_transcreation/I2I_trancreation.py`:

| Feature | I2I_transcreation.py | T2I_cube.py |
|---------|----------------------|-------------|
| Input | Images + prompts | Prompts only |
| Models | I2I editing models | T2I generation models |
| Dataset | Country-specific JSONs | CUBE_1k JSON |
| Output | Edited images + CSV | Generated images + JSON |
| Metadata | CSV with paths | JSON with all fields |

## Adding New Models

1. Create `models/T2I/your_model.py`:
```python
def generate_image(prompt, config):
    # Your implementation
    return pil_image
```

2. Run with:
```bash
python T2I_cube.py --model your_model
```

## Troubleshooting

**CUDA Out of Memory**:
- Reduce batch size (if applicable)
- Lower image resolution: `--height 512 --width 512`
- Use CPU offload (enabled by default)

**Module Not Found**:
```bash
pip install diffusers transformers accelerate torch pillow
```

**Model Download Issues**:
- Ensure you have Hugging Face credentials for gated models
- Check internet connection
- Models are cached in `~/.cache/huggingface/`

## Next Steps

After generation, use the metadata.json files to:
- Evaluate cultural representation
- Compare model outputs
- Analyze cross-cultural performance
- Feed into evaluation pipelines (e.g., `llm_judge/`)
