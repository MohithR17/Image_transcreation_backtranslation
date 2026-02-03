# T2I CUBE Evaluation

This script evaluates Text-to-Image (T2I) models on the CUBE_1k dataset.

## Dataset

The CUBE_1k dataset contains 1000 cultural concept prompts with the following structure:

```json
{
  "name": "carne de panela",
  "country": "Brazil",
  "domain": "cuisine",
  "prompt": "A high resolution image of carne de panela from Brazilian cuisine, realistic"
}
```

## Available Models

- **flux-dev**: High-quality FLUX.1 Dev model (50 steps, state-of-the-art quality)
- **qwen-image-2512**: Qwen vision-language T2I model (good cultural support)
- **flux-schnell**: Fast FLUX.1 Schnell model (4 steps, high quality) - optional
- **sdxl**: Stable Diffusion XL (50 steps, very high quality) - optional

See `../models/T2I/README.md` for more details.

## Usage

### Basic Usage

```bash
# Run with default model (flux-dev)
python T2I_cube.py

# Run with specific model
python T2I_cube.py --model qwen-image-2512

# Debug mode (only 20 samples)
python T2I_cube.py --debug

# Limit number of samples
python T2I_cube.py --max_samples 100
```

### Advanced Options

```bash
python T2I_cube.py \
  --model flux-dev \
  --cube_data data/cube_1k.json \
  --output_dir outputs \
  --num_inference_steps 50 \
  --guidance_scale 3.5 \
  --seed 42 \
  --height 1024 \
  --width 1024
```

### Run All Models

```bash
# Process with multiple models
python T2I_cube.py --model flux-dev
python T2I_cube.py --model qwen-image-2512
```

## Output Structure

Generated outputs are saved in the following structure:

```
CUBE_1k/
├── outputs/
│   ├── flux-dev/
│   │   ├── brazil_cuisine_carne_de_panela.png
│   │   ├── brazil_cuisine_bobo_de_camarao.png
│   │   ├── ...
│   │   └── metadata.json
│   └── qwen-image-2512/
│       ├── brazil_cuisine_carne_de_panela.png
│       ├── ...
│       └── metadata.json
```

## Metadata Format

The `metadata.json` file stores information about each generated image:

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

## Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--model` | `flux-dev` | T2I model to use |
| `--cube_data` | `data/cube_1k.json` | Path to CUBE dataset |
| `--output_dir` | `outputs` | Base output directory |
| `--debug` | `False` | Process only 20 samples |
| `--max_samples` | `None` | Maximum samples to process |
| `--num_inference_steps` | Model default | Number of denoising steps |
| `--guidance_scale` | Model default | Guidance scale |
| `--seed` | `42` | Random seed |
| `--height` | `1024` | Image height |
| `--width` | `1024` | Image width |

## Adding New Models

1. Create a new model file in `../models/T2I/` (e.g., `new_model.py`)
2. Implement the `generate_image(prompt, config)` function
3. Run the script with `--model new_model`

## Notes

- Images are skipped if they already exist (allows resuming)
- Metadata is saved incrementally (won't lose progress on crashes)
- CUDA OOM errors are caught and logged
- Progress is logged for every image processed
