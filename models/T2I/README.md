# Text-to-Image (T2I) Models

This directory contains text-to-image generation models for the CUBE evaluation.

## Model Interface

All T2I models must implement the following interface:

```python
def generate_image(prompt, config):
    """
    Generate an image from a text prompt.
    
    Args:
        prompt: Text prompt for image generation
        config: Configuration dictionary containing model parameters
        
    Returns:
        PIL.Image: Generated image
    """
```

## Available Models

### flux-dev
- **Model**: `black-forest-labs/FLUX.1-dev`
- **Type**: High-quality text-to-image model
- **Speed**: ~50 steps
- **Quality**: State-of-the-art
- **Size**: Large (requires good GPU)
- **Note**: Requires Hugging Face authentication

**Config Parameters:**
- `num_inference_steps`: Number of denoising steps (default: 50)
- `guidance_scale`: Guidance scale (default: 3.5)
- `height`: Output height (default: 1024)
- `width`: Output width (default: 1024)
- `seed`: Random seed for reproducibility

### qwen-image-2512
- **Model**: `Qwen/Qwen-Image-2512`
- **Type**: Vision-language text-to-image model
- **Speed**: ~50 steps
- **Quality**: Very High
- **Size**: Large
- **Features**: Good multilingual and cultural support

**Config Parameters:**
- `num_inference_steps`: Number of denoising steps (default: 50)
- `guidance_scale`: Guidance scale (default: 7.5)
- `negative_prompt`: Negative prompt (default: "")
- `height`: Output height (default: 1024)
- `width`: Output width (default: 1024)
- `seed`: Random seed for reproducibility

### flux-schnell (Optional)
- **Model**: `black-forest-labs/FLUX.1-schnell`
- **Type**: Fast distilled text-to-image model
- **Speed**: ~4 steps
- **Quality**: High
- **Size**: Large (requires good GPU)

**Config Parameters:**
- `num_inference_steps`: Number of denoising steps (default: 4)
- `guidance_scale`: Guidance scale (default: 0.0 for schnell)
- `height`: Output height (default: 1024)
- `width`: Output width (default: 1024)
- `seed`: Random seed for reproducibility

### sdxl (Optional)
- **Model**: `stabilityai/stable-diffusion-xl-base-1.0`
- **Type**: High-quality text-to-image model
- **Speed**: ~50 steps
- **Quality**: Very High
- **Size**: Large

**Config Parameters:**
- `num_inference_steps`: Number of denoising steps (default: 50)
- `guidance_scale`: Guidance scale (default: 7.5)
- `negative_prompt`: Negative prompt (default: "")
- `height`: Output height (default: 1024)
- `width`: Output width (default: 1024)
- `seed`: Random seed for reproducibility

## Usage

Models are automatically loaded by the T2I_cube.py script:

```python
from models.T2I import flux_dev

# Generate image
config = {
    "num_inference_steps": 50,
    "guidance_scale": 3.5,
    "seed": 42
}
image = flux_dev.generate_image("A high resolution image of sushi", config)
```

## Adding New Models

1. Create a new Python file in this directory (e.g., `new_model.py`)
2. Implement the `generate_image(prompt, config)` function
3. Add documentation in this README
4. Test with the T2I_cube.py script using `--model new_model`
