# Image Editing Models (I2I)

This directory contains Image-to-Image editing model implementations shared across all evaluation tasks.

**Location:** `/models/I2I/` (shared across project)

## Structure

Each model is in its own file and must implement:

```python
def edit_image(image, prompt, config):
    """
    Edit an image based on a text instruction.
    
    Args:
        image: PIL Image to edit
        prompt: Text instruction for editing
        config: Configuration dictionary containing model parameters
        
    Returns:
        PIL Image: Edited image
    """
    pass
```

## Available Models

### 1. InstructPix2Pix (`instructpix2pix.py`) ✅ Ready to Use
- **Model**: `timbrooks/instruct-pix2pix`
- **Base**: Stable Diffusion 1.5
- **Speed**: Fast
- **Quality**: Good
- **VRAM**: ~8GB
- **Status**: Works out of the box

### 2. SDXL InstructPix2Pix (`sdxl-instructpix2pix.py`) ✅ Ready to Use
- **Model**: `diffusers/sdxl-instructpix2pix-768`
- **Base**: Stable Diffusion XL
- **Speed**: Slow
- **Quality**: Best
- **VRAM**: ~16GB
- **Status**: Works out of the box

### 3. CosXL Edit (`cosxl-edit.py`) ✅ Ready to Use
- **Model**: `stabilityai/cosxl`
- **Base**: Stable Diffusion XL
- **Speed**: Medium
- **Quality**: Excellent
- **VRAM**: ~16GB
- **Status**: Works out of the box

### 4. Qwen-Image-Edit (`qwen-image-edit.py`) ✅ VLM-based Editing
- **Model**: `Qwen/Qwen-Image-Edit`
- **Base**: Qwen2-VL (Vision-Language Model)
- **Speed**: Medium
- **Quality**: Good (instruction understanding)
- **VRAM**: ~12-16GB
- **Status**: Ready to use
- **Features**: Better instruction understanding, multimodal approach

### 5. FLUX.2 Klein (`flux2-klein.py`) ✅ Fast & Efficient
- **Model**: `black-forest-labs/FLUX.2-klein-4B`
- **Base**: FLUX.2 (4B parameters)
- **Speed**: Fast
- **Quality**: Excellent
- **VRAM**: ~8-10GB (with optimizations)
- **Status**: Ready to use
- **Features**: Lightweight, efficient, good instruction following

### 6. MagicBrush (`magicbrush.py`) ⚠️ Requires Setup
- **Model**: `vinesmsuic/magicbrush-Jul7-LoRA-SD15-local` (community version)
- **Base**: Stable Diffusion
- **Speed**: Fast
- **Quality**: Better (trained on more diverse edits)
- **VRAM**: ~8GB
- **Status**: Requires downloading model or setting `MAGICBRUSH_MODEL_PATH`
- **Alternative**: Use `instructpix2pix` which has similar architecture

## Model Availability Issues

### MagicBrush Setup

If you want to use MagicBrush, you have two options:

**Option 1: Use the community LoRA version (default)**
```bash
# The model will try to download: vinesmsuic/magicbrush-Jul7-LoRA-SD15-local
python I2I_trancreation.py --config configs/part1/japan.yaml --model magicbrush
```

**Option 2: Download and use local checkpoint**
```bash
# Download the official checkpoint from MagicBrush repo
# Then set the environment variable:
export MAGICBRUSH_MODEL_PATH="/path/to/magicbrush/checkpoint"
python I2I_trancreation.py --config configs/part1/japan.yaml --model magicbrush
```

**Option 3: Use InstructPix2Pix instead** (Recommended if MagicBrush fails)
```bash
python I2I_trancreation.py --config configs/part1/japan.yaml --model instructpix2pix
```

## Adding a New Model

1. Create a new file in this directory: `models/I2I/your_model_name.py`
2. Implement the `edit_image(image, prompt, config)` function
3. Use the model in any evaluation:

```bash
# From eval/I2I_Image_transcreation/
python I2I_trancreation.py --config configs/part1/japan.yaml --model your_model_name
```

### Example Template

```python
"""
Your Model Name for image transcreation.
"""

import torch
import logging
# Import your model libraries here

# Global variable to cache the loaded model
_pipe = None
_device = None


def load_pipe(device="cuda"):
    """Load and cache your model pipeline."""
    global _pipe, _device
    
    if _pipe is None or _device != device:
        logging.info("Loading Your Model")
        
        # Load your model here
        _pipe = YourModel.from_pretrained("model-id")
        _pipe.to(device)
        
        _device = device
        logging.info("Model loaded successfully")
    
    return _pipe


def edit_image(image, prompt, config):
    """
    Edit image using your model.
    
    Args:
        image: PIL Image to edit
        prompt: Text instruction for editing
        config: Configuration dictionary
            
    Returns:
        PIL Image: Edited image
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = load_pipe(device)
    
    # Get parameters from config
    num_inference_steps = int(config.get("num_inference_steps", 100))
    # ... other parameters
    
    # Generate edited image
    with torch.no_grad():
        result = pipe(
            prompt=prompt,
            image=image,
            # your parameters here
        )
    
    return result.images[0]
```

## Benefits of This Structure

✅ **Separation of Concerns**: Data logic is separate from model logic  
✅ **Easy to Add Models**: Just create a new file  
✅ **No Code Changes Needed**: Add models without modifying main script  
✅ **Model Caching**: Models are loaded once and reused  
✅ **Clean Interface**: All models have the same interface  

## Usage

```bash
# Use InstructPix2Pix
python I2I_trancreation.py --config configs/part1/japan.yaml --model instructpix2pix

# Use MagicBrush
python I2I_trancreation.py --config configs/part1/japan.yaml --model magicbrush

# Use SDXL
python I2I_trancreation.py --config configs/part1/japan.yaml --model sdxl-instructpix2pix
```

## Config Parameters Used by Models

Models read these parameters from the config dictionary:

- `num_inference_steps`: Number of denoising steps (20-100)
- `image_guidance`: How much to preserve original (0.5-3.0)
- `text_guidance`: How much to follow prompt (5.0-15.0)
- `seed`: Random seed for reproducibility

Models can add their own custom parameters as needed.
