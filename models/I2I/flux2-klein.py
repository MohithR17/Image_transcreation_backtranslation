"""
FLUX.2 Klein 4B model for image editing.
Model: black-forest-labs/FLUX.2-klein-4B
A lightweight but powerful image editing model from Black Forest Labs.
"""

import torch
import logging
from diffusers import Flux2KleinPipeline
from PIL import Image

# Global variable to cache the loaded model
_pipe = None
_device = None


def load_pipe(device="cuda"):
    """Load and cache the FLUX.2 Klein pipeline."""
    global _pipe, _device
    
    if _pipe is None or _device != device:
        logging.info("Loading FLUX.2 Klein 4B model: black-forest-labs/FLUX.2-klein-4B")
        
        try:
            # FLUX.2 Klein is optimized for fast, distilled image generation/editing
            _pipe = Flux2KleinPipeline.from_pretrained(
                "black-forest-labs/FLUX.2-klein-4B",
                torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
            )
            
            if device == "cuda":
                # Enable CPU offload to save VRAM
                _pipe.enable_model_cpu_offload()
            
            _pipe.set_progress_bar_config(disable=True)
            _device = device
            logging.info("FLUX.2 Klein 4B model loaded successfully")
            
        except Exception as e:
            logging.error(f"Error loading FLUX.2 Klein: {e}")
            logging.info("Note: FLUX.2 Klein requires diffusers >= 0.30.0")
            raise
    
    return _pipe


def edit_image(image, prompt, config):
    """
    Edit image using FLUX.2 Klein 4B.
    
    FLUX.2 Klein is a distilled model designed for fast, high-quality image editing:
    - Efficient 4B parameter model (distilled from FLUX.1)
    - Fast 4-step inference
    - Low guidance scale (1.0)
    - Supports image-to-image editing with strength parameter
    
    Args:
        image: PIL Image to edit
        prompt: Text instruction for editing
        config: Configuration dictionary containing:
            - num_inference_steps: Number of denoising steps (default: 4 for distilled model)
            - guidance_scale: How closely to follow prompt (default: 1.0)
            - strength: How much to modify the image (default: 0.75, range 0-1)
                       0.0 = no change, 1.0 = complete regeneration
            - height: Output height (default: 1024)
            - width: Output width (default: 1024)
            - seed: Random seed for reproducibility (default: None)
            
    Returns:
        Edited PIL Image
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = load_pipe(device)
    
    # Extract parameters from config with Klein-optimized defaults
    num_inference_steps = config.get("num_inference_steps", 4)  # Distilled model is fast
    guidance_scale = config.get("guidance_scale", 1.0)  # Klein works best with low guidance
    strength = config.get("strength", 0.75)  # Balance between preservation and editing
    height = config.get("height", 1024)
    width = config.get("width", 1024)
    seed = config.get("seed", None)
    
    # Ensure image is RGB
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    # Prepare inputs
    inputs = {
        "prompt": prompt,
        "image": image,
        "height": height,
        "width": width,
        "guidance_scale": guidance_scale,
        "num_inference_steps": num_inference_steps,
    }
    
    # Add generator if seed is provided
    if seed is not None:
        inputs["generator"] = torch.Generator(device=device).manual_seed(seed)
    
    # Edit image
    logging.info(f"Editing image with prompt: '{prompt[:100]}...'")
    logging.info(f"Using strength={strength}, steps={num_inference_steps}, guidance={guidance_scale}")
    
    with torch.inference_mode():
        output = pipe(**inputs)
        edited_image = output.images[0]
    
    return edited_image


# Notes for usage:
# - FLUX.2 Klein is a 4B parameter distilled model from FLUX.1 (12B)
# - Optimized for speed: only 4 inference steps vs 50 for FLUX.1
# - Works best with guidance_scale=1.0 (distilled models don't need high guidance)
# - Supports image editing via the 'image' and 'strength' parameters
# - strength parameter: 0.0-1.0, controls how much of original image is preserved
# - Requires ~8-10GB VRAM with CPU offloading
# - Requires diffusers >= 0.30.0
# - Check https://huggingface.co/black-forest-labs/FLUX.2-klein-4B for latest info
