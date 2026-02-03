"""
Stable Diffusion XL model for text-to-image generation.
Model: stabilityai/stable-diffusion-xl-base-1.0
A powerful text-to-image model from Stability AI.
"""

import torch
import logging
from diffusers import StableDiffusionXLPipeline
from PIL import Image

# Global variable to cache the loaded model
_pipe = None
_device = None


def load_pipe(device="cuda"):
    """Load and cache the SDXL pipeline."""
    global _pipe, _device
    
    if _pipe is None or _device != device:
        logging.info("Loading SDXL model: stabilityai/stable-diffusion-xl-base-1.0")
        
        try:
            _pipe = StableDiffusionXLPipeline.from_pretrained(
                "stabilityai/stable-diffusion-xl-base-1.0",
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                variant="fp16" if device == "cuda" else None,
            )
            
            if device == "cuda":
                _pipe.to(device)
                # Enable memory optimizations
                _pipe.enable_model_cpu_offload()
            
            _pipe.set_progress_bar_config(disable=True)
            _device = device
            logging.info("SDXL model loaded successfully")
            
        except Exception as e:
            logging.error(f"Error loading SDXL: {e}")
            raise
    
    return _pipe


def generate_image(prompt, config):
    """
    Generate image using Stable Diffusion XL.
    
    SDXL is a powerful text-to-image generation model:
    - High-quality 1024x1024 outputs
    - Good prompt following
    - Supports negative prompts
    - Configurable guidance scale
    
    Args:
        prompt: Text prompt for image generation
        config: Configuration dictionary containing:
            - num_inference_steps: Number of denoising steps (default: 50)
            - guidance_scale: How closely to follow prompt (default: 7.5)
            - negative_prompt: Negative prompt to avoid certain features (default: "")
            - height: Output height (default: 1024)
            - width: Output width (default: 1024)
            - seed: Random seed for reproducibility (default: None)
            
    Returns:
        Generated PIL Image
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = load_pipe(device)
    
    # Extract parameters from config
    num_inference_steps = config.get("num_inference_steps", 50)
    guidance_scale = config.get("guidance_scale", 7.5)
    negative_prompt = config.get("negative_prompt", "")
    height = config.get("height", 1024)
    width = config.get("width", 1024)
    seed = config.get("seed", None)
    
    # Prepare inputs
    inputs = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "height": height,
        "width": width,
        "guidance_scale": guidance_scale,
        "num_inference_steps": num_inference_steps,
    }
    
    # Add generator if seed is provided
    if seed is not None:
        inputs["generator"] = torch.Generator(device=device).manual_seed(seed)
    
    # Generate image
    logging.info(f"Generating image with prompt: '{prompt[:100]}...'")
    
    with torch.inference_mode():
        output = pipe(**inputs)
        generated_image = output.images[0]
    
    return generated_image
