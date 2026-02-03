"""
FLUX.1 Dev model for text-to-image generation.
Model: black-forest-labs/FLUX.1-dev
A high-quality text-to-image model from Black Forest Labs.
"""

import torch
import logging
from diffusers import FluxPipeline
from PIL import Image

# Global variable to cache the loaded model
_pipe = None
_device = None


def load_pipe(device="cuda"):
    """Load and cache the FLUX.1 Dev pipeline."""
    global _pipe, _device
    
    if _pipe is None or _device != device:
        logging.info("Loading FLUX.1 Dev model: black-forest-labs/FLUX.1-dev")
        
        try:
            # FLUX.1 Dev is the main high-quality text-to-image model
            _pipe = FluxPipeline.from_pretrained(
                "black-forest-labs/FLUX.1-dev",
                torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
            )
            
            if device == "cuda":
                # Enable CPU offload to save VRAM
                _pipe.enable_model_cpu_offload()
            
            _pipe.set_progress_bar_config(disable=True)
            _device = device
            logging.info("FLUX.1 Dev model loaded successfully")
            
        except Exception as e:
            logging.error(f"Error loading FLUX.1 Dev: {e}")
            logging.info("Note: FLUX.1 Dev requires diffusers >= 0.30.0 and may require authentication")
            raise
    
    return _pipe


def generate_image(prompt, config):
    """
    Generate image using FLUX.1 Dev.
    
    FLUX.1 Dev is a high-quality text-to-image generation model:
    - State-of-the-art quality
    - 50-step inference (recommended)
    - Guidance scale around 3.5
    - High-quality 1024x1024 outputs
    - Requires Hugging Face authentication for download
    
    Args:
        prompt: Text prompt for image generation
        config: Configuration dictionary containing:
            - num_inference_steps: Number of denoising steps (default: 50)
            - guidance_scale: How closely to follow prompt (default: 3.5)
            - height: Output height (default: 1024)
            - width: Output width (default: 1024)
            - seed: Random seed for reproducibility (default: None)
            
    Returns:
        Generated PIL Image
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = load_pipe(device)
    
    # Extract parameters from config with Dev-optimized defaults
    num_inference_steps = config.get("num_inference_steps", 50)
    guidance_scale = config.get("guidance_scale", 3.5)
    height = config.get("height", 1024)
    width = config.get("width", 1024)
    seed = config.get("seed", None)
    
    # Prepare inputs
    inputs = {
        "prompt": prompt,
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
