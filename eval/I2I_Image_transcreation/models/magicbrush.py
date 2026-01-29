"""
MagicBrush model for image transcreation.
Model: osunlp/MagicBrush
"""

import torch
import logging
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler

# Global variable to cache the loaded model
_pipe = None
_device = None


def load_pipe(device="cuda"):
    """Load and cache the MagicBrush pipeline."""
    global _pipe, _device
    
    if _pipe is None or _device != device:
        logging.info("Loading MagicBrush model: osunlp/MagicBrush")
        
        _pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            "osunlp/MagicBrush",
            torch_dtype=torch.float16,
            safety_checker=None
        )
        _pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(_pipe.scheduler.config)
        _pipe.to(device)
        
        # Enable memory optimizations
        if hasattr(_pipe, "enable_attention_slicing"):
            _pipe.enable_attention_slicing()
        if hasattr(_pipe, "enable_vae_slicing"):
            _pipe.enable_vae_slicing()
        
        _device = device
        logging.info("MagicBrush model loaded successfully")
    
    return _pipe


def edit_image(image, prompt, config):
    """
    Edit image using MagicBrush.
    
    Args:
        image: PIL Image to edit
        prompt: Text instruction for editing
        config: Configuration dictionary containing:
            - num_inference_steps: Number of denoising steps
            - image_guidance: How much to preserve original image
            - text_guidance: How much to follow the text prompt
            
    Returns:
        PIL Image: Edited image
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = load_pipe(device)
    
    # Get parameters from config
    num_inference_steps = int(config.get("num_inference_steps", 100))
    image_guidance_scale = float(config.get("image_guidance", 1.5))
    guidance_scale = float(config.get("text_guidance", 7.5))
    
    # Generate edited image
    with torch.no_grad():
        result = pipe(
            prompt=prompt,
            image=image,
            num_inference_steps=num_inference_steps,
            image_guidance_scale=image_guidance_scale,
            guidance_scale=guidance_scale,
        )
    
    return result.images[0]
