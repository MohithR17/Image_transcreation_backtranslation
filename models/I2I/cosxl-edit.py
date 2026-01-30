"""
CosXL Edit model for image transcreation.
Model: stabilityai/cosxl (edit variant)
Good alternative to InstructPix2Pix with SDXL quality.
"""

import torch
import logging
from diffusers import StableDiffusionXLInstructPix2PixPipeline, EulerAncestralDiscreteScheduler

# Global variable to cache the loaded model
_pipe = None
_device = None


def load_pipe(device="cuda"):
    """Load and cache the CosXL Edit pipeline."""
    global _pipe, _device
    
    if _pipe is None or _device != device:
        logging.info("Loading CosXL Edit model")
        
        # CosXL is available and works well
        _pipe = StableDiffusionXLInstructPix2PixPipeline.from_pretrained(
            "stabilityai/cosxl",
            variant="fp16",
            torch_dtype=torch.float16,
        )
        _pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(_pipe.scheduler.config)
        _pipe.to(device)
        
        # Enable memory optimizations
        if hasattr(_pipe, "enable_attention_slicing"):
            _pipe.enable_attention_slicing()
        if hasattr(_pipe, "enable_vae_slicing"):
            _pipe.enable_vae_slicing()
        
        _device = device
        logging.info("CosXL Edit model loaded successfully")
    
    return _pipe


def edit_image(image, prompt, config):
    """
    Edit image using CosXL Edit.
    
    Args:
        image: PIL Image to edit
        prompt: Text instruction for editing
        config: Configuration dictionary containing:
            - num_inference_steps: Number of denoising steps
            - image_guidance_scale: How much to follow the input image
            - guidance_scale: How much to follow the text prompt
            
    Returns:
        Edited PIL Image
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = load_pipe(device)
    
    # Extract parameters from config
    num_inference_steps = config.get("num_inference_steps", 50)
    image_guidance_scale = config.get("image_guidance_scale", 1.5)
    guidance_scale = config.get("guidance_scale", 7.5)
    
    # Generate edited image
    edited_image = pipe(
        prompt=prompt,
        image=image,
        num_inference_steps=num_inference_steps,
        image_guidance_scale=image_guidance_scale,
        guidance_scale=guidance_scale,
    ).images[0]
    
    return edited_image
