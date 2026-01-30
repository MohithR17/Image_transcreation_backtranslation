"""
MagicBrush model for image transcreation.

NOTE: MagicBrush checkpoint needs to be downloaded manually.
Options:
1. Download from: https://huggingface.co/vinesmsuic/magicbrush-Jul7-LoRA-SD15-local
2. Or use InstructPix2Pix as MagicBrush is based on it
3. Set MAGICBRUSH_MODEL_PATH environment variable to local path

For now, using InstructPix2Pix as fallback (similar architecture).
"""

import torch
import logging
import os
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler

# Global variable to cache the loaded model
_pipe = None
_device = None


def load_pipe(device="cuda"):
    """Load and cache the MagicBrush pipeline."""
    global _pipe, _device
    
    if _pipe is None or _device != device:
        # Check if user has set a local MagicBrush model path
        magicbrush_path = os.environ.get("MAGICBRUSH_MODEL_PATH")
        
        if magicbrush_path and os.path.exists(magicbrush_path):
            model_id = magicbrush_path
            logging.info(f"Loading MagicBrush from local path: {magicbrush_path}")
        else:
            # Try the community LoRA version
            model_id = "vinesmsuic/magicbrush-Jul7-LoRA-SD15-local"
            logging.info(f"Loading MagicBrush model: {model_id}")
            logging.warning("If this fails, you can:")
            logging.warning("1. Download the model and set MAGICBRUSH_MODEL_PATH env variable")
            logging.warning("2. Or use 'instructpix2pix' model instead")
        
        _pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            model_id,
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
