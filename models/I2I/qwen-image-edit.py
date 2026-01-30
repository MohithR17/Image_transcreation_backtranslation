"""
Qwen-Image-Edit model for image transcreation.
Model: Qwen/Qwen-Image-Edit
A vision-language model for instruction-based image editing using diffusers.
"""

import torch
import logging
from diffusers import QwenImageEditPipeline
from PIL import Image

# Global variable to cache the loaded model
_pipeline = None
_device = None


def load_pipeline(device="cuda"):
    """Load and cache the Qwen-Image-Edit pipeline."""
    global _pipeline, _device
    
    if _pipeline is None or _device != device:
        logging.info("Loading Qwen-Image-Edit pipeline: Qwen/Qwen-Image-Edit")
        
        # Load pipeline
        _pipeline = QwenImageEditPipeline.from_pretrained("Qwen/Qwen-Image-Edit")
        _pipeline.to(torch.bfloat16)
        _pipeline.to(device)
        _pipeline.set_progress_bar_config(disable=True)
        
        _device = device
        logging.info("Qwen-Image-Edit pipeline loaded successfully")
    
    return _pipeline


def edit_image(image, prompt, config):
    """
    Edit image using Qwen-Image-Edit.
    
    Args:
        image: PIL Image to edit
        prompt: Text instruction for editing
        config: Configuration dictionary containing:
            - num_inference_steps: Number of denoising steps (default: 50)
            - true_cfg_scale: Guidance scale for editing (default: 4.0)
            - negative_prompt: Negative prompt (default: " ")
            - seed: Random seed for reproducibility (default: None)
            
    Returns:
        Edited PIL Image
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline = load_pipeline(device)
    
    # Extract parameters from config
    num_inference_steps = config.get("num_inference_steps", 50)
    true_cfg_scale = config.get("true_cfg_scale", 4.0)
    negative_prompt = config.get("negative_prompt", " ")
    seed = config.get("seed", None)
    
    # Ensure image is RGB
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    # Prepare inputs
    inputs = {
        "image": image,
        "prompt": prompt,
        "true_cfg_scale": true_cfg_scale,
        "negative_prompt": negative_prompt,
        "num_inference_steps": num_inference_steps,
    }
    
    # Add generator if seed is provided
    if seed is not None:
        inputs["generator"] = torch.manual_seed(seed)
    
    # Generate edited image
    logging.info(f"Editing image with prompt: '{prompt[:100]}...'")
    
    with torch.inference_mode():
        output = pipeline(**inputs)
        edited_image = output.images[0]
    
    return edited_image
