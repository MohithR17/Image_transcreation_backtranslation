"""
Qwen-Image-2512 model for text-to-image generation.
Model: Qwen/Qwen-Image-2512
A vision-language model for high-quality text-to-image generation.
"""

import torch
import logging
from diffusers import QwenImagePipeline
from PIL import Image

# Global variable to cache the loaded model
_pipeline = None
_device_map = None


def load_pipeline(config):
    """Load and cache the Qwen-Image-2512 pipeline."""
    global _pipeline, _device_map
    
    device_map = config.get("device_map", None)
    device = config.get("device", "cuda")
    
    if _pipeline is None or _device_map != device_map:
        logging.info("Loading Qwen-Image-2512 pipeline: Qwen/Qwen-Image-2512")
        
        try:
            # Load pipeline with device mapping for multi-GPU
            if device_map:
                # QwenImagePipeline only supports "balanced" or "cuda", not "auto"
                actual_device_map = "balanced" if device_map == "auto" else device_map
                _pipeline = QwenImagePipeline.from_pretrained(
                    "Qwen/Qwen-Image-2512",
                    torch_dtype=torch.bfloat16,
                    device_map=actual_device_map,  # Use "balanced" for multi-GPU
                )
                logging.info(f"Loaded with device_map: {actual_device_map}")
            else:
                _pipeline = QwenImagePipeline.from_pretrained(
                    "Qwen/Qwen-Image-2512",
                    torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
                )
                if device == "cuda":
                    _pipeline.to(device)
                    # Enable memory optimizations
                    _pipeline.enable_model_cpu_offload()
            
            _pipeline.set_progress_bar_config(disable=True)
            
            _device_map = device_map
            logging.info("Qwen-Image-2512 pipeline loaded successfully")
            
        except Exception as e:
            logging.error(f"Error loading Qwen-Image-2512: {e}")
            logging.info("Note: Qwen-Image-2512 requires the latest diffusers version")
            raise
    
    return _pipeline


def generate_image(prompt, config):
    """
    Generate image using Qwen-Image-2512.
    
    Qwen-Image-2512 is a powerful vision-language model for text-to-image generation:
    - High-quality outputs
    - Support for various resolutions
    - Good multilingual support
    - Trained on diverse cultural content
    
    Args:
        prompt: Text prompt for image generation
        config: Configuration dictionary containing:
            - num_inference_steps: Number of denoising steps (default: 50)
            - guidance_scale: Guidance scale for generation (default: 7.5)
            - negative_prompt: Negative prompt (default: "")
            - height: Output height (default: 1024)
            - width: Output width (default: 1024)
            - seed: Random seed for reproducibility (default: None)
            
    Returns:
        Generated PIL Image
    """
    device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    pipeline = load_pipeline(config)
    
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
        generator_device = "cuda:0" if device == "cuda" else device
        inputs["generator"] = torch.Generator(device=generator_device).manual_seed(seed)
    
    # Generate image
    logging.info(f"Generating image with prompt: '{prompt[:100]}...'")
    
    with torch.inference_mode():
        output = pipeline(**inputs)
        generated_image = output.images[0]
    
    return generated_image
