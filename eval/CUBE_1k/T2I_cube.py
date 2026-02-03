"""
T2I CUBE Evaluation Script

This script runs Text-to-Image (T2I) models on the CUBE_1k dataset and saves:
1. Generated images
2. Metadata (JSON) containing name, country, domain, prompt for each generation

Similar to I2I_transcreation.py but for text-to-image generation.
"""

import os
import sys
import torch
import argparse
import logging
import json
from PIL import Image

# Add parent directory to path to access shared models
# T2I_cube.py is in eval/CUBE_1k/, models are in models/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))


def load_cube_data(cube_json_path, debug=False, max_samples=None):
    """
    Load CUBE_1k dataset.
    
    Args:
        cube_json_path: Path to cube_1k.json file
        debug: If True, sample only a subset of data
        max_samples: Maximum number of samples to load (None for all)
        
    Returns:
        list: List of dictionaries with keys: name, country, domain, prompt
    """
    logging.info(f"Loading CUBE data from: {cube_json_path}")
    
    with open(cube_json_path, 'r') as f:
        data = json.load(f)
    
    logging.info(f"Total samples in CUBE_1k: {len(data)}")
    
    # Debug mode: use only 20 samples
    if debug:
        logging.info("Debug mode enabled. Using 20 random samples.")
        import random
        data = random.sample(data, min(20, len(data)))
    
    # Limit samples if specified
    if max_samples is not None and max_samples < len(data):
        logging.info(f"Limiting to {max_samples} samples")
        data = data[:max_samples]
    
    logging.info(f"Number of samples to process: {len(data)}")
    
    return data


def process_prompts(model_func, data, config, output_dir):
    """
    Generate images for all prompts using the provided model function.
    
    Args:
        model_func: Function that takes (prompt, config) and returns generated image
        data: List of dictionaries with name, country, domain, prompt
        config: Configuration dictionary
        output_dir: Directory to save outputs
        
    Returns:
        tuple: (successful, failed) - counts of successful and failed generations
    """
    successful = 0
    failed = 0
    
    # Create metadata list to store all results
    metadata_list = []
    metadata_path = os.path.join(output_dir, "metadata.json")
    
    for i, item in enumerate(data):
        try:
            name = item.get("name", "unknown")
            country = item.get("country", "unknown")
            domain = item.get("domain", "unknown")
            prompt = item.get("prompt", "")
            
            if not prompt:
                logging.warning(f"Skipping item {i+1}: No prompt found")
                failed += 1
                continue
            
            logging.info(f"Processing [{i+1}/{len(data)}]: {name} ({country}, {domain})")
            
            # Create safe filename from name
            safe_name = "".join(c if c.isalnum() or c in (' ', '-', '_') else '_' for c in name)
            safe_name = safe_name.replace(' ', '_').lower()
            
            # Check if output image already exists
            generated_image_path = os.path.join(output_dir, f"{country.lower()}_{domain.lower()}_{safe_name}.png")
            
            if os.path.exists(generated_image_path):
                logging.info(f"⏭️  Skipping - already exists: {generated_image_path}")
                
                # Add to metadata
                metadata_list.append({
                    "name": name,
                    "country": country,
                    "domain": domain,
                    "prompt": prompt,
                    "image_path": generated_image_path,
                    "status": "skipped_exists"
                })
                
                successful += 1
                
                # Save metadata incrementally
                with open(metadata_path, 'w') as f:
                    json.dump(metadata_list, f, indent=2)
                
                continue
            
            # Generate image using model-specific function
            generated_image = model_func(prompt, config)
            
            # Save generated image
            generated_image.save(generated_image_path)
            logging.info(f"✅ Saved: {generated_image_path}")
            
            # Add to metadata
            metadata_list.append({
                "name": name,
                "country": country,
                "domain": domain,
                "prompt": prompt,
                "image_path": generated_image_path,
                "status": "success"
            })
            
            successful += 1
            
            # Save metadata incrementally
            with open(metadata_path, 'w') as f:
                json.dump(metadata_list, f, indent=2)
            
        except torch.cuda.OutOfMemoryError as e:
            logging.warning(f"Skipping item {i+1} due to CUDA OOM error: {e}")
            
            metadata_list.append({
                "name": item.get("name", "unknown"),
                "country": item.get("country", "unknown"),
                "domain": item.get("domain", "unknown"),
                "prompt": item.get("prompt", ""),
                "image_path": "",
                "status": "cuda_oom"
            })
            
            failed += 1
            
            # Clear cache and continue
            torch.cuda.empty_cache()
            
            # Save metadata incrementally
            with open(metadata_path, 'w') as f:
                json.dump(metadata_list, f, indent=2)
            
            continue
            
        except Exception as e:
            logging.error(f"Error processing item {i+1}: {e}")
            
            metadata_list.append({
                "name": item.get("name", "unknown"),
                "country": item.get("country", "unknown"),
                "domain": item.get("domain", "unknown"),
                "prompt": item.get("prompt", ""),
                "image_path": "",
                "status": f"error: {str(e)}"
            })
            
            failed += 1
            
            # Save metadata incrementally
            with open(metadata_path, 'w') as f:
                json.dump(metadata_list, f, indent=2)
            
            continue
    
    logging.info("\n" + "="*50)
    logging.info(f"Processing complete!")
    logging.info(f"  Successful: {successful}/{len(data)}")
    logging.info(f"  Failed: {failed}/{len(data)}")
    logging.info(f"  Output directory: {output_dir}")
    logging.info(f"  Metadata: {metadata_path}")
    logging.info("="*50)
    
    return successful, failed


def main():
    # Initialize device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Set up logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Run T2I models on CUBE_1k dataset")
    parser.add_argument("--model", default="flux-dev", 
                        help="Model to use: flux-dev, qwen-image-2512, flux-schnell, sdxl, etc.")
    parser.add_argument("--cube_data", default="data/cube_1k.json",
                        help="Path to CUBE_1k JSON file")
    parser.add_argument("--output_dir", default="outputs",
                        help="Base output directory")
    parser.add_argument("--debug", action="store_true",
                        help="Debug mode: process only 20 samples")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Maximum number of samples to process")
    parser.add_argument("--num_inference_steps", type=int, default=None,
                        help="Number of inference steps (model-specific default if not set)")
    parser.add_argument("--guidance_scale", type=float, default=None,
                        help="Guidance scale (model-specific default if not set)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--height", type=int, default=1024,
                        help="Image height")
    parser.add_argument("--width", type=int, default=1024,
                        help="Image width")
   
    args = parser.parse_args()
    
    # Get model name
    model_name = args.model
    logging.info(f"Using model: {model_name}")
    logging.info(f"Device: {device}")
    
    # Create output directory based on model
    # Format: <output_dir>/<model_name>/
    output_dir = os.path.join(args.output_dir, model_name)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    logging.info(f"Output directory: {output_dir}")
    
    # Load CUBE data
    cube_data = load_cube_data(args.cube_data, args.debug, args.max_samples)
    
    if len(cube_data) == 0:
        logging.error("No data found in CUBE_1k!")
        return
    
    # Build config
    config = {
        "seed": args.seed,
        "height": args.height,
        "width": args.width,
        "device": device,
    }
    
    # Add optional parameters if specified
    if args.num_inference_steps is not None:
        config["num_inference_steps"] = args.num_inference_steps
    if args.guidance_scale is not None:
        config["guidance_scale"] = args.guidance_scale
    
    # Enable multi-GPU if available
    if torch.cuda.device_count() > 1:
        config["device_map"] = "auto"
        logging.info(f"Multi-GPU enabled: {torch.cuda.device_count()} GPUs detected")
    
    # Get the model-specific function
    try:
        # Import from shared models/T2I directory
        model_module = __import__(f"models.T2I.{model_name}", fromlist=[''])
        model_func = model_module.generate_image
        logging.info(f"Loaded model function from models/T2I/{model_name}.py")
    except ImportError as e:
        logging.error(f"Could not load model '{model_name}': {e}")
        logging.error(f"Please create models/T2I/{model_name}.py with a generate_image(prompt, config) function")
        return
    
    # Process all prompts using the model-specific function
    successful, failed = process_prompts(
        model_func,
        cube_data,
        config,
        output_dir
    )


if __name__ == "__main__":
    main()
