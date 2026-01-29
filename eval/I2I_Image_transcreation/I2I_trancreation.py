from PIL import Image, ImageOps
import os
import torch
import PIL
import argparse
import yaml
import random
import logging
import json
import requests
from io import BytesIO


def download_image(path):
    """Download image from URL or load from local path."""
    if path.startswith("http"):
        response = requests.get(path, timeout=120)
        if response.status_code == 200 and response.headers['Content-Type'].startswith('image'):
            image = PIL.Image.open(BytesIO(response.content))
            image = PIL.ImageOps.exif_transpose(image)
            image = image.convert("RGB")
            return image
        else:
            logging.info(f"Invalid response")
            return "error"
    image = PIL.Image.open(path)
    image = PIL.ImageOps.exif_transpose(image)
    image = image.convert("RGB")
    return image


def concatenate_images(source_img, target_img, save_path):
    """Concatenate source and target images side by side."""
    total_width = source_img.width + target_img.width
    max_height = max(source_img.height, target_img.height)

    concatenated_img = Image.new('RGB', (total_width, max_height))
    concatenated_img.paste(source_img, (0, 0))
    concatenated_img.paste(target_img, (source_img.width, 0))
    concatenated_img.save(save_path)


def resize_image(image, threshold_size=1024):
    """Resize image if it exceeds threshold size."""
    w, h = image.size
    if w > threshold_size or h > threshold_size:
        if w > h:
            new_w = threshold_size
            new_h = int(h * (threshold_size / w))
        else:
            new_h = threshold_size
            new_w = int(w * (threshold_size / h))
        image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
    return image


def load_source_images(source_countries_list, source_data_path, debug=False):
    """
    Load source image paths from country JSON files.
    
    Args:
        source_countries_list: List of country names to load images from
        source_data_path: Path to directory containing country JSON files
        debug: If True, sample only 20 random images
        
    Returns:
        tuple: (source_paths, source_countries, source_categories) - lists of image paths, corresponding countries, and categories
    """
    all_source_paths = []
    all_source_countries = []
    all_source_categories = []
    
    for country in source_countries_list:
        country_paths_file = os.path.join(source_data_path, f"{country}.json")
        
        if not os.path.exists(country_paths_file):
            logging.warning(f"Country file not found: {country_paths_file}")
            continue
            
        with open(country_paths_file) as f:
            data = json.load(f)
            # Get values from json file which is a dictionary of dictionaries
            for category in data:
                all_source_paths.extend(data[category].values())
                all_source_countries.extend([country] * len(data[category].values()))
                all_source_categories.extend([category] * len(data[category].values()))

    if debug:
        logging.info("Debug mode enabled. Using 20 random images.")
        # Sample together to maintain correspondence
        indices = random.sample(range(len(all_source_paths)), min(20, len(all_source_paths)))
        all_source_paths = [all_source_paths[i] for i in indices]
        all_source_countries = [all_source_countries[i] for i in indices]
        all_source_categories = [all_source_categories[i] for i in indices]
        
    logging.info(f"Number of images found: {len(all_source_paths)}")
    
    # Filter valid paths (existing files or URLs)
    source_paths = []
    source_countries = []
    source_categories = []
    for i in range(len(all_source_paths)):
        if os.path.exists(all_source_paths[i]) or all_source_paths[i].startswith("http"):
            source_paths.append(all_source_paths[i])
            source_countries.append(all_source_countries[i])
            source_categories.append(all_source_categories[i])
            
    logging.info(f"Number of valid images: {len(source_paths)}")
    
    return source_paths, source_countries, source_categories


def process_images(model_func, source_paths, source_countries, source_categories, config, output_dir):
    """
    Process all images using the provided model function.
    
    Args:
        model_func: Function that takes (image, prompt, config) and returns edited image
        source_paths: List of source image paths
        source_countries: List of source countries corresponding to images
        source_categories: List of source categories corresponding to images
        config: Configuration dictionary
        output_dir: Directory to save outputs
        
    Returns:
        tuple: (successful, failed) - counts of successful and failed images
    """
    successful = 0
    failed = 0
    
    metadata_path = os.path.join(output_dir, "metadata.csv")
    with open(metadata_path, "w") as f:
        f.write("src_image_path,src_country,src_category,tgt_image_path,prompt,status\n")
        
        for i, image_path in enumerate(source_paths):
            try:
                logging.info(f"Processing [{i+1}/{len(source_paths)}]: {image_path}")
                
                # Load and resize image
                image = download_image(image_path)
                if image == "error":
                    logging.warning(f"Failed to download image: {image_path}")
                    f.write(f"{image_path},{source_countries[i]},{source_categories[i]},,{config['prompt']},download_failed\n")
                    failed += 1
                    continue
                    
                image = resize_image(image)
                prompt = config["prompt"]
                src_country = source_countries[i]
                src_category = source_categories[i]
                
                # Generate edited image using model-specific function
                generated_image = model_func(image, prompt, config)
                
                # Save generated image
                generated_image_path = os.path.join(output_dir, f"{src_country}_{os.path.basename(image_path)}")
                generated_image.save(generated_image_path)
                
                # Optionally save concatenated image (source + generated)
                if config.get("save_concatenated", False):
                    concat_path = os.path.join(output_dir, f"concat_{src_country}_{os.path.basename(image_path)}")
                    concatenate_images(image, generated_image, concat_path)
                
                f.write(f"{image_path},{src_country},{src_category},{generated_image_path},{prompt},success\n")
                successful += 1
                
            except torch.cuda.OutOfMemoryError as e:
                logging.warning(f"Skipping image {image_path} due to CUDA OOM error: {e}")
                f.write(f"{image_path},{source_countries[i]},{source_categories[i]},,{config['prompt']},cuda_oom\n")
                failed += 1
                # Clear cache and continue
                torch.cuda.empty_cache()
                continue
            except Exception as e:
                logging.error(f"Error processing {image_path}: {e}")
                f.write(f"{image_path},{source_countries[i]},{source_categories[i]},,{config['prompt']},error\n")
                failed += 1
                continue
    
    logging.info("\n" + "="*50)
    logging.info(f"Processing complete!")
    logging.info(f"  Successful: {successful}/{len(source_paths)}")
    logging.info(f"  Failed: {failed}/{len(source_paths)}")
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

    # Read in config file to get args
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/part1/brazil.yaml", 
                        help="Path to config file.")
    parser.add_argument("--model", default="instructpix2pix", 
                        help="Model to use: instructpix2pix, magicbrush, sdxl-instructpix2pix, etc.")
   
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    random.seed(config["seed"])
    
    # Get model name
    model_name = args.model
    logging.info(f"Using model: {model_name}")
    
    # Get target country from config
    target_country = config.get("target_country", "unknown")
    
    # Create output directory based on model and target country
    # Format: ./outputs/part1/<model_name>/<target_country>/
    output_dir = f"./outputs/part1/{model_name}/{target_country.lower().replace(' ', '-')}"
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    logging.info(f"Output directory: {output_dir}")

    # Load source images (data logic only)
    source_paths, source_countries, source_categories = load_source_images(
        config["source_countries"],
        config["source_data_path"],
        config.get("debug", False)
    )
    
    if len(source_paths) == 0:
        logging.error("No valid images found!")
        return

    # Get the model-specific function
    try:
        model_module = __import__(f"models.{model_name}", fromlist=[''])
        model_func = model_module.edit_image
        logging.info(f"Loaded model function from models/{model_name}.py")
    except ImportError as e:
        logging.error(f"Could not load model '{model_name}': {e}")
        logging.error(f"Please create models/{model_name}.py with an edit_image(image, prompt, config) function")
        return
    
    # Process all images using the model-specific function
    successful, failed = process_images(
        model_func,
        source_paths,
        source_countries,
        source_categories,
        config,
        output_dir
    )


if __name__ == "__main__":
    main()
