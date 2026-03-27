import os
import sys
import json
from pathlib import Path

import argparse

# Add data_curation to path so we can import from generate_transcreation_images
parent_dir = str(Path(__file__).resolve().parent.parent)
sys.path.append(parent_dir)

try:
    from generate_transcreation_images import run_image_generation
except ImportError as e:
    print(f"Failed to import run_image_generation: {e}")
    sys.exit(1)

# Import the batch evaluation you created
from evaluate_transcreation_batch import process_batch

# Define your prompt variants here
PROMPT_VARIANTS = {
    "baseline": """You are generating a culturally adapted item photograph for a visual transcreation dataset.

The reference image shows "{source_name}" — a traditional item.

Your task: Generate a new photorealistic image showing "{target_item}" ({target_item_local}), the {target_region} cultural equivalent, adapted along the axis: "{axis}".

Why this is equivalent: {reason}

Apply these scene adjustments:
{scene_adj_text}

Generate a high-quality, photorealistic photograph. Match the general composition and framing of the reference image but replace all cultural elements with authentic {target_region} equivalents as described above.
Do not include any text or labels in the image. Do not include any text or labels.""",

    "balanced_realism": """You are generating a culturally adapted item photograph for a visual transcreation dataset. We need a perfect balance of structural preservation and lifelike photographic realism.

The reference image shows "{source_name}" — a traditional item.

Your task: Generate a highly realistic, vibrant photograph showing "{target_item}" ({target_item_local}), the {target_region} cultural equivalent, adapted along the axis: "{axis}".

Why this is equivalent: {reason}

Apply these scene adjustments carefully:
{scene_adj_text}

Generate a high-quality photograph that looks completely natural and authentic to {target_region}. Use the reference image only as inspiration for overall intent and subject placement. Preserve the general idea of the framing where appropriate, but prioritize real-world plausibility and cultural authenticity. You have creative freedom to introduce realistic lighting, highly natural textures, and authentic environmental depth to entirely avoid any artificial or stiff look.
Do not include any text or labels in the image.""",

    "realism_focused": """You are an expert, award-winning photographer specializing in cultural lifestyle imagery.

The user has provided a reference image of "{source_name}". DO NOT simply copy its style if it looks generic, illustrated, or dull.
Instead, generate a photorealistic, high-resolution authentic photograph of "{target_item}" ({target_item_local}), a traditional item from {target_region}.

Context: {axis} ({reason})

Crucially, meticulously apply these scene adjustments:
{scene_adj_text}

The output MUST look like a real photo taken on a high-end DSLR camera. Emphasize natural lighting, highly realistic textures, soft depth of field (bokeh), and authentic environmental details from {target_region}. NO text, NO watermarks, NO artificial cartoonish illustrative styles!""",


    "structure_preserved": """You are a highly skilled visual artist generating a precise culturally adapted counterpart.

Reference item: "{source_name}".
Target item: "{target_item}" ({target_item_local}) from {target_region}.
Reasoning: {reason}

We need to strictly preserve the EXACT geometric structure and framing of the original image, but completely and seamlessly swap the cultural identity to {target_region}. 
Ensure these adjustments are rigidly applied:
{scene_adj_text}

The image must be fully photorealistic, maintaining raw real-world photographic textures."""
}


def main():
    parser = argparse.ArgumentParser(description="Run transcreation prompt experiments.")
    parser.add_argument("--input_json", type=str, default="../output/morocco_to_india_transcreation.json",
                        help="Path to the source transcreation JSON")
    parser.add_argument("--use_subset", action="store_true", default=True,
                        help="If provided, run on a small subset (2 entities) for testing. Note: Default is True in this script version.")
    parser.add_argument("--full", action="store_true", 
                        help="Run on the full dataset (Overrides --use_subset default)")
    args = parser.parse_args()

    base_json = args.input_json
    use_subset = False if args.full else args.use_subset
    
    if not os.path.exists(base_json):
        print(f"[!] Warning: {base_json} not found. Please provide exactly which transcreation JSON you want to experiment on.")
        return

    print(f"Loading base JSON: {base_json} ...")
    with open(base_json, "r") as f:
        data = json.load(f)
        
    if use_subset:
        # Take a small subset for the experiment to save time and API costs (e.g., first 2 entities)
        first_cat = list(data.get("categories", {}).keys())[0] if data.get("categories") else None
        if not first_cat:
            print("No categories found.")
            return
            
        first_subcat = list(data["categories"][first_cat].keys())[0]
        all_entities = data["categories"][first_cat][first_subcat]
        
        subset_entities = {k: v for i, (k, v) in enumerate(all_entities.items()) if i < 2}
        subset_data = {
            "source_region": data.get("source_region"),
            "target_region": data.get("target_region"),
            "categories": {
                first_cat: {
                    first_subcat: subset_entities
                }
            }
        }
        working_json_path = "experiment_subset.json"
        with open(working_json_path, "w") as f:
            json.dump(subset_data, f, indent=2)
        print(f"Using SUBSET of 2 entities saved to {working_json_path}")
    else:
        # Use the full dataset
        working_json_path = base_json
        print(f"Using FULL dataset: {base_json}")
        
    results = {}

    for variant_name, prompt_template in PROMPT_VARIANTS.items():
        print(f"\n\n{'='*60}\nRUNNING EXPERIMENT VARIANT: {variant_name}\n{'='*60}")
        
        output_dir = f"experiment_outputs/{variant_name}/images"
        generated_json = f"experiment_outputs/{variant_name}/generated.json"
        evaluated_json = f"experiment_outputs/{variant_name}/evaluated.json"
        
        os.makedirs(f"experiment_outputs/{variant_name}", exist_ok=True)
        
        # 1. Generate Images
        print(f"\n[1] Generating images for variant: {variant_name}...")
        run_image_generation(
            transcreation_json_path=working_json_path,
            output_dir=output_dir,
            output_json_path=generated_json,
            delay=2.0,
            prompt_template=prompt_template
        )
        
        # 2. Evaluate
        print(f"\n[2] Evaluating images for variant: {variant_name}...")
        process_batch(generated_json, evaluated_json)
        
        # 3. Aggregate scores
        print(f"\n[3] Aggregating results for: {variant_name}...")
        with open(evaluated_json, "r") as f:
            eval_data = json.load(f)
            
        ir_scores, clip_scores, vlm_scores = [], [], []
        
        # Safely traverse and extract scores
        for c, subcats in eval_data.get("categories", {}).items():
            for sc, entities in subcats.items():
                for ent_name, ent_obj in entities.items():
                    for alt in ent_obj.get("alternatives", []):
                        metrics = alt.get("eval_metrics", {})
                        
                        # Use dict.get with a fallback Check against None specifically since scores can be 0 or negative
                        if metrics.get("image_reward") is not None: 
                            ir_scores.append(metrics["image_reward"])
                        if metrics.get("mc_clip") is not None: 
                            clip_scores.append(metrics["mc_clip"])
                        if metrics.get("vlm_judge") is not None: 
                            vlm_scores.append(metrics["vlm_judge"])
                        
        avg_ir = sum(ir_scores) / len(ir_scores) if len(ir_scores) > 0 else 0
        avg_clip = sum(clip_scores) / len(clip_scores) if len(clip_scores) > 0 else 0
        avg_vlm = sum(vlm_scores) / len(vlm_scores) if len(vlm_scores) > 0 else 0
        
        results[variant_name] = {
            "ImageReward": avg_ir,
            "MCCLIP": avg_clip,
            "VLM_Judge": avg_vlm
        }

    print("\n\n" + "="*50)
    print("FINAL EXPERIMENT RESULTS (AVERAGES)")
    print("="*50)
    for var, metrics in results.items():
        print(f"Variant: {var}")
        print(f"  ImageReward : {metrics['ImageReward']:.4f}")
        print(f"  MC-CLIP     : {metrics['MCCLIP']:.4f}")
        print(f"  VLM Judge   : {metrics['VLM_Judge']:.4f}\n")


if __name__ == "__main__":
    main()
