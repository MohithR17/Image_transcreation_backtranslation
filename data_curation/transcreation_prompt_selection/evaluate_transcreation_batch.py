import os
import sys
import json
import tempfile
import requests
from pathlib import Path
from tqdm import tqdm
from PIL import Image

sys.path.append(str(Path(__file__).parent.parent.parent / "vlm_judge"))
from tasks.image_transcreation import USER_PROMPT_TEMPLATE, SYSTEM_PROMPT, parse_response

# Global definitions to hold our imported metric modules
client = None
gemini_model_name = "gemini/gemini-3.1-pro-preview"
image_reward_model = None
open_clip_model = None
open_clip_preprocess = None
open_clip_tokenizer = None
torch_device = "cpu"

HEADERS = {
    "User-Agent": "CulturalDatasetPipeline/1.0 (research; contact: your@email.com)",
    "Referer":    "https://commons.wikimedia.org/",
    "Accept":     "image/webp,image/apng,image/*,*/*;q=0.8",
}

def _fetch_image_bytes(url: str) -> tuple[bytes, str]:
    """Download image bytes from a URL. Returns (bytes, mime_type) or None."""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=30)
        resp.raise_for_status()
        content_type = resp.headers.get("Content-Type", "image/jpeg").split(";")[0].strip()
        return resp.content, content_type
    except Exception as e:
        print(f"  [!] Could not fetch image from {url}: {e}")
        return None, None

def setup_metrics():
    global client, image_reward_model, open_clip_model, open_clip_preprocess, open_clip_tokenizer, torch_device
    
    print("\n--- Initializing Metric Models ---")
    
    # 1. Initialize Gemini Client
    try:
        from google import genai
        from google.genai import types
        api_key = os.environ.get("LITELLM_API_KEY")
        if not api_key:
            print("  [!] LITELLM_API_KEY not set. VLM Judge will be skipped.")
        else:
            client = genai.Client(
                api_key=api_key,
                http_options=types.HttpOptions(
                    base_url="https://cmu.litellm.ai",
                    headers={"Authorization": f"Bearer {api_key}"},
                ),
            )
            print("  [✓] VLM Check Initialized")
    except Exception as e:
        print(f"  [!] Failed to initialize VLM Check: {e}")

    # 2. Initialize ImageReward
    try:
        import ImageReward as ir
        image_reward_model = ir.load("ImageReward-v1.0")
        print("  [✓] ImageReward Initialized")
    except Exception as e:
        print(f"  [!] Failed to initialize ImageReward: {e}")

    # 3. Initialize CLIP
    try:
        import torch
        import open_clip
        torch_device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            open_clip_model, _, open_clip_preprocess = open_clip.create_model_and_transforms(
                "xlm-roberta-base-ViT-B-32", pretrained="laion5b_s13b_b90k"
            )
            open_clip_tokenizer = open_clip.get_tokenizer("xlm-roberta-base-ViT-B-32")
            # Fix for newer transformers versions where batch_encode_plus is removed
            if hasattr(open_clip_tokenizer, 'tokenizer') and not hasattr(open_clip_tokenizer.tokenizer, 'batch_encode_plus'):
                open_clip_tokenizer.tokenizer.batch_encode_plus = lambda texts, **kwargs: open_clip_tokenizer.tokenizer(texts, **kwargs)
        except Exception as inner_e:
            print(f"  [!] Failed to load xlm-roberta-base-ViT-B-32: {inner_e}")
            open_clip_model, _, open_clip_preprocess = open_clip.create_model_and_transforms(
                "ViT-B-32", pretrained="openai"
            )
            open_clip_tokenizer = open_clip.get_tokenizer("ViT-B-32")
        
        open_clip_model = open_clip_model.to(torch_device).eval()
        print("  [✓] MC-CLIP Initialized")
    except Exception as e:
        print(f"  [!] Failed to initialize MC-CLIP: {e}")
        
    print("----------------------------------\n")


def run_vlm_evaluator(source_image: str, target_image: str, src_culture: str, target_culture: str, category: str) -> dict:
    if client is None:
        return {"score": None, "status": "no_client"}
        
    try:
        from google.genai import types
        user_prompt = USER_PROMPT_TEMPLATE.format(
            src_country=src_culture,
            target_culture=target_culture,
            category=category
        )
        
        contents = [
            SYSTEM_PROMPT + "\n\n" + user_prompt,
            Image.open(source_image),
            Image.open(target_image)
        ]
        
        response = client.models.generate_content(
            model=gemini_model_name,
            contents=contents,
            config=types.GenerateContentConfig(temperature=0.3),
        )
        
        parsed = parse_response(response.text)
        if parsed['is_valid']:
            scores = {k: v['score'] for k, v in parsed['parsed'].items()}
            # Also capture the full parsed dictionary to get the exact reasoning for each criterion
            detailed_analysis = parsed['parsed']
            avg_score = sum(scores.values()) / len(scores) if scores else 0
            return {
                "score": avg_score, 
                "detailed_scores": scores, 
                "detailed_analysis": detailed_analysis, 
                "status": "ok"
            }
        else:
            return {"score": None, "status": "parse_error", "error": parsed.get('error'), "raw_response": response.text}
    except Exception as e:
        print(f"Error: {e}"); return {"score": None, "status": "error", "error": str(e)}


def run_image_reward(image_path: str, prompt: str) -> dict:
    if image_reward_model is None:
        return {"score": None, "status": "model_not_loaded"}
    try:
        score = image_reward_model.score(prompt, image_path)
        if isinstance(score, (list, tuple)):
            scalar = float(score[0])
        else:
            scalar = float(score)
        return {"score": scalar, "status": "ok"}
    except Exception as e:
        print(f"Error: {e}"); return {"score": None, "status": "error", "error": str(e)}


def run_clip_score(image_path: str, prompt: str) -> dict:
    if open_clip_model is None:
        return {"score": None, "status": "model_not_loaded"}
    import torch
    try:
        image = open_clip_preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(torch_device)
        text  = open_clip_tokenizer([prompt]).to(torch_device)

        with torch.no_grad():
            image_features = open_clip_model.encode_image(image)
            text_features  = open_clip_model.encode_text(text)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features  /= text_features.norm(dim=-1, keepdim=True)
            similarity = (image_features @ text_features.T).item()
            
        return {"score": similarity, "status": "ok"}
    except Exception as e:
        print(f"Error: {e}"); return {"score": None, "status": "error", "error": str(e)}


def process_batch(input_json_path: str, output_json_path: str):
    if os.path.exists(output_json_path):
        print(f"Resuming from existing output JSON: {output_json_path}")
        with open(output_json_path, 'r') as f:
            data = json.load(f)
    else:
        with open(input_json_path, 'r') as f:
            data = json.load(f)
        
    src_culture = data.get("source_region", "Unknown")
    target_culture = data.get("target_region", "Unknown")
    
    # We will accumulate items directly into the loaded structure so we can dump it easily
    # Flatten the tree to get a list of variations to evaluate
    flattened_alternatives = []
    
    categories = data.get("categories", {})
    for top_cat, subcats in categories.items():
        for subcat, items in subcats.items():
            for item_name, item_data in items.items():
                src_image_local = item_data.get("source_entity", {}).get("local_path")
                src_image_url = item_data.get("source_entity", {}).get("image_url")
                alts = item_data.get("alternatives", [])
                
                for alt_idx, alt in enumerate(alts):
                    metrics = alt.get("eval_metrics", {})
                    # Check if already fully evaluated to resume/skip
                    if metrics:
                        has_ir = metrics.get("image_reward") is not None
                        has_clip = metrics.get("mc_clip") is not None
                        has_vlm = metrics.get("vlm_judge") is not None
                        has_err = "error" in metrics
                        if (has_ir and has_clip and has_vlm) or has_err:
                            print(f"  [>] Skipping already evaluated: {item_name} -> {alt.get('target_item', f'Alt {alt_idx}')}")
                            continue

                    tgt_image = alt.get("generated_image_path")
                    prompt = alt.get("generation_prompt", "") # Ensure your schema specifies generation_prompt
                    
                    if not prompt: 
                         # Fallback prompt if omitted
                         prompt = f"An image of {alt.get('target_item', '')}"
                         
                    flattened_alternatives.append({
                        "ref_alt": alt,
                        "category": subcat,
                        "src_image_local": src_image_local,
                        "src_image_url": src_image_url,
                        "tgt_image": tgt_image,
                        "prompt": prompt
                    })
    
    print(f"Total generations to evaluate: {len(flattened_alternatives)}")
    setup_metrics()
    
    for task in tqdm(flattened_alternatives, desc="Evaluating Transcreations"):
        alt = task["ref_alt"]
        
        src_path_local = task["src_image_local"]
        src_path_url = task["src_image_url"]
        tgt_path = task["tgt_image"]
        prompt = task["prompt"]
        category = task["category"]
        
        alt["eval_metrics"] = {}
        
        if not (tgt_path and os.path.exists(tgt_path)):
            alt["eval_metrics"]["error"] = f"Target image not found: {tgt_path}"
            continue

        src_path_valid = None
        
        # Determine local cache path from URL if needed
        cache_dir = os.path.join(os.path.dirname(input_json_path), "cache_images")
        
        if src_path_local and os.path.exists(src_path_local):
            src_path_valid = src_path_local
        elif src_path_url:
            os.makedirs(cache_dir, exist_ok=True)
            # Create a safe filename using a hash of the URL to avoid path issues
            import hashlib
            url_hash = hashlib.md5(src_path_url.encode('utf-8')).hexdigest()
            cached_img_path = os.path.join(cache_dir, f"{url_hash}.png")
            
            if os.path.exists(cached_img_path):
                 src_path_valid = cached_img_path
            else:
                raw_bytes, _ = _fetch_image_bytes(src_path_url)
                if raw_bytes:
                    with open(cached_img_path, 'wb') as f:
                        f.write(raw_bytes)
                    src_path_valid = cached_img_path

        if not src_path_valid:
             alt["eval_metrics"]["error"] = f"Source image not found locally ({src_path_local}) or failed to download ({src_path_url})"
             continue

        # 1. Image Reward
        ir_res = run_image_reward(tgt_path, prompt)
        alt["eval_metrics"]["image_reward"] = ir_res.get("score")
        
        # 2. MC-CLIP
        clip_res = run_clip_score(tgt_path, prompt)
        alt["eval_metrics"]["mc_clip"] = clip_res.get("score")
        
        # 3. VLM Judge
        vlm_res = run_vlm_evaluator(src_path_valid, tgt_path, src_culture, target_culture, category)
        if vlm_res.get("status") == "ok":
            alt["eval_metrics"]["vlm_judge"] = vlm_res.get("score")
            alt["eval_metrics"]["vlm_detailed_scores"] = vlm_res.get("detailed_scores")
            alt["eval_metrics"]["vlm_detailed_analysis"] = vlm_res.get("detailed_analysis")
        else:
            alt["eval_metrics"]["vlm_judge"] = None
            alt["eval_metrics"]["vlm_error"] = vlm_res.get("error")

    # Output back to JSON completely preserved with eval scores
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, 'w') as f:
         json.dump(data, f, indent=2, ensure_ascii=False)
         
    print(f"Done! Evaluated data saved to {output_json_path}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python evaluate_transcreation_batch.py <input_json> <output_json>")
        sys.exit(1)
        
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    if not os.path.exists(input_file):
        print(f"Error: Input file {input_file} not found.")
        sys.exit(1)
        
    process_batch(input_file, output_file)
