import os
import json
import time
import argparse
import requests
import re
from pathlib import Path
from google import genai
from google.genai import types
import mimetypes

GEMINI_MODEL = "gemini/gemini-3.1-pro-preview"

def verify_entity(client, entity, image_path_or_url, region):
    prompt = f"""You are an expert cultural dataset curator. 
I am providing an image that supposedly represents the cultural entity: '{entity['name_en']}' (local name: {entity['name_local']}).
It is claimed to be a traditional item from {region}, belonging to the category '{entity['category']}' and subcategory '{entity['subcategory']}'.

Task:
1. Verify if '{entity['name_en']}' is genuinely an authentic culturalally relevant entity from {region}.
2. Check if the provided image accurately depicts this specific entity.

Respond ONLY with a valid JSON object matching this schema:
{{
  "verified": true or false,
  "reason": "Short explanation of your finding. Mention if the image matches and if the entity truly belongs to the region."
}}"""

    try:
        # Load from local file if it's a valid path, otherwise fetch from URL
        if os.path.exists(image_path_or_url):
            with open(image_path_or_url, "rb") as f:
                image_data = f.read()
            content_type = mimetypes.guess_type(image_path_or_url)[0] or 'image/jpeg'
        else:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
            # Add a small delay before fetching from URL to avoid rate limits
            time.sleep(1.5)
            resp = requests.get(image_path_or_url, headers=headers, stream=True, timeout=15)
            resp.raise_for_status()
            image_data = resp.content
            content_type = resp.headers.get('Content-Type', '')
            if not content_type.startswith('image/'):
                content_type = mimetypes.guess_type(image_path_or_url)[0] or 'image/jpeg'

        image_part = types.Part.from_bytes(data=image_data, mime_type=content_type)
        
        # Grounding tool can help the model look up if the entity is actually valid in that region
        grounding_tool = types.Tool(google_search=types.GoogleSearch())
        config = types.GenerateContentConfig(
            tools=[grounding_tool],
            temperature=0.1
        )

        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[prompt, image_part],
            config=config,
        )

        content = (response.text or "").strip()
        match = re.search(r"\{.*?\}", content, re.DOTALL)
        if match:
            data = json.loads(match.group(0))
            return bool(data.get("verified", False)), data.get("reason", "No reason provided")
        else:
            return False, f"Failed to parse JSON from response: {content}"
            
    except Exception as e:
        return False, f"Error during verification: {str(e)}"

def main():
    parser = argparse.ArgumentParser(description="Filter and verify JSON entities using Gemini Vision.")
    parser.add_argument("--input", required=True, help="Path to input JSON file (e.g., output/morocco.json)")
    parser.add_argument("--output", required=True, help="Path to output JSON file for verified entities")
    args = parser.parse_args()

    api_key = os.environ.get("LITELLM_API_KEY")
    if not api_key:
        print("Error: LITELLM_API_KEY environment variable not set.")
        return

    client = genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(
            base_url="https://cmu.litellm.ai",
            headers={"Authorization": f"Bearer {api_key}"},
        )
    )

    with open(args.input, 'r', encoding='utf-8') as f:
        data = json.load(f)

    region = data.get("region", "Unknown")
    new_data = {
        "region": region,
        "categories": {},
        "summary": {"total_entities": 0, "total_images": 0, "failed": []}
    }

    total_verified = 0
    total_images = 0

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for cat_name, subcats in data.get("categories", {}).items():
        new_data["categories"][cat_name] = {}
        for subcat_name, subcat_data in subcats.items():
            verified_entities = []
            entities = subcat_data.get("entities", [])
            
            print(f"\n--- Verifying {cat_name} / {subcat_name} ({len(entities)} entities) ---")
            
            for entity in entities:
                images = entity.get("images", [])
                if not images:
                    print(f"  [Skip] {entity['name_en']} - No images to verify against.")
                    continue
                
                # Use the first image for visual verification
                img_data = images[0]
                raw_local_path = img_data.get("local_path")
                img_source = None
                
                if raw_local_path:
                    # Construct absolute path relative to the script's directory (data_curation/)
                    script_dir = os.path.dirname(os.path.abspath(__file__))
                    full_path = os.path.join(script_dir, raw_local_path)
                    print(f"full_path: {full_path}, {os.path.exists(full_path)}, raw_local_path: {raw_local_path}")
                    
                    if os.path.exists(full_path):
                        img_source = full_path
                    elif os.path.exists(raw_local_path):
                        img_source = raw_local_path
                
                # If local_path is empty or missing, fallback to URL
                if not img_source:
                    img_source = img_data.get("url")
                    
                if not img_source:
                    print(f"  [Skip] {entity['name_en']} - No valid image path or URL found.")
                    continue
                    
                print(f"  [?] Verifying {entity['name_en']} (source: {'local' if os.path.exists(img_source) else 'url'})...")
                is_valid, reason = verify_entity(client, entity, img_source, region)
                
                if is_valid:
                    print(f"      [✓] Verified! Reason: {reason}")
                    entity['verified'] = True
                    entity['verification_method'] = 'gemini_multimodal'
                    # Capture the LLMs reason as extended description
                    entity['description'] = f"{entity.get('description', '')} | Verification: {reason}"
                    verified_entities.append(entity)
                    total_images += len(images)
                else:
                    print(f"      [✗] Rejected! Reason: {reason}")
                
                # To prevent rate-limiting from Wikimedia and Litellm
                time.sleep(2)
            
            new_data["categories"][cat_name][subcat_name] = {
                "total": len(verified_entities),
                "entities": verified_entities
            }
            total_verified += len(verified_entities)
            
            # Save incrementally after every subcategory
            new_data["summary"]["total_entities"] = total_verified
            new_data["summary"]["total_images"] = total_images
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(new_data, f, indent=2, ensure_ascii=False)

    print(f"\nDone! Verified {total_verified} entities out of the original set. Saved to {args.output}")

if __name__ == "__main__":
    main()
