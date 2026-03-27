"""
Image Transcreation Evaluation Script
======================================
Tests three metrics on a single generated image + prompt:
  1. ImageReward   — human aesthetic preference score
  2. MC-CLIP       — image-text alignment score
  3. VLM-VQA       — cultural detail accuracy (yes/no questions)

Setup:
  pip install image-reward open-clip-torch torch pillow transformers accelerate

For VLM-VQA you need an OpenAI API key (uses GPT-4o vision):
  export OPENAI_API_KEY="sk-..."

Usage:
  python evaluate_transcreation.py
"""

import os
import sys
import json
import base64
from pathlib import Path
from PIL import Image

# ── CONFIG — edit these ───────────────────────────────────────────────────────

IMAGE_PATH = "pizza.png"   # path to your generated image

# PROMPT = (
#     "You are generating a culturally adapted item photograph for a visual transcreation dataset.\n\n"
#     "The reference image shows \"Bissara\" — a traditional item.\n\n"
#     "Your task: Generate a new photorealistic image showing \"Dal Tadka\" (दाल तड़का), "
#     "the India cultural equivalent, adapted along the axis: \"Visual Similarity\".\n\n"
#     "Why this is equivalent: Both dishes are yellow-hued, pulse-based purees served in bowls "
#     "with a distinct 'tadka' (tempering) of oil and spices floating on the surface, "
#     "creating a near-identical visual profile.\n\n"
#     "Apply these scene adjustments to make the image feel authentically Indian:\n"
#     " \"Replace wooden bowls with stainless steel katoris\",\n"
#     "\"Swap the bread for stacked whole wheat rotis or phulkas\",\n"
#     "\"Replace olives with a small pile of mango pickle (achar) and sliced raw onions\",\n"
#     "\"Change the cumin powder in the small dish to a mix of dried red chilies and mustard seeds\"\n\n"
#     "Generate a high-quality, photorealistic photograph. Match the general composition and framing "
#     "of the reference image but replace all cultural elements (food, utensils, tableware, setting, "
#     "colors) with authentic Indian equivalents as described above.\n"
#     "Do not include any text or labels in the image."
# )

# PROMPT = PROMPT.replace("\n", " ")  # Flatten to single line for CLIP and ImageReward

PROMPT = "An Image of indian food dal tadka"

# Write yes/no questions that probe the cultural details you care about.
# Be specific — vague questions give noisy scores.
VQA_QUESTIONS = [
    "Is the food on the table consistent with Indian cuisine?",
    "Does the interior setting look like a typical Indian apartment?",
    "Are the people's clothing appropriate for an Indian household context?",
    "Does the lighting look like natural afternoon light?",
    "Is the overall scene realistic and not cartoonish or painted?",
]

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")   # or paste key here
GEMINI_MODEL        = "gemini/gemini-3.1-pro-preview"

# ─────────────────────────────────────────────────────────────────────────────


def run_image_reward(image_path: str, prompt: str) -> dict:
    """
    ImageReward: predicts human preference as a scalar.
    Input:  image file path + text prompt
    Output: float score (higher = more preferred; no fixed range, ~-2 to +2)
    """
    print("\n── ImageReward ──────────────────────────────")
    try:
        import ImageReward as ir  # package installs as 'image-reward', imports as 'ImageReward'
    except ImportError:
        try:
            import image_reward as ir  # some versions use underscore
        except ImportError:
            print("  [!] Could not import ImageReward.")
            print("      Try: pip install image-reward")
            print("      Then verify with: python -c \"import ImageReward\"")
            return {"score": None, "status": "missing_package"}

    try:
        model = ir.load("ImageReward-v1.0")
        # score() accepts (prompt, image_path) or (prompt, [image_path])
        score = model.score(prompt, image_path)
        # Normalise — some versions return a list, some a float
        if isinstance(score, (list, tuple)):
            scalar = float(score[0])
        else:
            scalar = float(score)
        print(f"  Score : {scalar:.4f}")
        print(f"  Meaning: {'above average preference' if scalar > 0 else 'below average preference'}")
        return {"score": scalar, "status": "ok"}
    except Exception as e:
        print(f"  [!] Error running ImageReward: {e}")
        return {"score": None, "status": "error", "detail": str(e)}


def run_clip_score(image_path: str, prompt: str) -> dict:
    """
    MC-CLIP score: cosine similarity between image embedding and text embedding.
    Uses open_clip with the multilingual ViT-B/32 checkpoint (mclip).
    Input:  image file path + text prompt
    Output: float 0–1 (higher = better alignment)
    """
    print("\n── MC-CLIP Score ────────────────────────────")
    try:
        import torch
        import open_clip

        # M-CLIP / multilingual model
        # Falls back to standard ViT-B-32 if mclip variant unavailable
        try:
            model, _, preprocess = open_clip.create_model_and_transforms(
                "M-CLIP/XLM-Roberta-Large-Vit-B-32",
                pretrained="openai"
            )
            tokenizer = open_clip.get_tokenizer("M-CLIP/XLM-Roberta-Large-Vit-B-32")
            model_name = "M-CLIP/XLM-Roberta-Large-Vit-B-32"
        except Exception:
            model, _, preprocess = open_clip.create_model_and_transforms(
                "ViT-B-32", pretrained="openai"
            )
            tokenizer = open_clip.get_tokenizer("ViT-B-32")
            model_name = "ViT-B-32 (standard CLIP, fallback)"

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device).eval()

        image = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
        text  = tokenizer([prompt]).to(device)

        with torch.no_grad():
            image_features = model.encode_image(image)
            text_features  = model.encode_text(text)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features  /= text_features.norm(dim=-1, keepdim=True)
            similarity = (image_features @ text_features.T).item()

        print(f"  Model : {model_name}")
        print(f"  Score : {similarity:.4f}")
        print(f"  Meaning: {'good alignment (>0.25)' if similarity > 0.25 else 'weak alignment'}")
        return {"score": similarity, "model": model_name, "status": "ok"}

    except ImportError:
        print("  [!] open_clip not installed. Run: pip install open-clip-torch")
        return {"score": None, "status": "missing_package"}
    except Exception as e:
        print(f"  [!] Error: {e}")
        return {"score": None, "status": "error", "detail": str(e)}


sys.path.append(str(Path(__file__).parent.parent.parent / "vlm_judge"))
from tasks.image_transcreation import USER_PROMPT_TEMPLATE, SYSTEM_PROMPT, parse_response

def _make_client():
    from google import genai
    from google.genai import types
    api_key = os.environ.get("LITELLM_API_KEY")
    if not api_key:
        raise EnvironmentError("LITELLM_API_KEY environment variable not set. E.g. export LITELLM_API_KEY='sk-...'")
    return genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(
            base_url="https://cmu.litellm.ai",
            headers={"Authorization": f"Bearer {api_key}"},
        ),
    )

def run_vlm_evaluator(source_image: str, target_image: str, prompt: str, target_culture: str = "Indian") -> dict:
    """
    VLM Judge score: uses Gemini to evaluate image transcreation based on standard criteria.
    """
    print("\n── VLM Judge Transcreation Score ────────────────")
    
    try:
        from google.genai import types
        client = _make_client()
        
        # We need a mocked metadata row that fits the template
        user_prompt = USER_PROMPT_TEMPLATE.format(
            src_country="Morocco", # Or wherever your source image is from
            target_culture=target_culture,
            category="Food"
        )
        
        # Structure content: Prompt text followed by the two images
        contents = [
            SYSTEM_PROMPT + "\n\n" + user_prompt,
            Image.open(source_image),
            Image.open(target_image)
        ]
        
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=contents,
            config=types.GenerateContentConfig(temperature=0.3),
        )
        
        parsed = parse_response(response.text)
        
        if parsed['is_valid']:
            print(f"Parsed content from VLM judge: {parsed['parsed']}")
            scores = {k: v['score'] for k, v in parsed['parsed'].items()}
            print(f"  Scores: {scores}")
            
            # Simple average score for summary
            avg_score = sum(scores.values()) / len(scores) if scores else 0
            
            return {
                "score": avg_score,
                "parsed": parsed['parsed'],
                "status": "ok"
            }
        else:
            print(f"  [!] Failed to parse VLM response: {parsed.get('error')}")
            print(f"  [!] Raw response: {response.text}")
            return {"score": None, "status": "parse_error", "error": parsed.get('error')}
            
    except ImportError:
         print("  [!] google-genai not installed. Run: pip install google-genai")
         return {"score": None, "status": "missing_package"}
    except Exception as e:
        print(f"  [!] Error: {e}")
        return {"score": None, "status": "error", "detail": str(e)}


def print_summary(ir_result, clip_result, vlm_result):
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)

    def fmt(val):
        return f"{val:.4f}" if isinstance(val, float) else str(val)

    ir_score   = ir_result.get("score")
    clip_score = clip_result.get("score")
    vlm_score  = vlm_result.get("score")

    print(f"  ImageReward  : {fmt(ir_score):>10}   (higher = more human-preferred)")
    print(f"  MC-CLIP      : {fmt(clip_score):>10}   (0–1, higher = better prompt alignment)")
    print(f"  VLM Judge avg: {fmt(vlm_score):>10}   (0-5, higher = better transcreation)")

    # Quick diagnosis
    print("\nDiagnosis:")
    if ir_score is not None and ir_score < 0:
        print("  → ImageReward is negative: image lacks visual realism. Add photorealism cues to prompt.")
    if clip_score is not None and clip_score < 0.22:
        print("  → CLIP score is low: image doesn't match prompt semantics. Make key elements more explicit.")
    if vlm_score is not None and vlm_score < 3.5:
        print("  → VLM Judge score is low: cultural details are incorrectly matched.")
    if all(v is None for v in [ir_score, clip_score, vlm_score]):
        print("  → All metrics skipped. Check setup above.")


def main():
    print(f"Target Image : {IMAGE_PATH}")
    print(f"Prompt: {PROMPT[:80]}{'...' if len(PROMPT) > 80 else ''}")

    if not Path(IMAGE_PATH).exists():
        print(f"\n[ERROR] Image not found at: {IMAGE_PATH}")
        print("Update IMAGE_PATH in the CONFIG section at the top of this file.")
        sys.exit(1)

    # ir_result   = run_image_reward(IMAGE_PATH, PROMPT)
    # clip_result = run_clip_score(IMAGE_PATH, PROMPT)

    ir_result   = 0
    clip_result = 0


    # You may need to provide a source reference image for accurate scoring. For now, we will
    # pass the target image twice as a fallback if you don't have a source image here.
    source_image = "Bissara.jpeg" if Path("Bissara.jpeg").exists() else IMAGE_PATH
    # source_image = "pizza.png" if Path("pizza.png").exists() else IMAGE_PATH

    vlm_result  = run_vlm_evaluator(source_image, IMAGE_PATH, PROMPT)

    print_summary(ir_result, clip_result, vlm_result)

    # Save results to JSON for later analysis
    output = {
        "image": IMAGE_PATH,
        "prompt": PROMPT,
        "image_reward": ir_result,
        "clip_score": clip_result,
        "vlm_judge": vlm_result,
    }
    out_path = "evaluation_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nFull results saved to: {out_path}")


if __name__ == "__main__":
    main()