"""
Transcreation Image Generator
================================
Reads a transcreation JSON (output of transcreation_suggest.py) and generates
one image per alternative using Gemini's image generation model.

For each source entity, the source image is passed along with the alternative's
axis, reason, and scene_adjustments to prompt Gemini to generate a culturally
adapted image for the target region.

Saves a new JSON mirroring the transcreation JSON structure but with an added
`generated_image_path` field for each alternative.

Usage:
    python generate_transcreation_images.py \\
        --transcreation_json output/morocco_to_india_transcreation.json \\
        --output_dir output/generated_images/morocco_to_india \\
        --output_json output/morocco_to_india_with_images.json
"""

import argparse
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Optional

import requests
from google import genai
from google.genai import types

# ── Config ────────────────────────────────────────────────────────────────────
IMAGE_GEN_MODEL = "gemini/gemini-3.1-flash-image-preview"
# Same headers used by cultural_entity_pipeline_new.py for Wikimedia downloads
HEADERS = {
    "User-Agent": "CulturalDatasetPipeline/1.0 (research; contact: your@email.com)",
    "Referer":    "https://commons.wikimedia.org/",
    "Accept":     "image/webp,image/apng,image/*,*/*;q=0.8",
}

logger = logging.getLogger("image_gen")


# ── Gemini client ─────────────────────────────────────────────────────────────
def _make_client() -> genai.Client:
    api_key = os.environ.get("LITELLM_API_KEY")
    if not api_key:
        raise EnvironmentError("LITELLM_API_KEY environment variable not set.")
    return genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(
            base_url="https://cmu.litellm.ai",
            headers={"Authorization": f"Bearer {api_key}"},
        ),
    )


# ── Image helpers ─────────────────────────────────────────────────────────────
def _fetch_image_bytes(url: str) -> Optional[tuple[bytes, str]]:
    """Download image bytes from a URL. Returns (bytes, mime_type) or None."""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=30)
        resp.raise_for_status()
        content_type = resp.headers.get("Content-Type", "image/jpeg").split(";")[0].strip()
        return resp.content, content_type
    except Exception as e:
        logger.warning("Could not fetch image from %s: %s", url, e)
        return None


def _ext_to_mime(ext: str) -> str:
    return {
        "jpg": "image/jpeg", "jpeg": "image/jpeg",
        "png": "image/png", "gif": "image/gif",
        "webp": "image/webp", "tif": "image/tiff", "tiff": "image/tiff",
    }.get(ext.lower().lstrip("."), "image/jpeg")


def _safe_filename(text: str) -> str:
    """Convert a string to a safe directory/file name component."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s]+", "_", text)
    return text[:60]


# ── Core generation ───────────────────────────────────────────────────────────
DEFAULT_PROMPT_TEMPLATE = """You are generating a culturally adapted item photograph for a visual transcreation dataset.

The reference image shows "{source_name}" — a traditional item.

Your task: Generate a new photorealistic image showing "{target_item}" ({target_item_local}), the {target_region} cultural equivalent, adapted along the axis: "{axis}".

Why this is equivalent: {reason}

Apply these scene adjustments to make the image feel authentically {target_region}:
{scene_adj_text}

Generate a high-quality, photorealistic photograph. Match the general composition and framing of the reference image but replace all cultural elements (food, utensils, tableware, setting, colors) with authentic {target_region} equivalents as described above.
Do not include any text or labels in the image."""

def generate_image_for_alternative(
    source_entity: dict,
    alternative: dict,
    target_region: str,
    client: genai.Client,
    prompt_template: str = None
) -> tuple[Optional[bytes], str]:
    """
    Calls Gemini image generation with:
      - Source entity image (as visual reference for composition/style)
      - A prompt describing the target cultural equivalent + scene adjustments

    Returns raw PNG bytes of the generated image, or None on failure.
    """
    source_name   = source_entity.get("name_en", "unknown")
    source_region = source_entity.get("region", "unknown")  # not in transcreation JSON directly
    image_url     = source_entity.get("image_url", "")

    axis              = alternative.get("axis", "")
    axis_desc         = alternative.get("axis_description", "")
    target_item       = alternative.get("target_item", "")
    target_item_local = alternative.get("target_item_local", "")
    reason            = alternative.get("reason", "")
    scene_adjustments = alternative.get("scene_adjustments", [])
    scene_adj_text    = "\n".join(f"- {a}" for a in scene_adjustments)

    template = prompt_template if prompt_template is not None else DEFAULT_PROMPT_TEMPLATE
    prompt = template.format(
        source_name=source_name,
        target_item=target_item,
        target_item_local=target_item_local,
        target_region=target_region,
        axis=axis,
        reason=reason,
        scene_adj_text=scene_adj_text
    )

    # Build contents: source image (if available) + prompt
    contents: list = []

    if image_url:
        img_data = _fetch_image_bytes(image_url)
        if img_data:
            raw_bytes, mime_type = img_data
            contents.append(types.Part.from_bytes(data=raw_bytes, mime_type=mime_type))
            print(f"    ↳ Source image fetched ({len(raw_bytes)//1024}KB, {mime_type})")
        else:
            print(f"    ↳ Could not fetch source image, generating text-only")

    contents.append(types.Part.from_text(text=prompt))

    try:
        response = client.models.generate_content(
            model=IMAGE_GEN_MODEL,
            contents=contents,
            config=types.GenerateContentConfig(
                response_modalities=["IMAGE"],
            ),
        )

        if not response.candidates:
            raise ValueError("No candidates returned")

        for part in response.parts:
            if part.inline_data is not None:
                return part.inline_data.data, prompt  # raw bytes

        raise ValueError("No image part found in response")

    except Exception as e:
        logger.error("Image generation failed for '%s' / '%s': %s", source_name, target_item, e)
        return None, prompt


# ── Main pipeline ─────────────────────────────────────────────────────────────
def run_image_generation(
    transcreation_json_path: str,
    output_dir: str,
    output_json_path: str,
    delay: float = 2.0,
    prompt_template: str = None,
) -> None:

    with open(transcreation_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    source_region = data.get("source_region", "unknown")
    target_region = data.get("target_region", "unknown")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load or init output JSON (for resumability)
    out_json_path = Path(output_json_path)
    if out_json_path.exists():
        with open(out_json_path, "r", encoding="utf-8") as f:
            output = json.load(f)
        print(f"[Resume] Loaded existing output from {out_json_path}")
    else:
        output = {
            "source_region": source_region,
            "target_region": target_region,
            "categories": {},
        }

    client = _make_client()

    total = done = failed = skipped = 0

    for category, subcategories in data.get("categories", {}).items():
        for subcategory, entities in subcategories.items():
            print(f"\n[{category} / {subcategory}]")

            for entity_name, entity_data in entities.items():
                source_entity = entity_data.get("source_entity", {})
                alternatives  = entity_data.get("alternatives", [])

                # Ensure output structure exists
                out_entity = (
                    output.setdefault("categories", {})
                          .setdefault(category, {})
                          .setdefault(subcategory, {})
                          .setdefault(entity_name, {
                              "source_entity": source_entity,
                              "alternatives": [],
                          })
                )

                # Build a lookup of already-generated alternatives by index
                existing_alts = {
                    alt.get("_alt_index"): alt
                    for alt in out_entity.get("alternatives", [])
                    if "generated_image_path" in alt
                }

                updated_alts = []
                for idx, alt in enumerate(alternatives):
                    total += 1
                    target_item = alt.get("target_item", f"alt_{idx}")

                    # Skip if already generated
                    if idx in existing_alts:
                        print(f"  [skip] '{entity_name}' alt {idx+1}/5 '{target_item}' already done")
                        updated_alts.append(existing_alts[idx])
                        skipped += 1
                        continue

                    print(f"  → '{entity_name}' alt {idx+1}/5: '{target_item}' [{alt.get('axis','')}]")

                    img_bytes, generation_prompt = generate_image_for_alternative(
                        source_entity=source_entity,
                        alternative=alt,
                        target_region=target_region,
                        client=client,
                        prompt_template=prompt_template
                    )

                    # Build output record — copy all fields from alt, add path
                    alt_record = dict(alt)
                    alt_record["_alt_index"] = idx

                    if img_bytes:
                        # Save image: output_dir/<category>/<subcategory>/<entity>/<idx>_<target_item>.png
                        img_subdir = out_dir / _safe_filename(category) / _safe_filename(subcategory) / _safe_filename(entity_name)
                        img_subdir.mkdir(parents=True, exist_ok=True)
                        img_filename = f"{idx+1:02d}_{_safe_filename(target_item)}.png"
                        img_path = img_subdir / img_filename

                        with open(img_path, "wb") as f:
                            f.write(img_bytes)

                        alt_record["generated_image_path"] = str(img_path)
                        alt_record["generation_prompt"] = generation_prompt
                        done += 1
                        print(f"  [✓] Saved → {img_path}")
                    else:
                        alt_record["generated_image_path"] = None
                        failed += 1
                        print(f"  [✗] Generation failed for '{target_item}'")

                    updated_alts.append(alt_record)

                    # Save after every image
                    out_entity["alternatives"] = updated_alts
                    output["summary"] = {
                        "total": total, "done": done,
                        "failed": failed, "skipped": skipped,
                    }
                    out_json_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(out_json_path, "w", encoding="utf-8") as f:
                        json.dump(output, f, indent=2, ensure_ascii=False)

                    time.sleep(delay)

                out_entity["alternatives"] = updated_alts

    output["summary"] = {
        "total": total, "done": done,
        "failed": failed, "skipped": skipped,
    }
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"IMAGE GENERATION COMPLETE: {source_region} → {target_region}")
    print(f"  Total    : {total}")
    print(f"  Generated: {done}")
    print(f"  Failed   : {failed}")
    print(f"  Skipped  : {skipped}")
    print(f"  Output   → {out_json_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate culturally adapted images using Gemini I2I.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--transcreation_json", required=True,
        help="Path to transcreation JSON (output of transcreation_suggest.py)")
    parser.add_argument("--output_dir", default=None,
        help="Directory to save generated images. "
             "Defaults to output/generated_images/<source>_to_<target>")
    parser.add_argument("--output_json", default=None,
        help="Output JSON path. "
             "Defaults to output/<source>_to_<target>_with_images.json")
    parser.add_argument("--delay", type=float, default=2.0,
        help="Seconds to wait between Gemini calls (default: 2.0)")
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    args = parse_args()

    # Derive defaults from transcreation JSON filename
    stem = Path(args.transcreation_json).stem  # e.g. "morocco_to_india_transcreation"
    slug = stem.replace("_transcreation", "")   # e.g. "morocco_to_india"

    output_dir  = args.output_dir  or f"output/generated_images/{slug}"
    output_json = args.output_json or f"output/{slug}_with_images.json"

    run_image_generation(
        transcreation_json_path=args.transcreation_json,
        output_dir=output_dir,
        output_json_path=output_json,
        delay=args.delay,
    )
