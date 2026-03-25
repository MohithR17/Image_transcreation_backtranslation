"""
Transcreation Alternative Suggester
=====================================
Takes a source region's cultural entity JSON and a target region,
then uses Gemini (with the entity's image) to suggest 5 culturally
equivalent alternatives in the target region — each along a different
axis (visual, sensory, functional, contextual, emotional).

Usage:
    python transcreation_suggest.py \
        --source_json output/morocco.json \
        --target_region India \
        --output_json output/morocco_to_india_transcreation.json
"""

import argparse
import base64
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
GEMINI_MODEL = "gemini/gemini-3-flash-preview"
HEADERS = {"User-Agent": "TranscreationPipeline/1.0 (research; contact@example.com)"}

logger = logging.getLogger("transcreation")


# ── Gemini client factory ─────────────────────────────────────────────────────
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
def _load_image_bytes(image_info: dict) -> Optional[tuple[bytes, str]]:
    """
    Returns (raw_bytes, mime_type) for an entity image.
    Tries local_path first, then downloads from URL.
    """
    local_path = image_info.get("local_path", "")
    url = image_info.get("url", "")

    # Try local file first
    if local_path and Path(local_path).exists():
        suffix = Path(local_path).suffix.lower().lstrip(".")
        mime = _ext_to_mime(suffix)
        with open(local_path, "rb") as f:
            return f.read(), mime

    # Download from URL
    if url:
        try:
            resp = requests.get(url, headers=HEADERS, timeout=30)
            resp.raise_for_status()
            content_type = resp.headers.get("Content-Type", "image/jpeg").split(";")[0]
            return resp.content, content_type
        except Exception as e:
            logger.warning("Could not fetch image from %s: %s", url, e)

    return None


def _ext_to_mime(ext: str) -> str:
    return {
        "jpg": "image/jpeg", "jpeg": "image/jpeg",
        "png": "image/png", "gif": "image/gif",
        "webp": "image/webp", "tif": "image/tiff", "tiff": "image/tiff",
    }.get(ext, "image/jpeg")


# ── Core suggestion logic ─────────────────────────────────────────────────────
def suggest_alternatives(
    entity: dict,
    target_region: str,
    client: genai.Client,
) -> list[dict]:
    """
    Calls Gemini with the entity image + metadata and asks for 5 alternatives
    in target_region, each along a different cultural-equivalence axis.
    Returns a list of alternative dicts.
    """
    name        = entity.get("name_en", "unknown")
    source      = entity.get("region", "unknown")
    category    = entity.get("category", "")
    subcategory = entity.get("subcategory", "")
    description = entity.get("description", "")
    images      = entity.get("images", [])

    prompt = f"""You are a cultural transcreation expert helping to adapt visual content across cultures.

Source item: "{name}" from {source}
Category: {category} → {subcategory}
Description: {description}

Look at the image provided and suggest 5 culturally equivalent items from {target_region} that could replace "{name}" in a visual scene.

Each alternative should preserve a DIFFERENT dimension of equivalence. Explore freely — you might consider dimensions like:
- Visual similarity (shape, color, form)
- Sensory similarity (taste, smell, texture)
- Functional/contextual role (used the same way, same occasion)
- Emotional or symbolic meaning (what it represents culturally)
- Social context (who uses it, when, where)

Do NOT hardcode these axes — choose the 5 most insightful dimensions for THIS specific item.

For each alternative, also suggest small scene-level adjustments (utensils, tableware, setting, colors, surroundings) to make the full image feel authentically {target_region}.

Respond ONLY with a valid JSON array of exactly 5 items, no markdown fences:
[
  {{
    "axis": "Name of the equivalence dimension",
    "axis_description": "Brief explanation of why this axis matters for this item",
    "target_item": "Name of the {target_region} equivalent",
    "target_item_local": "Name in local language/script if applicable",
    "reason": "Why this item is equivalent along this axis",
    "scene_adjustments": ["adjustment 1", "adjustment 2", "..."]
  }}
]"""

    contents: list = [prompt]

    # Attach image if available
    if images:
        img_data = _load_image_bytes(images[0])
        if img_data:
            raw_bytes, mime_type = img_data
            contents = [
                types.Part.from_bytes(data=raw_bytes, mime_type=mime_type),
                prompt,
            ]
        else:
            logger.warning("  [!] Could not load image for '%s', proceeding text-only", name)

    print(f"  → Querying Gemini for '{name}' ({source} → {target_region})...")

    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=contents,
            config=types.GenerateContentConfig(temperature=0.7),
        )

        if not response.candidates:
            raise ValueError("Gemini returned no candidates")

        content = (response.text or "").strip()
        if not content:
            raise ValueError("Gemini returned empty response")

        match = re.search(r"\[\s*\{.*?\}\s*\]", content, re.DOTALL)
        if not match:
            raise ValueError(f"Could not extract JSON from response:\n{content}")

        alternatives = json.loads(match.group(0))
        logger.info("  → %d alternatives for '%s'", len(alternatives), name)
        return alternatives

    except json.JSONDecodeError as e:
        logger.error("JSON parse error for '%s': %s", name, e)
        return []
    except Exception as e:
        logger.error("Gemini error for '%s': %s", name, e)
        return []


# ── Main pipeline ─────────────────────────────────────────────────────────────
def run_transcreation(
    source_json_path: str,
    target_region: str,
    output_json_path: str,
    delay: float = 1.0,
) -> None:

    with open(source_json_path, "r", encoding="utf-8") as f:
        source_data = json.load(f)

    source_region = source_data.get("region", "Unknown")
    client = _make_client()

    output = {
        "source_region": source_region,
        "target_region": target_region,
        "categories": {},
    }

    # Load existing output to allow resuming
    out_path = Path(output_json_path)
    if out_path.exists():
        with open(out_path, "r", encoding="utf-8") as f:
            output = json.load(f)
        print(f"[Resume] Loaded existing output from {out_path}")

    total_entities = total_done = total_failed = 0

    for category, subcategories in source_data.get("categories", {}).items():
        for subcategory, subcat_data in subcategories.items():
            entities = subcat_data.get("entities", [])
            print(f"\n[{category} / {subcategory}] {len(entities)} entities")

            for entity in entities:
                total_entities += 1
                name = entity.get("name_en", "?")

                # Skip if already done
                existing = (
                    output.get("categories", {})
                    .get(category, {})
                    .get(subcategory, {})
                    .get(name)
                )
                if existing:
                    print(f"  [skip] '{name}' already processed")
                    total_done += 1
                    continue

                alternatives = suggest_alternatives(entity, target_region, client)

                result = {
                    "source_entity": {
                        "qid":          entity.get("qid"),
                        "name_en":      name,
                        "name_local":   entity.get("name_local"),
                        "region":       entity.get("region"),
                        "category":     entity.get("category"),
                        "subcategory":  entity.get("subcategory"),
                        "description":  entity.get("description"),
                        "wikipedia_url": entity.get("wikipedia_url"),
                        "image_url":    entity.get("images", [{}])[0].get("url", "") if entity.get("images") else "",
                    },
                    "alternatives": alternatives,
                }

                output.setdefault("categories", {}) \
                      .setdefault(category, {}) \
                      .setdefault(subcategory, {})[name] = result

                if alternatives:
                    total_done += 1
                    print(f"  [✓] '{name}' → {len(alternatives)} alternatives")
                else:
                    total_failed += 1
                    print(f"  [✗] '{name}' → no alternatives returned")

                # Save after every entity (safe progress)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(output, f, indent=2, ensure_ascii=False)

                time.sleep(delay)

    output["summary"] = {
        "total_entities": total_entities,
        "total_done": total_done,
        "total_failed": total_failed,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"TRANSCREATION COMPLETE: {source_region} → {target_region}")
    print(f"  Total entities : {total_entities}")
    print(f"  Succeeded      : {total_done}")
    print(f"  Failed         : {total_failed}")
    print(f"  Output         → {out_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="Suggest culturally equivalent transcreation alternatives using Gemini.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--source_json",   required=True,
        help="Path to source region JSON (e.g. output/morocco.json)")
    parser.add_argument("--target_region", required=True,
        help="Target culture/region (e.g. 'India', 'Japan', 'USA')")
    parser.add_argument("--output_json",   default=None,
        help="Output JSON path. Defaults to output/<source>_to_<target>_transcreation.json")
    parser.add_argument("--delay", type=float, default=1.0,
        help="Seconds to wait between Gemini calls (default: 1.0)")
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    args = parse_args()

    source_stem = Path(args.source_json).stem  # e.g. "morocco"
    target_slug = args.target_region.replace(" ", "_").lower()  # e.g. "india"

    output_json = args.output_json or f"output/{source_stem}_to_{target_slug}_transcreation.json"

    run_transcreation(
        source_json_path=args.source_json,
        target_region=args.target_region,
        output_json_path=output_json,
        delay=args.delay,
    )
