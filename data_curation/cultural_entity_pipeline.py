"""
Cultural Entity Image Scraping Pipeline
========================================
Fetches culturally-specific entities and their images for a given region and category.

Pipeline:
  1. Gemini 2.0 Flash + Google Search → discover real, web-grounded entity names
  2. Wikidata search API               → fuzzy match name → get QID
  3. Wikidata entity API               → verify region link + get P18 image filename
  4. Wikimedia imageinfo API           → resolve filename → proper download URL
  5. Wikipedia pageimages API          → fallback if no Wikidata image
  6. Download images to disk

Usage:
    export LITELLM_API_KEY=sk-XXX      # NeuLab LiteLLM proxy key

    # Single subcategory
    python cultural_entity_pipeline.py --region India --category "Food & Drink" --subcategory "Food / Cuisine"

    # All categories for a region
    python cultural_entity_pipeline.py --region India
"""

import argparse
import json
import logging
import os
import re
import time
import requests
from google import genai
from google.genai import types
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

GEMINI_MODEL        = "gemini/gemini-2.0-flash"    # Gemini via NeuLab LiteLLM proxy
LLM_CANDIDATES      = 20                           # how many entities to ask for (we keep top 10)

WIKIDATA_SEARCH_URL = "https://www.wikidata.org/w/api.php"
WIKIDATA_ENTITY_URL = "https://www.wikidata.org/w/api.php"
WIKIPEDIA_API_URL   = "https://en.wikipedia.org/w/api.php"
WIKIMEDIA_API_URL   = "https://commons.wikimedia.org/w/api.php"

ENTITIES_PER_CATEGORY = 10                         # final entity count per subcategory

HEADERS = {
    "User-Agent": "CulturalEntityPipeline/1.0 (research; contact@example.com)"
}

# ─────────────────────────────────────────────
# LOGGING SETUP
# ─────────────────────────────────────────────

def setup_logger(output_dir: str) -> logging.Logger:
    """
    Sets up a logger that writes to both console and a file.
    LLM prompts/responses are logged to <output_dir>/llm_calls.log
    so every retry is fully reproducible and auditable.
    Safe to call multiple times — won't add duplicate handlers.
    """
    log_dir = Path(output_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "llm_calls.log"

    logger = logging.getLogger("pipeline")
    logger.setLevel(logging.DEBUG)

    # Avoid adding duplicate handlers on repeated calls (e.g. in batch mode)
    if logger.handlers:
        return logger

    # File handler — full detail including prompts
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))

    # Console handler — INFO and above only (no prompt spam in terminal)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(message)s"))

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


# Module-level logger (initialised properly in run_pipeline/run_batch)
logger = logging.getLogger("pipeline")

# ─────────────────────────────────────────────
# REGION → WIKIDATA QID (for verification)
# ─────────────────────────────────────────────

REGION_QID_MAP = {
    "India":         "Q668",
    "Japan":         "Q17",
    "China":         "Q148",
    "Brazil":        "Q155",
    "Mexico":        "Q96",
    "Nigeria":       "Q1033",
    "Egypt":         "Q79",
    "France":        "Q142",
    "Germany":       "Q183",
    "USA":           "Q30",
    "South Korea":   "Q884",
    "Indonesia":     "Q252",
    "Ethiopia":      "Q115",
    "Iran":          "Q794",
    "Turkey":        "Q43",
    "Morocco":       "Q1028",
    "Peru":          "Q419",
    "Thailand":      "Q869",
    "Vietnam":       "Q881",
    "Ghana":         "Q117",
}

# Wikidata properties used for region verification (any one is sufficient)
REGION_PROPERTIES = [
    "P495",   # country of origin
    "P17",    # country
    "P131",   # located in administrative entity
    "P27",    # country of citizenship
    "P276",   # location
    "P19",    # place of birth
    "P2341",  # indigenous to
    "P921",   # main subject
    "P366",   # has use
    "P1532",  # country for sport
]

# ─────────────────────────────────────────────
# DATA STRUCTURES
# ─────────────────────────────────────────────

@dataclass
class EntityImage:
    url: str
    source: str        # "wikidata", "wikipedia"
    license: str = ""
    width: int = 0
    height: int = 0
    filename: str = ""
    local_path: str = ""

@dataclass
class CulturalEntity:
    qid: str
    name_en: str
    name_local: str
    region: str
    category: str
    subcategory: str
    wikidata_url: str
    wikipedia_url: str = ""
    description: str = ""
    verified: bool = False              # True if verified by Wikidata OR Wikipedia categories
    verification_method: str = "none"   # "wikidata", "wikipedia_category", "both", "none"
    images: list[EntityImage] = field(default_factory=list)


# ─────────────────────────────────────────────
# STEP 0+1: GEMINI — grounded entity discovery via Google Search
# ─────────────────────────────────────────────

def discover_entities_via_gemini(
    region: str,
    category: str,
    subcategory: str,
    n: int = LLM_CANDIDATES,
    failed_entities: Optional[list[dict]] = None,
    already_found: Optional[list[str]] = None,
) -> list[dict]:
    """
    Use Gemini + Google Search grounding (native google-genai SDK) to discover
    real, verifiable cultural entities.

    Google Search grounding means Gemini actually searches the web and returns
    entities it found on Wikipedia/Wikidata — not hallucinated from training data.
    Uses LITELLM_API_KEY via NeuLab proxy — no separate Gemini key needed.

    Returns list of {name_en, name_local, description} dicts.
    Raises ValueError on invalid JSON, RuntimeError on API failure.
    """
    api_key = os.environ.get("LITELLM_API_KEY")
    if not api_key:
        raise EnvironmentError("LITELLM_API_KEY environment variable not set.")

    client = genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(
            base_url = "https://cmu.litellm.ai",
            headers  = {"Authorization": f"Bearer {api_key}"},
        )
    )

    grounding_tool = types.Tool(google_search=types.GoogleSearch())
    config = types.GenerateContentConfig(
        tools       = [grounding_tool],
        temperature = 0.7 if failed_entities else 0.3,
    )

    is_retry = bool(failed_entities)

    if is_retry:
        failed_lines  = "\n".join(f'  - "{f["name"]}": {f["reason"]}' for f in failed_entities)
        already_lines = ", ".join(f'"{name}"' for name in (already_found or []))
        prompt = f"""You are building a cultural dataset of {region} entities.

We already have these verified entities for {subcategory}:
{already_lines}

The following {len(failed_entities)} entities failed verification — please suggest {n} REPLACEMENT entities different from all the above:

Failed entities and reasons:
{failed_lines}

Search the web and return {n} real {region} {subcategory} entities that:
- Have their own Wikipedia article in English
- Are visually distinctive and commonly photographed
- Are NOT any of the already-found or failed entities above
- Use the exact Wikipedia article title as name_en (no parentheticals, no compound names)

Respond ONLY with a valid JSON array, no markdown fences:
[
  {{
    "name_en": "Exact Wikipedia article title",
    "name_local": "Name in local language/script",
    "description": "One sentence description"
  }}
]"""
    else:
        prompt = f"""Search the web and find {n} real, well-documented {region} cultural entities in the subcategory: {subcategory} (part of {category}).

Requirements:
- Each entity must have its own dedicated Wikipedia article in English
- Must be visually distinctive and commonly depicted in photographs
- Must be traditional or iconic to {region} specifically
- Include a mix of well-known and lesser-known entities
- "name_en" must be the EXACT Wikipedia article title — search Wikipedia to verify
- Do NOT use parentheticals, alternate names, or compound names in name_en
  Good: "Atay", "Bissara", "Mahia"
  Bad: "Atay (Moroccan tea)", "Bissara soup", "Mahia fig brandy"

Respond ONLY with a valid JSON array of exactly {n} items, no markdown fences:
[
  {{
    "name_en": "Exact Wikipedia article title",
    "name_local": "Name in local language/script",
    "description": "One sentence description"
  }}
]"""

    print(f"[Gemini+Search] {'Retry' if is_retry else 'Discovering'}: {region} | {subcategory}"
          + (f" (need {n} replacements)" if is_retry else ""))

    logger.debug(
        "\n%s\n[GEMINI CALL] region=%s | subcategory=%s | is_retry=%s | n=%s\n"
        "--- PROMPT ---\n%s\n--- END PROMPT ---",
        "=" * 70, region, subcategory, is_retry, n, prompt
    )

    try:
        response = client.models.generate_content(
            model    = GEMINI_MODEL,
            contents = prompt,
            config   = config,
        )

        print(f"Search Query: {response.candidates[0].grounding_metadata.web_search_queries}")
        content = response.text.strip()

        logger.debug("--- GEMINI RAW RESPONSE ---\n%s\n--- END ---", content)

        # Strip markdown fences if present
        # Find the first JSON array
        match = re.search(r"\[\s*\{.*?\}\s*\]", content, re.DOTALL)
        if not match:
            # No array? Try to find any JSON-like block
            match = re.search(r"\{.*?\}", content, re.DOTALL)

        if not match:
            raise ValueError(
                f"Could not extract JSON from Gemini response — raw text:\n{content}"
            )

        content = match.group(0).strip()

        logger.debug("Extracted JSON content: %s", content)

        suggestions = json.loads(content)
        logger.info("  → %d entities suggested by Gemini", len(suggestions))
        return suggestions

    except json.JSONDecodeError as e:
        logger.error("Invalid JSON from Gemini: %s\nRaw: %s", e, content if 'content' in dir() else "N/A")
        raise ValueError(f"Invalid JSON from Gemini: {e}") from e
    except Exception as e:
        logger.error("Gemini API error: %s", e)
        raise RuntimeError(f"Gemini API error: {e}") from e


# ─────────────────────────────────────────────
# STEP 2: WIKIDATA SEARCH — fuzzy match → QID
# ─────────────────────────────────────────────

def search_wikidata(name: str) -> Optional[str]:
    """
    Search Wikidata for an entity by name. Returns best-match QID or None.
    Uses wbsearchentities — expects a clean, simple name (e.g. "Pani Puri" not "Pani Puri (Golgappa)").
    """
    params = {
        "action":   "wbsearchentities",
        "search":   name,
        "language": "en",
        "limit":    5,
        "format":   "json",
    }
    try:
        resp = requests.get(WIKIDATA_SEARCH_URL, params=params, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        results = resp.json().get("search", [])
        if results:
            return results[0]["id"]
    except Exception as e:
        print(f"  [!] Wikidata search failed for '{name}': {e}")
    return None


# ─────────────────────────────────────────────
# STEP 3: WIKIDATA ENTITY — verify + get image filename
# ─────────────────────────────────────────────

def fetch_wikidata_entity(qid: str, region_qid: str) -> Optional[dict]:
    """
    Fetch full entity data from Wikidata. Returns dict with:
      - verified: bool (does entity have any property linking to the region?)
      - image_filename: str or None
      - wikipedia_url: str or None
      - description: str
    """
    params = {
        "action": "wbgetentities",
        "ids":    qid,
        "props":  "claims|sitelinks|descriptions",
        "format": "json",
    }
    try:
        resp = requests.get(WIKIDATA_ENTITY_URL, params=params, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        entity_data = resp.json().get("entities", {}).get(qid, {})
    except Exception as e:
        print(f"  [!] Wikidata entity fetch failed for {qid}: {e}")
        return None

    claims = entity_data.get("claims", {})

    # ── Verify region link via any of the region properties
    verified = False
    for prop in REGION_PROPERTIES:
        for v in claims.get(prop, []):
            try:
                if v["mainsnak"]["datavalue"]["value"]["id"] == region_qid:
                    verified = True
                    break
            except (KeyError, TypeError):
                continue
        if verified:
            break

    # ── Extract P18 image filename
    image_filename = None
    p18 = claims.get("P18", [])
    if p18:
        try:
            image_filename = p18[0]["mainsnak"]["datavalue"]["value"]
        except (KeyError, TypeError):
            pass

    # ── Extract Wikipedia URL from sitelinks
    wikipedia_url = ""
    en_wiki = entity_data.get("sitelinks", {}).get("enwiki", {})
    if en_wiki:
        title = en_wiki.get("title", "").replace(" ", "_")
        wikipedia_url = f"https://en.wikipedia.org/wiki/{title}"

    # ── Extract English description
    description = entity_data.get("descriptions", {}).get("en", {}).get("value", "")

    return {
        "verified":       verified,
        "image_filename": image_filename,
        "wikipedia_url":  wikipedia_url,
        "description":    description,
    }


# ─────────────────────────────────────────────
# STEP 3b: WIKIPEDIA CATEGORY — verify region link
# ─────────────────────────────────────────────

def verify_via_wikipedia_categories(wikipedia_url: str, region: str) -> bool:
    """
    Check if a Wikipedia article's categories mention the region.
    e.g. "Lassi" → categories include "Indian beverages" → verified for India.

    This is Option C's second verification path, complementing Wikidata property checks.
    Source is human-curated Wikipedia categories, citable in a paper.
    """
    if not wikipedia_url:
        return False

    title = wikipedia_url.split("/wiki/")[-1]
    params = {
        "action":  "query",
        "titles":  title,
        "prop":    "categories",
        "cllimit": 50,
        "format":  "json",
    }
    try:
        resp = requests.get(WIKIPEDIA_API_URL, params=params, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        pages = resp.json().get("query", {}).get("pages", {})
        for page in pages.values():
            cats = [c["title"].lower() for c in page.get("categories", [])]
            region_lower = region.lower()
            # Check if any category mentions the region name
            if any(region_lower in cat for cat in cats):
                return True
    except Exception as e:
        print(f"  [!] Wikipedia category check failed: {e}")

    return False


# ─────────────────────────────────────────────
# ─────────────────────────────────────────────

def resolve_wikimedia_image_url(filename: str) -> Optional[dict]:
    """
    Convert a Wikimedia Commons filename to a proper download URL via imageinfo API.
    """
    clean = filename.replace("File:", "").replace("file:", "").strip()
    params = {
        "action":  "query",
        "titles":  f"File:{clean}",
        "prop":    "imageinfo",
        "iiprop":  "url|size|extmetadata",
        "format":  "json",
    }
    try:
        resp = requests.get(WIKIMEDIA_API_URL, params=params, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        pages = resp.json().get("query", {}).get("pages", {})
        for page in pages.values():
            info = page.get("imageinfo", [{}])[0]
            url  = info.get("url", "")
            if url:
                meta        = info.get("extmetadata", {})
                license_str = meta.get("LicenseShortName", {}).get("value", "unknown")
                return {
                    "url":     url,
                    "license": license_str,
                    "width":   info.get("width", 0),
                    "height":  info.get("height", 0),
                }
    except Exception as e:
        print(f"  [!] Wikimedia URL resolution failed for '{filename}': {e}")
    return None


# ─────────────────────────────────────────────
# STEP 5: WIKIPEDIA — fallback image
# ─────────────────────────────────────────────

def fetch_wikipedia_image(wikipedia_url: str) -> Optional[str]:
    """
    Fetch full-resolution image URL from a Wikipedia article.
    Uses piprop=original (as in the reference script) which returns the
    original upload URL directly — no redirect, no 403s.
    Falls back to Wikimedia Commons search if Wikipedia page has no image.
    """
    if not wikipedia_url:
        return None

    title = wikipedia_url.split("/wiki/")[-1]

    # ── Try Wikipedia article original image first (piprop=original)
    try:
        resp = requests.get(
            WIKIPEDIA_API_URL,
            params={
                "action":    "query",
                "titles":    title,
                "prop":      "pageimages",
                "piprop":    "original",   # full-res original, not thumbnail
                "pilicense": "any",
                "format":    "json",
            },
            headers=HEADERS,
            timeout=15,
        )
        resp.raise_for_status()
        pages = resp.json().get("query", {}).get("pages", {})
        for page in pages.values():
            original = page.get("original", {})
            url = original.get("source", "")
            if url:
                return url
    except Exception as e:
        print(f"  [!] Wikipedia image fetch failed for '{title}': {e}")

    # ── Fallback: Wikimedia Commons search by name
    try:
        name = title.replace("_", " ")
        resp = requests.get(
            WIKIMEDIA_API_URL,
            params={
                "action":      "query",
                "list":        "search",
                "srsearch":    f"{name} filetype:bitmap",
                "srnamespace": 6,      # File namespace
                "srlimit":     3,
                "format":      "json",
            },
            headers=HEADERS,
            timeout=15,
        )
        resp.raise_for_status()
        results = resp.json().get("query", {}).get("search", [])
        if results:
            file_title = results[0]["title"]
            info_resp  = requests.get(
                WIKIMEDIA_API_URL,
                params={
                    "action":  "query",
                    "titles":  file_title,
                    "prop":    "imageinfo",
                    "iiprop":  "url|size",
                    "format":  "json",
                },
                headers=HEADERS,
                timeout=15,
            )
            info_resp.raise_for_status()
            pages = info_resp.json().get("query", {}).get("pages", {})
            for page in pages.values():
                info = page.get("imageinfo", [{}])[0]
                url  = info.get("url", "")
                if url:
                    return url
    except Exception as e:
        print(f"  [!] Wikimedia Commons search failed for '{title}': {e}")

    return None


# ─────────────────────────────────────────────
# STEP 6: IMAGE DOWNLOAD
# ─────────────────────────────────────────────

def download_image(url: str, save_path: Path) -> bool:
    """
    Download an image to disk. Returns True on success.
    Wikimedia requires a descriptive User-Agent and Referer header.
    NOTE: Download failures are soft warnings — the URL is still stored in JSON.
    If running behind a proxy that blocks upload.wikimedia.org, run the
    download step locally using the URLs saved in the JSON output.
    """
    headers = {
        "User-Agent": "CulturalDatasetPipeline/1.0 (research; contact: your@email.com)",
        "Referer":    "https://commons.wikimedia.org/",
        "Accept":     "image/webp,image/apng,image/*,*/*;q=0.8",
    }
    try:
        resp = requests.get(url, headers=headers, timeout=30, stream=True)
        resp.raise_for_status()
        with open(save_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
        return True
    except Exception as e:
        logger.warning("Download failed (URL still saved in JSON): %s — %s", url, e)
        return False


# ─────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────

def _process_suggestions(
    suggestions: list[dict],
    region: str,
    category: str,
    subcategory: str,
    region_qid: str,
    seen_names: set,
) -> tuple[list[CulturalEntity], list[dict]]:
    """
    Process a batch of LLM suggestions through Wikidata search → verify → image resolution.

    Returns:
        verified: list of verified CulturalEntity objects
        failed:   list of {name, reason} dicts for entities that failed (used as retry feedback)
    """
    verified = []
    failed   = []

    for suggestion in suggestions:
        name_en     = suggestion.get("name_en", "").strip()
        name_local  = suggestion.get("name_local", "").strip()
        description = suggestion.get("description", "").strip()

        if not name_en or name_en.lower() in seen_names:
            continue
        seen_names.add(name_en.lower())

        # Search Wikidata
        qid = search_wikidata(name_en)
        time.sleep(0.3)

        if not qid:
            print(f"  [Wikidata] '{name_en}': not found")
            failed.append({"name": name_en, "reason": "not found on Wikidata — name may not match Wikipedia exactly"})
            continue

        # Fetch entity + verify region
        entity_data = fetch_wikidata_entity(qid, region_qid)
        time.sleep(0.3)

        if not entity_data:
            failed.append({"name": name_en, "reason": "could not fetch entity data from Wikidata"})
            continue

        entity = CulturalEntity(
            qid                 = qid,
            name_en             = name_en,
            name_local          = name_local,
            region              = region,
            category            = category,
            subcategory         = subcategory,
            wikidata_url        = f"https://www.wikidata.org/wiki/{qid}",
            wikipedia_url       = entity_data["wikipedia_url"],
            description         = entity_data["description"] or description,
            verified            = False,
            verification_method = "none",
        )

        # Option C: Wikidata properties OR Wikipedia categories
        wikidata_verified  = entity_data["verified"]
        wikipedia_verified = False

        if not wikidata_verified and entity_data["wikipedia_url"]:
            wikipedia_verified = verify_via_wikipedia_categories(entity_data["wikipedia_url"], region)
            time.sleep(0.3)

        if wikidata_verified and wikipedia_verified:
            entity.verified = True
            entity.verification_method = "both"
        elif wikidata_verified:
            entity.verified = True
            entity.verification_method = "wikidata"
        elif wikipedia_verified:
            entity.verified = True
            entity.verification_method = "wikipedia_category"
        else:
            failed.append({
                "name":   name_en,
                "reason": f"found on Wikidata ({qid}) but could not verify link to {region} "
                          f"via Wikidata properties or Wikipedia categories"
            })

        # Resolve Wikimedia image URL
        img_filename = entity_data.get("image_filename")
        if img_filename:
            img_info = resolve_wikimedia_image_url(img_filename)
            if img_info:
                entity.images.append(EntityImage(
                    url      = img_info["url"],
                    source   = "wikidata",
                    license  = img_info["license"],
                    width    = img_info["width"],
                    height   = img_info["height"],
                    filename = img_filename,
                ))
            time.sleep(0.2)

        # Wikipedia image fallback
        if not entity.images and entity.wikipedia_url:
            wp_url = fetch_wikipedia_image(entity.wikipedia_url)
            if wp_url:
                entity.images.append(EntityImage(url=wp_url, source="wikipedia", license="unknown"))
                print(f"  [Wikipedia] '{name_en}': fallback image found")
            time.sleep(0.3)

        # Hard failure if no image URL found at all — image is the primary deliverable
        if not entity.images:
            print(f"  [✗ no image URL] '{name_en}' ({qid}) — skipping")
            failed.append({
                "name":   name_en,
                "reason": "verified but no image URL found on Wikidata (P18), Wikipedia, or Wikimedia Commons — suggest a more visually documented entity"
            })
            continue

        status = f"✓ {entity.verification_method}" if entity.verified else "~ unverified"
        print(f"  [{status}] {name_en} ({qid}) | image_url={'yes' if entity.images else 'no'}")

        if entity.verified:
            verified.append(entity)

    return verified, failed


def run_pipeline(
    region: str,
    category: str,
    subcategory: str,
    output_dir: str = "output",
    download_images: bool = True,
    max_retries: int = 3,
) -> list[CulturalEntity]:
    """
    Full pipeline for one region + subcategory, with targeted LLM self-retry.

    Flow:
      1. Ask GPT-5 for LLM_CANDIDATES entities
      2. Process all through Wikidata search + Option C verification
      3. If < ENTITIES_PER_CATEGORY verified, retry up to max_retries times:
           - Send only the failed entity names + exact reasons back to GPT-5
           - Ask for replacements only for the missing slots
           - Already-verified entities are preserved across retries
      4. Raise ValueError if still insufficient after all retries

    Args:
        region:          Country/region name (must be in REGION_QID_MAP)
        category:        Top-level category string
        subcategory:     Subcategory string
        output_dir:      Root output directory
        download_images: Whether to download images to disk
        max_retries:     Max retry attempts if not enough verified (default 3)
    """

    print(f"\n{'='*60}")
    print(f"Pipeline: {region} | {category} | {subcategory}")
    print(f"{'='*60}")

    region_qid = REGION_QID_MAP.get(region)
    if not region_qid:
        raise ValueError(f"Region '{region}' not in REGION_QID_MAP. Available: {list(REGION_QID_MAP.keys())}")

    entities   = []    # accumulates verified entities across all attempts
    seen_names = set()

    # ── Initial Gemini+Search entity discovery
    suggestions = discover_entities_via_gemini(region, category, subcategory, n=LLM_CANDIDATES)

    for attempt in range(max_retries + 1):
        new_verified, failed = _process_suggestions(
            suggestions, region, category, subcategory, region_qid, seen_names
        )
        entities.extend(new_verified)
        entities = entities[:ENTITIES_PER_CATEGORY]

        if len(entities) >= ENTITIES_PER_CATEGORY:
            break

        still_needed = ENTITIES_PER_CATEGORY - len(entities)

        if attempt == max_retries:
            break  # exhausted retries

        if not failed:
            print(f"  [!] Only {len(entities)} verified but no failures to retry on — stopping")
            break

        # ── Retry: ask Gemini to replace only as many as still needed
        print(f"\n[Retry {attempt + 1}/{max_retries}] "
              f"{len(entities)} verified so far, need {still_needed} more. "
              f"Sending {still_needed} failed entities back to Gemini.")
        logger.debug(
            "[Retry %d/%d] %d verified, need %d more. Failed entities:\n%s",
            attempt + 1, max_retries, len(entities), still_needed,
            json.dumps(failed[:still_needed], indent=2, ensure_ascii=False)
        )

        try:
            suggestions = discover_entities_via_gemini(
                region, category, subcategory,
                n               = still_needed,
                failed_entities = failed[:still_needed],
                already_found   = [e.name_en for e in entities],
            )
        except (ValueError, RuntimeError) as e:
            print(f"  [!] Gemini retry call failed: {e} — stopping retries")
            break

    if len(entities) < ENTITIES_PER_CATEGORY:
        raise ValueError(
            f"Only {len(entities)}/{ENTITIES_PER_CATEGORY} verified entities found for "
            f"{region} | {subcategory} after {max_retries} retries."
        )

    print(f"\n[✓] Total entities: {len(entities)} "
          f"(wikidata={sum(1 for e in entities if e.verification_method == 'wikidata')}, "
          f"wikipedia_category={sum(1 for e in entities if e.verification_method == 'wikipedia_category')}, "
          f"both={sum(1 for e in entities if e.verification_method == 'both')})")
    print(f"[✓] Image URLs resolved: {sum(1 for e in entities if e.images)}/{len(entities)}")
    downloaded = sum(1 for e in entities for img in e.images if img.local_path)
    print(f"[✓] Images downloaded:   {downloaded}/{len(entities)}"
          + (" (download separately using URLs in JSON if behind proxy)" if downloaded < len(entities) else ""))

    # ── Download images
    if download_images:
        safe_region = region.replace(" ", "_").lower()
        safe_cat    = subcategory.replace("/", "_").replace(" ", "_").lower()
        img_root    = Path(output_dir) / safe_region / safe_cat

        for entity in entities:
            entity_dir = img_root / entity.qid
            entity_dir.mkdir(parents=True, exist_ok=True)

            for i, img in enumerate(entity.images):
                ext       = img.url.split(".")[-1].split("?")[0][:4] or "jpg"
                fname     = f"{i+1:02d}_{img.source}.{ext}"
                save_path = entity_dir / fname

                if download_image(img.url, save_path):
                    img.local_path = str(save_path)
                    img.filename   = fname
                    print(f"  [↓] {entity.name_en} → {fname}")
                time.sleep(0.2)

    print(f"[✓] Total images collected: {sum(len(e.images) for e in entities)}")
    return entities


# ─────────────────────────────────────────────
# BATCH MODE — all subcategories for a region
# ─────────────────────────────────────────────

ALL_CATEGORIES = [
    ("Food & Drink",                 "Food / Cuisine"),
    ("Food & Drink",                 "Beverages"),
    ("Food & Drink",                 "Cooking methods / Kitchen tools"),
    ("Clothing & Appearance",        "Clothing / Costume"),
    ("Clothing & Appearance",        "Jewelry / Accessories"),
    ("Clothing & Appearance",        "Hairstyles / Makeup"),
    ("Shelter & Built Environment",  "Residential dwellings"),
    ("Shelter & Built Environment",  "Religious / Sacred buildings"),
    ("Shelter & Built Environment",  "Public & civic buildings"),
    ("Visual Arts & Crafts",         "Fine arts (painting, murals, sculpture)"),
    ("Visual Arts & Crafts",         "Handicrafts / Decorative objects"),
    ("Visual Arts & Crafts",         "Textiles / Patterns"),
    ("Rituals & Ceremonies",         "Religious worship / Prayer"),
    ("Rituals & Ceremonies",         "Life cycle events (birth, marriage, death)"),
    ("Rituals & Ceremonies",         "Seasonal / Calendar festivals"),
    ("Performing Arts",              "Music / Instruments"),
    ("Performing Arts",              "Dance"),
    ("Performing Arts",              "Theater / Storytelling"),
    ("Daily Life & Work",            "Markets / Trade / Commerce"),
    ("Daily Life & Work",            "Agricultural / Craft labor"),
    ("Daily Life & Work",            "Education / Learning"),
    ("Sports & Games",               "Sporting events / Competitions"),
    ("Sports & Games",               "Sports clothing / Equipment"),
    ("Sports & Games",               "Fan culture / Stadiums"),
]


def run_batch(
    region: str,
    output_dir: str = "output",
    download_images: bool = True,
    max_retries: int = 3,
) -> None:
    """
    Run the full pipeline for ALL subcategories for a region.
    Saves one combined JSON: output/<region>.json
    """
    combined       = {"region": region, "categories": {}}
    total_entities = total_images = 0
    failed         = []

    print(f"\nBatch mode: {region} — {len(ALL_CATEGORIES)} subcategories\n")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{region.replace(' ', '_').lower()}.json"

    logger.info("Batch mode: %s — %d subcategories", region, len(ALL_CATEGORIES))

    for category, subcategory in ALL_CATEGORIES:
        try:
            entities = run_pipeline(
                region          = region,
                category        = category,
                subcategory     = subcategory,
                output_dir      = output_dir,
                download_images = download_images,
                max_retries     = max_retries,
            )
            combined["categories"].setdefault(category, {})[subcategory] = {
                "total":    len(entities),
                "entities": [asdict(e) for e in entities],
            }
            total_entities += len(entities)
            total_images   += sum(len(e.images) for e in entities)

        except Exception as ex:
            print(f"  [!] Failed — {category} / {subcategory}: {ex}")
            failed.append({"category": category, "subcategory": subcategory, "error": str(ex)})

        # ── Save after every subcategory so progress is never lost
        combined["summary"] = {
            "total_entities": total_entities,
            "total_images":   total_images,
            "failed":         failed,
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(combined, f, indent=2, ensure_ascii=False)
        print(f"  [saved] → {json_path}")

        time.sleep(1)

    print(f"\n{'='*60}")
    print(f"BATCH COMPLETE: {region}")
    print(f"{'='*60}")
    print(f"  Total entities : {total_entities}")
    print(f"  Total images   : {total_images}")
    print(f"  Failed         : {len(failed)}")
    print(f"  Combined JSON  → {json_path}")
    if failed:
        print("\n  Failed subcategories:")
        for f_ in failed:
            print(f"    - {f_['category']} / {f_['subcategory']}: {f_['error']}")


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Cultural Entity Image Pipeline (GPT-5 + Wikidata + Wikimedia)",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--region", required=True,
        help=f"Country/region. Available: {list(REGION_QID_MAP.keys())}")
    parser.add_argument("--category", default=None,
        help='Top-level category e.g. "Food & Drink". Omit to run all categories.')
    parser.add_argument("--subcategory", default=None,
        help='Subcategory e.g. "Food / Cuisine". Required if --category is set.')
    parser.add_argument("--output_dir", default="output",
        help="Root output directory (default: ./output)")
    parser.add_argument("--no_download", action="store_true",
        help="Skip downloading images (metadata only)")
    parser.add_argument("--max_retries", type=int, default=3,
        help="Max LLM retry attempts if not enough verified entities (default: 3)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    setup_logger(args.output_dir)  # initialise once for the entire run

    if args.category is None and args.subcategory is None:
        run_batch(
            region          = args.region,
            output_dir      = args.output_dir,
            download_images = not args.no_download,
            max_retries     = args.max_retries,
        )
    elif args.category and args.subcategory:
        # Single subcategory mode
        entities = run_pipeline(
            region          = args.region,
            category        = args.category,
            subcategory     = args.subcategory,
            output_dir      = args.output_dir,
            download_images = not args.no_download,
            max_retries     = args.max_retries,
        )

        # Save/update country-level JSON even in single mode
        out_dir    = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        json_path  = out_dir / f"{args.region.replace(' ', '_').lower()}.json"

        # Load existing if present, else start fresh
        if json_path.exists():
            with open(json_path, "r", encoding="utf-8") as f:
                combined = json.load(f)
        else:
            combined = {"region": args.region, "categories": {}}

        combined["categories"].setdefault(args.category, {})[args.subcategory] = {
            "total":    len(entities),
            "entities": [asdict(e) for e in entities],
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(combined, f, indent=2, ensure_ascii=False)

        print(f"[✓] Saved → {json_path}")
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        for e in entities:
            tag = "✓" if e.verified else "~"
            print(f"  {tag} {e.qid:<12} {e.name_en:<40} images={len(e.images)}")
    else:
        print("[!] Provide both --category and --subcategory, or neither to run all.")
        exit(1)