"""
Cultural Entity Image Scraping Pipeline
========================================
Improved Image Strategy:
  1. Wikipedia lead image (primary)
  2. Wikidata P18 (secondary)
  3. Wikimedia Commons generator search (region-aware fallback)

Usage
-----
Prerequisites:
  export LITELLM_API_KEY="your_api_key"
  pip install requests google-genai

1. Run ALL subcategories for a region (batch mode — resumable):
   python cultural_entity_pipeline_new.py --region Morocco

   Output is saved incrementally to output/morocco.json.
   If the script is interrupted and re-run, already-completed subcategories
   are skipped automatically.

2. Run a SINGLE subcategory:
   python cultural_entity_pipeline_new.py \\
       --region Morocco \\
       --category "Food & Drink" \\
       --subcategory "Food / Cuisine"

3. Skip image downloads (URLs still saved in JSON):
   python cultural_entity_pipeline_new.py --region Morocco --no_download

4. Change output directory:
   python cultural_entity_pipeline_new.py --region Morocco --output_dir my_output

5. Change max Wikidata/Wikipedia verification retries (default 3):
   python cultural_entity_pipeline_new.py --region Morocco --max_retries 5

Available regions:
  India, Japan, China, Brazil, Mexico, Nigeria, Egypt, France, Germany, USA,
  South Korea, Indonesia, Ethiopia, Iran, Turkey, Morocco, Peru, Thailand,
  Vietnam, Ghana

Output format (output/<region>.json):
  {
    "region": "Morocco",
    "categories": {
      "Food & Drink": {
        "Food / Cuisine": {
          "total": 10,
          "entities": [ { "qid": ..., "name_en": ..., "images": [...], ... } ]
        }
      }
    },
    "summary": { "total_entities": ..., "total_images": ..., "failed": [...] }
  }
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

GEMINI_MODEL        = "gemini/gemini-3-flash-preview"
LLM_CANDIDATES      = 20

WIKIDATA_SEARCH_URL = "https://www.wikidata.org/w/api.php"
WIKIDATA_ENTITY_URL = "https://www.wikidata.org/w/api.php"
WIKIPEDIA_API_URL   = "https://en.wikipedia.org/w/api.php"
WIKIMEDIA_API_URL   = "https://commons.wikimedia.org/w/api.php"

ENTITIES_PER_CATEGORY = 10

HEADERS = {
    "User-Agent": "CulturalEntityPipeline/2.0 (research; contact@example.com)"
}


@dataclass
class EntityImage:
    url: str
    source: str
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
    verified: bool = False
    verification_method: str = "none"
    images: list[EntityImage] = field(default_factory=list)


def fetch_wikipedia_image(wikipedia_url: str):
    """PRIMARY image source: Wikipedia lead image (piprop=original)."""
    if not wikipedia_url:
        return None
    title = wikipedia_url.split("/wiki/")[-1]
    try:
        resp = requests.get(
            WIKIPEDIA_API_URL,
            params={"action": "query", "titles": title, "prop": "pageimages",
                    "piprop": "original", "format": "json"},
            headers=HEADERS, timeout=15,
        )
        resp.raise_for_status()
        pages = resp.json().get("query", {}).get("pages", {})
        for page in pages.values():
            url = page.get("original", {}).get("source", "")
            if url:
                return EntityImage(url=url, source="wikipedia", license="unknown")
    except Exception as e:
        print(f"[!] Wikipedia image failed: {e}")
    return None


def resolve_wikimedia_image_url(filename: str):
    """SECONDARY image source: Wikidata P18 resolution."""
    clean = filename.replace("File:", "").strip()
    try:
        resp = requests.get(
            WIKIMEDIA_API_URL,
            params={"action": "query", "titles": f"File:{clean}", "prop": "imageinfo",
                    "iiprop": "url|size|extmetadata", "format": "json"},
            headers=HEADERS, timeout=15,
        )
        resp.raise_for_status()
        pages = resp.json().get("query", {}).get("pages", {})
        for page in pages.values():
            info = page.get("imageinfo", [{}])[0]
            url  = info.get("url")
            if not url:
                continue
            meta        = info.get("extmetadata", {})
            license_str = meta.get("LicenseShortName", {}).get("value", "unknown")
            return EntityImage(url=url, source="wikidata", license=license_str,
                               width=info.get("width", 0), height=info.get("height", 0),
                               filename=clean)
    except Exception as e:
        print(f"[!] Wikidata image resolution failed: {e}")
    return None


def search_commons_image(name: str, region: str):
    """FALLBACK: Wikimedia Commons generator search (region-aware)."""
    query = f"{name} {region}"
    try:
        resp = requests.get(
            WIKIMEDIA_API_URL,
            params={"action": "query", "generator": "search", "gsrsearch": query,
                    "gsrnamespace": 6, "gsrlimit": 5, "prop": "imageinfo",
                    "iiprop": "url|size|extmetadata", "format": "json"},
            headers=HEADERS, timeout=20,
        )
        resp.raise_for_status()
        pages = resp.json().get("query", {}).get("pages", {})
        for page in pages.values():
            info = page.get("imageinfo", [{}])[0]
            url  = info.get("url")
            if not url:
                continue
            meta        = info.get("extmetadata", {})
            license_str = meta.get("LicenseShortName", {}).get("value", "unknown")
            return EntityImage(url=url, source="commons_search", license=license_str,
                               width=info.get("width", 0), height=info.get("height", 0),
                               filename=page.get("title", ""))
    except Exception as e:
        print(f"[!] Commons search failed: {e}")
    return None


def download_image(url: str, save_path) -> bool:
    """Download an image to disk. Soft failure — URL still saved in JSON."""
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


def setup_logger(output_dir: str) -> logging.Logger:
    log_dir = Path(output_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "llm_calls.log"
    logger = logging.getLogger("pipeline")
    logger.setLevel(logging.DEBUG)
    if logger.handlers:
        return logger
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


logger = logging.getLogger("pipeline")

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

REGION_PROPERTIES = [
    "P495", "P17", "P131", "P27", "P276",
    "P19", "P2341", "P921", "P366", "P1532",
]


def discover_entities_via_gemini(
    region: str,
    category: str,
    subcategory: str,
    n: int = LLM_CANDIDATES,
    failed_entities: Optional[list[dict]] = None,
    already_found: Optional[list[str]] = None,
) -> list[dict]:
    """
    Use Gemini + Google Search grounding to discover real, verifiable cultural entities.
    Uses LITELLM_API_KEY via NeuLab proxy.
    """
    api_key = os.environ.get("LITELLM_API_KEY")
    if not api_key:
        raise EnvironmentError("LITELLM_API_KEY environment variable not set.")

    client = genai.Client(
        api_key = api_key,
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

    # Shared type-constraint clause injected into both prompt branches
    type_constraint = f"""\
IMPORTANT — entity type rule:
Each entity you return must itself BE a "{subcategory}" — meaning its Wikipedia article \
is primarily ABOUT a {subcategory} item, not merely associated with or producing one.
Ask yourself: "Is this thing a specific, named instance or example of '{subcategory}'?" \
— if the answer is no, leave it out.
For example, do NOT return a place, city, region, or person simply because they are \
known for or linked to {subcategory} items — only return the items themselves.
If you cannot find {n} valid items, return fewer rather than returning wrong-type entities."""

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

{type_constraint}

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

{type_constraint}

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
    print(f"  → Waiting for Gemini API response (may take 15–30s)...")
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


        # Guard: no candidates (safety filter or empty response)
        if not response.candidates:
            raise ValueError(f"Gemini returned no candidates (possible safety filter or empty response)")

        try:
            grounding = response.candidates[0].grounding_metadata
            if grounding and grounding.web_search_queries:
                print(f"Search Query: {grounding.web_search_queries}")
        except (AttributeError, IndexError):
            pass

        content = (response.text or "").strip()
        if not content:
            raise ValueError("Gemini returned an empty response text")

        logger.debug("--- GEMINI RAW RESPONSE ---\n%s\n--- END ---", content)

        match = re.search(r"\[\s*\{.*?\}\s*\]", content, re.DOTALL)
        if not match:
            match = re.search(r"\{.*?\}", content, re.DOTALL)
        if not match:
            raise ValueError(f"Could not extract JSON from Gemini response — raw text:\n{content}")

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


def search_wikidata(name: str) -> Optional[str]:
    params = {"action": "wbsearchentities", "search": name, "language": "en",
              "limit": 5, "format": "json"}
    try:
        resp = requests.get(WIKIDATA_SEARCH_URL, params=params, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        results = resp.json().get("search", [])
        if results:
            return results[0]["id"]
    except Exception as e:
        print(f"  [!] Wikidata search failed for '{name}': {e}")
    return None


def fetch_wikidata_entity(qid: str, region_qid: str) -> Optional[dict]:
    params = {"action": "wbgetentities", "ids": qid,
              "props": "claims|sitelinks|descriptions", "format": "json"}
    try:
        resp = requests.get(WIKIDATA_ENTITY_URL, params=params, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        entity_data = resp.json().get("entities", {}).get(qid, {})
    except Exception as e:
        print(f"  [!] Wikidata entity fetch failed for {qid}: {e}")
        return None

    claims = entity_data.get("claims", {})

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

    image_filename = None
    p18 = claims.get("P18", [])
    if p18:
        try:
            image_filename = p18[0]["mainsnak"]["datavalue"]["value"]
        except (KeyError, TypeError):
            pass

    wikipedia_url = ""
    en_wiki = entity_data.get("sitelinks", {}).get("enwiki", {})
    if en_wiki:
        title = en_wiki.get("title", "").replace(" ", "_")
        wikipedia_url = f"https://en.wikipedia.org/wiki/{title}"

    description = entity_data.get("descriptions", {}).get("en", {}).get("value", "")

    return {"verified": verified, "image_filename": image_filename,
            "wikipedia_url": wikipedia_url, "description": description}


def verify_via_wikipedia_categories(wikipedia_url: str, region: str) -> bool:
    if not wikipedia_url:
        return False
    title = wikipedia_url.split("/wiki/")[-1]
    params = {"action": "query", "titles": title, "prop": "categories",
              "cllimit": 50, "format": "json"}
    try:
        resp = requests.get(WIKIPEDIA_API_URL, params=params, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        pages = resp.json().get("query", {}).get("pages", {})
        for page in pages.values():
            cats = [c["title"].lower() for c in page.get("categories", [])]
            if any(region.lower() in cat for cat in cats):
                return True
    except Exception as e:
        print(f"  [!] Wikipedia category check failed: {e}")
    return False


def _process_suggestions(
    suggestions: list[dict],
    region: str,
    category: str,
    subcategory: str,
    region_qid: str,
    seen_names: set,
) -> tuple[list[CulturalEntity], list[dict]]:

    verified = []
    failed   = []

    for suggestion in suggestions:
        name_en     = suggestion.get("name_en", "").strip()
        name_local  = suggestion.get("name_local", "").strip()
        description = suggestion.get("description", "").strip()

        if not name_en or name_en.lower() in seen_names:
            continue
        seen_names.add(name_en.lower())

        qid = search_wikidata(name_en)
        time.sleep(0.3)

        if not qid:
            print(f"  [Wikidata] '{name_en}': not found")
            failed.append({"name": name_en,
                           "reason": "not found on Wikidata — name may not match Wikipedia exactly"})
            continue

        entity_data = fetch_wikidata_entity(qid, region_qid)
        time.sleep(0.3)

        if not entity_data:
            failed.append({"name": name_en, "reason": "could not fetch entity data from Wikidata"})
            continue

        entity = CulturalEntity(
            qid=qid, name_en=name_en, name_local=name_local, region=region,
            category=category, subcategory=subcategory,
            wikidata_url=f"https://www.wikidata.org/wiki/{qid}",
            wikipedia_url=entity_data["wikipedia_url"],
            description=entity_data["description"] or description,
            verified=False, verification_method="none",
        )

        wikidata_verified  = entity_data["verified"]
        wikipedia_verified = False

        if not wikidata_verified and entity.wikipedia_url:
            wikipedia_verified = verify_via_wikipedia_categories(entity.wikipedia_url, region)
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
            failed.append({"name": name_en,
                           "reason": f"found on Wikidata ({qid}) but could not verify link to {region}"})
            continue

        # IMAGE ACQUISITION (priority order)

        # 1. Wikipedia lead image (PRIMARY)
        if entity.wikipedia_url:
            wp_img = fetch_wikipedia_image(entity.wikipedia_url)
            if wp_img:
                entity.images.append(wp_img)
                print(f"  [Image] Wikipedia lead image found")
            time.sleep(0.2)

        # 2. Wikidata P18 (SECONDARY)
        if not entity.images:
            img_filename = entity_data.get("image_filename")
            if img_filename:
                wd_img = resolve_wikimedia_image_url(img_filename)
                if wd_img:
                    entity.images.append(wd_img)
                    print(f"  [Image] Wikidata P18 used")
            time.sleep(0.2)

        # 3. Commons search fallback
        if not entity.images:
            commons_img = search_commons_image(name_en, region)
            if commons_img:
                entity.images.append(commons_img)
                print(f"  [Image] Commons search used")
            time.sleep(0.2)

        # Hard failure if still no image
        if not entity.images:
            print(f"  [✗ no image] '{name_en}' ({qid})")
            failed.append({"name": name_en,
                           "reason": "verified but no image found on Wikipedia, Wikidata, or Commons search"})
            continue

        print(f"  [✓ {entity.verification_method}] {name_en} ({qid}) | image=yes")
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

    print(f"\n{'='*60}")
    print(f"Pipeline: {region} | {category} | {subcategory}")
    print(f"{'='*60}")

    region_qid = REGION_QID_MAP.get(region)
    if not region_qid:
        raise ValueError(f"Region '{region}' not in REGION_QID_MAP.")

    entities      = []
    seen_names    = set()
    all_failed    = []  # accumulated failures across ALL rounds

    suggestions = discover_entities_via_gemini(region, category, subcategory, n=LLM_CANDIDATES)

    for attempt in range(max_retries + 1):
        new_verified, round_failed = _process_suggestions(
            suggestions, region, category, subcategory, region_qid, seen_names
        )
        entities.extend(new_verified)
        entities   = entities[:ENTITIES_PER_CATEGORY]
        all_failed.extend(round_failed)  # accumulate failures across rounds

        if len(entities) >= ENTITIES_PER_CATEGORY:
            break

        still_needed = ENTITIES_PER_CATEGORY - len(entities)

        if attempt == max_retries:
            break

        if not all_failed:
            print(f"  [!] Only {len(entities)} verified but no failures to retry on — stopping")
            break

        # Always ask for at least 5, or double still_needed — whichever is more
        n_retry = max(still_needed * 2, 5)
        print(f"\n[Retry {attempt + 1}/{max_retries}] "
              f"{len(entities)} verified so far, need {still_needed} more. "
              f"Sending {len(all_failed)} accumulated failed entities as context, asking for {n_retry} replacements.")
        logger.debug(
            "[Retry %d/%d] %d verified, need %d more. All accumulated failures:\n%s",
            attempt + 1, max_retries, len(entities), still_needed,
            json.dumps(all_failed, indent=2, ensure_ascii=False)
        )

        try:
            suggestions = discover_entities_via_gemini(
                region, category, subcategory,
                n               = n_retry,
                failed_entities = all_failed,        # ALL accumulated failures
                already_found   = [e.name_en for e in entities],
            )
        except (ValueError, RuntimeError) as e:
            print(f"  [!] Gemini retry call failed: {e} — stopping retries")
            break

    if len(entities) < ENTITIES_PER_CATEGORY:
        logger.warning(
            "Only %d/%d verified entities found for %s | %s after %d retries. Continuing with partial results.",
            len(entities), ENTITIES_PER_CATEGORY, region, subcategory, max_retries
        )
        print(f"  [!] Warning: Only {len(entities)}/{ENTITIES_PER_CATEGORY} entities found — saving partial results.")

    print(f"\n[✓] Total entities: {len(entities)} "
          f"(wikidata={sum(1 for e in entities if e.verification_method == 'wikidata')}, "
          f"wikipedia_category={sum(1 for e in entities if e.verification_method == 'wikipedia_category')}, "
          f"both={sum(1 for e in entities if e.verification_method == 'both')})")
    print(f"[✓] Image URLs resolved: {sum(1 for e in entities if e.images)}/{len(entities)}")
    downloaded = sum(1 for e in entities for img in e.images if img.local_path)
    print(f"[✓] Images downloaded:   {downloaded}/{len(entities)}"
          + (" (download separately using URLs in JSON if behind proxy)" if downloaded < len(entities) else ""))

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


def run_batch(region: str, output_dir: str = "output",
              download_images: bool = True, max_retries: int = 3) -> None:
    combined       = {"region": region, "categories": {}}
    total_entities = total_images = 0
    failed         = []

    print(f"\nBatch mode: {region} — {len(ALL_CATEGORIES)} subcategories\n")

    out_dir   = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{region.replace(' ', '_').lower()}.json"

    # Load existing output so already-completed subcategories can be skipped
    if json_path.exists():
        with open(json_path, "r", encoding="utf-8") as f:
            combined = json.load(f)
        total_entities = combined.get("summary", {}).get("total_entities", 0)
        total_images   = combined.get("summary", {}).get("total_images", 0)
        print(f"  [resume] Loaded existing output from {json_path}")
    else:
        combined = {"region": region, "categories": {}}

    logger.info("Batch mode: %s — %d subcategories", region, len(ALL_CATEGORIES))

    for category, subcategory in ALL_CATEGORIES:
        # Skip if already present in the output JSON
        existing = combined.get("categories", {}).get(category, {}).get(subcategory)
        if existing and existing.get("total", 0) > 0:
            print(f"  [skip] Already have {existing['total']} entities for {category} / {subcategory}")
            total_entities += existing["total"]
            total_images   += sum(len(e.get("images", [])) for e in existing.get("entities", []))
            continue

        try:
            entities = run_pipeline(region=region, category=category, subcategory=subcategory,
                                    output_dir=output_dir, download_images=download_images,
                                    max_retries=max_retries)
            combined["categories"].setdefault(category, {})[subcategory] = {
                "total": len(entities), "entities": [asdict(e) for e in entities]}
            total_entities += len(entities)
            total_images   += sum(len(e.images) for e in entities)
        except Exception as ex:
            print(f"  [!] Failed — {category} / {subcategory}: {ex}")
            failed.append({"category": category, "subcategory": subcategory, "error": str(ex)})

        combined["summary"] = {"total_entities": total_entities,
                               "total_images": total_images, "failed": failed}
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(combined, f, indent=2, ensure_ascii=False)
        print(f"  [saved] → {json_path}")
        time.sleep(1)

    print(f"\n{'='*60}\nBATCH COMPLETE: {region}\n{'='*60}")
    print(f"  Total entities : {total_entities}")
    print(f"  Total images   : {total_images}")
    print(f"  Failed         : {len(failed)}")
    print(f"  Combined JSON  → {json_path}")
    if failed:
        print("\n  Failed subcategories:")
        for f_ in failed:
            print(f"    - {f_['category']} / {f_['subcategory']}: {f_['error']}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Cultural Entity Image Pipeline (Gemini+Search + Wikidata + Wikimedia)",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--region", required=True,
        help=f"Country/region. Available: {list(REGION_QID_MAP.keys())}")
    parser.add_argument("--category", default=None)
    parser.add_argument("--subcategory", default=None)
    parser.add_argument("--output_dir", default="output")
    parser.add_argument("--no_download", action="store_true")
    parser.add_argument("--max_retries", type=int, default=3)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    setup_logger(args.output_dir)

    if args.category is None and args.subcategory is None:
        run_batch(region=args.region, output_dir=args.output_dir,
                  download_images=not args.no_download, max_retries=args.max_retries)
    elif args.category and args.subcategory:
        entities = run_pipeline(region=args.region, category=args.category,
                                subcategory=args.subcategory, output_dir=args.output_dir,
                                download_images=not args.no_download, max_retries=args.max_retries)

        out_dir   = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        json_path = out_dir / f"{args.region.replace(' ', '_').lower()}.json"

        if json_path.exists():
            with open(json_path, "r", encoding="utf-8") as f:
                combined = json.load(f)
        else:
            combined = {"region": args.region, "categories": {}}

        combined["categories"].setdefault(args.category, {})[args.subcategory] = {
            "total": len(entities), "entities": [asdict(e) for e in entities]}
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(combined, f, indent=2, ensure_ascii=False)

        print(f"[✓] Saved → {json_path}")
        print(f"\n{'='*60}\nSUMMARY\n{'='*60}")
        for e in entities:
            tag = "✓" if e.verified else "~"
            print(f"  {tag} {e.qid:<12} {e.name_en:<40} images={len(e.images)}")
    else:
        print("[!] Provide both --category and --subcategory, or neither to run all.")
        exit(1)