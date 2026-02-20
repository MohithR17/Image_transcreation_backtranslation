"""
Quick test: fetch Wikimedia image for the first entry in cultural_artifacts.json
Run: python test_fetch.py
"""

import json
import io
import requests
from PIL import Image

USER_AGENT = "CulturalDatasetPipeline/1.0 (research)"

# ── Load first entry ──────────────────────────────────────────────────────────
with open("cultural_artifacts.json", "r") as f:
    data = json.load(f)["data"]

country   = next(iter(data))
category  = next(iter(data[country]))
subcategory = next(iter(data[country][category]))
triplet   = data[country][category][subcategory][0]
item_name = triplet["final_item"]

print(f"Country:     {country}")
print(f"Category:    {category}")
print(f"Subcategory: {subcategory}")
print(f"Item:        {item_name}")
print()

# ── Fetch image URL from Wikipedia ───────────────────────────────────────────
print(f"Fetching from Wikipedia...")
resp = requests.get(
    "https://en.wikipedia.org/w/api.php",
    params={
        "action":    "query",
        "titles":    item_name,
        "prop":      "pageimages",
        "piprop":    "original",
        "pilicense": "any",
        "format":    "json",
    },
    headers={"User-Agent": USER_AGENT},
    timeout=10,
)

pages = resp.json()["query"]["pages"]
page  = next(iter(pages.values()))

if "original" not in page:
    print(f"No image found on Wikipedia for '{item_name}'")
    exit(1)

url = page["original"]["source"]
print(f"Image URL:   {url}")

# ── Download & save ───────────────────────────────────────────────────────────
print(f"Downloading...")
img_resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=10)
img      = Image.open(io.BytesIO(img_resp.content)).convert("RGB")

filename = f"test_{item_name.replace(' ', '_')}.jpg"
img.save(filename)

print(f"Saved:       {filename}  ({img.width}×{img.height}px)")