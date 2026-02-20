"""
Cultural Artifacts Extraction Pipeline
=======================================
Produces exactly 10 entries per country × subcategory slot.
Each entry is a triplet:

  wikidata         llm_replacement   final_item
  ───────────────────────────────────────────────────────────
  "Sushi"          null              "Sushi"       wikidata valid
  "Hamburger"      "Onigiri"         "Onigiri"     wikidata flagged, LLM wins
  null             "Acarajé"         "Acarajé"     pure gap-fill (wikidata empty)

Rules:
  - wikidata    = original Wikidata item name (null if slot was empty / gap-fill)
  - llm_replacement = Gemini suggestion (null if wikidata was valid)
  - final_item  = what to USE — always non-null, always one of the above two

Pipeline steps:
  1. Wikidata SPARQL — fetch up to 20 candidates ranked by sitelinks
  2. Gemini batch verification — validate each candidate, provide replacement if flagged
  3. Gemini gap-fill — fill remaining empty rows until exactly 10 entries

DEPENDENCIES:
  pip install SPARQLWrapper requests pandas google-generativeai

SETUP:
  export GEMINI_API_KEY=your_key

  
RUN:
  python cultural_artifacts_pipeline.py
  python cultural_artifacts_pipeline.py --countries Japan India
  python cultural_artifacts_pipeline.py --sample Japan
"""

import json
import time
import argparse
import os
import requests
import pandas as pd
import google.generativeai as genai
from SPARQLWrapper import SPARQLWrapper, JSON

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

COUNTRIES = {
    "Brazil":        "Q155",
    "India":         "Q668",
    "Japan":         "Q17",
    "Nigeria":       "Q1033",
    "Portugal":      "Q45",
    "Turkey":        "Q43",
    "United States": "Q30",
}

# category → subcategory → (wikidata_root_qid, concept_label, description)
TAXONOMY = {
    "Food & Drink": {
        "Food / Cuisine": (
            "Q2095", "food",
            "traditional dishes, street food, and national culinary specialties"
        ),
        "Beverages": (
            "Q40050", "drink",
            "traditional drinks including teas, juices, spirits, and non-alcoholic beverages"
        ),
        "Cooking methods / Kitchen tools": (
            "Q164747", "cooking technique",
            "traditional cooking vessels, utensils, clay pots, grills, and preparation methods"
        ),
    },
    "Clothing & Appearance": {
        "Clothing / Costume": (
            "Q11460", "clothing",
            "traditional garments, national dress, and everyday ethnic clothing"
        ),
        "Jewelry / Accessories": (
            "Q161439", "jewellery",
            "traditional jewelry, ornaments, headwear, and cultural accessories"
        ),
        "Hairstyles / Makeup": (
            "Q327496", "hairstyle",
            "traditional hairstyles, face painting, and cultural cosmetic practices"
        ),
    },
    "Shelter & Built Environment": {
        "Residential dwellings": (
            "Q3947", "house",
            "traditional and vernacular house types, huts, compounds, and domestic architecture"
        ),
        "Religious / Sacred buildings": (
            "Q1021645", "religious building",
            "temples, mosques, churches, shrines, and sacred structures"
        ),
        "Public & civic buildings": (
            "Q41176", "building",
            "famous public buildings, markets, civic plazas, and urban landmarks"
        ),
    },
    "Visual Arts & Crafts": {
        "Fine arts (painting, murals, sculpture)": (
            "Q3305213", "painting",
            "traditional visual art styles, famous paintings, murals, and sculptures"
        ),
        "Handicrafts / Decorative objects": (
            "Q2582501", "handicraft",
            "traditional crafts, pottery, basketry, woodcarving, and decorative objects"
        ),
        "Textiles / Patterns": (
            "Q28823", "textile",
            "traditional fabrics, weaving styles, printed patterns, and embroidery"
        ),
    },
    "Rituals & Ceremonies": {
        "Religious worship / Prayer": (
            "Q200538", "ritual",
            "religious practices, prayer customs, offerings, and sacred rituals"
        ),
        "Life cycle events (birth, marriage, death)": (
            "Q8445", "wedding",
            "birth ceremonies, wedding traditions, and funeral customs"
        ),
        "Seasonal / Calendar festivals": (
            "Q132241", "festival",
            "national, seasonal, and religious festivals and public celebrations"
        ),
    },
    "Performing Arts": {
        "Music / Instruments": (
            "Q34379", "musical instrument",
            "traditional instruments and nationally distinctive music genres"
        ),
        "Dance": (
            "Q11639", "dance",
            "traditional, folk, and ceremonial dance styles"
        ),
        "Theater / Storytelling": (
            "Q11635", "theatrical form",
            "traditional theater, puppetry, oral storytelling, and performance traditions"
        ),
    },
    "Daily Life & Work": {
        "Markets / Trade / Commerce": (
            "Q213441", "marketplace",
            "traditional markets, bazaars, street vendors, and trade customs"
        ),
        "Agricultural / Craft labor": (
            "Q11398", "agriculture",
            "traditional farming techniques, fishing practices, and craft production"
        ),
        "Education / Learning": (
            "Q8513", "educational institution",
            "traditional schools, madrasas, learning institutions, and academic customs"
        ),
    },
    "Sports & Games": {
        "Sporting events / Competitions": (
            "Q16510064", "sporting event",
            "famous national sports, sporting tournaments, and athletic traditions"
        ),
        "Sports clothing / Equipment": (
            "Q483110", "sport",
            "sports equipment, traditional sporting gear, and athletic clothing"
        ),
        "Fan culture / Stadiums": (
            "Q483110", "sport",
            "stadium culture, supporter traditions, fan rituals, and spectator customs"
        ),
    },
}

TARGET        = 10    # exactly 10 triplet entries per slot
SPARQL_LIMIT  = 20    # wikidata candidates to fetch (> TARGET to allow for filtering)
SLEEP_SPARQL  = 1.2   # seconds between SPARQL calls
SLEEP_GEMINI  = 2.0   # seconds between Gemini calls (free tier: 15 RPM)


# ─────────────────────────────────────────────────────────────────────────────
# TRIPLET HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def make_triplet(wikidata=None, llm_replacement=None, wikidata_qid=None,
                 wikidata_sitelinks=0, reason="") -> dict:
    """
    Build a single triplet entry. final_item is always derived automatically:
      - If wikidata is valid (no llm_replacement) → final_item = wikidata
      - If wikidata was flagged (llm_replacement set) → final_item = llm_replacement
      - If pure gap-fill (wikidata is None) → final_item = llm_replacement
    """
    final_item = wikidata if llm_replacement is None else llm_replacement
    return {
        "wikidata":         wikidata,          # str or null
        "wikidata_qid":     wikidata_qid,      # Wikidata QID or null
        "wikidata_sitelinks": wikidata_sitelinks,
        "llm_replacement":  llm_replacement,   # str or null
        "final_item":       final_item,        # always non-null
        "reason":           reason,            # Gemini's explanation
    }


# ─────────────────────────────────────────────────────────────────────────────
# GEMINI INIT
# ─────────────────────────────────────────────────────────────────────────────

def init_gemini():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY not set.\n"
            "Get a free key at: https://aistudio.google.com/app/apikey\n"
            "Then run: export GEMINI_API_KEY=your_key"
        )
    genai.configure(api_key=api_key)
    return genai.GenerativeModel("gemini-2.0-flash")


# ─────────────────────────────────────────────────────────────────────────────
# WIKIDATA SPARQL
# ─────────────────────────────────────────────────────────────────────────────

def fetch_wikidata_items(country_qid: str, concept_qid: str) -> list[dict]:
    """
    Fetch concept items for a country from Wikidata ranked by sitelink count.
    Returns list of {name, qid, sitelinks}.
    """
    query = f"""
    SELECT DISTINCT ?item ?itemLabel (COUNT(?sitelink) AS ?sitelinks) WHERE {{
      {{
        ?item wdt:P31/wdt:P279* wd:{concept_qid} .
        ?item wdt:P495 wd:{country_qid} .
      }} UNION {{
        ?item wdt:P31/wdt:P279* wd:{concept_qid} .
        ?item wdt:P17 wd:{country_qid} .
      }}
      ?sitelink schema:about ?item .
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en" . }}
    }}
    GROUP BY ?item ?itemLabel
    ORDER BY DESC(?sitelinks)
    LIMIT {SPARQL_LIMIT}
    """
    sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
    sparql.addCustomHttpHeader("User-Agent", "CulturalDatasetPipeline/1.0 (research)")
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)

    try:
        bindings = sparql.query().convert()["results"]["bindings"]
    except Exception as e:
        print(f"    [SPARQL error] {e}")
        return []

    items, seen = [], set()
    for b in bindings:
        label     = b.get("itemLabel", {}).get("value", "")
        qid       = b.get("item",      {}).get("value", "").split("/")[-1]
        sitelinks = int(b.get("sitelinks", {}).get("value", 0))
        key       = label.lower().strip()
        if label and not label.startswith("Q") and len(label) > 2 and key not in seen:
            seen.add(key)
            items.append({"name": label, "qid": qid, "sitelinks": sitelinks})

    return items


# ─────────────────────────────────────────────────────────────────────────────
# GEMINI: BATCH VERIFICATION → returns triplets
# ─────────────────────────────────────────────────────────────────────────────

def verify_and_build_triplets(
    model,
    country:     str,
    subcategory: str,
    description: str,
    wd_items:    list[dict],   # raw Wikidata candidates
) -> list[dict]:
    """
    Send all Wikidata candidates to Gemini in one batch call.
    Returns a list of triplet dicts (one per Wikidata item).
    """
    if not wd_items:
        return []

    names_json = json.dumps([i["name"] for i in wd_items], ensure_ascii=False)

    prompt = f"""You are a cultural knowledge expert auditing a dataset.

Context:
- Country: {country}
- Subcategory: "{subcategory}"
- What belongs here: {description}

Review each item and decide:
  "valid"   → genuinely fits this subcategory for {country}
  "flagged" → does NOT belong (wrong country, wrong category, too generic, nonsensical)
               → also provide a specific REPLACEMENT that does belong

Items:
{names_json}

Replacement rules:
- Must be specific to {country}
- Must fit "{subcategory}" — {description}
- Must be visually distinctive
- Must NOT duplicate any name already in the list
- Should have a Wikipedia article if possible

Return ONLY a valid JSON array, one object per item, same order as input:
  "original":     exact original name
  "verdict":      "valid" or "flagged"
  "replacement":  null if valid; replacement string if flagged
  "reason":       one short sentence, both cases reason why it's valid or why the original was flagged and how the replacement fits

Example:
[
  {{"original": "Sushi", "verdict": "valid", "replacement": null, "reason": "Classic Japanese dish."}},
  {{"original": "Hamburger", "verdict": "flagged", "replacement": "Onigiri", "reason": "American, not Japanese."}}
]"""

    try:
        raw = model.generate_content(prompt).text.strip()
        if "```" in raw:
            for part in raw.split("```"):
                part = part.strip().lstrip("json").strip()
                if part.startswith("["):
                    raw = part
                    break
        verdicts = json.loads(raw)
    except Exception as e:
        print(f"    [Gemini verify error] {e} — treating all as valid")
        verdicts = [{"original": i["name"], "verdict": "valid",
                     "replacement": None, "reason": ""} for i in wd_items]

    triplets = []
    for idx, v in enumerate(verdicts):
        wd_item     = wd_items[idx] if idx < len(wd_items) else {}
        verdict     = v.get("verdict", "valid")
        replacement = v.get("replacement")  # None if valid
        reason      = v.get("reason", "")

        if verdict == "valid":
            triplets.append(make_triplet(
                wikidata          = wd_item.get("name"),
                llm_replacement   = None,
                wikidata_qid      = wd_item.get("qid"),
                wikidata_sitelinks= wd_item.get("sitelinks", 0),
                reason            = reason,
            ))
            print(f"    ✓  {wd_item.get('name')}  →  final: {wd_item.get('name')}")
        else:
            triplets.append(make_triplet(
                wikidata          = wd_item.get("name"),
                llm_replacement   = replacement,
                wikidata_qid      = wd_item.get("qid"),
                wikidata_sitelinks= wd_item.get("sitelinks", 0),
                reason            = reason,
            ))
            print(f"    ⚑  {wd_item.get('name')}  →  final: {replacement}  ({reason})")

    return triplets


# ─────────────────────────────────────────────────────────────────────────────
# GEMINI: GAP-FILL → returns triplets with wikidata=null
# ─────────────────────────────────────────────────────────────────────────────

def gemini_gap_fill(
    model,
    country:     str,
    subcategory: str,
    description: str,
    existing_finals: list[str],  # all final_item values already in slot
    needed:      int,
) -> list[dict]:
    """
    Generate `needed` new items. Returns triplets where wikidata=null
    and llm_replacement=final_item (pure LLM entries).
    """
    prompt = f"""You are a cultural knowledge expert building a research dataset.

Task: Suggest exactly {needed} culturally distinctive items for:
- Country: {country}
- Subcategory: "{subcategory}"
- What belongs here: {description}
- Already in slot (do NOT repeat): {json.dumps(existing_finals, ensure_ascii=False)}

Requirements:
1. Specific to {country}
2. Visually distinctive (good for image generation)
3. No repeats from the already-in-slot list
4. Prefer items with a Wikipedia article
5. Authentic, avoid stereotypes

Return ONLY a valid JSON array of strings. No markdown, no explanation.
Example: ["Item One", "Item Two", "Item Three"]"""

    try:
        raw = model.generate_content(prompt).text.strip()
        if "```" in raw:
            for part in raw.split("```"):
                part = part.strip().lstrip("json").strip()
                if part.startswith("["):
                    raw = part
                    break
        items_list = json.loads(raw)
        return [
            make_triplet(
                wikidata        = None,
                llm_replacement = str(i).strip(),
                reason          = "gap-fill",
            )
            for i in items_list
            if isinstance(i, str) and len(i.strip()) > 2
        ][:needed]
    except Exception as e:
        print(f"    [Gemini gap-fill error] {e}")
        return []


# ─────────────────────────────────────────────────────────────────────────────
# WIKIPEDIA CHECK
# ─────────────────────────────────────────────────────────────────────────────

def check_wikipedia_exists(name: str) -> bool:
    try:
        resp = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params={"action": "query", "titles": name, "prop": "info", "format": "json"},
            timeout=5,
            headers={"User-Agent": "CulturalDatasetPipeline/1.0 (research)"},
        )
        return "-1" not in resp.json()["query"]["pages"]
    except Exception:
        return True  # fail open


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(
    countries_filter: list = None,
    output_json:      str  = "cultural_artifacts.json",
    output_csv:       str  = "cultural_artifacts.csv",
) -> dict:

    active_countries = {
        k: v for k, v in COUNTRIES.items()
        if countries_filter is None or k in countries_filter
    }
    total_slots = len(active_countries) * sum(len(s) for s in TAXONOMY.values())

    print(f"\nCountries:     {', '.join(active_countries.keys())}")
    print(f"Subcategories: 24 (8 categories × 3 each)")
    print(f"Total slots:   {total_slots}")
    print(f"Target:        {TARGET} triplet entries per slot")
    print("=" * 65)

    model = init_gemini()
    print("✓ Gemini 2.0 Flash initialized (free tier)\n")

    results  = {}
    stats    = {"wd_valid": 0, "wd_flagged": 0, "llm_gap": 0, "slots": 0, "full": 0}
    slot_num = 0

    for country, country_qid in active_countries.items():
        results[country] = {}
        print(f"\n{'─'*65}")
        print(f"  {country}")
        print(f"{'─'*65}")

        for category, subcategories in TAXONOMY.items():
            results[country][category] = {}

            for subcategory, (concept_qid, _, description) in subcategories.items():
                slot_num += 1
                print(f"\n[{slot_num:3}/{total_slots}] {subcategory}")

                # ── Step 1: Wikidata SPARQL ───────────────────────────────────
                wd_items = fetch_wikidata_items(country_qid, concept_qid)
                time.sleep(SLEEP_SPARQL)
                print(f"    Wikidata: {len(wd_items)} candidates")

                # ── Step 2: Gemini batch verification → triplets ──────────────
                if wd_items:
                    print("    Verifying with Gemini...")
                    triplets = verify_and_build_triplets(
                        model, country, subcategory, description,
                        wd_items[:TARGET]  # cap at TARGET to avoid over-verifying
                    )
                    time.sleep(SLEEP_GEMINI)
                else:
                    triplets = []

                # ── Step 3: Gemini gap-fill until exactly TARGET triplets ──────
                needed = TARGET - len(triplets)
                if needed > 0:
                    # Pass all current final_items to avoid duplicates
                    existing_finals = [t["final_item"] for t in triplets
                                       if t["final_item"]]
                    print(f"    Gap-filling {needed} entries...")
                    fill_triplets = gemini_gap_fill(
                        model, country, subcategory, description,
                        existing_finals, needed * 2  # request extra in case some fail wiki check
                    )
                    time.sleep(SLEEP_GEMINI)

                    added = 0
                    for t in fill_triplets:
                        if added >= needed:
                            break
                        final = t["final_item"]
                        if final and check_wikipedia_exists(final):
                            triplets.append(t)
                            added += 1
                            print(f"    +  null  →  final: {final}")
                        else:
                            print(f"    ✗  '{final}' — no Wikipedia, skipped")

                # ── Trim to exactly TARGET ────────────────────────────────────
                triplets = triplets[:TARGET]

                # ── Tally ─────────────────────────────────────────────────────
                for t in triplets:
                    if   t["wikidata"] and t["llm_replacement"] is None: stats["wd_valid"]   += 1
                    elif t["wikidata"] and t["llm_replacement"]:          stats["wd_flagged"] += 1
                    else:                                                  stats["llm_gap"]    += 1

                found = len(triplets)
                print(f"    ── {found}/{TARGET} entries", end="")
                if found < TARGET:
                    print("  ⚠ sparse", end="")
                print()

                if found >= TARGET:
                    stats["full"] += 1
                stats["slots"] += 1

                results[country][category][subcategory] = triplets

    # ── Build output ──────────────────────────────────────────────────────────
    output = {
        "metadata": {
            "pipeline":            "wikidata + gemini-2.0-flash verify + gap-fill",
            "entries_per_slot":    TARGET,
            "countries":           list(active_countries.keys()),
            "categories":          list(TAXONOMY.keys()),
            "subcategories_total": 24,
            "total_slots":         stats["slots"],
            "full_slots":          stats["full"],
            "coverage_pct":        round(stats["full"] / max(stats["slots"], 1) * 100, 1),
            "wikidata_valid":      stats["wd_valid"],
            "wikidata_flagged":    stats["wd_flagged"],
            "llm_gapfill":         stats["llm_gap"],
        },
        "data": results,
    }

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n✓ JSON saved: {output_json}")

    _to_dataframe(output).to_csv(output_csv, index=False)
    print(f"✓ CSV saved:  {output_csv}")

    _print_summary(output)
    return output


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _to_dataframe(output: dict) -> pd.DataFrame:
    """
    Flatten to tidy CSV. Each row = one triplet.
    Columns: country, category, subcategory, rank,
             wikidata, wikidata_qid, wikidata_sitelinks,
             llm_replacement, final_item, reason
    """
    rows = []
    for country, categories in output["data"].items():
        for category, subcategories in categories.items():
            for subcategory, triplets in subcategories.items():
                for rank, t in enumerate(triplets, 1):
                    rows.append({
                        "country":             country,
                        "category":            category,
                        "subcategory":         subcategory,
                        "rank":                rank,
                        "wikidata":            t.get("wikidata")            or "",
                        "wikidata_qid":        t.get("wikidata_qid")        or "",
                        "wikidata_sitelinks":  t.get("wikidata_sitelinks",  0),
                        "llm_replacement":     t.get("llm_replacement")     or "",
                        "final_item":          t.get("final_item")          or "",
                        "reason":              t.get("reason")              or "",
                    })
    return pd.DataFrame(rows)


def _print_summary(output: dict):
    m = output["metadata"]
    print(f"\n{'='*65}")
    print("  PIPELINE COMPLETE")
    print(f"{'='*65}")
    print(f"  Total slots:       {m['total_slots']}")
    print(f"  Full slots (=10):  {m['full_slots']}  ({m['coverage_pct']}% coverage)")
    print(f"  Wikidata valid:    {m['wikidata_valid']}")
    print(f"  Wikidata flagged:  {m['wikidata_flagged']}")
    print(f"  LLM gap-fill:      {m['llm_gapfill']}")
    print()
    print(f"  {'Country':<18} {'Entries':>8}  {'WD✓':>6}  {'WD⚑':>6}  {'Fill':>6}")
    print(f"  {'─'*50}")
    for country, categories in output["data"].items():
        entries = wd_v = wd_f = gap = 0
        for subcats in categories.values():
            for triplets in subcats.values():
                for t in triplets:
                    entries += 1
                    if   t["wikidata"] and t["llm_replacement"] is None: wd_v += 1
                    elif t["wikidata"] and t["llm_replacement"]:          wd_f += 1
                    else:                                                  gap  += 1
        print(f"  {country:<18} {entries:>8}  {wd_v:>6}  {wd_f:>6}  {gap:>6}")
    print()


def print_sample(output: dict, country: str):
    print(f"\nSample — {country}:")
    for category, subcategories in output["data"].get(country, {}).items():
        print(f"\n  [{category}]")
        for subcat, triplets in subcategories.items():
            print(f"    {subcat}  ({len(triplets)} entries)")
            for t in triplets[:4]:
                wd   = t["wikidata"]        or "(null)"
                llm  = t["llm_replacement"] or "(null)"
                fin  = t["final_item"]
                flag = "⚑" if t["wikidata"] and t["llm_replacement"] else \
                       "+" if not t["wikidata"] else "✓"
                print(f"      {flag}  wikidata={wd:<25} llm={llm:<25} → {fin}")
            if len(triplets) > 4:
                print(f"      ... +{len(triplets)-4} more")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Cultural Artifacts — Wikidata + Gemini Flash triplet pipeline"
    )
    parser.add_argument(
        "--countries", nargs="+", choices=list(COUNTRIES.keys()), default=None,
        help="Countries to process (default: all 7)"
    )
    parser.add_argument("--output-json", default="cultural_artifacts.json")
    parser.add_argument("--output-csv",  default="cultural_artifacts.csv")
    parser.add_argument(
        "--sample", nargs="?", const="Japan", metavar="COUNTRY",
        help="Print sample output for a country after run"
    )
    args = parser.parse_args()

    print("Cultural Artifacts Extraction Pipeline")
    print("=" * 65)

    output = run_pipeline(
        countries_filter=args.countries,
        output_json=args.output_json,
        output_csv=args.output_csv,
    )

    if args.sample:
        c = args.sample if args.sample in output["data"] else list(output["data"].keys())[0]
        print_sample(output, c)