from flask import Flask, render_template, request, jsonify, send_file, abort
from pathlib import Path
import json
import mimetypes
import os

# App root
app = Flask(__name__, template_folder="templates", static_folder="static")

WORKSPACE_ROOT = Path(__file__).parent.parent / "transcreation_prompt_selection"
EXPERIMENT_DIR = WORKSPACE_ROOT / "experiment_outputs"

PROMPT_VARIANTS = ["baseline", "balanced_realism", "realism_focused", "structure_preserved"]

# Build a flat index of entities by lowercase name for quick lookup
ENTITY_INDEX = {}
SOURCE_REGION = "Unknown"
TARGET_REGION = "Unknown"

def load_data():
    global ENTITY_INDEX, SOURCE_REGION, TARGET_REGION
    ENTITY_INDEX.clear()
    
    # We will use the baseline file to build the initial structure, then enrich with all variants
    for variant in PROMPT_VARIANTS:
        json_path = EXPERIMENT_DIR / variant / "evaluated.json"
        # fallback if evaluated is not there yet but generated is
        if not json_path.exists():
            json_path = EXPERIMENT_DIR / variant / "generated.json"
        
        if not json_path.exists():
            continue
            
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        SOURCE_REGION = data.get("source_region", SOURCE_REGION)
        TARGET_REGION = data.get("target_region", TARGET_REGION)
            
        categories = data.get("categories", {})
        for category, subcats in categories.items():
            for subcat, entities in subcats.items():
                for entity_name, entity_record in entities.items():
                    key = entity_name.strip().lower()
                    
                    if key not in ENTITY_INDEX:
                        # Initialize entity record
                        ENTITY_INDEX[key] = {
                            "category": category,
                            "subcategory": subcat,
                            "name": entity_name,
                            "source_entity": entity_record.get("source_entity", {}),
                            "alternatives": []
                        }
                        # Pre-populate empty alternatives
                        for alt in entity_record.get("alternatives", []):
                            alt_base = {
                                "axis": alt.get("axis"),
                                "target_item": alt.get("target_item"),
                                "_alt_index": alt.get("_alt_index"),
                                "reason": alt.get("reason"),
                                "scene_adjustments": alt.get("scene_adjustments"),
                                "variants": {}
                            }
                            ENTITY_INDEX[key]["alternatives"].append(alt_base)
                            
                    # Inject variant data for each alternative
                    for alt in entity_record.get("alternatives", []):
                        target_item = alt.get("target_item")
                        # Find the pre-populated alternative by target item instead of index
                        # to prevent mix-ups if the order differs between variant JSON files
                        for rec in ENTITY_INDEX[key]["alternatives"]:
                            if rec.get("target_item") == target_item:
                                rec["variants"][variant] = {
                                    "generated_image_path": alt.get("generated_image_path"),
                                    "generation_prompt": alt.get("generation_prompt"),
                                    "eval_metrics": alt.get("eval_metrics", {})
                                }
                                break

load_data()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/names")
def api_names():
    # return list of available source entity names (for autocomplete)
    names = sorted([v["name"] for v in ENTITY_INDEX.values()])
    return jsonify(names)

@app.route("/api/entity")
def api_entity():
    # Reload data simply to get latest if generating in background
    load_data()
    name = request.args.get("name", "").strip().lower()
    if not name:
        return jsonify({"error": "missing name"}), 400

    item = ENTITY_INDEX.get(name)
    if not item:
        # try fuzzy substring match: return best match if found
        for k, v in ENTITY_INDEX.items():
            if name in k:
                item = v
                break

    if not item:
        return jsonify({"error": "entity not found"}), 404

    return jsonify({
        "category": item["category"],
        "subcategory": item["subcategory"],
        "name": item["name"],
        "data": item,
        "source_region": SOURCE_REGION,
        "target_region": TARGET_REGION,
    })

@app.route("/image")
def serve_image():
    """Serve a local image file path safely.
    Query param: path=/absolute/or/relative/path
    """
    raw = request.args.get("path", "")
    if not raw:
        return jsonify({"error": "missing path"}), 400

    p = Path(raw)
    if not p.is_absolute():
        p = (WORKSPACE_ROOT / raw).resolve()
    else:
        p = p.resolve()

    try:
        p_relative = p.relative_to(WORKSPACE_ROOT)
    except Exception:
        return jsonify({"error": f"file outside workspace not allowed: {raw}"}), 403

    if not p.exists() or not p.is_file():
        return jsonify({"error": "file not found", "path": str(p)}), 404

    mime_type, _ = mimetypes.guess_type(str(p))
    return send_file(str(p), mimetype=mime_type or "application/octet-stream")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="127.0.0.1", port=port, debug=True)
