from flask import Flask, render_template, request, jsonify, send_file, abort
from pathlib import Path
import json
import mimetypes
import os

# App root
app = Flask(__name__, template_folder="templates", static_folder="static")

# Default data file (expected to be produced by the generate script)
# Location: data_curation/output/<stem>_with_images.json
DATA_FILE = Path(__file__).parent.parent / "output" / "morocco_to_india_with_images.json"
WORKSPACE_ROOT = Path(__file__).parent.parent.resolve()

# Load data at startup (will raise if missing)
if not DATA_FILE.exists():
    raise RuntimeError(f"Transcreation-with-images JSON not found: {DATA_FILE}\nRun the generation script first or point DATA_FILE to the correct path.")

with open(DATA_FILE, "r", encoding="utf-8") as f:
    DATA = json.load(f)

# Build a flat index of entities by lowercase name for quick lookup
ENTITY_INDEX = {}
for category, subcats in DATA.get("categories", {}).items():
    for subcat, entities in subcats.items():
        for entity_name, entity_record in entities.items():
            key = entity_name.strip().lower()
            ENTITY_INDEX[key] = {
                "category": category,
                "subcategory": subcat,
                "name": entity_name,
                "record": entity_record,
            }

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

    # Return source_entity + alternatives as-is
    return jsonify({
        "category": item["category"],
        "subcategory": item["subcategory"],
        "name": item["name"],
        "data": item["record"],
        "source_region": DATA.get("source_region"),
        "target_region": DATA.get("target_region"),
    })

@app.route("/image")
def serve_image():
    """Serve a local image file path safely.
    Query param: path=/absolute/or/relative/path
    Only files within the workspace root are allowed.
    """
    raw = request.args.get("path", "")
    if not raw:
        return jsonify({"error": "missing path"}), 400

    # Support paths coming from JSON (which may be relative)
    p = Path(raw)
    if not p.is_absolute():
        p = (WORKSPACE_ROOT / raw).resolve()
    else:
        p = p.resolve()

    try:
        p_relative = p.relative_to(WORKSPACE_ROOT)
    except Exception:
        return jsonify({"error": "file outside workspace not allowed"}), 403

    if not p.exists() or not p.is_file():
        return jsonify({"error": "file not found", "path": str(p)}), 404

    mime_type, _ = mimetypes.guess_type(str(p))
    return send_file(str(p), mimetype=mime_type or "application/octet-stream")

if __name__ == "__main__":
    # simple dev server
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="127.0.0.1", port=port, debug=True)
