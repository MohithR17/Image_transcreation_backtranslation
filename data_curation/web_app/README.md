# Transcreation Preview Web App

A tiny Flask app that reads the generated transcreation JSON (with generated images)
and provides a simple searchable UI for previewing a source entity and its 5
alternatives with images.

How to run locally

1. Activate your conda env that has Flask installed (or install Flask):

   conda activate image-transcreation
   pip install flask

2. From the `data_curation` folder run:

   python web_app/app.py

3. Open http://127.0.0.1:5000 in your browser and search for a source entity (e.g. "Harira").

Notes

- The app serves images that are referenced by paths in the JSON. It restricts
  served files to the repository workspace directory for safety.
- If your JSON is in a different location, edit `DATA_FILE` in `app.py` to point
  to it, or copy your JSON to `output/morocco_to_india_with_images.json`.
