# Phase 5 - DOTA YOLO26n-OBB integration

This package adds a separate Phase 5 oriented-object-detection pipeline to the existing
Satellite-Image-Intelligence Flask project. It does not replace Phase 3 or Phase 4.

## Files
- `app.py`: complete Flask application preserving RGB upload, segmentation, multispectral
  composite/NDVI, and adding `/detection`.
- `models/object_detection.py`: lazy-loads `checkpoints/detection/best.pt` and runs OBB inference.
- `templates/detection.html`: upload and results page.
- `notebooks/phase5_dota_training.ipynb`: Google Colab training notebook.
- `requirements-phase5.txt`: Phase 5 dependencies.

## Install
Run from the project root:
```powershell
pip install -r requirements-phase5.txt
```
If your project already has a requirements file, merge the listed packages rather than
replacing unrelated dependencies.

## Checkpoint placement
After completing the Colab notebook, copy the downloaded `best.pt` to:
`checkpoints/detection/best.pt`

Create the folder if needed. Do not put the checkpoint in `models/`.

Optional environment override in PowerShell:
```powershell
$env:DETECTION_CHECKPOINT = "C:\full\path\to\best.pt"
```

## Start Flask
```powershell
python app.py
```
Open `http://127.0.0.1:5000/detection`.

## Existing project dependencies
This app still expects your existing:
- `config.py` with `MAX_CONTENT_LENGTH`
- `models/segmentation.py` exposing `segment_image(image_path, output_dir)`
- `models/multispectral.py` exposing:
  - `inspect_raster(path)`
  - `create_composite(path, red_band, green_band, blue_band, output_path)`
  - `calculate_ndvi(path, red_band, nir_band, output_path)`
- Existing templates `index.html`, `results.html`, and `multispectral.html`.

The updated `app.py` corrects the multispectral call argument order to match those
function signatures and moves segmentation outputs into the folders already served
by Flask.

## Add a navigation link
Add this link to the navigation bar in your existing `index.html` (and other shared navs):
```html
<a href="{{ url_for('detection_page') }}">Object Detection</a>
```

## DOTA dataset licence note
Ultralytics' DOTA documentation states that DOTA v1.0 images and annotations are for
academic use and commercial use is prohibited. Check the dataset terms before using
the dataset or derivative data commercially.
