# Satellite Image Intelligence

A Flask-based geospatial image-analysis application supporting RGB land-cover segmentation, multispectral GeoTIFF analysis, oriented object detection, analytics, and downloadable PDF reports.

## Features

| Module | Functionality |
| --- | --- |
| RGB segmentation | Upload JPEG/PNG/WebP, generate a land-cover mask and overlay, and inspect pixel statistics |
| Multispectral analysis | Inspect GeoTIFF bands, create RGB composites and NDVI visualizations, and run four-band land-cover inference |
| Object detection | Run trained YOLO oriented-bounding-box (OBB) detection on JPEG/PNG imagery |
| Analytics | Review recorded analysis events and summary statistics |
| PDF reports | Generate and download reports for completed analyses |

**Model interpretation caveat:** The RGB segmentation checkpoint produces five output channels, but the mapping between channel indices and semantic class names has **not yet been verified** against the training metadata. Treat displayed RGB class labels as provisional. Pixel counts and percentages refer to the predicted channel assignments, not independently validated land-cover ground truth.

## Technology

- Python 3.12, Flask, Jinja templates, HTML/CSS/JavaScript
- PyTorch, torchvision, segmentation-models-pytorch, Ultralytics YOLO OBB
- Rasterio, NumPy, Pillow, OpenCV, Matplotlib, pandas
- ReportLab for PDF generation

AiTLAS and Google Earth Engine were used in earlier experimentation/training workflows; they are **not assumed to be required** to run the current Flask application. Check notebook-specific requirements separately.

## Project layout

```text
Satellite-Image-Intelligence/
├── app.py
├── config.py
├── requirements.txt
├── models/
│   ├── segmentation.py
│   ├── multispectral.py
│   ├── multispectral_inference.py
│   ├── object_detection.py
│   ├── analytics.py
│   └── report_generator.py
├── templates/
├── static/
├── tests/
├── notebooks/                  # If present
├── checkpoints/                # Local only; not committed
│   ├── segmentation/
│   │   └── landcover_segmentation.pth
│   ├── multispectral/          # Use path expected by multispectral_inference.py
│   └── detection/
│       └── best.pt
├── uploads/                    # Generated at runtime; ignored
├── outputs/                    # Generated at runtime; ignored
├── previews/                   # Generated at runtime; ignored
└── reports/                    # Generated at runtime; ignored
```

The exact multispectral checkpoint name/path must be checked in `models/multispectral_inference.py` before documenting it as fixed.

## Local setup (Windows PowerShell)

```powershell
cd Satellite-Image-Intelligence
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Place the trained model checkpoints in the locations expected by the corresponding model modules. Checkpoints are not included in the repository. The application preloads the RGB segmentation model at startup, so startup will fail if its checkpoint is absent or incompatible.

Start the application:

```powershell
python app.py
```

Visit <http://127.0.0.1:5000/>. Useful routes: `/multispectral`, `/detection`, `/analytics`, `/reports`, and `/health`.

## Testing

```powershell
python -m py_compile app.py
python -m unittest discover -s tests -v
```

At the time of the latest local verification, **21 tests passed**. Additional manual checks succeeded for RGB segmentation, multispectral inference, OBB detection, analytics, and PDF downloads. These checks are not a full security audit or scientific accuracy evaluation.

## Input and model constraints

- RGB segmentation accepts the formats configured by `config.py`; uploads are normalized before inference.
- The multispectral inference workflow expects four distinct bands in **Red (B04), Green (B03), Blue (B02), NIR (B08)** order, compatible raw Sentinel-2 reflectance DN, a projected metric CRS, and approximately 10 m pixels. Arbitrary four-band TIFFs may be incompatible.
- The detection workflow accepts JPEG/PNG and requires `checkpoints/detection/best.pt` unless overridden in its model configuration.
- Upload size, raster dimensions, and other limits are controlled by `config.py` and enforced by the Flask application.

## Security and deployment

The app has upload validation and local debug mode is disabled by default in the updated configuration. **Do not expose the current development server directly to the public internet.** Before public deployment, implement an access policy for uploaded imagery and reports, CSRF protection for state-changing requests, request throttling, resource limits, and a production WSGI server (such as Waitress on Windows). Review privacy and retention requirements for uploaded imagery and analytics history.

Never commit `.env` files, credentials, user imagery, generated reports, or large model checkpoints. Review the `.gitignore` file and `git status` before pushing.

## Screenshots

Add your own verified screenshots under `screenshots/` (homepage, RGB segmentation, multispectral analysis, OBB detection, analytics, and reports). Do not label mockups as real inference results.

## Data and model licenses

Check the license and usage terms of every dataset, checkpoint, and library before redistribution or commercial use. In particular, verify the terms applicable to any DOTA training data and derived checkpoints.

## Project status

Local end-to-end workflows have been exercised successfully. Remaining work includes deeper security testing, deployment configuration, and final acceptance testing.
