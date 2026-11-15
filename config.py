"""Satellite Image Intelligence — Phase 8.4 configuration.

Backward-compatible constants for the existing Flask app and AI modules.
Environment overrides are optional; defaults preserve current behavior.
"""
import os
from pathlib import Path

BASE_DIR = str(Path(__file__).resolve().parent)


def _positive_int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero")
    return value


def _boolean_env(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be true or false")


UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
RGB_UPLOAD_FOLDER = os.path.join(UPLOAD_FOLDER, "rgb")
MULTISPECTRAL_UPLOAD_FOLDER = os.path.join(UPLOAD_FOLDER, "multispectral")
DETECTION_UPLOAD_FOLDER = os.path.join(UPLOAD_FOLDER, "detection")

OUTPUT_FOLDER = os.path.join(BASE_DIR, "outputs")
SEGMENTATION_OUTPUT_FOLDER = os.path.join(OUTPUT_FOLDER, "segmentation")
MASK_FOLDER = os.path.join(SEGMENTATION_OUTPUT_FOLDER, "masks")
OVERLAY_FOLDER = os.path.join(SEGMENTATION_OUTPUT_FOLDER, "overlays")
MULTISPECTRAL_OUTPUT_FOLDER = os.path.join(OUTPUT_FOLDER, "multispectral")
COMPOSITE_FOLDER = os.path.join(MULTISPECTRAL_OUTPUT_FOLDER, "composites")
NDVI_FOLDER = os.path.join(MULTISPECTRAL_OUTPUT_FOLDER, "ndvi")
PROCESSED_FOLDER = os.path.join(MULTISPECTRAL_OUTPUT_FOLDER, "processed")
MULTISPECTRAL_PREDICTION_FOLDER = os.path.join(MULTISPECTRAL_OUTPUT_FOLDER, "predictions")
DETECTION_OUTPUT_FOLDER = os.path.join(OUTPUT_FOLDER, "detection")
DETECTION_RESULT_FOLDER = os.path.join(DETECTION_OUTPUT_FOLDER, "results")
PREVIEW_FOLDER = os.path.join(BASE_DIR, "previews")
REPORT_FOLDER = os.path.join(BASE_DIR, "reports")

# Default stays 20 MiB to preserve current application behavior.
# Changing this limit does not eliminate memory/CPU risks from large rasters.
MAX_CONTENT_LENGTH = _positive_int_env("MAX_UPLOAD_MB", 20) * 1024 * 1024
MAX_IMAGE_DIMENSION = _positive_int_env("MAX_IMAGE_DIMENSION", 50000)

RGB_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}
TIFF_EXTENSIONS = {"tif", "tiff"}
ALLOWED_EXTENSIONS = RGB_EXTENSIONS | TIFF_EXTENSIONS

# This flag is available for a future production launcher. Existing app.py
# currently calls app.run(debug=True), so this setting alone does NOT disable debug.
FLASK_DEBUG = _boolean_env("FLASK_DEBUG", False)

REQUIRED_DIRECTORIES = [
    RGB_UPLOAD_FOLDER,
    MULTISPECTRAL_UPLOAD_FOLDER,
    DETECTION_UPLOAD_FOLDER,
    SEGMENTATION_OUTPUT_FOLDER,
    MASK_FOLDER,
    OVERLAY_FOLDER,
    MULTISPECTRAL_OUTPUT_FOLDER,
    COMPOSITE_FOLDER,
    NDVI_FOLDER,
    PROCESSED_FOLDER,
    MULTISPECTRAL_PREDICTION_FOLDER,
    DETECTION_OUTPUT_FOLDER,
    DETECTION_RESULT_FOLDER,
    PREVIEW_FOLDER,
    REPORT_FOLDER,
]

for directory in REQUIRED_DIRECTORIES:
    os.makedirs(directory, exist_ok=True)
