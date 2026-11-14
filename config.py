
import os

# ==================================================
# MAIN PROJECT DIRECTORY
# ==================================================

BASE_DIR = os.path.abspath(os.path.dirname(__file__))


# ==================================================
# UPLOAD DIRECTORIES
# ==================================================

UPLOAD_FOLDER = os.path.join(
    BASE_DIR,
    "uploads"
)

RGB_UPLOAD_FOLDER = os.path.join(
    UPLOAD_FOLDER,
    "rgb"
)

MULTISPECTRAL_UPLOAD_FOLDER = os.path.join(
    UPLOAD_FOLDER,
    "multispectral"
)


# ==================================================
# OUTPUT DIRECTORIES
# ==================================================

OUTPUT_FOLDER = os.path.join(
    BASE_DIR,
    "outputs"
)

SEGMENTATION_OUTPUT_FOLDER = os.path.join(
    OUTPUT_FOLDER,
    "segmentation"
)

MASK_FOLDER = os.path.join(
    SEGMENTATION_OUTPUT_FOLDER,
    "masks"
)

OVERLAY_FOLDER = os.path.join(
    SEGMENTATION_OUTPUT_FOLDER,
    "overlays"
)

MULTISPECTRAL_OUTPUT_FOLDER = os.path.join(
    OUTPUT_FOLDER,
    "multispectral"
)

COMPOSITE_FOLDER = os.path.join(
    MULTISPECTRAL_OUTPUT_FOLDER,
    "composites"
)

NDVI_FOLDER = os.path.join(
    MULTISPECTRAL_OUTPUT_FOLDER,
    "ndvi"
)

PROCESSED_FOLDER = os.path.join(
    MULTISPECTRAL_OUTPUT_FOLDER,
    "processed"
)


# ==================================================
# PREVIEW DIRECTORY
# ==================================================

PREVIEW_FOLDER = os.path.join(
    BASE_DIR,
    "previews"
)


# ==================================================
# UPLOAD LIMIT
# ==================================================

# Maximum upload size: 20 MB

MAX_CONTENT_LENGTH = 20 * 1024 * 1024


# ==================================================
# ACCEPTED FILE EXTENSIONS
# ==================================================

RGB_EXTENSIONS = {
    "png",
    "jpg",
    "jpeg"
}

TIFF_EXTENSIONS = {
    "tif",
    "tiff"
}

ALLOWED_EXTENSIONS = (
    RGB_EXTENSIONS | TIFF_EXTENSIONS
)


# ==================================================
# IMAGE DIMENSION LIMIT
# ==================================================

MAX_IMAGE_DIMENSION = 50000


# ==================================================
# CREATE REQUIRED DIRECTORIES
# ==================================================

REQUIRED_DIRECTORIES = [
    RGB_UPLOAD_FOLDER,
    MULTISPECTRAL_UPLOAD_FOLDER,
    SEGMENTATION_OUTPUT_FOLDER,
    MASK_FOLDER,
    OVERLAY_FOLDER,
    MULTISPECTRAL_OUTPUT_FOLDER,
    COMPOSITE_FOLDER,
    NDVI_FOLDER,
    PROCESSED_FOLDER,
    PREVIEW_FOLDER
]

for directory in REQUIRED_DIRECTORIES:
    os.makedirs(directory, exist_ok=True)
