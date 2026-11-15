
"""
Satellite Image Intelligence
Phase 5 - Oriented Object Detection

Uses a trained Ultralytics YOLO OBB checkpoint.

Checkpoint:
    checkpoints/detection/best.pt

Main functions:
    load_detection_model()
    detect_objects(image_path, output_dir, confidence=0.25,
                   image_size=1024)

Supported input formats:
    JPG, JPEG, PNG, TIFF

Returns:
    {
        "annotated_filename": "...",
        "detections": [...],
        "class_counts": {...},
        "total_detections": 0,
        "image_width": 0,
        "image_height": 0,
        "confidence_threshold": 0.25,
        "model_name": "...",
        "model_type": "YOLO OBB"
    }
"""

import os
import uuid

from collections import Counter
from pathlib import Path
from threading import Lock

import cv2
import numpy as np
import torch

from PIL import Image, UnidentifiedImageError
from ultralytics import YOLO


# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

CHECKPOINT_PATH = Path(
    os.environ.get(
        "DETECTION_CHECKPOINT",
        str(
            BASE_DIR
            / "checkpoints"
            / "detection"
            / "best.pt"
        )
    )
).expanduser()

DEFAULT_OUTPUT_DIR = (
    BASE_DIR
    / "outputs"
    / "detection"
)

DEVICE = "0" if torch.cuda.is_available() else "cpu"

DEFAULT_IMAGE_SIZE = 1024
DEFAULT_CONFIDENCE = 0.25

SUPPORTED_IMAGE_FORMATS = {
    "JPEG",
    "PNG",
    "TIFF",
}

_model = None
_model_lock = Lock()
_inference_lock = Lock()


# ============================================================
# 2. LOAD TRAINED OBJECT DETECTION MODEL
# ============================================================

def load_detection_model():
    """
    Load the trained YOLO OBB checkpoint.

    The model is cached so Flask does not reload it for
    every image request.

    Returns:
        ultralytics.YOLO
    """

    global _model

    if _model is not None:
        return _model

    with _model_lock:

        if _model is not None:
            return _model

        if not CHECKPOINT_PATH.is_file():
            raise FileNotFoundError(
                "Phase 5 object detection checkpoint is missing.\n"
                f"Expected location: {CHECKPOINT_PATH}\n"
                "Place your trained best.pt file inside "
                "checkpoints/detection/."
            )

        try:
            model = YOLO(str(CHECKPOINT_PATH))

            task = getattr(model, "task", None)

            if task != "obb":
                raise RuntimeError(
                    "The checkpoint is not an oriented "
                    "bounding-box model.\n"
                    f"Detected task: {task!r}\n"
                    "Expected task: 'obb'."
                )

            _model = model

            print(
                "[Phase 5] Object detection model loaded."
            )
            print(
                f"[Phase 5] Checkpoint: {CHECKPOINT_PATH}"
            )
            print(
                f"[Phase 5] Device: {DEVICE}"
            )
            print(
                f"[Phase 5] Task: {_model.task}"
            )
            print(
                f"[Phase 5] Classes: {_model.names}"
            )

            return _model

        except Exception as exc:
            raise RuntimeError(
                "Failed to load the Phase 5 object "
                f"detection checkpoint: {exc}"
            ) from exc


# ============================================================
# 3. VALIDATE AND PREPARE INPUT IMAGE
# ============================================================

def prepare_detection_image(image_path):
    """
    Validate an input satellite image and convert it
    to a three-channel BGR NumPy array.

    Ultralytics expects BGR channel ordering for
    NumPy image inputs.

    Returns:
        (bgr_image, width, height)
    """

    image_path = Path(image_path)

    if not image_path.is_file():
        raise FileNotFoundError(
            f"Satellite image not found: {image_path}"
        )

    try:
        with Image.open(image_path) as image:

            image_format = image.format

            if image_format not in SUPPORTED_IMAGE_FORMATS:
                raise ValueError(
                    "Unsupported image format. "
                    "Upload a valid JPG, JPEG, PNG or TIFF."
                )

            width, height = image.size

            if width <= 0 or height <= 0:
                raise ValueError(
                    "Invalid satellite image dimensions."
                )

            # Convert grayscale, RGBA and other
            # PIL-supported modes into RGB.
            rgb_image = image.convert("RGB")

            # Create a NumPy array.
            rgb_array = np.asarray(rgb_image)

            # OpenCV/Ultralytics NumPy input uses BGR.
            bgr_image = cv2.cvtColor(
                rgb_array,
                cv2.COLOR_RGB2BGR
            )

            return (
                np.ascontiguousarray(bgr_image),
                int(width),
                int(height)
            )

    except Image.DecompressionBombError as exc:
        raise ValueError(
            "The image is too large to process safely. "
            "Try a smaller satellite image tile."
        ) from exc

    except UnidentifiedImageError as exc:
        raise ValueError(
            "The uploaded file is not a readable image."
        ) from exc

    except OSError as exc:
        raise ValueError(
            f"Could not read satellite image: {exc}"
        ) from exc


# ============================================================
# 4. VALIDATE INFERENCE PARAMETERS
# ============================================================

def validate_detection_parameters(
    confidence,
    image_size
):
    """
    Validate model inference parameters.

    Returns:
        (confidence, image_size)
    """

    try:
        confidence = float(confidence)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Confidence must be a number."
        ) from exc

    if not 0.0 < confidence <= 1.0:
        raise ValueError(
            "Confidence must be greater than 0 "
            "and less than or equal to 1."
        )

    try:
        image_size = int(image_size)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Image size must be an integer."
        ) from exc

    if image_size < 32 or image_size > 2048:
        raise ValueError(
            "Image size must be between "
            "32 and 2048 pixels."
        )

    return confidence, image_size


# ============================================================
# 5. EXTRACT ORIENTED BOUNDING BOXES
# ============================================================

def extract_obb_detections(result):
    """
    Extract rotated bounding boxes, class names,
    confidence scores and polygon coordinates.

    Returns:
        (detections, class_counts)
    """

    detections = []
    class_counter = Counter()

    obb = getattr(result, "obb", None)

    if obb is None or len(obb) == 0:
        return detections, {}

    polygons = (
        obb.xyxyxyxy
        .detach()
        .cpu()
        .numpy()
    )

    class_ids = (
        obb.cls
        .detach()
        .cpu()
        .int()
        .tolist()
    )

    scores = (
        obb.conf
        .detach()
        .cpu()
        .tolist()
    )

    class_names = result.names

    for polygon, class_id, score in zip(
        polygons,
        class_ids,
        scores
    ):

        class_id = int(class_id)
        score = float(score)

        if isinstance(class_names, dict):
            class_name = class_names.get(
                class_id,
                str(class_id)
            )
        else:
            class_name = (
                class_names[class_id]
                if 0 <= class_id < len(class_names)
                else str(class_id)
            )

        polygon_points = [
            [
                round(float(x), 2),
                round(float(y), 2)
            ]
            for x, y in polygon
        ]

        detection = {
            "class_id": class_id,
            "class_name": str(class_name),
            "confidence": round(score, 4),
            "confidence_percent": round(
                score * 100,
                2
            ),
            "polygon": polygon_points
        }

        detections.append(detection)

        class_counter[str(class_name)] += 1

    return (
        detections,
        dict(sorted(class_counter.items()))
    )


# ============================================================
# 6. SAVE ANNOTATED DETECTION IMAGE
# ============================================================

def save_annotated_image(result, output_dir):
    """
    Draw oriented bounding boxes and save the
    annotated image as a JPEG.

    Returns:
        Filename of the saved image.
    """

    output_dir = Path(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    annotated_bgr = result.plot()

    if annotated_bgr is None:
        raise RuntimeError(
            "Could not generate the annotated "
            "object detection image."
        )

    output_filename = (
        f"detection_{uuid.uuid4().hex}.jpg"
    )

    output_path = output_dir / output_filename

    success = cv2.imwrite(
        str(output_path),
        annotated_bgr,
        [
            cv2.IMWRITE_JPEG_QUALITY,
            92
        ]
    )

    if not success:
        raise OSError(
            "Failed to save annotated detection "
            f"image: {output_path}"
        )

    return output_filename


# ============================================================
# 7. MAIN OBJECT DETECTION FUNCTION
# ============================================================

def detect_objects(
    image_path,
    output_dir=None,
    confidence=DEFAULT_CONFIDENCE,
    image_size=DEFAULT_IMAGE_SIZE
):
    """
    Perform satellite image object detection.

    Steps:
        1. Validate input image.
        2. Prepare a BGR image.
        3. Load the trained YOLO OBB model.
        4. Run object detection.
        5. Extract rotated bounding boxes.
        6. Save annotated image.
        7. Return JSON-safe results.

    Args:
        image_path:
            Path to the input satellite image.

        output_dir:
            Directory for annotated images.
            Defaults to outputs/detection/.

        confidence:
            Minimum prediction confidence.

        image_size:
            YOLO inference image size.

    Returns:
        dict containing detection results.
    """

    image_path = Path(image_path)

    if output_dir is None:
        output_dir = DEFAULT_OUTPUT_DIR

    output_dir = Path(output_dir)

    confidence, image_size = (
        validate_detection_parameters(
            confidence,
            image_size
        )
    )

    # --------------------------------------------------------
    # Step 1: Validate and prepare image
    # --------------------------------------------------------

    bgr_image, image_width, image_height = (
        prepare_detection_image(image_path)
    )

    # --------------------------------------------------------
    # Step 2: Load trained model
    # --------------------------------------------------------

    model = load_detection_model()

    # --------------------------------------------------------
    # Step 3: Run inference
    # --------------------------------------------------------

    try:
        with _inference_lock:
            with torch.inference_mode():
                predictions = model.predict(
                    source=bgr_image,
                    imgsz=image_size,
                    conf=confidence,
                    device=DEVICE,
                    verbose=False
                )

                if not predictions:
                    raise RuntimeError(
                        "The detection model returned "
                        "no prediction results."
                    )

                result = predictions[0]

                # ------------------------------------------------
                # Step 4: Extract detections
                # ------------------------------------------------

                detections, class_counts = (
                    extract_obb_detections(result)
                )

                # ------------------------------------------------
                # Step 5: Save annotated image
                # ------------------------------------------------

                annotated_filename = (
                    save_annotated_image(
                        result,
                        output_dir
                    )
                )

    except Exception as exc:
        raise RuntimeError(
            "Phase 5 object detection inference "
            f"failed: {exc}"
        ) from exc

    # --------------------------------------------------------
    # Step 6: Build result dictionary
    # --------------------------------------------------------

    model_name = (
        f"{model.model.__class__.__name__} "
        "(YOLO OBB)"
    )

    result_data = {
        "annotated_filename": annotated_filename,

        "detections": detections,

        "class_counts": class_counts,

        "total_detections": len(detections),

        "image_width": image_width,

        "image_height": image_height,

        "confidence_threshold": confidence,

        "model_name": model_name,

        "model_type": "YOLO OBB"
    }

    return result_data


# ============================================================
# 8. STANDALONE MODEL VERIFICATION
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("SATELLITE IMAGE INTELLIGENCE")
    print("PHASE 5 - OBJECT DETECTION")
    print("=" * 60)

    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print(f"Checkpoint exists: {CHECKPOINT_PATH.is_file()}")
    print(f"Device: {DEVICE}")

    try:
        model = load_detection_model()

        print("\nModel verification successful.")
        print(f"Model task: {model.task}")
        print(f"Detection classes: {model.names}")

    except Exception as exc:
        print(f"\nModel verification failed: {exc}")
        raise
