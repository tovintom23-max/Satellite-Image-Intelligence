"""
Phase 5 - DOTA Oriented Object Detection
Uses the fine-tuned Ultralytics YOLO OBB checkpoint.
"""
import os
import uuid
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent.parent
CHECKPOINT_PATH = Path(
    os.environ.get(
        "DETECTION_CHECKPOINT",
        BASE_DIR / "checkpoints" / "detection" / "best.pt"
    )
)

DEVICE = "0" if __import__("torch").cuda.is_available() else "cpu"
DEFAULT_IMAGE_SIZE = 1024
_model = None


def load_detection_model():
    """Load the trained OBB model once per Flask process."""
    global _model

    if _model is not None:
        return _model

    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(
            "Phase 5 checkpoint was not found.\n"
            f"Expected: {CHECKPOINT_PATH}\n"
            "Train the Colab notebook and copy best.pt into "
            "checkpoints/detection/best.pt."
        )

    _model = YOLO(str(CHECKPOINT_PATH))

    task = getattr(_model, "task", None)
    if task != "obb":
        _model = None
        raise RuntimeError(
            f"Expected an OBB checkpoint, but model task is {task!r}. "
            "Use the YOLO26n-OBB training checkpoint."
        )

    return _model


def detect_objects(
    image_path,
    output_dir,
    confidence=0.25,
    image_size=DEFAULT_IMAGE_SIZE
):
    """
    Detect aerial objects and save an annotated image.

    Returns JSON-safe data for the Flask template.
    """
    image_path = Path(image_path)
    output_dir = Path(output_dir)

    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    with Image.open(image_path) as im:
        image_width, image_height = im.size
        if im.format not in ("JPEG", "PNG", "TIFF"):
            raise ValueError("Upload a JPG, JPEG, PNG or TIFF image.")
        rgb_image = im.convert("RGB")
        # YOLO accepts paths; PIL validation confirms a readable image.

    model = load_detection_model()

    predictions = model.predict(
        source=str(image_path),
        imgsz=int(image_size),
        conf=float(confidence),
        device=DEVICE,
        verbose=False
    )

    if not predictions:
        raise RuntimeError("The detection model returned no prediction result.")

    result = predictions[0]
    obb = getattr(result, "obb", None)

    detections = []
    class_counter = Counter()

    if obb is not None and len(obb) > 0:
        polygons = obb.xyxyxyxy.cpu().numpy()
        class_ids = obb.cls.int().cpu().tolist()
        scores = obb.conf.cpu().tolist()

        for polygon, class_id, score in zip(polygons, class_ids, scores):
            class_name = result.names.get(int(class_id), str(class_id))
            points = [
                [round(float(x), 2), round(float(y), 2)]
                for x, y in polygon
            ]

            detections.append({
                "class_id": int(class_id),
                "class_name": str(class_name),
                "confidence": round(float(score), 4),
                "confidence_percent": round(float(score) * 100, 2),
                "polygon": points
            })
            class_counter[str(class_name)] += 1

    # Ultralytics plot() returns a BGR numpy image with OBB polygons.
    annotated_bgr = result.plot()
    if annotated_bgr is None:
        raise RuntimeError("Could not create the annotated detection image.")

    output_filename = f"detection_{uuid.uuid4().hex}.jpg"
    output_path = output_dir / output_filename

    if not cv2.imwrite(str(output_path), annotated_bgr):
        raise OSError(f"Could not save annotated image: {output_path}")

    return {
        "annotated_filename": output_filename,
        "detections": detections,
        "class_counts": dict(sorted(class_counter.items())),
        "total_detections": len(detections),
        "image_width": int(image_width),
        "image_height": int(image_height),
        "confidence_threshold": float(confidence),
        "model_name": "YOLO26n-OBB fine-tuned on DOTA v1.0"
    }
