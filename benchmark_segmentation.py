
"""
Phase 8.4 — RGB Segmentation Performance Benchmark

Run:
    python benchmark_segmentation.py "path/to/image.png"

Uses the existing segmentation model and checkpoint.
Does not modify app.py or model architecture.
"""

import argparse
import tempfile
import time
from pathlib import Path

from PIL import Image

from models.segmentation import (
    load_segmentation_model,
    predict_segmentation,
    segment_image,
)


def elapsed(start):
    return time.perf_counter() - start


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark RGB segmentation performance"
    )
    parser.add_argument(
        "image",
        help="Path to a JPG or PNG satellite image"
    )
    args = parser.parse_args()

    image_path = Path(args.image).resolve()

    if not image_path.is_file():
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    print("\n=== PHASE 8.4 SEGMENTATION BENCHMARK ===")

    # 1. Model loading
    start = time.perf_counter()
    load_segmentation_model()
    model_time = elapsed(start)

    print(f"Model initialization: {model_time:.3f} seconds")

    # 2. Image decoding
    start = time.perf_counter()

    with Image.open(image_path) as source:
        image = source.convert("RGB")

    decode_time = elapsed(start)

    print(f"Image decoding:       {decode_time:.3f} seconds")
    print(f"Image dimensions:     {image.width} x {image.height}")

    # 3. First prediction
    start = time.perf_counter()
    prediction = predict_segmentation(image)
    first_time = elapsed(start)

    print(f"First prediction:     {first_time:.3f} seconds")

    # 4. Second prediction (warm model)
    start = time.perf_counter()
    predict_segmentation(image)
    second_time = elapsed(start)

    print(f"Second prediction:    {second_time:.3f} seconds")

    # 5. Complete processing pipeline
    with tempfile.TemporaryDirectory() as temp_dir:
        start = time.perf_counter()

        result = segment_image(
            str(image_path),
            temp_dir
        )

        pipeline_time = elapsed(start)

        print(
            f"Full pipeline:        {pipeline_time:.3f} seconds"
        )

        print("\nGenerated outputs:")
        print("Mask:", result["mask_filename"])
        print("Overlay:", result["overlay_filename"])

    print("\n=== SUMMARY ===")

    if model_time > second_time:
        print("Model initialization is relatively expensive.")

    if second_time > 5:
        print("Warm CPU inference is taking over 5 seconds.")

    print(
        "Note: Full pipeline includes image decoding, "
        "prediction and output generation."
    )


if __name__ == "__main__":
    main()
