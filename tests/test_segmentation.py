
import unittest
import tempfile

from pathlib import Path

import numpy as np
from PIL import Image

from models.segmentation import (
    segment_image,
    create_coloured_mask,
    get_class_colour
)


class TestSegmentation(unittest.TestCase):

    def setUp(self):

        # Create a temporary directory for testing.
        self.temp_dir = tempfile.TemporaryDirectory()

        self.test_dir = Path(self.temp_dir.name)

        self.image_path = self.test_dir / "test_satellite.png"

        self.output_dir = self.test_dir / "outputs"

        # Create a sample RGB satellite-like image.
        image = Image.new(
            "RGB",
            (128, 128),
            (100, 150, 80)
        )

        image.save(self.image_path)

    def tearDown(self):

        # Remove temporary test files.
        self.temp_dir.cleanup()

    # ------------------------------------------------
    # FAKE PREDICTOR
    # ------------------------------------------------

    def fake_predictor(self, image):

        """
        Simulates a segmentation model.

        Left half  = Class 0 (Background)
        Right half = Class 1 (Buildings)
        """

        mask = np.zeros(
            (image.height, image.width),
            dtype=np.uint8
        )

        mask[:, :64] = 0

        mask[:, 64:] = 1

        return mask

    # ------------------------------------------------
    # TEST 1: COLOURED MASK
    # ------------------------------------------------

    def test_coloured_mask(self):

        mask = np.array([
            [0, 0],
            [1, 1]
        ], dtype=np.uint8)

        result = create_coloured_mask(mask)

        # Check image dimensions.
        self.assertEqual(
            result.size,
            (2, 2)
        )

        # Check that two classes have different colours.
        self.assertNotEqual(
            get_class_colour(0),
            get_class_colour(1)
        )

    # ------------------------------------------------
    # TEST 2: SEGMENTATION OUTPUT FILES
    # ------------------------------------------------

    def test_segmentation_outputs(self):

        result = segment_image(
            self.image_path,
            self.output_dir,
            predictor=self.fake_predictor
        )

        # Check returned dictionary keys.
        self.assertIn("mask_filename", result)

        self.assertIn("overlay_filename", result)

        self.assertIn("class_statistics", result)

        # Check generated files.
        mask_path = (
            self.output_dir / result["mask_filename"]
        )

        overlay_path = (
            self.output_dir / result["overlay_filename"]
        )

        self.assertTrue(mask_path.exists())

        self.assertTrue(overlay_path.exists())

    # ------------------------------------------------
    # TEST 3: OUTPUT IMAGE DIMENSIONS
    # ------------------------------------------------

    def test_output_dimensions(self):

        result = segment_image(
            self.image_path,
            self.output_dir,
            predictor=self.fake_predictor
        )

        mask_path = (
            self.output_dir / result["mask_filename"]
        )

        overlay_path = (
            self.output_dir / result["overlay_filename"]
        )

        with Image.open(mask_path) as mask:

            self.assertEqual(
                mask.size,
                (128, 128)
            )

        with Image.open(overlay_path) as overlay:

            self.assertEqual(
                overlay.size,
                (128, 128)
            )

        # Check dimensions returned by the function.
        self.assertEqual(
            result["image_width"],
            128
        )

        self.assertEqual(
            result["image_height"],
            128
        )

    # ------------------------------------------------
    # TEST 4: CLASS STATISTICS
    # ------------------------------------------------

    def test_class_statistics(self):

        result = segment_image(
            self.image_path,
            self.output_dir,
            predictor=self.fake_predictor
        )

        statistics = result["class_statistics"]

        # Only two classes should be present.
        self.assertEqual(
            len(statistics),
            2
        )

        # Verify class names.
        self.assertIn(
            "Background",
            statistics
        )

        self.assertIn(
            "Buildings",
            statistics
        )

        # Verify statistics structure.
        for class_name, class_data in statistics.items():

            self.assertIn(
                "pixel_count",
                class_data
            )

            self.assertIn(
                "percentage",
                class_data
            )

            self.assertIn(
                "color",
                class_data
            )

    # ------------------------------------------------
    # TEST 5: PIXEL COUNTS
    # ------------------------------------------------

    def test_pixel_counts(self):

        result = segment_image(
            self.image_path,
            self.output_dir,
            predictor=self.fake_predictor
        )

        statistics = result["class_statistics"]

        # Each half contains 128 x 64 = 8192 pixels.
        self.assertEqual(
            statistics["Background"]["pixel_count"],
            8192
        )

        self.assertEqual(
            statistics["Buildings"]["pixel_count"],
            8192
        )

        # Total pixels must equal image dimensions.
        total_pixels = sum(
            item["pixel_count"]
            for item in statistics.values()
        )

        self.assertEqual(
            total_pixels,
            128 * 128
        )

    # ------------------------------------------------
    # TEST 6: CLASS PERCENTAGES
    # ------------------------------------------------

    def test_class_percentages(self):

        result = segment_image(
            self.image_path,
            self.output_dir,
            predictor=self.fake_predictor
        )

        statistics = result["class_statistics"]

        self.assertEqual(
            statistics["Background"]["percentage"],
            50.0
        )

        self.assertEqual(
            statistics["Buildings"]["percentage"],
            50.0
        )

        total_percentage = sum(
            item["percentage"]
            for item in statistics.values()
        )

        self.assertAlmostEqual(
            total_percentage,
            100.0,
            places=1
        )

    # ------------------------------------------------
    # TEST 7: MISSING IMAGE
    # ------------------------------------------------

    def test_missing_image(self):

        missing_image = self.test_dir / "missing.png"

        with self.assertRaises(FileNotFoundError):

            segment_image(
                missing_image,
                self.output_dir,
                predictor=self.fake_predictor
            )


# ====================================================
# RUN UNIT TESTS
# ====================================================

if __name__ == "__main__":

    unittest.main(verbosity=2)
