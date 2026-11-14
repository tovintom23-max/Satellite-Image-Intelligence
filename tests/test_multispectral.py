
import tempfile
import unittest
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image
from rasterio.transform import from_origin

from models.multispectral import (
    inspect_raster,
    create_composite,
    calculate_ndvi
)


class TestMultispectral(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """
        Create a temporary four-band GeoTIFF.

        Band 1 = Red
        Band 2 = Green
        Band 3 = Blue
        Band 4 = Near Infrared (NIR)
        """

        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.test_dir = Path(cls.temp_dir.name)

        cls.raster_path = cls.test_dir / "test_multispectral.tif"

        height = 100
        width = 100

        # Create synthetic multispectral data.
        red = np.full((height, width), 1000, dtype=np.uint16)
        green = np.full((height, width), 1500, dtype=np.uint16)
        blue = np.full((height, width), 800, dtype=np.uint16)
        nir = np.full((height, width), 3000, dtype=np.uint16)

        # Modify a 20x20 pixel region.
        red[20:40, 20:40] = 1200
        nir[20:40, 20:40] = 4000

        transform = from_origin(
            77.0,
            13.0,
            10,
            10
        )

        # Create the four-band GeoTIFF.
        with rasterio.open(
            cls.raster_path,
            "w",
            driver="GTiff",
            height=height,
            width=width,
            count=4,
            dtype="uint16",
            crs="EPSG:4326",
            transform=transform
        ) as dataset:

            dataset.write(red, 1)
            dataset.write(green, 2)
            dataset.write(blue, 3)
            dataset.write(nir, 4)

            dataset.set_band_description(1, "Red")
            dataset.set_band_description(2, "Green")
            dataset.set_band_description(3, "Blue")
            dataset.set_band_description(4, "Near Infrared")

    @classmethod
    def tearDownClass(cls):
        """Remove temporary test files."""
        cls.temp_dir.cleanup()

    # ================================================
    # TEST 1: Raster file creation
    # ================================================

    def test_raster_file_exists(self):

        self.assertTrue(self.raster_path.exists())

    # ================================================
    # TEST 2: Raster opening and dimensions
    # ================================================

    def test_raster_can_be_opened(self):

        with rasterio.open(self.raster_path) as dataset:

            self.assertEqual(dataset.count, 4)
            self.assertEqual(dataset.width, 100)
            self.assertEqual(dataset.height, 100)

            self.assertEqual(
                dataset.descriptions,
                ("Red", "Green", "Blue", "Near Infrared")
            )

    # ================================================
    # TEST 3: Raster metadata inspection
    # ================================================

    def test_inspect_raster(self):

        metadata = inspect_raster(str(self.raster_path))

        self.assertIsInstance(metadata, dict)

        self.assertEqual(metadata["width"], 100)
        self.assertEqual(metadata["height"], 100)
        self.assertEqual(metadata["count"], 4)

        self.assertEqual(
            metadata["bands"][0]["description"],
            "Red"
        )

        self.assertEqual(
            metadata["bands"][3]["description"],
            "Near Infrared"
        )

        self.assertTrue(metadata["georeferenced"])

    # ================================================
    # TEST 4: RGB composite generation
    # ================================================

    def test_create_composite(self):

        output_path = self.test_dir / "test_composite.png"

        result = create_composite(
            str(self.raster_path),
            red_band=1,
            green_band=2,
            blue_band=3,
            output_path=str(output_path)
        )

        self.assertTrue(output_path.exists())

        self.assertIsInstance(result, dict)

        self.assertEqual(result["width"], 100)
        self.assertEqual(result["height"], 100)

        with Image.open(output_path) as image:

            self.assertEqual(image.format, "PNG")
            self.assertEqual(image.mode, "RGB")
            self.assertEqual(image.size, (100, 100))

    # ================================================
    # TEST 5: NDVI calculation
    # ================================================

    def test_calculate_ndvi(self):

        output_path = self.test_dir / "test_ndvi.png"

        result = calculate_ndvi(
            str(self.raster_path),
            red_band=1,
            nir_band=4,
            output_path=str(output_path)
        )

        self.assertTrue(output_path.exists())

        self.assertIsInstance(result, dict)

        self.assertIn("valid_pixels", result)
        self.assertIn("mean", result)
        self.assertIn("min", result)
        self.assertIn("max", result)

        self.assertEqual(result["valid_pixels"], 10000)

        with Image.open(output_path) as image:

            self.assertEqual(image.format, "PNG")
            self.assertEqual(image.mode, "RGB")
            self.assertEqual(image.size, (100, 100))

    # ================================================
    # TEST 6: NDVI mathematical validation
    # ================================================

    def test_ndvi_expected_value(self):

        output_path = self.test_dir / "ndvi_stats.png"

        result = calculate_ndvi(
            str(self.raster_path),
            red_band=1,
            nir_band=4,
            output_path=str(output_path)
        )

        # Normal pixels:
        # NDVI = (3000 - 1000) / (3000 + 1000)
        # NDVI = 0.5

        # Modified pixels:
        # NDVI = (4000 - 1200) / (4000 + 1200)
        # NDVI = 0.5384615

        normal_pixels = 9600
        modified_pixels = 400

        normal_ndvi = 0.5
        modified_ndvi = 2800 / 5200

        expected_mean = (
            normal_pixels * normal_ndvi +
            modified_pixels * modified_ndvi
        ) / 10000

        self.assertAlmostEqual(
            result["mean"],
            round(expected_mean, 4),
            places=4
        )

        self.assertAlmostEqual(
            result["max"],
            0.5385,
            places=3
        )

        self.assertAlmostEqual(
            result["min"],
            0.5,
            places=4
        )

    # ================================================
    # TEST 7: Invalid composite band
    # ================================================

    def test_invalid_composite_band(self):

        output_path = self.test_dir / "invalid_composite.png"

        with self.assertRaises(ValueError):

            create_composite(
                str(self.raster_path),
                red_band=1,
                green_band=2,
                blue_band=5,
                output_path=str(output_path)
            )

    # ================================================
    # TEST 8: Invalid NDVI band
    # ================================================

    def test_invalid_ndvi_band(self):

        output_path = self.test_dir / "invalid_ndvi.png"

        with self.assertRaises(ValueError):

            calculate_ndvi(
                str(self.raster_path),
                red_band=1,
                nir_band=6,
                output_path=str(output_path)
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
