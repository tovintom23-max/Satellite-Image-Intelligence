"""
Phase 4: Multispectral GeoTIFF utilities.

This module does not run an AI model. It inspects a raster, creates
visual composites, and calculates NDVI from user-selected red/NIR bands.
All band numbers exposed to users are 1-based Rasterio band indexes.
"""
from pathlib import Path
import numpy as np
import rasterio
from rasterio.enums import Resampling
from PIL import Image


def inspect_raster(path):
    with rasterio.open(path) as src:
        if src.count < 1:
            raise ValueError("The raster contains no bands.")
        band_details = []
        for index in range(1, src.count + 1):
            description = src.descriptions[index - 1]
            tags = src.tags(index)
            band_details.append({
                "index": index,
                "description": description or tags.get("NAME") or tags.get("BAND_NAME") or f"Band {index}",
                "dtype": src.dtypes[index - 1],
                "nodata": src.nodatavals[index - 1],
                "colorinterp": str(src.colorinterp[index - 1])
            })
        return {
            "width": src.width,
            "height": src.height,
            "count": src.count,
            "crs": str(src.crs) if src.crs else "Not available",
            "transform": str(src.transform),
            "driver": src.driver,
            "bounds": str(src.bounds),
            "resolution": [abs(float(src.res[0])), abs(float(src.res[1]))],
            "dtype": ", ".join(sorted(set(src.dtypes))),
            "nodata": str(src.nodata) if src.nodata is not None else "Not declared",
            "georeferenced": src.crs is not None,
            "bands": band_details
        }


def _read_band(src, band_index, max_side=1400):
    if not 1 <= band_index <= src.count:
        raise ValueError(f"Band {band_index} is not available. This file has {src.count} bands.")
    scale = min(1.0, max_side / max(src.width, src.height))
    out_w = max(1, int(src.width * scale))
    out_h = max(1, int(src.height * scale))
    arr = src.read(
        band_index,
        out_shape=(out_h, out_w),
        resampling=Resampling.bilinear,
        masked=True
    ).astype("float32").filled(np.nan)
    return arr


def _stretch_to_uint8(array, low_percentile=2, high_percentile=98):
    valid = np.isfinite(array)
    if not np.any(valid):
        return np.zeros(array.shape, dtype=np.uint8)
    low, high = np.nanpercentile(array[valid], [low_percentile, high_percentile])
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        return np.zeros(array.shape, dtype=np.uint8)
    scaled = (array - low) / (high - low) * 255.0
    return np.clip(np.nan_to_num(scaled, nan=0, posinf=255, neginf=0), 0, 255).astype(np.uint8)


def create_composite(path, red_band, green_band, blue_band, output_path):
    with rasterio.open(path) as src:
        red = _read_band(src, int(red_band))
        green = _read_band(src, int(green_band))
        blue = _read_band(src, int(blue_band))
    rgb = np.stack([
        _stretch_to_uint8(red),
        _stretch_to_uint8(green),
        _stretch_to_uint8(blue)
    ], axis=-1)
    Image.fromarray(rgb, mode="RGB").save(output_path, format="PNG")
    return {"width": int(rgb.shape[1]), "height": int(rgb.shape[0])}


def calculate_ndvi(path, red_band, nir_band, output_path):
    with rasterio.open(path) as src:
        red = _read_band(src, int(red_band))
        nir = _read_band(src, int(nir_band))
    if red.shape != nir.shape:
        raise ValueError("Selected bands could not be aligned to the same preview dimensions.")
    denominator = nir + red
    valid = np.isfinite(red) & np.isfinite(nir) & (np.abs(denominator) > 1e-10)
    ndvi = np.full(red.shape, np.nan, dtype=np.float32)
    ndvi[valid] = (nir[valid] - red[valid]) / denominator[valid]
    ndvi = np.clip(ndvi, -1.0, 1.0)
    # A simple display ramp: low NDVI = dark, high NDVI = bright green.
    normalized = np.nan_to_num((ndvi + 1.0) / 2.0, nan=0.0)
    green = np.clip(normalized * 255, 0, 255).astype(np.uint8)
    red_channel = np.clip((1.0 - normalized) * 180, 0, 180).astype(np.uint8)
    blue = np.zeros_like(green)
    rgba = np.stack([red_channel, green, blue], axis=-1)
    Image.fromarray(rgba, mode="RGB").save(output_path, format="PNG")
    valid_values = ndvi[np.isfinite(ndvi)]
    stats = {
        "valid_pixels": int(valid_values.size),
        "mean": round(float(np.mean(valid_values)), 4) if valid_values.size else None,
        "min": round(float(np.min(valid_values)), 4) if valid_values.size else None,
        "max": round(float(np.max(valid_values)), 4) if valid_values.size else None
    }
    return stats
