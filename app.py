import logging
import uuid
from pathlib import Path

import numpy as np
import rasterio

from PIL import Image, UnidentifiedImageError
from rasterio.enums import Resampling

from flask import (
    Flask,
    render_template,
    request,
    jsonify,
    send_from_directory,
    url_for,
    redirect,
    abort,
)

from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge, HTTPException

from config import MAX_CONTENT_LENGTH

from models.segmentation import segment_image

from models.multispectral import (
    inspect_raster,
    create_composite,
    calculate_ndvi,
)

from models.multispectral_inference import predict_multispectral

from models.object_detection import detect_objects

from models.analytics import (
    record_event,
    get_dashboard_data,
    read_analytics,
)

from models.report_generator import generate_analysis_report


# ==========================================================
# 1. PROJECT DIRECTORIES
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent

RGB_UPLOAD_FOLDER = BASE_DIR / "uploads" / "rgb"

MULTISPECTRAL_UPLOAD_FOLDER = (
    BASE_DIR / "uploads" / "multispectral"
)

DETECTION_UPLOAD_FOLDER = (
    BASE_DIR / "uploads" / "detection"
)

SEGMENTATION_OUTPUT_FOLDER = (
    BASE_DIR / "outputs" / "segmentation"
)

MASK_FOLDER = SEGMENTATION_OUTPUT_FOLDER / "masks"
OVERLAY_FOLDER = SEGMENTATION_OUTPUT_FOLDER / "overlays"

MULTISPECTRAL_OUTPUT_FOLDER = (
    BASE_DIR / "outputs" / "multispectral"
)

COMPOSITE_FOLDER = (
    MULTISPECTRAL_OUTPUT_FOLDER / "composites"
)

NDVI_FOLDER = (
    MULTISPECTRAL_OUTPUT_FOLDER / "ndvi"
)

PROCESSED_FOLDER = (
    MULTISPECTRAL_OUTPUT_FOLDER / "processed"
)

MULTISPECTRAL_PREDICTION_FOLDER = (
    MULTISPECTRAL_OUTPUT_FOLDER / "predictions"
)

DETECTION_OUTPUT_FOLDER = (
    BASE_DIR / "outputs" / "detection"
)

DETECTION_RESULT_FOLDER = (
    DETECTION_OUTPUT_FOLDER / "results"
)

PREVIEW_FOLDER = BASE_DIR / "previews"

# Phase 7
REPORT_FOLDER = BASE_DIR / "reports"

REQUIRED_FOLDERS = [
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

for folder in REQUIRED_FOLDERS:
    folder.mkdir(parents=True, exist_ok=True)


# ==========================================================
# 2. FLASK INITIALIZATION
# ==========================================================

app = Flask(__name__)

app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

app.config["RGB_UPLOAD_FOLDER"] = str(
    RGB_UPLOAD_FOLDER
)

app.config["MULTISPECTRAL_UPLOAD_FOLDER"] = str(
    MULTISPECTRAL_UPLOAD_FOLDER
)

app.config["DETECTION_UPLOAD_FOLDER"] = str(
    DETECTION_UPLOAD_FOLDER
)

app.config["REPORT_FOLDER"] = str(
    REPORT_FOLDER
)

logging.basicConfig(level=logging.INFO)


# ==========================================================
# 3. SAFE ANALYTICS RECORDING
# ==========================================================

def safely_record_event(event_type, filename, details=None):
    """
    Record analytics without interrupting the application
    if analytics storage fails.
    """
    try:
        return record_event(
            event_type,
            filename,
            details or {},
        )
    except Exception:
        app.logger.exception(
            "Unable to record analytics event: %s",
            event_type,
        )


# ==========================================================
# 4. FILE VALIDATION HELPERS
# ==========================================================

RGB_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}
TIFF_EXTENSIONS = {"tif", "tiff"}


def get_extension(filename):
    if "." not in filename:
        return ""
    return filename.rsplit(".", 1)[1].lower()


def allowed_rgb_file(filename):
    return get_extension(filename) in RGB_EXTENSIONS


def allowed_tiff_file(filename):
    return get_extension(filename) in TIFF_EXTENSIONS


def validate_rgb_image(image_path):
    """Check image content, rather than trusting the file extension.

    This helper does not modify the source image. The upload route
    separately normalizes supported images to RGB PNG.
    """
    try:
        with Image.open(image_path) as image:
            detected_format = image.format
            if detected_format not in ("JPEG", "PNG", "WEBP"):
                raise ValueError(
                    f"Unsupported image content ({detected_format}). "
                    "Please upload JPEG, PNG or WebP."
                )
            image.verify()
        return detected_format
    except Image.DecompressionBombError as error:
        raise ValueError("Image dimensions exceed the safety limit.") from error
    except (UnidentifiedImageError, OSError, SyntaxError) as error:
        raise ValueError("The uploaded file is not a valid supported image.") from error


def normalize_rgb_image(source_path, destination_path):
    """Validate image and write a real RGB PNG to destination_path.

    Source and destination must be distinct. The destination is removed
    if the conversion fails, avoiding partially written uploads.
    """
    validate_rgb_image(source_path)
    if source_path.resolve() == destination_path.resolve():
        raise ValueError("Source and destination must be different files.")
    try:
        with Image.open(source_path) as image:
            if image.width < 1 or image.height < 1:
                raise ValueError("Invalid image dimensions.")
            rgb = image.convert("RGB")
            rgb.save(destination_path, format="PNG")
            return {"width": rgb.width, "height": rgb.height,
                    "bands": 3, "format": "PNG", "mode": "RGB",
                    "georeferenced": False, "crs": "Not available"}
    except Exception:
        destination_path.unlink(missing_ok=True)
        raise


def validate_tiff(image_path):
    try:
        with rasterio.open(image_path) as dataset:

            if dataset.count < 1:
                raise ValueError(
                    "The raster contains no bands."
                )

            if dataset.width < 1 or dataset.height < 1:
                raise ValueError(
                    "Invalid raster dimensions."
                )

            return {
                "width": dataset.width,
                "height": dataset.height,
                "bands": dataset.count,
                "crs": (
                    str(dataset.crs)
                    if dataset.crs
                    else "Not available"
                ),
                "georeferenced": dataset.crs is not None,
                "resolution": list(dataset.res),
                "data_types": list(dataset.dtypes),
                "nodata": dataset.nodata,
                "format": "GeoTIFF / TIFF",
            }

    except rasterio.errors.RasterioIOError as error:
        raise ValueError(
            "Unable to read this TIFF file."
        ) from error


# ==========================================================
# 5. IMAGE PREVIEW HELPERS
# ==========================================================

def raster_to_rgb(image_path, max_size=1200):

    with rasterio.open(image_path) as dataset:

        width = dataset.width
        height = dataset.height

        scale = min(
            1.0,
            max_size / max(width, height),
        )

        output_width = max(1, int(width * scale))
        output_height = max(1, int(height * scale))

        indexes = (
            [1, 2, 3]
            if dataset.count >= 3
            else [1]
        )

        data = dataset.read(
            indexes=indexes,
            out_shape=(
                len(indexes),
                output_height,
                output_width,
            ),
            resampling=Resampling.bilinear,
        )

        channels = []

        for band in data:

            valid = np.isfinite(band)

            if not np.any(valid):
                channel = np.zeros(
                    (output_height, output_width),
                    dtype=np.uint8,
                )
            else:
                low, high = np.percentile(
                    band[valid],
                    [2, 98],
                )

                if high <= low:
                    channel = np.zeros(
                        (output_height, output_width),
                        dtype=np.uint8,
                    )
                else:
                    normalized = (
                        (band.astype(np.float32) - low)
                        / (high - low)
                    ) * 255

                    normalized = np.nan_to_num(
                        normalized,
                        nan=0,
                        posinf=255,
                        neginf=0,
                    )

                    channel = np.clip(
                        normalized,
                        0,
                        255,
                    ).astype(np.uint8)

            channels.append(channel)

        if len(channels) == 1:
            rgb = np.stack(
                [channels[0]] * 3,
                axis=-1,
            )
        else:
            rgb = np.stack(
                channels,
                axis=-1,
            )

        return rgb, width, height


def create_rgb_preview(image_path, preview_path):

    with Image.open(image_path) as image:
        image.thumbnail(
            (1200, 1200),
            Image.Resampling.LANCZOS,
        )

        image.convert("RGB").save(
            preview_path,
            format="PNG",
        )


def create_tiff_preview(image_path, preview_path):

    rgb, width, height = raster_to_rgb(
        image_path,
        1200,
    )

    Image.fromarray(rgb).save(
        preview_path,
        format="PNG",
    )

    metadata = validate_tiff(image_path)
    metadata["width"] = width
    metadata["height"] = height

    return metadata


def create_processing_copy(
    image_path,
    output_path,
    max_size=2048,
):

    with Image.open(image_path) as image:

        original_width, original_height = image.size

        image = image.convert("RGB")

        image.thumbnail(
            (max_size, max_size),
            Image.Resampling.LANCZOS,
        )

        processing_width, processing_height = image.size

        image.save(
            output_path,
            format="JPEG",
            quality=90,
            optimize=True,
        )

    return {
        "original_width": original_width,
        "original_height": original_height,
        "processing_width": processing_width,
        "processing_height": processing_height,
    }


# ==========================================================
# 6. HOME PAGE
# ==========================================================

@app.route("/")
def home():
    return render_template("index.html")


# ==========================================================
# 7. PHASE 6 ANALYTICS DASHBOARD
# ==========================================================

@app.route("/analytics")
def analytics_dashboard():

    try:
        dashboard_data = get_dashboard_data()

        return render_template(
            "analytics.html",
            data=dashboard_data,
        )

    except Exception:
        app.logger.exception(
            "Unable to load analytics dashboard."
        )

        return render_template(
            "analytics.html",
            error="Unable to load analytics data.",
            data={},
        ), 500


# ==========================================================
# 8. PHASE 7 REPORT GENERATION
# ==========================================================

REPORTABLE_EVENT_TYPES = {
    "segmentation",
    "multispectral",
    "detection",
}


def get_reportable_events():
    """
    Read completed analysis records from Phase 6.
    """

    history = read_analytics()

    if isinstance(history, dict):
        events = history.get(
            "events",
            history.get("history", []),
        )

    elif isinstance(history, list):
        events = history

    else:
        events = []

    if not isinstance(events, list):
        raise ValueError(
            "Analytics history must contain a list."
        )

    return [
        event
        for event in events
        if isinstance(event, dict)
        and event.get(
            "event_type",
            event.get("type"),
        ) in REPORTABLE_EVENT_TYPES
    ]


def get_reportable_events_safe():
    try:
        return list(reversed(get_reportable_events()))
    except Exception:
        return []


@app.route("/reports")
def reports_dashboard():

    try:
        events = get_reportable_events()

        return render_template(
            "reports.html",
            events=list(reversed(events)),
        )

    except Exception:
        app.logger.exception(
            "Unable to load reports dashboard."
        )

        return render_template(
            "reports.html",
            events=[],
            error="Unable to load analysis history.",
        ), 500


@app.route(
    "/reports/generate/<int:event_id>",
    methods=["POST"],
)
def generate_report(event_id):

    try:
        events = get_reportable_events()

        selected_event = next(
            (
                event
                for event in events
                if str(event.get("id")) == str(event_id)
            ),
            None,
        )

        if selected_event is None:
            abort(
                404,
                description="Analysis record not found.",
            )

        report_filename = (
            f"analysis_report_{event_id}_"
            f"{uuid.uuid4().hex[:8]}.pdf"
        )

        report_path = REPORT_FOLDER / report_filename

        generate_analysis_report(
            event=selected_event,
            base_dir=BASE_DIR,
            output_path=report_path,
        )

        if not report_path.is_file():
            raise RuntimeError(
                "The report generator did not create a PDF."
            )

        return redirect(
            url_for(
                "download_report",
                filename=report_filename,
            )
        )

    except HTTPException:
        raise

    except Exception:
        app.logger.exception(
            "PDF generation failed for event %s.",
            event_id,
        )

        return render_template(
            "reports.html",
            events=get_reportable_events_safe(),
            error=(
                "PDF generation failed. "
                "Check the Flask terminal."
            ),
        ), 500


@app.route("/reports/download/<filename>")
def download_report(filename):

    safe_name = secure_filename(filename)

    if (
        safe_name != filename
        or not safe_name.lower().endswith(".pdf")
    ):
        abort(404)

    return send_from_directory(
        REPORT_FOLDER,
        safe_name,
        as_attachment=True,
        mimetype="application/pdf",
    )


# ==========================================================
# 9. PHASE 3 RGB IMAGE UPLOAD
# ==========================================================

@app.route("/upload", methods=["POST"])
def upload_image():
    """Accept supported RGB uploads and normalize to a real PNG."""
    if "satellite_image" not in request.files:
        return render_template("index.html", error="Please select an image."), 400

    file = request.files["satellite_image"]
    if not file.filename:
        return render_template("index.html", error="No file was selected."), 400

    original_name = secure_filename(file.filename)
    if not original_name or not allowed_rgb_file(original_name):
        return render_template(
            "index.html", error="Upload a JPG, JPEG, PNG or WebP image."
        ), 400

    token = uuid.uuid4().hex
    incoming_path = RGB_UPLOAD_FOLDER / f"incoming_{token}.{get_extension(original_name)}"
    unique_name = f"{token}.png"
    image_path = RGB_UPLOAD_FOLDER / unique_name
    preview_name = f"preview_{uuid.uuid4().hex}.png"
    preview_path = PREVIEW_FOLDER / preview_name

    try:
        file.save(str(incoming_path))
        if incoming_path.stat().st_size == 0:
            raise ValueError("The uploaded image is empty.")

        metadata = normalize_rgb_image(incoming_path, image_path)
        create_rgb_preview(image_path, preview_path)

        safely_record_event("upload", unique_name, {
            "original_filename": original_name,
            "image_type": "RGB Image",
            "stored_format": "PNG",
        })

        return render_template("index.html", result={
            "filename": original_name,
            "stored_filename": unique_name,
            "image_type": "RGB Image",
            "metadata": metadata,
            "preview_url": url_for("uploaded_preview", filename=preview_name),
            "message": "RGB image uploaded successfully.",
        })

    except (ValueError, OSError) as error:
        image_path.unlink(missing_ok=True)
        preview_path.unlink(missing_ok=True)
        app.logger.warning("RGB upload rejected (%s): %s", original_name, error)
        return render_template(
            "index.html", error=f"Image validation failed: {error}"
        ), 400
    except Exception:
        image_path.unlink(missing_ok=True)
        preview_path.unlink(missing_ok=True)
        app.logger.exception("RGB upload failed.")
        return render_template(
            "index.html", error="Unexpected RGB upload error."
        ), 500
    finally:
        incoming_path.unlink(missing_ok=True)


# ==========================================================
# 10. PHASE 3 LAND-COVER SEGMENTATION
# ==========================================================

@app.route("/segment", methods=["POST"])
def run_segmentation():
    # Only process images previously uploaded into the RGB directory.
    raw_filename = request.form.get("filename", "")
    filename = secure_filename(raw_filename)
    if not filename or filename != raw_filename or not allowed_rgb_file(filename):
        return render_template("index.html", error="Select a valid previously uploaded RGB image."), 400

    image_path = RGB_UPLOAD_FOLDER / filename
    if not image_path.is_file():
        return render_template("index.html", error="Uploaded image not found."), 404

    processing_path = PREVIEW_FOLDER / f"processing_{uuid.uuid4().hex}.jpg"
    try:
        validate_rgb_image(image_path)
        image_info = create_processing_copy(image_path, processing_path, max_size=2048)
        result = segment_image(str(processing_path), str(SEGMENTATION_OUTPUT_FOLDER))

        mask_name = result["mask_filename"]
        overlay_name = result["overlay_filename"]
        mask_source = SEGMENTATION_OUTPUT_FOLDER / mask_name
        overlay_source = SEGMENTATION_OUTPUT_FOLDER / overlay_name
        if not mask_source.is_file() or not overlay_source.is_file():
            raise RuntimeError("Segmentation did not generate both output images.")

        mask_destination = MASK_FOLDER / mask_name
        overlay_destination = OVERLAY_FOLDER / overlay_name
        mask_source.replace(mask_destination)
        overlay_source.replace(overlay_destination)

        # IMPORTANT: The checkpoint has five output channels but its original
        # training label order has not been verified. Preserve provisional labels
        # in the application and explicitly flag their status in reports.
        class_statistics = result.get("class_statistics", {})
        event = safely_record_event(
            "segmentation", filename, {
                "original_filename": filename,
                "image_width": result["image_width"],
                "image_height": result["image_height"],
                "original_width": image_info["original_width"],
                "original_height": image_info["original_height"],
                "class_statistics": class_statistics,
                "original_image": f"uploads/rgb/{filename}",
                "mask_image": f"outputs/segmentation/masks/{mask_name}",
                "overlay_image": f"outputs/segmentation/overlays/{overlay_name}",
                "model_architecture": "U-Net with ResNet-50 encoder",
                "checkpoint_epoch": 6,
                "class_mapping_verified": False,
                "class_mapping_note": (
                    "Five output channels are confirmed; semantic class names "
                    "and channel order have not been verified against training metadata."
                ),
                "statistics_unit": "pixels",
            }
        )
        return render_template(
            "results.html",
            original_filename=filename,
            mask_filename=mask_name,
            overlay_filename=overlay_name,
            class_statistics=class_statistics,
            image_width=result["image_width"],
            image_height=result["image_height"],
            original_width=image_info["original_width"],
            original_height=image_info["original_height"],
            report_event_id=event.get("id") if isinstance(event, dict) else None,
            class_mapping_verified=False,
        )
    except (ValueError, OSError) as error:
        app.logger.warning("Segmentation input rejected: %s", error)
        return render_template("index.html", error=f"Segmentation input error: {error}"), 400
    except Exception:
        app.logger.exception("Land-cover segmentation failed.")
        return render_template(
            "index.html",
            error="Segmentation failed. Check the Flask terminal for details.",
        ), 500
    finally:
        processing_path.unlink(missing_ok=True)


# ==========================================================
# 11. PHASE 4 MULTISPECTRAL PAGE
# ==========================================================

@app.route("/multispectral")
def multispectral_home():
    return render_template(
        "multispectral.html"
    )


# ==========================================================
# 12. PHASE 4 MULTISPECTRAL ANALYSIS
# ==========================================================

@app.route(
    "/multispectral/analyze",
    methods=["POST"],
)
def run_multispectral_visualization():

    if "multispectral_image" not in request.files:
        return render_template(
            "multispectral.html",
            error="Please select a GeoTIFF file.",
        ), 400

    file = request.files["multispectral_image"]

    if not file.filename:
        return render_template(
            "multispectral.html",
            error="No file was selected.",
        ), 400

    original_name = secure_filename(file.filename)

    if not original_name or not allowed_tiff_file(
        original_name
    ):
        return render_template(
            "multispectral.html",
            error="Upload a .tif or .tiff file.",
        ), 400

    unique_name = (
        f"{uuid.uuid4().hex}."
        f"{get_extension(original_name)}"
    )

    image_path = (
        MULTISPECTRAL_UPLOAD_FOLDER / unique_name
    )

    output_id = uuid.uuid4().hex

    composite_name = (
        f"composite_{output_id}.png"
    )

    ndvi_preview_name = (
        f"ndvi_{output_id}.png"
    )

    composite_path = (
        COMPOSITE_FOLDER / composite_name
    )

    ndvi_preview_path = (
        NDVI_FOLDER / ndvi_preview_name
    )

    try:
        file.save(str(image_path))

        metadata = validate_tiff(image_path)

        safely_record_event(
            "upload",
            unique_name,
            {
                "original_filename": original_name,
                "image_type": "Multispectral GeoTIFF",
                "bands": metadata["bands"],
            },
        )

        inspection = inspect_raster(
            str(image_path)
        )

        if not isinstance(inspection, dict):
            raise ValueError(
                "inspect_raster() must return a dictionary."
            )

        red_band = request.form.get(
            "red_band", 1, type=int
        )

        green_band = request.form.get(
            "green_band", 2, type=int
        )

        blue_band = request.form.get(
            "blue_band", 3, type=int
        )

        nir_band = request.form.get(
            "nir_band", 4, type=int
        )

        band_count = metadata["bands"]

        requested_bands = [
            red_band,
            green_band,
            blue_band,
            nir_band,
        ]

        if any(
            band < 1 or band > band_count
            for band in requested_bands
        ):
            raise ValueError(
                "Selected bands must be between "
                f"1 and {band_count}."
            )

        if len({
            red_band,
            green_band,
            blue_band,
        }) != 3:
            raise ValueError(
                "Select three different RGB bands."
            )

        if red_band == nir_band:
            raise ValueError(
                "Red and NIR must be different bands."
            )

        # The trained U-Net expects four distinct channels in this order:
        # Red (B04), Green (B03), Blue (B02), NIR (B08).
        if len(set(requested_bands)) != 4:
            raise ValueError(
                "AI inference requires four distinct Red, Green, Blue and NIR bands."
            )

        composite_result = create_composite(
            str(image_path),
            red_band,
            green_band,
            blue_band,
            str(composite_path),
        )

        ndvi_result = calculate_ndvi(
            str(image_path),
            red_band,
            nir_band,
            str(ndvi_preview_path),
        )

        # Phase 4 AI: select the correct bands in training order and
        # preserve georeferencing for the segmentation mask.
        # This model was trained on raw Sentinel-2 reflectance DN (0..10000).
        # Arbitrary 4-band GeoTIFFs or reflectance already scaled to 0..1
        # are NOT compatible without conversion and validation.
        inference_input = PROCESSED_FOLDER / f"inference_{output_id}.tif"
        try:
            with rasterio.open(image_path) as src:
                if src.crs is None or src.crs.is_geographic:
                    raise ValueError(
                        "AI inference requires a GeoTIFF in a projected metric CRS "
                        "(for Bengaluru, EPSG:32643)."
                    )
                if any(abs(r - 10.0) > 1.0 for r in src.res):
                    raise ValueError(
                        "AI inference expects Sentinel-2 imagery at approximately "
                        "10-metre pixel resolution."
                    )
                # Check a representative small sample before predicting.
                sample = src.read(
                    indexes=requested_bands,
                    out_shape=(4, min(src.height, 256), min(src.width, 256)),
                    resampling=Resampling.nearest,
                ).astype(np.float32)
                sample_valid = np.isfinite(sample)
                if src.nodata is not None:
                    sample_valid &= sample != src.nodata
                values = sample[sample_valid & (sample >= 0)]
                if values.size == 0:
                    raise ValueError("The raster has no valid reflectance samples.")
                if float(np.percentile(values, 99)) <= 1.5:
                    raise ValueError(
                        "The raster appears to contain reflectance scaled to 0–1. "
                        "This checkpoint expects raw Sentinel-2 DN (typically 0–10000)."
                    )

                profile = src.profile.copy()
                profile.update(count=4, driver="GTiff", dtype="float32")
                with rasterio.open(inference_input, "w", **profile) as dst:
                    # Stream by block to avoid holding large GeoTIFFs in RAM.
                    for _, window in dst.block_windows(1):
                        bands = src.read(
                            indexes=requested_bands,
                            window=window,
                            out_dtype="float32",
                        )
                        dst.write(bands, window=window)
                    for i, name in enumerate(("B04", "B03", "B02", "B08"), 1):
                        dst.set_band_description(i, name)

            prediction = predict_multispectral(
                image_path=str(inference_input),
                output_dir=str(MULTISPECTRAL_PREDICTION_FOLDER),
            )
        finally:
            inference_input.unlink(missing_ok=True)

        prediction_preview_name = Path(prediction["mask_preview"]).name
        prediction_geotiff_name = Path(prediction["mask_geotiff"]).name
        prediction_preview_url = url_for(
            "multispectral_prediction", filename=prediction_preview_name
        )
        prediction_geotiff_url = url_for(
            "multispectral_prediction", filename=prediction_geotiff_name
        )

        # Phase 7: Store multispectral report data.

        safely_record_event(
            "multispectral",
            unique_name,
            {
                "original_filename": original_name,
                "red_band": red_band,
                "green_band": green_band,
                "blue_band": blue_band,
                "nir_band": nir_band,
                "metadata": metadata,
                "ndvi_statistics": ndvi_result,
                "class_statistics": prediction["class_statistics"],
                "valid_pixels": prediction["valid_pixels"],
                "prediction_image": (
                    "outputs/multispectral/predictions/"
                    + prediction_preview_name
                ),
                "prediction_mask_geotiff": (
                    "outputs/multispectral/predictions/"
                    + prediction_geotiff_name
                ),
                "composite_image": (
                    "outputs/multispectral/composites/"
                    + composite_name
                ),
                "ndvi_image": (
                    "outputs/multispectral/ndvi/"
                    + ndvi_preview_name
                ),
            },
        )

        result = {
            "filename": original_name,
            "stored_filename": unique_name,
            "metadata": metadata,
            "inspection": inspection,
            "red_band": red_band,
            "green_band": green_band,
            "blue_band": blue_band,
            "nir_band": nir_band,
            "composite_filename": composite_name,
            "ndvi_filename": ndvi_preview_name,
            "ndvi_preview_filename": ndvi_preview_name,
            "composite_result": composite_result,
            "ndvi_result": ndvi_result,
            "prediction": prediction,
            "class_statistics": prediction["class_statistics"],
            "prediction_preview_url": prediction_preview_url,
            "prediction_geotiff_url": prediction_geotiff_url,
        }

        return render_template(
            "multispectral.html",
            result=result,
        )

    except (
        ValueError,
        rasterio.errors.RasterioIOError,
    ) as error:

        image_path.unlink(missing_ok=True)
        composite_path.unlink(missing_ok=True)
        ndvi_preview_path.unlink(missing_ok=True)

        app.logger.warning(
            "Multispectral validation error: %s",
            error,
        )

        return render_template(
            "multispectral.html",
            error=str(error),
        ), 400

    except Exception:

        image_path.unlink(missing_ok=True)
        composite_path.unlink(missing_ok=True)
        ndvi_preview_path.unlink(missing_ok=True)

        app.logger.exception(
            "Multispectral analysis failed."
        )

        return render_template(
            "multispectral.html",
            error=(
                "Multispectral analysis failed. "
                "Check the Flask terminal."
            ),
        ), 500


# ==========================================================
# 13. PHASE 5 OBJECT DETECTION
# ==========================================================

@app.route(
    "/detection",
    methods=["GET", "POST"],
)
def detection_page():

    if request.method == "GET":
        return render_template(
            "detection.html"
        )

    if "detection_image" not in request.files:
        return render_template(
            "detection.html",
            error="Please choose an image.",
        ), 400

    file = request.files["detection_image"]

    if not file.filename:
        return render_template(
            "detection.html",
            error="No file was selected.",
        ), 400

    original_name = secure_filename(file.filename)

    if not original_name or not allowed_rgb_file(
        original_name
    ):
        return render_template(
            "detection.html",
            error=(
                "Object detection accepts JPG, "
                "JPEG and PNG images."
            ),
        ), 400

    stored_name = (
        f"{uuid.uuid4().hex}."
        f"{get_extension(original_name)}"
    )

    image_path = (
        DETECTION_UPLOAD_FOLDER / stored_name
    )

    try:
        file.save(str(image_path))

        validate_rgb_image(image_path)

        safely_record_event(
            "upload",
            stored_name,
            {
                "original_filename": original_name,
                "image_type": "Object Detection Image",
            },
        )

        confidence = request.form.get(
            "confidence",
            "0.25",
            type=float,
        )

        if confidence is None:
            confidence = 0.25

        if not 0.05 <= confidence <= 0.95:
            raise ValueError(
                "Confidence must be between 0.05 and 0.95."
            )

        result = detect_objects(
            str(image_path),
            str(DETECTION_RESULT_FOLDER),
            confidence=confidence,
            image_size=1024,
        )

        result["original_filename"] = original_name
        result["stored_filename"] = stored_name

        result["annotated_url"] = url_for(
            "detection_output",
            filename=result["annotated_filename"],
        )

        result["original_url"] = url_for(
            "detection_upload",
            filename=stored_name,
        )

        # Phase 7: Store full detection report data.

        safely_record_event(
            "detection",
            stored_name,
            {
                "original_filename": original_name,
                "class_counts": result.get(
                    "class_counts",
                    {},
                ),
                "total_detections": result.get(
                    "total_detections",
                    0,
                ),
                "confidence_threshold": confidence,
                "image_width": result.get(
                    "image_width"
                ),
                "image_height": result.get(
                    "image_height"
                ),
                "detections": result.get(
                    "detections",
                    [],
                ),
                "original_image": (
                    f"uploads/detection/{stored_name}"
                ),
                "annotated_image": (
                    "outputs/detection/results/"
                    + result["annotated_filename"]
                ),
            },
        )

        return render_template(
            "detection.html",
            result=result,
        )

    except (
        ValueError,
        OSError,
        FileNotFoundError,
    ) as error:

        app.logger.warning(
            "Object detection request error: %s",
            error,
        )

        return render_template(
            "detection.html",
            error=str(error),
        ), 400

    except Exception:

        app.logger.exception(
            "Object detection failed."
        )

        return render_template(
            "detection.html",
            error=(
                "Object detection failed. "
                "Check the checkpoint and terminal."
            ),
        ), 500


# ==========================================================
# 14. SERVE UPLOADED AND GENERATED FILES
# ==========================================================

@app.route("/previews/<filename>")
def uploaded_preview(filename):

    filename = secure_filename(filename)

    if not filename.startswith("preview_"):
        abort(404)

    return send_from_directory(
        PREVIEW_FOLDER,
        filename,
    )


@app.route("/uploads/rgb/<filename>")
def uploaded_rgb_image(filename):

    return send_from_directory(
        RGB_UPLOAD_FOLDER,
        secure_filename(filename),
    )


@app.route("/uploads/multispectral/<filename>")
def uploaded_multispectral_image(filename):

    return send_from_directory(
        MULTISPECTRAL_UPLOAD_FOLDER,
        secure_filename(filename),
    )


@app.route("/uploads/detection/<filename>")
def detection_upload(filename):

    return send_from_directory(
        DETECTION_UPLOAD_FOLDER,
        secure_filename(filename),
    )


@app.route(
    "/outputs/segmentation/masks/<filename>"
)
def segmentation_mask(filename):

    return send_from_directory(
        MASK_FOLDER,
        secure_filename(filename),
    )


@app.route(
    "/outputs/segmentation/overlays/<filename>"
)
def segmentation_overlay(filename):

    return send_from_directory(
        OVERLAY_FOLDER,
        secure_filename(filename),
    )


@app.route(
    "/outputs/multispectral/composites/<filename>"
)
def multispectral_composite(filename):

    return send_from_directory(
        COMPOSITE_FOLDER,
        secure_filename(filename),
    )


@app.route(
    "/outputs/multispectral/ndvi/<filename>"
)
def multispectral_ndvi(filename):

    return send_from_directory(
        NDVI_FOLDER,
        secure_filename(filename),
    )


@app.route("/outputs/multispectral/predictions/<filename>")
def multispectral_prediction(filename):
    safe_name = secure_filename(filename)
    if safe_name != filename or Path(safe_name).suffix.lower() not in {".png", ".tif"}:
        abort(404)
    return send_from_directory(
        MULTISPECTRAL_PREDICTION_FOLDER,
        safe_name,
    )


@app.route(
    "/outputs/detection/results/<filename>"
)
def detection_output(filename):

    return send_from_directory(
        DETECTION_RESULT_FOLDER,
        secure_filename(filename),
    )


# ==========================================================
# 15. HEALTH CHECK
# ==========================================================

@app.route("/health")
def health():

    return jsonify({
        "status": "success",
        "application": "Satellite Image Intelligence",
        "message": "Flask backend is running.",
        "phase3": (
            "RGB segmentation route available"
        ),
        "phase4": (
            "Multispectral analysis and trained U-Net prediction routes available"
        ),
        "phase5": (
            "DOTA OBB detection route available"
        ),
        "phase6": (
            "Analytics dashboard route available"
        ),
        "phase7": (
            "PDF report routes available"
        ),
    })


# ==========================================================
# 16. FILE SIZE ERROR HANDLER
# ==========================================================

@app.errorhandler(RequestEntityTooLarge)
def file_too_large(error):

    message = (
        "File is too large. "
        "Check MAX_CONTENT_LENGTH in config.py."
    )

    if request.path.startswith("/detection"):
        return render_template(
            "detection.html",
            error=message,
        ), 413

    if request.path.startswith("/multispectral"):
        return render_template(
            "multispectral.html",
            error=message,
        ), 413

    return render_template(
        "index.html",
        error=message,
    ), 413


# ==========================================================
# 17. RUN FLASK APPLICATION
# ==========================================================

if __name__ == "__main__":
    app.run(debug=True)

