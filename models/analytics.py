
"""
Phase 6: Analytics and Dashboard Data Management

Stores application events in:
data/analytics_history.json

Tracks:
- Image uploads
- RGB segmentation
- Multispectral analysis
- Object detection
- Segmentation class statistics
- Detection class counts
- NDVI statistics
- Recent activity

This JSON implementation is suitable for local development
and single-process use. A database is recommended for
production and concurrent multi-worker deployments.
"""

import json
import logging

from pathlib import Path
from datetime import datetime, timezone
from threading import Lock


# ==========================================================
# CONFIGURATION
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_FOLDER = BASE_DIR / "data"

ANALYTICS_FILE = DATA_FOLDER / "analytics_history.json"

SUPPORTED_EVENTS = {
    "upload",
    "segmentation",
    "multispectral",
    "detection"
}

RECENT_ACTIVITY_LIMIT = 10

logger = logging.getLogger(__name__)

# Prevent overlapping writes within this Python process.
_ANALYTICS_LOCK = Lock()


# ==========================================================
# INITIALIZE STORAGE
# ==========================================================

def initialize_analytics_storage():
    """
    Create the data directory and JSON file if they
    do not already exist.
    """

    DATA_FOLDER.mkdir(
        parents=True,
        exist_ok=True
    )

    if not ANALYTICS_FILE.exists():

        with ANALYTICS_FILE.open(
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                [],
                file,
                indent=4
            )


# ==========================================================
# READ ANALYTICS HISTORY
# ==========================================================

def read_analytics():
    """
    Read all analytics events from the JSON file.

    Returns an empty list if the file does not exist
    or contains invalid JSON.
    """

    initialize_analytics_storage()

    try:

        with ANALYTICS_FILE.open(
            "r",
            encoding="utf-8"
        ) as file:

            history = json.load(file)

        if not isinstance(history, list):

            logger.warning(
                "Analytics history must contain a JSON list."
            )

            return []

        return history

    except json.JSONDecodeError:

        logger.exception(
            "Analytics JSON file contains invalid JSON."
        )

        return []

    except OSError:

        logger.exception(
            "Unable to read analytics history."
        )

        return []


# ==========================================================
# WRITE ANALYTICS HISTORY
# ==========================================================

def write_analytics(history):
    """
    Safely write analytics history using a temporary file
    followed by an atomic replacement.
    """

    initialize_analytics_storage()

    temporary_file = ANALYTICS_FILE.with_suffix(".tmp")

    try:

        with temporary_file.open(
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                history,
                file,
                indent=4,
                ensure_ascii=False,
                default=str
            )

        temporary_file.replace(ANALYTICS_FILE)

    except OSError:

        temporary_file.unlink(missing_ok=True)

        logger.exception(
            "Unable to write analytics history."
        )

        raise


# ==========================================================
# RECORD EVENT
# ==========================================================

def record_event(event_type, filename, details=None):
    """
    Record a successful application event.

    Supported event types:
    upload
    segmentation
    multispectral
    detection
    """

    if event_type not in SUPPORTED_EVENTS:

        raise ValueError(
            f"Unsupported analytics event: {event_type}"
        )

    if not filename:

        raise ValueError(
            "An event filename is required."
        )

    event = {
        "id": None,
        "event_type": event_type,
        "filename": str(filename),
        "timestamp": datetime.now(
            timezone.utc
        ).isoformat(),

        "details": details or {}
    }

    with _ANALYTICS_LOCK:

        history = read_analytics()

        event["id"] = (
            history[-1].get("id", 0) + 1
            if history and isinstance(history[-1], dict)
            and isinstance(history[-1].get("id"), int)
            else len(history) + 1
        )

        history.append(event)

        write_analytics(history)

    return event


# ==========================================================
# EXTRACT CLASS COUNTS
# ==========================================================

def extract_class_counts(statistics):
    """
    Convert segmentation or detection statistics into
    a simple dictionary of class names and numeric counts.

    Supports structures such as:

    {
        "Buildings": 1200,
        "Vegetation": 3500
    }

    Or:

    {
        "Buildings": {"pixels": 1200},
        "Vegetation": {"pixel_count": 3500}
    }
    """

    class_counts = {}

    if not isinstance(statistics, dict):
        return class_counts

    possible_count_keys = (
        "count",
        "pixels",
        "pixel_count",
        "instances",
        "total"
    )

    for class_name, value in statistics.items():

        count = None

        if isinstance(value, (int, float)):

            count = value

        elif isinstance(value, dict):

            for key in possible_count_keys:

                candidate = value.get(key)

                if isinstance(candidate, (int, float)):

                    count = candidate
                    break

        if count is None:
            continue

        if isinstance(count, bool):
            continue

        if count < 0:
            continue

        class_counts[str(class_name)] = (
            class_counts.get(str(class_name), 0) + count
        )

    return class_counts


# ==========================================================
# AGGREGATE CLASS COUNTS
# ==========================================================

def aggregate_class_counts(history, event_type, details_key):
    """
    Aggregate class counts from all events of a given type.
    """

    totals = {}

    for event in history:

        if not isinstance(event, dict):
            continue

        if event.get("event_type") != event_type:
            continue

        details = event.get("details", {})

        if not isinstance(details, dict):
            continue

        statistics = details.get(
            details_key,
            {}
        )

        counts = extract_class_counts(statistics)

        for class_name, count in counts.items():

            totals[class_name] = (
                totals.get(class_name, 0) + count
            )

    return totals


# ==========================================================
# CALCULATE AVERAGE NDVI
# ==========================================================

def calculate_average_ndvi(history):
    """
    Calculate weighted average NDVI across successful
    multispectral analyses.

    Each image's mean is weighted by its valid pixel count.

    This avoids giving a small image the same influence
    as a much larger image.
    """

    weighted_total = 0.0
    total_valid_pixels = 0

    for event in history:

        if not isinstance(event, dict):
            continue

        if event.get("event_type") != "multispectral":
            continue

        details = event.get("details", {})

        if not isinstance(details, dict):
            continue

        statistics = details.get(
            "ndvi_statistics",
            {}
        )

        if not isinstance(statistics, dict):
            continue

        mean_value = statistics.get("mean")

        valid_pixels = statistics.get("valid_pixels")

        if not isinstance(mean_value, (int, float)):
            continue

        if isinstance(mean_value, bool):
            continue

        if not isinstance(valid_pixels, (int, float)):
            continue

        if isinstance(valid_pixels, bool) or valid_pixels <= 0:
            continue

        if not -1.0 <= mean_value <= 1.0:
            continue

        weighted_total += mean_value * valid_pixels

        total_valid_pixels += valid_pixels

    if total_valid_pixels == 0:
        return None

    return round(
        weighted_total / total_valid_pixels,
        4
    )


# ==========================================================
# FORMAT RECENT ACTIVITY
# ==========================================================

def get_recent_activity(history, limit=RECENT_ACTIVITY_LIMIT):
    """
    Return the newest analytics events first.
    """

    recent_events = []

    valid_events = [
        event
        for event in history
        if isinstance(event, dict)
    ]

    for event in reversed(valid_events[-limit:]):

        details = event.get("details", {})

        if not isinstance(details, dict):
            details = {}

        recent_events.append({

            "id": event.get("id"),

            "event_type": event.get(
                "event_type",
                "unknown"
            ),

            "filename": event.get(
                "filename",
                "Unknown"
            ),

            "timestamp": event.get(
                "timestamp",
                "Not available"
            ),

            "original_filename": details.get(
                "original_filename",
                event.get("filename", "Unknown")
            )

        })

    return recent_events


# ==========================================================
# GET DASHBOARD DATA
# ==========================================================

def get_dashboard_data():
    """
    Calculate all information required by analytics.html.
    """

    history = read_analytics()

    # Count event types.
    total_uploads = 0
    segmentation_count = 0
    multispectral_count = 0
    detection_count = 0

    for event in history:

        if not isinstance(event, dict):
            continue

        event_type = event.get("event_type")

        if event_type == "upload":

            total_uploads += 1

        elif event_type == "segmentation":

            segmentation_count += 1

        elif event_type == "multispectral":

            multispectral_count += 1

        elif event_type == "detection":

            detection_count += 1

    # Total completed analyses.
    total_analyses = (
        segmentation_count
        + multispectral_count
        + detection_count
    )

    # Segmentation class distribution.
    segmentation_classes = aggregate_class_counts(
        history,
        "segmentation",
        "class_statistics"
    )

    # Object detection class distribution.
    detection_classes = aggregate_class_counts(
        history,
        "detection",
        "class_counts"
    )

    # Phase 4: pixel totals across AI multispectral predictions.
    # These are pixel counts, not object counts. Class statistics are
    # kept separate from the five-class RGB segmentation model.
    multispectral_classes = aggregate_class_counts(
        history, "multispectral", "class_statistics"
    )
    multispectral_ai_count = sum(
        1 for event in history
        if isinstance(event, dict)
        and event.get("event_type") == "multispectral"
        and isinstance(event.get("details"), dict)
        and bool(event["details"].get("class_statistics"))
    )
    multispectral_total_pixels = sum(multispectral_classes.values())
    multispectral_class_percentages = {
        name: round(100 * count / multispectral_total_pixels, 2)
        if multispectral_total_pixels else 0.0
        for name, count in multispectral_classes.items()
    }
    # Physical area is meaningful only for appropriately projected,
    # consistently georeferenced imagery. Do not add pixel areas from
    # geographic CRS rasters or unverified area estimates.
    multispectral_class_area_m2 = {}
    for event in history:
        if not isinstance(event, dict) or event.get("event_type") != "multispectral":
            continue
        details = event.get("details") or {}
        if not isinstance(details, dict):
            continue
        stats = details.get("class_statistics") or {}
        if not isinstance(stats, dict):
            continue
        for name, item in stats.items():
            if not isinstance(item, dict):
                continue
            area = item.get("area_m2")
            if isinstance(area, (int, float)) and not isinstance(area, bool) and area >= 0:
                multispectral_class_area_m2[name] = (
                    multispectral_class_area_m2.get(name, 0.0) + area
                )
    multispectral_class_area_m2 = {
        name: round(area, 2)
        for name, area in multispectral_class_area_m2.items()
    }

    # Weighted NDVI average.
    average_ndvi = calculate_average_ndvi(history)

    # Recent activity.
    recent_activity = get_recent_activity(history)

    return {

        "total_uploads": total_uploads,

        "total_analyses": total_analyses,

        "segmentation_count": segmentation_count,

        "multispectral_count": multispectral_count,

        "detection_count": detection_count,

        "segmentation_classes": segmentation_classes,

        "detection_classes": detection_classes,

        "multispectral_ai_count": multispectral_ai_count,
        "multispectral_classes": multispectral_classes,
        "multispectral_class_percentages": multispectral_class_percentages,
        "multispectral_class_area_m2": multispectral_class_area_m2,
        "average_ndvi": average_ndvi,

        "recent_activity": recent_activity

    }


# ==========================================================
# INITIALIZE MODULE STORAGE
# ==========================================================

initialize_analytics_storage()
