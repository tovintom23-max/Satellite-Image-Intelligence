
import os
from pathlib import Path
from datetime import datetime

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image as ReportImage,
    PageBreak,
    KeepTogether
)


# ==========================================================
# REPORT CONFIGURATION
# ==========================================================

PAGE_WIDTH, PAGE_HEIGHT = A4

PRIMARY_COLOR = colors.HexColor("#173B57")
SECONDARY_COLOR = colors.HexColor("#2878A5")
LIGHT_BLUE = colors.HexColor("#EAF3F8")
LIGHT_GREY = colors.HexColor("#F3F5F7")


def _safe_text(value):
    """Convert values into safe printable strings."""

    if value is None:
        return "Not available"

    if isinstance(value, float):
        return f"{value:.4f}"

    if isinstance(value, (dict, list, tuple)):
        return str(value)

    return str(value)


def _relative_file(base_dir, relative_path):
    """
    Resolve a relative project path safely.

    Paths outside the project directory are rejected.
    """

    if not relative_path:
        return None

    base = Path(base_dir).resolve()
    candidate = (base / relative_path).resolve()

    try:
        candidate.relative_to(base)
    except ValueError:
        return None

    if candidate.is_file():
        return candidate

    return None


def _add_image(story, base_dir, relative_path, caption):
    """Add an image to the PDF if it exists."""

    image_path = _relative_file(base_dir, relative_path)

    if image_path is None:
        return

    try:
        image = ReportImage(
            str(image_path),
            width=150 * mm,
            height=90 * mm,
            kind="proportional"
        )

        story.append(Paragraph(
            caption,
            ParagraphStyle(
                name="ImageCaption",
                parent=getSampleStyleSheet()["Normal"],
                textColor=SECONDARY_COLOR,
                fontSize=9,
                spaceAfter=5
            )
        ))

        story.append(image)
        story.append(Spacer(1, 8 * mm))

    except Exception:
        # An unsupported image should not prevent report creation.
        story.append(Paragraph(
            f"{caption}: Image could not be embedded.",
            getSampleStyleSheet()["Normal"]
        ))


def _add_table(story, rows, column_widths=None):
    """Add a consistently styled data table."""

    if not rows:
        return

    table = Table(
        rows,
        colWidths=column_widths,
        repeatRows=1
    )

    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), PRIMARY_COLOR),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("LEADING", (0, 0), (-1, -1), 11),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#D6DDE3")),
        ("BACKGROUND", (0, 1), (-1, -1), colors.white),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [
            colors.white,
            LIGHT_GREY
        ]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
        ("RIGHTPADDING", (0, 0), (-1, -1), 7),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7)
    ]))

    story.append(table)
    story.append(Spacer(1, 7 * mm))


def _add_key_value_table(story, values):
    rows = [["Property", "Value"]]

    for key, value in values.items():
        rows.append([
            _safe_text(key),
            _safe_text(value)
        ])

    _add_table(
        story,
        rows,
        [55 * mm, 115 * mm]
    )


def _add_class_table(story, class_counts, first_column="Class"):
    rows = [[first_column, "Count"]]

    if isinstance(class_counts, dict):
        for class_name, count in class_counts.items():
            if isinstance(count, dict):
                count = count.get(
                    "count",
                    count.get(
                        "pixels",
                        count.get("pixel_count", count.get("instances", 0))
                    )
                )

            rows.append([
                _safe_text(class_name),
                _safe_text(count)
            ])

    if len(rows) == 1:
        rows.append(["No class statistics available", "—"])

    _add_table(
        story,
        rows,
        [85 * mm, 85 * mm]
    )


def _page_footer(canvas, document):
    """Draw page number and report footer."""

    canvas.saveState()

    canvas.setStrokeColor(colors.HexColor("#D6DDE3"))
    canvas.line(
        20 * mm,
        15 * mm,
        PAGE_WIDTH - 20 * mm,
        15 * mm
    )

    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#667788"))

    canvas.drawString(
        20 * mm,
        10 * mm,
        "Satellite Image Intelligence"
    )

    canvas.drawRightString(
        PAGE_WIDTH - 20 * mm,
        10 * mm,
        f"Page {document.page}"
    )

    canvas.restoreState()


# ==========================================================
# MAIN REPORT GENERATOR
# ==========================================================

def generate_analysis_report(event, base_dir, output_path):
    """
    Generate a PDF report from one analytics history event.

    Supported event types:
        segmentation
        multispectral
        detection
        upload

    Image paths in event details must be relative to base_dir.
    """

    if not isinstance(event, dict):
        raise ValueError("Invalid analytics event.")

    event_type = str(event.get("event_type", "")).lower()
    details = event.get("details") or {}

    if event_type not in {
        "segmentation",
        "multispectral",
        "detection",
        "upload"
    }:
        raise ValueError("Unsupported report event type.")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    styles = getSampleStyleSheet()

    styles.add(ParagraphStyle(
        name="ReportTitle",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=21,
        leading=27,
        textColor=PRIMARY_COLOR,
        alignment=TA_CENTER,
        spaceAfter=8 * mm
    ))

    styles.add(ParagraphStyle(
        name="ReportSubtitle",
        parent=styles["Normal"],
        fontSize=10,
        textColor=colors.HexColor("#667788"),
        alignment=TA_CENTER,
        spaceAfter=10 * mm
    ))

    styles.add(ParagraphStyle(
        name="SectionHeading",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=17,
        textColor=SECONDARY_COLOR,
        spaceBefore=5 * mm,
        spaceAfter=3 * mm
    ))

    styles.add(ParagraphStyle(
        name="ReportBody",
        parent=styles["BodyText"],
        fontSize=9,
        leading=14
    ))

    story = []

    # ------------------------------------------------------
    # TITLE
    # ------------------------------------------------------

    report_titles = {
        "segmentation": "RGB Land-Cover Segmentation Report",
        "multispectral": "Multispectral Analysis Report",
        "detection": "Aerial Object Detection Report",
        "upload": "Satellite Image Upload Report"
    }

    story.append(Spacer(1, 15 * mm))

    story.append(Paragraph(
        "SATELLITE IMAGE INTELLIGENCE",
        styles["ReportSubtitle"]
    ))

    story.append(Paragraph(
        report_titles[event_type],
        styles["ReportTitle"]
    ))

    story.append(Paragraph(
        "Automated Satellite and Aerial Image Analysis",
        styles["ReportSubtitle"]
    ))

    # ------------------------------------------------------
    # GENERAL INFORMATION
    # ------------------------------------------------------

    timestamp = event.get("timestamp")

    if timestamp:
        try:
            timestamp = datetime.fromisoformat(
                str(timestamp).replace("Z", "+00:00")
            ).strftime("%d %B %Y, %H:%M:%S")
        except ValueError:
            timestamp = str(timestamp)
    else:
        timestamp = datetime.now().strftime(
            "%d %B %Y, %H:%M:%S"
        )

    story.append(Paragraph(
        "1. Report Information",
        styles["SectionHeading"]
    ))

    _add_key_value_table(story, {
        "Report ID": event.get("id"),
        "Analysis Type": event_type.title(),
        "Original Filename": details.get(
            "original_filename",
            event.get("filename")
        ),
        "Stored Filename": event.get("filename"),
        "Analysis Date": timestamp
    })

    # ------------------------------------------------------
    # RGB SEGMENTATION
    # ------------------------------------------------------

    if event_type == "segmentation":

        story.append(Paragraph(
            "2. Segmentation Results",
            styles["SectionHeading"]
        ))

        _add_key_value_table(story, {
            "Image Width": details.get("image_width"),
            "Image Height": details.get("image_height")
        })

        story.append(Paragraph(
            "Land-Cover Class Statistics",
            styles["Heading3"]
        ))

        _add_class_table(
            story,
            details.get("class_statistics", {})
        )

        story.append(Paragraph(
            "3. Image Results",
            styles["SectionHeading"]
        ))

        _add_image(
            story,
            base_dir,
            details.get("original_image"),
            "Original RGB Image"
        )

        _add_image(
            story,
            base_dir,
            details.get("mask_image"),
            "Segmentation Mask"
        )

        _add_image(
            story,
            base_dir,
            details.get("overlay_image"),
            "Segmentation Overlay"
        )

    # ------------------------------------------------------
    # MULTISPECTRAL ANALYSIS
    # ------------------------------------------------------

    elif event_type == "multispectral":

        story.append(Paragraph(
            "2. Raster and Band Information",
            styles["SectionHeading"]
        ))

        metadata = details.get("metadata") or {}

        band_values = {
            "Red Band": details.get("red_band"),
            "Green Band": details.get("green_band"),
            "Blue Band": details.get("blue_band"),
            "NIR Band": details.get("nir_band")
        }

        raster_values = {
            "Width": metadata.get("width"),
            "Height": metadata.get("height"),
            "Total Bands": metadata.get("bands"),
            "CRS": metadata.get("crs"),
            "Georeferenced": metadata.get("georeferenced")
        }

        _add_key_value_table(story, raster_values)
        _add_key_value_table(story, band_values)

        story.append(Paragraph(
            "3. NDVI Statistics",
            styles["SectionHeading"]
        ))

        ndvi = details.get("ndvi_statistics") or {}

        _add_key_value_table(story, {
            "Valid Pixels": ndvi.get("valid_pixels"),
            "Mean NDVI": ndvi.get("mean"),
            "Minimum NDVI": ndvi.get("min"),
            "Maximum NDVI": ndvi.get("max")
        })

        story.append(Paragraph(
            "4. Multispectral Image Results",
            styles["SectionHeading"]
        ))

        _add_image(
            story,
            base_dir,
            details.get("composite_image"),
            "RGB False-Colour Composite"
        )

        _add_image(
            story,
            base_dir,
            details.get("ndvi_image"),
            "NDVI Visualization"
        )

        # Phase 4 AI: four-class land-cover segmentation (if available).
        class_statistics = details.get("class_statistics") or {}
        if isinstance(class_statistics, dict) and class_statistics:
            story.append(Paragraph(
                "5. AI Land-Cover Prediction",
                styles["SectionHeading"]
            ))
            story.append(Paragraph(
                "Four-class U-Net prediction: Other, Built-up, Vegetation and Water. "
                "Percentages exclude invalid pixels. Areas are estimates based on "
                "the source raster pixel size.",
                styles["ReportBody"]
            ))
            rows = [["Land-cover class", "Pixels", "Percent", "Area (m2)"]]
            for name in ("Other", "Built-up", "Vegetation", "Water"):
                item = class_statistics.get(name, {})
                if not isinstance(item, dict):
                    item = {"pixels": item}
                rows.append([
                    _safe_text(name),
                    _safe_text(item.get("pixels", item.get("pixel_count"))),
                    _safe_text(item.get("percentage")),
                    _safe_text(item.get("area_m2")),
                ])
            _add_table(story, rows, [48 * mm, 32 * mm, 35 * mm, 55 * mm])
            _add_key_value_table(story, {
                "Valid Classified Pixels": details.get("valid_pixels"),
                "Model Input": "Sentinel-2 B04, B03, B02, B08",
            })
            _add_image(
                story, base_dir, details.get("prediction_image"),
                "AI Land-Cover Prediction (Other / Built-up / Vegetation / Water)"
            )
        else:
            story.append(Paragraph(
                "AI classification is unavailable for this older analysis record.",
                styles["ReportBody"]
            ))

    # ------------------------------------------------------
    # OBJECT DETECTION
    # ------------------------------------------------------

    elif event_type == "detection":

        story.append(Paragraph(
            "2. Object Detection Results",
            styles["SectionHeading"]
        ))

        _add_key_value_table(story, {
            "Image Width": details.get("image_width"),
            "Image Height": details.get("image_height"),
            "Total Detections": details.get("total_detections", 0),
            "Confidence Threshold": details.get(
                "confidence_threshold"
            )
        })

        story.append(Paragraph(
            "Detected Object Classes",
            styles["Heading3"]
        ))

        _add_class_table(
            story,
            details.get("class_counts", {}),
            first_column="Object Class"
        )

        detections = details.get("detections", [])

        if isinstance(detections, list) and detections:
            story.append(Paragraph(
                "Individual Detections",
                styles["Heading3"]
            ))

            rows = [["Class", "Confidence"]]

            for detection in detections[:100]:
                rows.append([
                    _safe_text(detection.get("class_name")),
                    _safe_text(
                        detection.get(
                            "confidence_percent",
                            detection.get("confidence")
                        )
                    )
                ])

            _add_table(story, rows, [85 * mm, 85 * mm])

        story.append(Paragraph(
            "3. Detection Image Results",
            styles["SectionHeading"]
        ))

        _add_image(
            story,
            base_dir,
            details.get("original_image"),
            "Original Aerial Image"
        )

        _add_image(
            story,
            base_dir,
            details.get("annotated_image"),
            "Annotated Object Detection Image"
        )

    # ------------------------------------------------------
    # UPLOAD-ONLY REPORT
    # ------------------------------------------------------

    elif event_type == "upload":

        story.append(Paragraph(
            "2. Upload Information",
            styles["SectionHeading"]
        ))

        _add_key_value_table(story, {
            "Image Type": details.get("image_type"),
            "Number of Bands": details.get("bands"),
            "Stored Filename": event.get("filename")
        })

    # ------------------------------------------------------
    # DISCLAIMER
    # ------------------------------------------------------

    story.append(Spacer(1, 8 * mm))

    story.append(Paragraph(
        "Report Notes",
        styles["SectionHeading"]
    ))

    story.append(Paragraph(
        "This report presents the results recorded by the "
        "Satellite Image Intelligence application. Results are "
        "dependent on the input imagery, selected bands, model "
        "checkpoint and analysis configuration. Model predictions "
        "should be reviewed before being used for operational "
        "or scientific decisions.",
        styles["ReportBody"]
    ))

    # ------------------------------------------------------
    # BUILD PDF
    # ------------------------------------------------------

    document = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        rightMargin=20 * mm,
        leftMargin=20 * mm,
        topMargin=18 * mm,
        bottomMargin=22 * mm,
        title=report_titles[event_type],
        author="Satellite Image Intelligence"
    )

    document.build(
        story,
        onFirstPage=_page_footer,
        onLaterPages=_page_footer
    )

    return str(output_path)
