
"""
Satellite Image Intelligence
Phase 3 — RGB Land-Cover Segmentation

Model:
    U-Net with ResNet encoder
    segmentation_models_pytorch

Checkpoint:
    checkpoints/segmentation/checkpoint.pth.tar

Main Flask function:
    segment_image(image_path, output_dir, predictor=None)

Returns:
    {
        "mask_filename": str,
        "overlay_filename": str,
        "class_statistics": dict,
        "image_width": int,
        "image_height": int
    }
"""

import os
import uuid
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


# ============================================================
# 1. CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

CHECKPOINT_PATH = Path(
    os.environ.get(
        "SEGMENTATION_CHECKPOINT",
        str(
            BASE_DIR
            / "checkpoints"
            / "segmentation"
            / "checkpoint.pth.tar"
        )
    )
)

# CPU-friendly default for an 8 GB RAM laptop.
DEVICE = torch.device(
    os.environ.get("SEGMENTATION_DEVICE", "cpu")
)

INPUT_SIZE = 512
NUM_CLASSES = 5

# These are the application's existing display labels.
# Verify their mapping against the checkpoint's training dataset.
CLASS_NAMES = {
    0: "Background",
    1: "Buildings",
    2: "Woodlands",
    3: "Water",
    4: "Road"
}

CLASS_COLORS = {
    0: (128, 128, 128),
    1: (255, 0, 0),
    2: (34, 139, 34),
    3: (30, 144, 255),
    4: (255, 215, 0)
}

# Set to "imagenet" or "zero_one" after confirming
# the normalization used during model training.
PREPROCESSING = os.environ.get(
    "SEGMENTATION_PREPROCESSING",
    "imagenet"
).lower()

# Avoid excessive CPU thread usage.
torch.set_num_threads(
    min(4, os.cpu_count() or 1)
)

_model = None


# ============================================================
# 2. CLASS COLOUR HELPERS
# ============================================================

def get_class_colour(class_id):
    """Return the RGB colour for a class ID."""
    return CLASS_COLORS.get(
        int(class_id),
        (255, 255, 255)
    )


def get_class_color(class_id):
    """American spelling compatibility."""
    return get_class_colour(class_id)


# ============================================================
# 3. CHECKPOINT INSPECTION
# ============================================================

def read_checkpoint():
    """
    Safely read the checkpoint on CPU.

    This checkpoint contains a dictionary with a
    'state_dict' entry.
    """

    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(
            "Phase 3 checkpoint not found:\n"
            f"{CHECKPOINT_PATH}"
        )

    try:
        checkpoint = torch.load(
            str(CHECKPOINT_PATH),
            map_location="cpu",
            weights_only=True
        )
    except Exception as error:
        raise RuntimeError(
            "Unable to read the checkpoint safely. "
            "Verify that it is a valid PyTorch checkpoint. "
            f"Details: {error}"
        ) from error

    if not isinstance(checkpoint, dict):
        raise ValueError(
            "Expected a checkpoint dictionary."
        )

    state_dict = checkpoint.get(
        "state_dict",
        checkpoint.get("model_state_dict")
    )

    if not isinstance(state_dict, dict):
        raise ValueError(
            "Checkpoint does not contain a valid state_dict."
        )

    cleaned_state_dict = {}

    for key, value in state_dict.items():
        if not isinstance(key, str):
            raise ValueError(
                "Invalid checkpoint parameter name."
            )

        if key.startswith("module."):
            key = key[len("module."):]

        cleaned_state_dict[key] = value

    return checkpoint, cleaned_state_dict


# ============================================================
# 4. IDENTIFY RESNET ENCODER
# ============================================================

def identify_encoder(state_dict):
    """
    Infer the most likely ResNet encoder from
    the checkpoint's layer structure.

    ResNet-50:
        3, 4, 6, 3 bottleneck blocks

    ResNet-101:
        3, 4, 23, 3 bottleneck blocks

    ResNet-152:
        3, 8, 36, 3 bottleneck blocks
    """

    first_layer = state_dict.get(
        "encoder.conv1.weight"
    )

    if first_layer is None:
        raise ValueError(
            "Checkpoint does not contain "
            "'encoder.conv1.weight'."
        )

    if tuple(first_layer.shape) != (64, 3, 7, 7):
        raise ValueError(
            "Unexpected first encoder layer shape: "
            f"{tuple(first_layer.shape)}"
        )

    block_counts = []

    for stage in range(1, 5):
        indices = set()

        prefix = f"encoder.layer{stage}."

        for key in state_dict:
            if key.startswith(prefix):
                remainder = key[len(prefix):]
                block_id = remainder.split(".")[0]

                if block_id.isdigit():
                    indices.add(int(block_id))

        block_counts.append(len(indices))

    architecture_map = {
        (3, 4, 6, 3): "resnet50",
        (3, 4, 23, 3): "resnet101",
        (3, 8, 36, 3): "resnet152"
    }

    architecture = architecture_map.get(
        tuple(block_counts)
    )

    if architecture is None:
        raise ValueError(
            "Unsupported or unrecognized ResNet encoder. "
            f"Detected block counts: {block_counts}"
        )

    return architecture


# ============================================================
# 5. CREATE MODEL
# ============================================================

def create_segmentation_model(
    encoder_name="resnet50",
    num_classes=NUM_CLASSES
):
    """
    Create a segmentation_models_pytorch U-Net.

    No pretrained encoder weights are downloaded:
    the checkpoint supplies the trained weights.
    """

    try:
        import segmentation_models_pytorch as smp
    except ImportError as error:
        raise ImportError(
            "segmentation_models_pytorch is not installed. "
            "Run: python -m pip install "
            "segmentation-models-pytorch"
        ) from error

    model = smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=None,
        in_channels=3,
        classes=num_classes,
        encoder_depth=5,
        decoder_channels=(256, 128, 64, 32, 16),
        decoder_use_batchnorm=True,
        activation=None
    )

    return model


# ============================================================
# 6. LOAD TRAINED MODEL
# ============================================================

def load_segmentation_model():
    """
    Load the Phase 3 model once and cache it.

    Uses strict=True to prevent inference with
    missing or incorrectly mapped weights.
    """

    global _model

    if _model is not None:
        return _model

    print("\nLoading Phase 3 segmentation model...")

    checkpoint, state_dict = read_checkpoint()

    encoder_name = identify_encoder(state_dict)

    print("Detected encoder:", encoder_name)
    print("Checkpoint epoch:", checkpoint.get("epoch"))

    output_weight = state_dict.get(
        "segmentation_head.0.weight"
    )

    if output_weight is None:
        raise ValueError(
            "Checkpoint is missing the segmentation head."
        )

    checkpoint_classes = int(output_weight.shape[0])

    if checkpoint_classes != NUM_CLASSES:
        raise ValueError(
            f"Checkpoint has {checkpoint_classes} classes, "
            f"but this application expects {NUM_CLASSES}."
        )

    model = create_segmentation_model(
        encoder_name=encoder_name,
        num_classes=NUM_CLASSES
    )

    try:
        model.load_state_dict(
            state_dict,
            strict=True
        )
    except RuntimeError as error:
        raise RuntimeError(
            "Checkpoint architecture mismatch.\n"
            "The detected ResNet encoder may be correct, "
            "but the U-Net decoder or the installed SMP "
            "version may differ from the training setup.\n"
            "Do not use strict=False to bypass this error.\n"
            f"Original error:\n{error}"
        ) from error

    model = model.to(DEVICE)
    model.eval()

    _model = model

    print("Phase 3 model loaded successfully!")
    print("Device:", DEVICE)
    print("Classes:", NUM_CLASSES)

    return _model


# ============================================================
# 7. IMAGE PREPROCESSING
# ============================================================

def preprocess_image(image):
    """
    Convert an RGB PIL image to a model input tensor.

    The normalization must match training.
    """

    image = image.convert("RGB")

    resized = image.resize(
        (INPUT_SIZE, INPUT_SIZE),
        Image.Resampling.BILINEAR
    )

    image_array = np.asarray(
        resized,
        dtype=np.float32
    ) / 255.0

    if PREPROCESSING == "imagenet":
        mean = np.array(
            [0.485, 0.456, 0.406],
            dtype=np.float32
        )

        std = np.array(
            [0.229, 0.224, 0.225],
            dtype=np.float32
        )

        image_array = (
            image_array - mean
        ) / std

    elif PREPROCESSING == "zero_one":
        pass

    else:
        raise ValueError(
            "Unsupported SEGMENTATION_PREPROCESSING. "
            "Use 'imagenet' or 'zero_one'."
        )

    image_array = np.transpose(
        image_array,
        (2, 0, 1)
    )

    tensor = torch.from_numpy(
        np.ascontiguousarray(image_array)
    ).unsqueeze(0)

    return tensor.to(DEVICE)


# ============================================================
# 8. RUN SEGMENTATION INFERENCE
# ============================================================

def predict_segmentation(image):
    """
    Predict a segmentation mask for an RGB PIL image.

    Returns:
        2D NumPy array of class IDs at original size.
    """

    model = load_segmentation_model()

    image = image.convert("RGB")

    original_width, original_height = image.size

    input_tensor = preprocess_image(image)

    with torch.inference_mode():
        logits = model(input_tensor)

        if isinstance(logits, dict):
            logits = logits.get("out")

        if isinstance(logits, (tuple, list)):
            logits = logits[0]

        if not isinstance(logits, torch.Tensor):
            raise ValueError(
                "Model did not return a tensor."
            )

        if logits.ndim != 4:
            raise ValueError(
                "Expected output shape [B, C, H, W]."
            )

        if logits.shape[1] != NUM_CLASSES:
            raise ValueError(
                f"Expected {NUM_CLASSES} classes, "
                f"received {logits.shape[1]}."
            )

        # Predict at model resolution first to reduce
        # peak memory usage on CPU.
        mask = torch.argmax(
            logits,
            dim=1
        )

        mask = mask.unsqueeze(1).float()

        mask = F.interpolate(
            mask,
            size=(original_height, original_width),
            mode="nearest"
        )

        mask = mask.squeeze(0).squeeze(0)

    return mask.cpu().numpy().astype(np.uint8)


# ============================================================
# 9. CLASS STATISTICS
# ============================================================

def calculate_class_statistics(prediction_mask):
    """
    Calculate pixel counts and percentages for
    classes present in the prediction.
    """

    prediction_mask = np.asarray(prediction_mask)

    if prediction_mask.ndim != 2:
        raise ValueError(
            "Prediction mask must be a 2D array."
        )

    total_pixels = prediction_mask.size

    if total_pixels == 0:
        raise ValueError(
            "Prediction mask is empty."
        )

    class_statistics = {}

    for class_id in np.unique(prediction_mask):
        class_id = int(class_id)

        if class_id not in CLASS_NAMES:
            raise ValueError(
                f"Unknown class ID: {class_id}"
            )

        pixel_count = int(
            np.count_nonzero(
                prediction_mask == class_id
            )
        )

        percentage = (
            pixel_count / total_pixels
        ) * 100.0

        color = get_class_colour(class_id)

        color_hex = "#{:02X}{:02X}{:02X}".format(
            *color
        )

        class_statistics[
            CLASS_NAMES[class_id]
        ] = {
            "pixel_count": pixel_count,
            "percentage": round(percentage, 2),
            "color": color_hex
        }

    return class_statistics


# ============================================================
# 10. CREATE COLOURED MASK
# ============================================================

def create_colored_mask(prediction_mask):
    """
    Convert a 2D class-ID mask to an RGB PIL image.
    """

    prediction_mask = np.asarray(prediction_mask)

    if prediction_mask.ndim != 2:
        raise ValueError(
            "Prediction mask must be 2D."
        )

    unique_classes = set(
        np.unique(prediction_mask).tolist()
    )

    invalid_classes = unique_classes - set(
        CLASS_NAMES.keys()
    )

    if invalid_classes:
        raise ValueError(
            f"Unknown class IDs: {invalid_classes}"
        )

    height, width = prediction_mask.shape

    colored_mask = np.zeros(
        (height, width, 3),
        dtype=np.uint8
    )

    for class_id, color in CLASS_COLORS.items():
        colored_mask[
            prediction_mask == class_id
        ] = color

    return Image.fromarray(
        colored_mask,
        mode="RGB"
    )


def create_coloured_mask(prediction_mask):
    """British spelling compatibility."""
    return create_colored_mask(prediction_mask)


# ============================================================
# 11. CREATE SEGMENTATION OVERLAY
# ============================================================

def create_overlay(original_image, colored_mask):
    """
    Blend the original RGB image and coloured mask.
    """

    original_image = original_image.convert("RGB")
    colored_mask = colored_mask.convert("RGB")

    if original_image.size != colored_mask.size:
        raise ValueError(
            "Image and mask dimensions must match."
        )

    return Image.blend(
        original_image,
        colored_mask,
        alpha=0.45
    )


# ============================================================
# 12. COMPLETE SEGMENTATION WORKFLOW
# ============================================================

def segment_image(
    image_path,
    output_dir,
    predictor=None
):
    """
    Main function called by Flask.

    predictor:
        Optional testing function that receives a
        PIL RGB image and returns a 2D class-ID mask.

    The predictor argument preserves compatibility
    with the existing automated tests.
    """

    image_path = Path(image_path)
    output_dir = Path(output_dir)

    if not image_path.is_file():
        raise FileNotFoundError(
            f"Input image not found: {image_path}"
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    try:
        with Image.open(image_path) as uploaded:
            if uploaded.format not in ("JPEG", "PNG", "WEBP"):
                raise ValueError(
                    "Phase 3 accepts JPG and PNG images."
                )

            original_image = uploaded.convert("RGB")

        image_width, image_height = original_image.size

        if image_width <= 0 or image_height <= 0:
            raise ValueError(
                "Invalid image dimensions."
            )

        if predictor is not None:
            prediction_mask = predictor(
                original_image
            )
        else:
            prediction_mask = predict_segmentation(
                original_image
            )

        prediction_mask = np.asarray(
            prediction_mask
        )

        if prediction_mask.ndim != 2:
            raise ValueError(
                "Prediction mask must be 2D."
            )

        if prediction_mask.shape != (
            image_height,
            image_width
        ):
            raise ValueError(
                "Prediction mask dimensions do not "
                "match the original image."
            )

        if not np.issubdtype(
            prediction_mask.dtype,
            np.integer
        ):
            if (
                not np.all(np.isfinite(prediction_mask))
                or not np.all(
                    prediction_mask
                    == np.floor(prediction_mask)
                )
            ):
                raise ValueError(
                    "Prediction mask must contain "
                    "integer class IDs."
                )

        if (
            prediction_mask.min() < 0
            or prediction_mask.max() >= NUM_CLASSES
        ):
            raise ValueError(
                "Prediction contains invalid class IDs."
            )

        prediction_mask = prediction_mask.astype(
            np.uint8
        )

        class_statistics = calculate_class_statistics(
            prediction_mask
        )

        colored_mask = create_colored_mask(
            prediction_mask
        )

        overlay_image = create_overlay(
            original_image,
            colored_mask
        )

        unique_id = uuid.uuid4().hex

        mask_filename = (
            f"segmentation_{unique_id}.png"
        )

        overlay_filename = (
            f"overlay_{unique_id}.png"
        )

        colored_mask.save(
            output_dir / mask_filename,
            format="PNG"
        )

        overlay_image.save(
            output_dir / overlay_filename,
            format="PNG"
        )

        return {
            "mask_filename": mask_filename,
            "overlay_filename": overlay_filename,
            "class_statistics": class_statistics,
            "image_width": image_width,
            "image_height": image_height
        }

    except Exception as error:
        raise RuntimeError(
            f"Segmentation failed: {error}"
        ) from error


# ============================================================
# 13. DIRECT MODEL TEST
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Satellite Image Intelligence")
    print("Phase 3 — Segmentation Model Test")
    print("=" * 60)

    print("Checkpoint:", CHECKPOINT_PATH)
    print("Checkpoint exists:", CHECKPOINT_PATH.is_file())

    model = load_segmentation_model()

    print("\nModel ready for inference.")
    print("Encoder:", type(model.encoder).__name__)
    print("Device:", DEVICE)
