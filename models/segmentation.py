
"""
Satellite Image Intelligence
Land-Cover Segmentation Module

Expected model checkpoint:
    checkpoints/segmentation/landcover_segmentation.pth

Expected function:
    segment_image(image_path, output_dir, predictor=None)

Returns:
    {
        "mask_filename": "...",
        "overlay_filename": "...",
        "class_statistics": {...},
        "image_width": 0,
        "image_height": 0
    }
"""

import os
import uuid
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image


# ============================================================
# 1. CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

CHECKPOINT_PATH = Path(
    os.environ.get(
        "SEGMENTATION_CHECKPOINT",
        BASE_DIR / "checkpoints" / "segmentation"
        / "landcover_segmentation.pth"
    )
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

INPUT_SIZE = 512

NUM_CLASSES = 5

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


# ============================================================
# 2. COLOUR HELPER FUNCTIONS
# ============================================================

def get_class_colour(class_id):
    """
    Returns the RGB colour for a class ID.
    """

    return CLASS_COLORS.get(
        int(class_id),
        (255, 255, 255)
    )


def get_class_color(class_id):
    """
    American spelling compatibility.
    """

    return get_class_colour(class_id)


# ============================================================
# 3. SEGMENTATION MODEL ARCHITECTURE
# ============================================================

class ConvBlock(nn.Module):
    """
    Two convolution layers with batch normalization
    and ReLU activation.
    """

    def __init__(self, input_channels, output_channels):

        super().__init__()

        self.block = nn.Sequential(

            nn.Conv2d(
                input_channels,
                output_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(output_channels),

            nn.ReLU(inplace=True),

            nn.Conv2d(
                output_channels,
                output_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(output_channels),

            nn.ReLU(inplace=True)
        )

    def forward(self, x):

        return self.block(x)


class LandCoverSegmentationModel(nn.Module):
    """
    Lightweight encoder-decoder segmentation model.

    Input:
        RGB image tensor [B, 3, H, W]

    Output:
        Class logits [B, 5, H, W]
    """

    def __init__(self, num_classes=NUM_CLASSES):

        super().__init__()

        # Encoder
        self.encoder1 = ConvBlock(3, 32)
        self.encoder2 = ConvBlock(32, 64)
        self.encoder3 = ConvBlock(64, 128)

        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = ConvBlock(128, 256)

        # Decoder
        self.decoder3 = ConvBlock(256 + 128, 128)
        self.decoder2 = ConvBlock(128 + 64, 64)
        self.decoder1 = ConvBlock(64 + 32, 32)

        # Classification layer
        self.output_layer = nn.Conv2d(
            32,
            num_classes,
            kernel_size=1
        )

    def forward(self, x):

        # Encoder
        e1 = self.encoder1(x)

        e2 = self.encoder2(
            self.pool(e1)
        )

        e3 = self.encoder3(
            self.pool(e2)
        )

        # Bottleneck
        b = self.bottleneck(
            self.pool(e3)
        )

        # Decoder level 3
        d3 = F.interpolate(
            b,
            size=e3.shape[2:],
            mode="bilinear",
            align_corners=False
        )

        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.decoder3(d3)

        # Decoder level 2
        d2 = F.interpolate(
            d3,
            size=e2.shape[2:],
            mode="bilinear",
            align_corners=False
        )

        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.decoder2(d2)

        # Decoder level 1
        d1 = F.interpolate(
            d2,
            size=e1.shape[2:],
            mode="bilinear",
            align_corners=False
        )

        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.decoder1(d1)

        return self.output_layer(d1)


# ============================================================
# 4. MODEL LOADING
# ============================================================

_model = None


def load_segmentation_model():
    """
    Loads the trained segmentation model once.

    Supports:
    - TorchScript model.
    - Complete saved nn.Module.
    - State dictionary matching the architecture above.
    """

    global _model

    if _model is not None:
        return _model

    if not CHECKPOINT_PATH.is_file():

        raise FileNotFoundError(
            "Segmentation checkpoint was not found.\n"
            f"Expected location: {CHECKPOINT_PATH}\n"
            "Place your trained model checkpoint at this location."
        )

    try:

        # Try TorchScript first.
        try:

            model = torch.jit.load(
                str(CHECKPOINT_PATH),
                map_location=DEVICE
            )

            model.eval()
            _model = model

            print("TorchScript segmentation model loaded.")

            return _model

        except (RuntimeError, ValueError, OSError):
            pass

        # Load a trusted local checkpoint.
        try:

            checkpoint = torch.load(
                str(CHECKPOINT_PATH),
                map_location=DEVICE,
                weights_only=False
            )

        except TypeError:

            checkpoint = torch.load(
                str(CHECKPOINT_PATH),
                map_location=DEVICE
            )

        # Full model checkpoint.
        if isinstance(checkpoint, nn.Module):

            model = checkpoint

        else:

            # Extract state dictionary.
            if isinstance(checkpoint, dict):

                if "model_state_dict" in checkpoint:

                    state_dict = checkpoint["model_state_dict"]

                elif "state_dict" in checkpoint:

                    state_dict = checkpoint["state_dict"]

                else:

                    state_dict = checkpoint

            else:

                raise ValueError(
                    "Unsupported checkpoint format."
                )

            if not isinstance(state_dict, dict):

                raise ValueError(
                    "Checkpoint does not contain a valid state dictionary."
                )

            # Remove DataParallel prefixes.
            cleaned_state_dict = {}

            for key, value in state_dict.items():

                if key.startswith("module."):

                    key = key[len("module."):]

                cleaned_state_dict[key] = value

            # Initialize architecture.
            model = LandCoverSegmentationModel(
                num_classes=NUM_CLASSES
            )

            try:

                model.load_state_dict(
                    cleaned_state_dict,
                    strict=True
                )

            except RuntimeError as error:

                raise RuntimeError(
                    "The checkpoint architecture does not match "
                    "LandCoverSegmentationModel.\n"
                    "The model architecture and class count must "
                    "match the architecture used during training.\n"
                    f"Original error: {error}"
                ) from error

        model = model.to(DEVICE)
        model.eval()

        _model = model

        print(
            f"Segmentation model loaded successfully on {DEVICE}."
        )

        return _model

    except Exception as error:

        _model = None

        raise RuntimeError(
            f"Could not load segmentation model: {error}"
        ) from error


# ============================================================
# 5. IMAGE PREPROCESSING
# ============================================================

def preprocess_image(image):
    """
    Converts an RGB PIL image into a normalized PyTorch tensor.
    """

    resized_image = image.resize(
        (INPUT_SIZE, INPUT_SIZE),
        Image.Resampling.BILINEAR
    )

    image_array = np.asarray(
        resized_image,
        dtype=np.float32
    ) / 255.0

    # ImageNet normalization.
    mean = np.array(
        [0.485, 0.456, 0.406],
        dtype=np.float32
    )

    std = np.array(
        [0.229, 0.224, 0.225],
        dtype=np.float32
    )

    image_array = (image_array - mean) / std

    # HWC -> CHW
    image_array = np.transpose(
        image_array,
        (2, 0, 1)
    )

    tensor = torch.from_numpy(
        image_array
    ).unsqueeze(0)

    return tensor.to(DEVICE)


# ============================================================
# 6. PREDICTION
# ============================================================

def predict_segmentation(image):
    """
    Runs model inference and returns a class-ID mask.

    Output:
        NumPy array with shape [original_height, original_width].
    """

    model = load_segmentation_model()

    original_width, original_height = image.size

    input_tensor = preprocess_image(image)

    with torch.inference_mode():

        prediction = model(input_tensor)

        # Dictionary output support.
        if isinstance(prediction, dict):

            if "out" in prediction:

                prediction = prediction["out"]

            else:

                raise ValueError(
                    "Model output dictionary must contain an 'out' key."
                )

        # Tuple/list output support.
        if isinstance(prediction, (tuple, list)):

            prediction = prediction[0]

        if not isinstance(prediction, torch.Tensor):

            raise ValueError(
                "The segmentation model did not return a tensor."
            )

        if prediction.ndim != 4:

            raise ValueError(
                "Expected model output shape [B, C, H, W], "
                f"but received {tuple(prediction.shape)}."
            )

        if prediction.shape[1] != NUM_CLASSES:

            raise ValueError(
                f"Expected {NUM_CLASSES} output classes, "
                f"but the model returned {prediction.shape[1]}."
            )

        # Resize logits to original dimensions.
        prediction = F.interpolate(
            prediction,
            size=(original_height, original_width),
            mode="bilinear",
            align_corners=False
        )

        prediction_mask = torch.argmax(
            prediction,
            dim=1
        )

    return prediction_mask.squeeze(0).cpu().numpy().astype(
        np.uint8
    )


# ============================================================
# 7. CLASS STATISTICS
# ============================================================

def calculate_class_statistics(prediction_mask):
    """
    Calculates pixel counts and percentages for
    classes actually present in the prediction.
    """

    prediction_mask = np.asarray(prediction_mask)

    total_pixels = prediction_mask.size

    if total_pixels == 0:

        raise ValueError(
            "Prediction mask contains no pixels."
        )

    class_statistics = {}

    predicted_class_ids = np.unique(prediction_mask)

    for class_id in predicted_class_ids:

        class_id = int(class_id)

        if class_id not in CLASS_NAMES:

            raise ValueError(
                f"Unknown class ID: {class_id}"
            )

        class_name = CLASS_NAMES[class_id]

        pixel_count = int(
            np.count_nonzero(
                prediction_mask == class_id
            )
        )

        percentage = (
            pixel_count / total_pixels
        ) * 100

        color = get_class_colour(class_id)

        class_statistics[class_name] = {
            "pixel_count": pixel_count,
            "percentage": round(percentage, 2),
            "color": "#{:02X}{:02X}{:02X}".format(
                *color
            )
        }

    return class_statistics


# ============================================================
# 8. CREATE COLOURED SEGMENTATION MASK
# ============================================================

def create_colored_mask(prediction_mask):
    """
    Converts class IDs into an RGB segmentation image.
    """

    prediction_mask = np.asarray(prediction_mask)

    if prediction_mask.ndim != 2:

        raise ValueError(
            "Prediction mask must be a 2D array."
        )

    invalid_classes = (
        set(np.unique(prediction_mask).tolist())
        - set(CLASS_NAMES.keys())
    )

    if invalid_classes:

        raise ValueError(
            f"Unknown class IDs in mask: {invalid_classes}"
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
    """
    British spelling compatibility for existing tests.
    """

    return create_colored_mask(prediction_mask)


# ============================================================
# 9. CREATE ORIGINAL IMAGE OVERLAY
# ============================================================

def create_overlay(original_image, colored_mask):
    """
    Blends the original satellite image with the
    coloured segmentation mask.
    """

    original_image = original_image.convert("RGB")

    colored_mask = colored_mask.convert("RGB")

    if original_image.size != colored_mask.size:

        raise ValueError(
            "Original image and segmentation mask "
            "must have identical dimensions."
        )

    overlay = Image.blend(
        original_image,
        colored_mask,
        alpha=0.45
    )

    return overlay


# ============================================================
# 10. MAIN SEGMENTATION FUNCTION
# ============================================================

def segment_image(image_path, output_dir, predictor=None):
    """
    Complete land-cover segmentation workflow.

    Args:
        image_path:
            Path to the uploaded RGB satellite image.

        output_dir:
            Folder where generated files are saved.

        predictor:
            Optional custom prediction function.
            Used for testing. Receives a PIL RGB image
            and must return a 2D class-ID NumPy array.

    Returns:
        Dictionary containing:
            mask_filename
            overlay_filename
            class_statistics
            image_width
            image_height
    """

    image_path = Path(image_path)

    output_dir = Path(output_dir)

    # Validate input image before entering try block.
    if not image_path.is_file():

        raise FileNotFoundError(
            f"Input image not found: {image_path}"
        )

    # Create output directory.
    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    try:

        # Open uploaded image.
        with Image.open(image_path) as uploaded_image:

            if uploaded_image.format not in ["JPEG", "PNG"]:
                raise ValueError(
                    "Segmentation requires a JPG or PNG image."
                )

            original_image = uploaded_image.convert("RGB")

        image_width, image_height = original_image.size

        if image_width == 0 or image_height == 0:

            raise ValueError(
                "Uploaded image has invalid dimensions."
            )

        # Generate prediction.
        if predictor is not None:

            prediction_mask = predictor(original_image)

        else:

            prediction_mask = predict_segmentation(
                original_image
            )

        # Convert prediction to NumPy array.
        prediction_mask = np.asarray(
            prediction_mask
        )

        # Validate prediction shape.
        if prediction_mask.ndim != 2:

            raise ValueError(
                "Prediction mask must be a 2D array."
            )

        if prediction_mask.shape != (
            image_height,
            image_width
        ):

            raise ValueError(
                "Prediction mask dimensions do not match "
                "the original image dimensions."
            )

        # Validate that class IDs are integers.
        if not np.issubdtype(
            prediction_mask.dtype,
            np.integer
        ):

            if not np.all(
                np.isfinite(prediction_mask)
            ) or not np.all(
                prediction_mask == np.floor(prediction_mask)
            ):

                raise ValueError(
                    "Prediction mask must contain integer class IDs."
                )

        # Convert to uint8 after checking values.
        if prediction_mask.min() < 0 or prediction_mask.max() > 255:

            raise ValueError(
                "Prediction mask contains class IDs outside "
                "the supported range."
            )

        prediction_mask = prediction_mask.astype(
            np.uint8
        )

        # Validate class IDs.
        invalid_classes = (
            set(np.unique(prediction_mask).tolist())
            - set(CLASS_NAMES.keys())
        )

        if invalid_classes:

            raise ValueError(
                f"Unknown class IDs in prediction: {invalid_classes}"
            )

        # Calculate class statistics.
        class_statistics = calculate_class_statistics(
            prediction_mask
        )

        # Generate coloured mask.
        colored_mask = create_colored_mask(
            prediction_mask
        )

        # Generate overlay.
        overlay_image = create_overlay(
            original_image,
            colored_mask
        )

        # Unique output filenames.
        unique_id = uuid.uuid4().hex

        mask_filename = f"segmentation_{unique_id}.png"

        overlay_filename = f"overlay_{unique_id}.png"

        mask_path = output_dir / mask_filename

        overlay_path = output_dir / overlay_filename

        # Save mask.
        colored_mask.save(
            mask_path,
            format="PNG"
        )

        # Save overlay.
        overlay_image.save(
            overlay_path,
            format="PNG"
        )

        # Return results to Flask.
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
