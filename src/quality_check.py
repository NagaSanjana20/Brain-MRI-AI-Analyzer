"""Basic uploaded-image quality checks."""

from typing import Any

from PIL import Image


MINIMUM_WIDTH = 100
MINIMUM_HEIGHT = 100


def check_image_quality(
    image: Image.Image,
) -> dict[str, Any]:
    """
    Perform basic technical checks on an uploaded image.

    This does not confirm that the image is a valid clinical MRI.
    """

    warnings = []
    errors = []

    if not isinstance(image, Image.Image):
        return {
            "is_valid": False,
            "errors": ["The uploaded file is not a readable image."],
            "warnings": [],
        }

    width, height = image.size

    if width < MINIMUM_WIDTH or height < MINIMUM_HEIGHT:
        errors.append(
            "The image resolution is too small. "
            f"Minimum recommended size is "
            f"{MINIMUM_WIDTH} × {MINIMUM_HEIGHT} pixels."
        )

    if width != height:
        warnings.append(
            "The uploaded image is not square. "
            "It will be resized to 224 × 224 pixels."
        )

    if image.mode not in ("L", "RGB", "RGBA"):
        warnings.append(
            f"The image uses {image.mode} color mode. "
            "It will be converted to RGB."
        )

    return {
        "is_valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "width": width,
        "height": height,
        "mode": image.mode,
    }