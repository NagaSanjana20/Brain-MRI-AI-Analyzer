"""Grad-CAM explainability utilities."""

from typing import Optional

import numpy as np
from PIL import Image
import torch.nn as nn

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import (
    ClassifierOutputTarget,
)

from src.preprocessing import (
    IMAGE_SIZE,
    preprocess_image,
)


def generate_gradcam(
    model: nn.Module,
    image: Image.Image,
    target_class: int,
    target_layer: Optional[nn.Module] = None,
) -> np.ndarray:
    """
    Generate a Grad-CAM overlay for a prediction.

    Returns:
        RGB NumPy image with values from 0 to 255.
    """

    model.eval()

    if target_layer is None:
        target_layer = model.layer4[-1]

    input_tensor = preprocess_image(image)

    resized_image = image.convert("RGB").resize(
        IMAGE_SIZE
    )

    original_image_array = np.asarray(
        resized_image,
        dtype=np.float32,
    ) / 255.0

    targets = [
        ClassifierOutputTarget(target_class)
    ]

    with GradCAM(
        model=model,
        target_layers=[target_layer],
    ) as cam:

        grayscale_cam = cam(
            input_tensor=input_tensor,
            targets=targets,
        )[0]

    visualization = show_cam_on_image(
        original_image_array,
        grayscale_cam,
        use_rgb=True,
    )

    return visualization