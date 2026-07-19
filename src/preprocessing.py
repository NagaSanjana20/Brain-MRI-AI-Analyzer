"""Image preprocessing utilities for NeuroVision-AI."""

from PIL import Image
import torch
from torchvision import transforms


IMAGE_SIZE = (224, 224)


# This must match the preprocessing used to train the current model.
MRI_TRANSFORM = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.Lambda(lambda image: image.convert("RGB")),
    transforms.ToTensor(),
])


def preprocess_image(image: Image.Image) -> torch.Tensor:
    """
    Convert a PIL image into a model-ready tensor.

    Returns:
        Tensor with shape [1, 3, 224, 224].
    """

    if not isinstance(image, Image.Image):
        raise TypeError("The supplied input must be a PIL image.")

    processed_tensor = MRI_TRANSFORM(image)

    # Add batch dimension:
    # [3, 224, 224] -> [1, 3, 224, 224]
    return processed_tensor.unsqueeze(0)