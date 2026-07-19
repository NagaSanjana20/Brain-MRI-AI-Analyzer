"""Model creation and loading utilities."""

from pathlib import Path
from typing import Union

import torch
import torch.nn as nn
from torchvision import models


NUMBER_OF_CLASSES = 4

CLASS_NAMES = [
    "Glioma",
    "Meningioma",
    "No Tumor",
    "Pituitary Tumor",
]


def create_model() -> nn.Module:
    """
    Create the ResNet18 architecture used during training.
    """

    model = models.resnet18(weights=None)

    model.fc = nn.Linear(
        in_features=model.fc.in_features,
        out_features=NUMBER_OF_CLASSES,
    )

    return model


def load_trained_model(
    model_path: Union[str, Path],
    device: Union[str, torch.device] = "cpu",
) -> nn.Module:
    """
    Load the saved ResNet18 weights.

    Args:
        model_path: Path to the .pth model file.
        device: CPU or CUDA device.

    Returns:
        Loaded model in evaluation mode.
    """

    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file was not found: {model_path}"
        )

    device = torch.device(device)

    model = create_model()

    state_dict = torch.load(
        model_path,
        map_location=device,
    )

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model