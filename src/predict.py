"""MRI prediction utilities."""

from typing import Any

from PIL import Image
import torch
import torch.nn as nn

from src.model import CLASS_NAMES
from src.preprocessing import preprocess_image


def predict_mri(
    model: nn.Module,
    image: Image.Image,
    device: str = "cpu",
) -> dict[str, Any]:
    """
    Predict the MRI category and return all class probabilities.

    Returns:
        Dictionary containing:
        - predicted_class
        - predicted_index
        - confidence
        - probabilities
    """

    device_object = torch.device(device)

    model = model.to(device_object)
    model.eval()

    image_tensor = preprocess_image(image)
    image_tensor = image_tensor.to(device_object)

    with torch.no_grad():
        logits = model(image_tensor)

        probabilities_tensor = torch.softmax(
            logits,
            dim=1,
        )[0]

        predicted_index = int(
            torch.argmax(probabilities_tensor).item()
        )

        confidence = float(
            probabilities_tensor[predicted_index].item()
        )

    probabilities = {
        class_name: float(probability)
        for class_name, probability in zip(
            CLASS_NAMES,
            probabilities_tensor.cpu().tolist(),
        )
    }

    return {
        "predicted_class": CLASS_NAMES[predicted_index],
        "predicted_index": predicted_index,
        "confidence": confidence,
        "probabilities": probabilities,
    }