"""Reusable training utilities."""

from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def train_one_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str = "cpu",
) -> float:
    """
    Train the model for one epoch.

    Returns:
        Average training loss.
    """

    device_object = torch.device(device)

    model.to(device_object)
    model.train()

    running_loss = 0.0

    for images, labels in data_loader:

        images = images.to(device_object)
        labels = labels.to(device_object)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            labels,
        )

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    average_loss = (
        running_loss / len(data_loader)
    )

    return average_loss


def evaluate_model_accuracy(
    model: nn.Module,
    data_loader: DataLoader,
    device: str = "cpu",
) -> float:
    """
    Evaluate model accuracy on a DataLoader.
    """

    device_object = torch.device(device)

    model.to(device_object)
    model.eval()

    correct_predictions = 0
    total_images = 0

    with torch.no_grad():

        for images, labels in data_loader:

            images = images.to(device_object)
            labels = labels.to(device_object)

            outputs = model(images)

            predictions = torch.argmax(
                outputs,
                dim=1,
            )

            correct_predictions += (
                predictions == labels
            ).sum().item()

            total_images += labels.size(0)

    if total_images == 0:
        return 0.0

    return correct_predictions / total_images


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    number_of_epochs: int = 5,
    device: str = "cpu",
) -> Tuple[list[float], list[float]]:
    """
    Run the complete model training loop.

    Returns:
        training_losses and validation_accuracies.
    """

    training_losses = []
    validation_accuracies = []

    for epoch in range(number_of_epochs):

        average_loss = train_one_epoch(
            model=model,
            data_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        validation_accuracy = (
            evaluate_model_accuracy(
                model=model,
                data_loader=validation_loader,
                device=device,
            )
        )

        training_losses.append(
            average_loss
        )

        validation_accuracies.append(
            validation_accuracy
        )

        print(
            f"Epoch {epoch + 1}/{number_of_epochs} | "
            f"Loss: {average_loss:.4f} | "
            f"Validation Accuracy: "
            f"{validation_accuracy:.2%}"
        )

    return (
        training_losses,
        validation_accuracies,
    )