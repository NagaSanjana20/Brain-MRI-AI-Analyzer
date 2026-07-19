"""Reusable model-evaluation functions."""

from typing import Sequence

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)


def calculate_accuracy(
    true_labels: Sequence[int],
    predicted_labels: Sequence[int],
) -> float:
    """Calculate classification accuracy."""

    return float(
        accuracy_score(
            true_labels,
            predicted_labels,
        )
    )


def create_classification_report(
    true_labels: Sequence[int],
    predicted_labels: Sequence[int],
    class_names: Sequence[str],
) -> pd.DataFrame:
    """Create a classification report DataFrame."""

    report = classification_report(
        true_labels,
        predicted_labels,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    return pd.DataFrame(report).transpose()


def create_confusion_matrix(
    true_labels: Sequence[int],
    predicted_labels: Sequence[int],
) -> np.ndarray:
    """Create a confusion matrix."""

    return confusion_matrix(
        true_labels,
        predicted_labels,
    )


def create_evaluation_summary(
    true_labels: Sequence[int],
    predicted_labels: Sequence[int],
) -> pd.DataFrame:
    """Create a compact evaluation summary."""

    total_images = len(true_labels)

    correct_predictions = sum(
        true_label == predicted_label
        for true_label, predicted_label
        in zip(
            true_labels,
            predicted_labels,
        )
    )

    incorrect_predictions = (
        total_images - correct_predictions
    )

    accuracy = calculate_accuracy(
        true_labels,
        predicted_labels,
    )

    return pd.DataFrame({
        "Metric": [
            "Total Images",
            "Correct Predictions",
            "Incorrect Predictions",
            "Accuracy",
        ],
        "Value": [
            total_images,
            correct_predictions,
            incorrect_predictions,
            accuracy,
        ],
    })