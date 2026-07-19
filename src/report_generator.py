"""PDF report generation for NeuroVision-AI."""

from io import BytesIO
from typing import Mapping, Sequence, Union

import numpy as np
from PIL import Image

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas


ProbabilityInput = Union[
    Mapping[str, float],
    Sequence[float],
]


def _convert_to_image_reader(
    image: Union[Image.Image, np.ndarray],
) -> ImageReader:
    """Convert a PIL or NumPy image into ReportLab format."""

    if isinstance(image, np.ndarray):
        image = Image.fromarray(
            image.astype("uint8")
        )

    if not isinstance(image, Image.Image):
        raise TypeError(
            "The report image must be a PIL image "
            "or NumPy array."
        )

    buffer = BytesIO()

    image.convert("RGB").save(
        buffer,
        format="PNG",
    )

    buffer.seek(0)

    return ImageReader(buffer)


def create_pdf_report(
    original_image: Union[Image.Image, np.ndarray],
    gradcam_image: Union[Image.Image, np.ndarray],
    predicted_class: str,
    confidence: float,
    all_probabilities: ProbabilityInput,
) -> bytes:
    """
    Create a downloadable AI analysis PDF report.
    """

    pdf_buffer = BytesIO()

    pdf = canvas.Canvas(
        pdf_buffer,
        pagesize=letter,
    )

    page_width, page_height = letter

    # Header
    pdf.setFillColor(
        colors.HexColor("#1F4E78")
    )

    pdf.rect(
        0,
        page_height - 85,
        page_width,
        85,
        fill=True,
        stroke=False,
    )

    pdf.setFillColor(colors.white)
    pdf.setFont("Helvetica-Bold", 20)

    pdf.drawString(
        40,
        page_height - 52,
        "Brain MRI AI Analysis Report",
    )

    # Prediction result
    pdf.setFillColor(colors.black)
    pdf.setFont("Helvetica-Bold", 14)

    pdf.drawString(
        40,
        page_height - 125,
        "AI Analysis Result",
    )

    pdf.setFont("Helvetica", 12)

    pdf.drawString(
        40,
        page_height - 150,
        f"Predicted category: {predicted_class}",
    )

    pdf.drawString(
        40,
        page_height - 172,
        f"Model confidence: {confidence:.2%}",
    )

    # Probabilities
    pdf.setFont("Helvetica-Bold", 12)

    pdf.drawString(
        40,
        page_height - 208,
        "Class probabilities",
    )

    pdf.setFont("Helvetica", 10)

    if isinstance(all_probabilities, Mapping):
        probability_items = (
            all_probabilities.items()
        )
    else:
        class_names = [
            "Glioma",
            "Meningioma",
            "No Tumor",
            "Pituitary Tumor",
        ]

        probability_items = zip(
            class_names,
            all_probabilities,
        )

    probability_y = page_height - 228

    for class_name, probability in probability_items:

        pdf.drawString(
            50,
            probability_y,
            f"{class_name}: {float(probability):.2%}",
        )

        probability_y -= 18

    # Original MRI
    pdf.setFont("Helvetica-Bold", 12)

    pdf.drawString(
        40,
        probability_y - 10,
        "Uploaded MRI",
    )

    original_reader = _convert_to_image_reader(
        original_image
    )

    pdf.drawImage(
        original_reader,
        40,
        probability_y - 230,
        width=220,
        height=200,
        preserveAspectRatio=True,
        anchor="c",
    )

    # Grad-CAM
    pdf.drawString(
        320,
        probability_y - 10,
        "Grad-CAM Explanation",
    )

    gradcam_reader = _convert_to_image_reader(
        gradcam_image
    )

    pdf.drawImage(
        gradcam_reader,
        320,
        probability_y - 230,
        width=220,
        height=200,
        preserveAspectRatio=True,
        anchor="c",
    )

    # Explainability note
    note_y = probability_y - 270

    pdf.setFont("Helvetica", 9)

    explanation_text = (
        "Grad-CAM highlights image regions that most influenced "
        "the model prediction. It does not represent a clinically "
        "validated tumor boundary."
    )

    pdf.drawString(
        40,
        note_y,
        explanation_text[:95],
    )

    pdf.drawString(
        40,
        note_y - 14,
        explanation_text[95:],
    )

    # Disclaimer
    pdf.setFillColor(
        colors.HexColor("#8B0000")
    )

    pdf.setFont("Helvetica-Bold", 9)

    pdf.drawString(
        40,
        65,
        "Medical disclaimer:",
    )

    pdf.setFillColor(colors.black)
    pdf.setFont("Helvetica", 8)

    pdf.drawString(
        40,
        50,
        "This software is for educational and research purposes only.",
    )

    pdf.drawString(
        40,
        38,
        "It must not be used as a replacement for professional medical diagnosis.",
    )

    pdf.save()

    pdf_buffer.seek(0)

    return pdf_buffer.getvalue()