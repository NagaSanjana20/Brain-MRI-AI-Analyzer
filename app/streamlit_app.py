from pathlib import Path

import numpy as np
import streamlit as st
import torch
import torch.nn as nn

from PIL import Image
from torchvision import models, transforms

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from io import BytesIO
from datetime import datetime

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

# --------------------------------------------------
# Page configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Brain MRI AI Analyzer",
    page_icon="🧠",
    layout="wide"
)


# --------------------------------------------------
# Project paths
# --------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "resnet18_brain_tumor.pth"
)

# --------------------------------------------------
# External validation summary
# These values come from evaluation on 200 outside MRI images
# --------------------------------------------------
OUTSIDE_VALIDATION_TOTAL = 200
OUTSIDE_VALIDATION_CORRECT = 184
OUTSIDE_VALIDATION_INCORRECT = 16
OUTSIDE_VALIDATION_ACCURACY = 0.92

OUTSIDE_CLASS_RECALL = {
    "Glioma": 0.70,
    "Meningioma": 0.98,
    "No Tumor": 1.00,
    "Pituitary Tumor": 1.00
}

# --------------------------------------------------
# Class names
# Keep the same order used during training
# --------------------------------------------------
CLASS_NAMES = [
    "Glioma",
    "Meningioma",
    "No Tumor",
    "Pituitary Tumor"
]


# --------------------------------------------------
# Image preprocessing
# --------------------------------------------------
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Lambda(
        lambda image: image.convert("RGB")
    ),
    transforms.ToTensor()
])


# --------------------------------------------------
# Load trained model
# --------------------------------------------------
@st.cache_resource
def load_trained_model():
    model = models.resnet18(weights=None)

    model.fc = nn.Linear(
        in_features=512,
        out_features=4
    )

    state_dict = torch.load(
        MODEL_PATH,
        map_location="cpu"
    )

    model.load_state_dict(state_dict)
    model.eval()

    return model


# --------------------------------------------------
# Predict uploaded MRI
# --------------------------------------------------
def predict_mri(model, original_image):
    input_tensor = image_transform(
        original_image
    )

    input_tensor = input_tensor.unsqueeze(0)

    with torch.no_grad():
        outputs = model(input_tensor)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        predicted_index = torch.argmax(
            probabilities,
            dim=1
        ).item()

    predicted_class = CLASS_NAMES[
        predicted_index
    ]

    confidence = probabilities[
        0,
        predicted_index
    ].item()

    all_probabilities = probabilities[
        0
    ].tolist()

    return (
        predicted_index,
        predicted_class,
        confidence,
        all_probabilities
    )


# --------------------------------------------------
# Generate Grad-CAM explanation
# --------------------------------------------------
def generate_gradcam(
    model,
    original_image,
    predicted_index
):
    input_tensor = image_transform(
        original_image
    )

    input_tensor = input_tensor.unsqueeze(0)

    target_layers = [
        model.layer4[-1]
    ]

    cam = GradCAM(
        model=model,
        target_layers=target_layers
    )

    grayscale_cam = cam(
        input_tensor=input_tensor,
        targets=[
            ClassifierOutputTarget(
                predicted_index
            )
        ]
    )

    grayscale_cam = grayscale_cam[0]

    display_image = original_image.resize(
        (224, 224)
    )

    rgb_image = (
        np.array(
            display_image,
            dtype=np.float32
        )
        / 255.0
    )

    visualization = show_cam_on_image(
        rgb_image,
        grayscale_cam,
        use_rgb=True
    )

    return visualization

def create_pdf_report(
    original_image,
    gradcam_image,
    predicted_class,
    confidence,
    all_probabilities
):
    pdf_buffer = BytesIO()

    pdf = canvas.Canvas(
        pdf_buffer,
        pagesize=letter
    )

    page_width, page_height = letter

    # --------------------------------------------------
    # Document information
    # --------------------------------------------------
    pdf.setTitle("Brain MRI AI Analysis Report")
    generated_time = datetime.now()

    # --------------------------------------------------
    # Colors
    # --------------------------------------------------
    dark_blue = colors.HexColor("#123B63")
    medical_blue = colors.HexColor("#1976D2")
    light_blue = colors.HexColor("#EAF4FF")
    very_light_blue = colors.HexColor("#F6FAFE")
    dark_text = colors.HexColor("#1F2937")
    muted_text = colors.HexColor("#5F6B7A")
    border_color = colors.HexColor("#D6E3F0")
    warning_background = colors.HexColor("#FFF7E6")
    warning_border = colors.HexColor("#F4B740")

    # --------------------------------------------------
    # Background
    # --------------------------------------------------
    pdf.setFillColor(colors.white)
    pdf.rect(
        0,
        0,
        page_width,
        page_height,
        fill=1,
        stroke=0
    )

    # --------------------------------------------------
    # Header banner
    # --------------------------------------------------
    pdf.setFillColor(dark_blue)
    pdf.rect(
        0,
        page_height - 105,
        page_width,
        105,
        fill=1,
        stroke=0
    )

    pdf.setFillColor(colors.white)
    pdf.setFont("Helvetica-Bold", 22)

    pdf.drawString(
        42,
        page_height - 48,
        "Brain MRI AI Analyzer"
    )

    pdf.setFont("Helvetica", 11)

    pdf.drawString(
        42,
        page_height - 68,
        "AI-Assisted Brain MRI Analysis Report"
    )

    pdf.setFont("Helvetica", 9)

    pdf.drawRightString(
        page_width - 42,
        page_height - 47,
        generated_time.strftime("%B %d, %Y")
    )

    pdf.drawRightString(
        page_width - 42,
        page_height - 64,
        generated_time.strftime("%I:%M %p")
    )

    # --------------------------------------------------
    # Report information card
    # --------------------------------------------------
    card_top = page_height - 130
    card_height = 94

    pdf.setFillColor(light_blue)
    pdf.setStrokeColor(border_color)
    pdf.roundRect(
        42,
        card_top - card_height,
        page_width - 84,
        card_height,
        12,
        fill=1,
        stroke=1
    )

    pdf.setFillColor(dark_text)
    pdf.setFont("Helvetica-Bold", 13)
    pdf.drawString(
        60,
        card_top - 25,
        "AI Analysis Result"
    )

    pdf.setFont("Helvetica", 10)
    pdf.setFillColor(muted_text)
    pdf.drawString(
        60,
        card_top - 44,
        "Predicted condition"
    )

    pdf.setFillColor(dark_blue)
    pdf.setFont("Helvetica-Bold", 17)
    pdf.drawString(
        60,
        card_top - 66,
        predicted_class
    )

    pdf.setFont("Helvetica", 10)
    pdf.setFillColor(muted_text)
    pdf.drawString(
        365,
        card_top - 44,
        "Model confidence"
    )

    pdf.setFillColor(medical_blue)
    pdf.setFont("Helvetica-Bold", 17)
    pdf.drawString(
        365,
        card_top - 66,
        f"{confidence:.2%}"
    )

    # --------------------------------------------------
    # Class probability section
    # --------------------------------------------------
    probability_title_y = card_top - card_height - 28

    pdf.setFillColor(dark_text)
    pdf.setFont("Helvetica-Bold", 13)
    pdf.drawString(
        42,
        probability_title_y,
        "Class Probability Summary"
    )

    table_top = probability_title_y - 14
    row_height = 26
    table_width = page_width - 84
    label_width = 250

    for index, (class_name, probability) in enumerate(
        zip(CLASS_NAMES, all_probabilities)
    ):
        row_y = table_top - ((index + 1) * row_height)

        if index % 2 == 0:
            pdf.setFillColor(very_light_blue)
        else:
            pdf.setFillColor(colors.white)

        pdf.setStrokeColor(border_color)
        pdf.rect(
            42,
            row_y,
            table_width,
            row_height,
            fill=1,
            stroke=1
        )

        pdf.setFillColor(dark_text)
        pdf.setFont("Helvetica", 10)

        pdf.drawString(
            54,
            row_y + 8,
            class_name
        )

        # Probability bar background
        bar_x = 42 + label_width
        bar_y = row_y + 8
        bar_width = 210
        bar_height = 9

        pdf.setFillColor(colors.HexColor("#DCE8F3"))
        pdf.roundRect(
            bar_x,
            bar_y,
            bar_width,
            bar_height,
            4,
            fill=1,
            stroke=0
        )

        pdf.setFillColor(medical_blue)
        pdf.roundRect(
            bar_x,
            bar_y,
            max(2, bar_width * probability),
            bar_height,
            4,
            fill=1,
            stroke=0
        )

        pdf.setFillColor(dark_text)
        pdf.setFont("Helvetica-Bold", 9)

        pdf.drawRightString(
            page_width - 54,
            row_y + 8,
            f"{probability:.2%}"
        )

    # --------------------------------------------------
    # Convert images to memory buffers
    # --------------------------------------------------
    original_buffer = BytesIO()

    original_image.resize(
        (224, 224)
    ).save(
        original_buffer,
        format="PNG"
    )

    original_buffer.seek(0)

    gradcam_pil = Image.fromarray(
        gradcam_image
    )

    gradcam_buffer = BytesIO()

    gradcam_pil.save(
        gradcam_buffer,
        format="PNG"
    )

    gradcam_buffer.seek(0)

    # --------------------------------------------------
    # MRI image section
    # --------------------------------------------------
    image_heading_y = table_top - (4 * row_height) - 32

    pdf.setFillColor(dark_text)
    pdf.setFont("Helvetica-Bold", 13)

    pdf.drawString(
        42,
        image_heading_y,
        "MRI Visual Analysis"
    )

    image_y = image_heading_y - 185
    image_size = 170

    # Image cards
    pdf.setFillColor(very_light_blue)
    pdf.setStrokeColor(border_color)

    pdf.roundRect(
        42,
        image_y - 26,
        238,
        210,
        10,
        fill=1,
        stroke=1
    )

    pdf.roundRect(
        332,
        image_y - 26,
        238,
        210,
        10,
        fill=1,
        stroke=1
    )

    pdf.setFillColor(dark_text)
    pdf.setFont("Helvetica-Bold", 10)

    pdf.drawCentredString(
        161,
        image_y + 165,
        "Original MRI"
    )

    pdf.drawCentredString(
        451,
        image_y + 165,
        "Grad-CAM Explanation"
    )

    pdf.drawImage(
        ImageReader(original_buffer),
        76,
        image_y - 7,
        width=image_size,
        height=image_size,
        preserveAspectRatio=True,
        mask="auto"
    )

    pdf.drawImage(
        ImageReader(gradcam_buffer),
        366,
        image_y - 7,
        width=image_size,
        height=image_size,
        preserveAspectRatio=True,
        mask="auto"
    )

    # --------------------------------------------------
    # Interpretation box
    # --------------------------------------------------
    explanation_y = image_y - 62

    pdf.setFillColor(light_blue)
    pdf.setStrokeColor(border_color)

    pdf.roundRect(
        42,
        explanation_y - 55,
        page_width - 84,
        55,
        10,
        fill=1,
        stroke=1
    )

    pdf.setFillColor(dark_text)
    pdf.setFont("Helvetica-Bold", 10)

    pdf.drawString(
        56,
        explanation_y - 17,
        "Explainability note"
    )

    pdf.setFont("Helvetica", 8.5)
    pdf.setFillColor(muted_text)

    explanation_lines = [
        "The Grad-CAM image highlights regions that most influenced the model's prediction.",
        "It is an explanation of model attention and not a clinically validated tumor boundary."
    ]

    text_y = explanation_y - 32

    for line in explanation_lines:
        pdf.drawString(
            56,
            text_y,
            line
        )
        text_y -= 12

    # --------------------------------------------------
    # Medical disclaimer
    # --------------------------------------------------
    disclaimer_y = explanation_y - 130

    pdf.setFillColor(warning_background)
    pdf.setStrokeColor(warning_border)

    pdf.roundRect(
        42,
        disclaimer_y,
        page_width - 84,
        72,
        10,
        fill=1,
        stroke=1
    )

    pdf.setFillColor(colors.HexColor("#7A4E00"))
    pdf.setFont("Helvetica-Bold", 10)

    pdf.drawString(
        56,
        disclaimer_y + 52,
        "Important Medical Disclaimer"
    )

    pdf.setFont("Helvetica", 8.5)

    disclaimer_lines = [
        "This report was generated by an educational artificial intelligence system.",
        "It is not a medical diagnosis and must not replace evaluation by a qualified",
        "radiologist, neurologist, oncologist, or other healthcare professional."
    ]

    disclaimer_text_y = disclaimer_y + 36

    for line in disclaimer_lines:
        pdf.drawString(
            56,
            disclaimer_text_y,
            line
        )
        disclaimer_text_y -= 11

    # --------------------------------------------------
    # Footer
    # --------------------------------------------------
    pdf.setStrokeColor(border_color)
    pdf.line(
        42,
        30,
        page_width - 42,
        30
    )

    pdf.setFillColor(muted_text)
    pdf.setFont("Helvetica", 7.5)

    pdf.drawString(
        42,
        17,
        "Generated by Brain MRI AI Analyzer"
    )

    pdf.drawRightString(
        page_width - 42,
        17,
        "Educational and research use only"
    )

    pdf.save()
    pdf_buffer.seek(0)

    return pdf_buffer.getvalue()

# --------------------------------------------------
# Session state
# --------------------------------------------------
if "page" not in st.session_state:
    st.session_state.page = "home"


# --------------------------------------------------
# Custom styling
# --------------------------------------------------
st.markdown(
    """
    <style>
    .hero-title {
        font-size: 3.2rem;
        font-weight: 700;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
        text-align: center;
    }

    .hero-subtitle {
        font-size: 1.25rem;
        color: #5f6b7a;
        margin-bottom: 2rem;
        text-align: center;
    }

    .brain-icon {
        font-size: 7rem;
        margin-top: 2rem;
        margin-bottom: 1rem;
        text-align: center;
    }

    .center-box {
        max-width: 850px;
        margin: 0 auto;
        text-align: center;
    }

    div.stButton > button {
        background-color: #0A84FF;
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.9rem 2rem;
        font-size: 1.1rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }

    div.stButton > button:hover {
        background-color: #0066CC;
        color: white;
        transform: scale(1.03);
    }

    div.stButton > button:focus {
        border: none;
        box-shadow: none;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# -----------------------------
# Page 1: Home
# -----------------------------
if st.session_state.page == "home":

    left, center, right = st.columns([1, 2.8, 1])

    with center:

        st.markdown(
            """
            <div style="text-align:center;">

            <div class="brain-icon">🧠</div>

            <div class="hero-title">
                Brain MRI AI Analyzer
            </div>

            <div class="hero-subtitle">
                AI-Powered Brain MRI Analysis System
            </div>

            </div>
            """,
            unsafe_allow_html=True
        )

        st.info("""
### Welcome to **Brain MRI AI Analyzer**

This application uses a trained Artificial Intelligence model to analyze brain MRI images and classify them into one of four categories:
• Glioma
• Meningioma
• No Tumor
• Pituitary Tumor

---

### After the analysis, the application will display:
 • Predicted Class
 • Confidence Score
 • Class Probabilities
 • Grad-CAM Explainability Heatmap
""")
        st.success(
            "External validation completed: 184 of 200 outside MRI images "
            "were classified correctly, giving an overall accuracy of 92%."
        )

        st.write("")

        if st.button(
            "🧠 Start AI Analysis",
            use_container_width=True
        ):
            st.session_state.page = "analysis"
            st.rerun()



# --------------------------------------------------
# Page 2: Analysis
# --------------------------------------------------
elif st.session_state.page == "analysis":

    st.markdown(
        '<div class="hero-title">Brain MRI Analysis</div>',
        unsafe_allow_html=True
    )

    information_left, information_center, information_right = (
        st.columns([1, 3, 1])
    )

    with information_center:
        st.info(
            "This AI application analyzes an uploaded brain MRI and "
            "classifies it as Glioma, Meningioma, No Tumor, or "
            "Pituitary Tumor. After analysis, it will display the "
            "predicted class, confidence score, class probabilities, "
            "and an explainability heatmap."
        )

        st.warning(
            "This application is an educational AI demonstration. "
            "It is not a medical diagnosis and must not replace "
            "evaluation by a qualified healthcare professional."
        )
        
        st.success(
            "External validation result: 92%accuracy on 200 outside MRI "
            "images, with 50 images from each supported category."
        )

        st.warning(
            "Known limitation: Glioma was the most difficult category in "
            "outside validation, with 70% recall. Some glioma scans were "
            "misclassified as meningioma or no tumor."
        )

    st.write("")

    upload_left, upload_center, upload_right = st.columns(
        [1, 2, 1]
    )

    with upload_center:
        uploaded_file = st.file_uploader(
            "Upload Brain MRI",
            type=["jpg", "jpeg", "png"]
        )


    # --------------------------------------------------
    # Display and analyze uploaded MRI
    # --------------------------------------------------
    if uploaded_file is not None:

        try:
            original_image = Image.open(
                uploaded_file
            ).convert("RGB")

        except Exception:
            st.error(
                "The selected file could not be opened. "
                "Please upload a valid JPG, JPEG, or PNG image."
            )

        else:
            st.write("")

            image_left, image_center, image_right = st.columns(
                [1, 2, 1]
            )

            with image_center:
                st.image(
                    original_image,
                    caption="Uploaded Brain MRI",
                    use_container_width=True
                )

                analyze_button = st.button(
                    "Analyze MRI",
                    type="primary",
                    use_container_width=True
                )


            if analyze_button:

                if not MODEL_PATH.exists():
                    st.error(
                        "The trained model file was not found at:\n\n"
                        f"{MODEL_PATH}"
                    )

                else:
                    with st.spinner(
                        "Analyzing the MRI image..."
                    ):
                        model = load_trained_model()

                        (
                            predicted_index,
                            predicted_class,
                            confidence,
                            all_probabilities
                        ) = predict_mri(
                            model,
                            original_image
                        )

                        gradcam_image = generate_gradcam(
                            model,
                            original_image,
                            predicted_index
                        )

                    st.write("")

                    result_left, result_center, result_right = (
                        st.columns([1, 3, 1])
                    )

                    with result_center:
                        st.success(
                            "MRI analysis completed."
                        )

                        st.markdown(
                            "## AI Analysis Result"
                        )

                        prediction_column, confidence_column = (
                            st.columns(2)
                        )

                        with prediction_column:
                            st.metric(
                                label="Predicted Class",
                                value=predicted_class
                            )

                        with confidence_column:
                            st.metric(
                                label="Confidence",
                                value=f"{confidence:.2%}"
                            )

                        st.markdown(
                            "### Class Probabilities"
                        )

                        for class_name, probability in zip(
                            CLASS_NAMES,
                            all_probabilities
                        ):
                            st.write(
                                f"**{class_name}: "
                                f"{probability:.2%}**"
                            )

                            st.progress(
                                float(probability)
                            )

                        st.info(
                            "The confidence score represents the "
                            "model's calculated certainty. It is not a medical dignosis "
                            "and should not be used as a substitute for "
                            "professional medical evaluation."
                        )

                        st.markdown(
                            "### Explainability Heatmap"
                        )

                        original_column, heatmap_column = st.columns(
                            2
                        )

                        with original_column:
                            st.image(
                                original_image,
                                caption="Original MRI",
                                use_container_width=True
                            )

                        with heatmap_column:
                            st.image(
                                gradcam_image,
                                caption="Grad-CAM Explanation",
                                use_container_width=True
                            )

                        st.info(
                            "The Red highlighted areas show the most of the tumor regions, " 
                            "yellow highlighted region shows the moderate tumor regions, "
                            "and the blue highligted regions shows the least tumor regions."
                            " They do not represent a "
                            "clinically validated tumor boundary."
                        )
                        pdf_report = create_pdf_report(
                            original_image=original_image,
                            gradcam_image=gradcam_image,
                            predicted_class=predicted_class,
                            confidence=confidence,
                            all_probabilities=all_probabilities
                            )
                        
                        st.download_button(
                            label="📄 Download AI Analysis Report",
                            data=pdf_report,
                            file_name="brain_mri_ai_analysis_report.pdf",
                            mime="application/pdf",
                            use_container_width=True
                            )
                        
                        with st.expander(
                            "ℹ️ About the model and outside validation"
                        ):
                            st.markdown(
                                f"""
**Model:** ResNet18 brain MRI classifier

**Supported categories:**
- Glioma
- Meningioma
- No Tumor
- Pituitary Tumor

**Outside validation results:**
- Total outside images: {OUTSIDE_VALIDATION_TOTAL}
- Correct predictions: {OUTSIDE_VALIDATION_CORRECT}
- Incorrect predictions: {OUTSIDE_VALIDATION_INCORRECT}
- Overall accuracy: {OUTSIDE_VALIDATION_ACCURACY:.2%}

**Class recall:**
- Glioma: {OUTSIDE_CLASS_RECALL["Glioma"]:.2%}
- Meningioma: {OUTSIDE_CLASS_RECALL["Meningioma"]:.2%}
- No Tumor: {OUTSIDE_CLASS_RECALL["No Tumor"]:.2%}
- Pituitary Tumor: {OUTSIDE_CLASS_RECALL["Pituitary Tumor"]:.2%}

The outside validation images were not used for model training.
The model may still make incorrect or highly confident predictions.
"""
                            )

                        st.error(
                            "Medical disclaimer: This application is an "
                            "educational AI project and is not a medical "
                            "diagnostic system. Always consult a qualified "
                            "medical professional."
                        )

    st.write("")
    st.write("")

    back_left, back_center, back_right = st.columns(
        [2, 1.4, 2]
    )

    with back_center:
        if st.button(
            "Back to Home",
            use_container_width=True
        ):
            st.session_state.page = "home"
            st.rerun()