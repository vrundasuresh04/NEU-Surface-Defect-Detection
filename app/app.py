import os
import numpy as np
import streamlit as st
import torch
import torch.nn.functional as F

from PIL import Image
from torchvision import models, transforms
import matplotlib.pyplot as plt


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="NEU Surface Defect Detection",
    page_icon="🔍",
    layout="wide"
)


# ============================================================
# 2. PATH CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "best_resnet18_neu_det.pth"
)


# ============================================================
# 3. DEVICE
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# 4. CLASS NAMES
# ============================================================

classes = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches"
]


# ============================================================
# 5. IMAGE TRANSFORMATION
# ============================================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# 6. LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    model = models.resnet18(weights=None)

    # Six NEU-DET classes
    model.fc = torch.nn.Linear(
        model.fc.in_features,
        len(classes)
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device
    )

    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):

        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]

        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]

        else:
            state_dict = checkpoint

    else:
        state_dict = checkpoint.state_dict()

    # Remove "module." if model was saved using DataParallel
    new_state_dict = {}

    for key, value in state_dict.items():

        if key.startswith("module."):
            key = key[7:]

        new_state_dict[key] = value

    model.load_state_dict(
        new_state_dict,
        strict=True
    )

    model.to(device)
    model.eval()

    return model


model = load_model()


# ============================================================
# 7. GRAD-CAM FUNCTION
# ============================================================

def generate_gradcam(model, image_tensor, target_class):

    activations = []
    gradients = []

    # ResNet18 final convolutional layer
    target_layer = model.layer4[-1].conv2

    # --------------------------------------------------------
    # Forward hook
    # --------------------------------------------------------

    def forward_hook(module, input, output):

        activations.append(
            output.detach()
        )

    # --------------------------------------------------------
    # Backward hook
    # --------------------------------------------------------

    def backward_hook(module, grad_input, grad_output):

        gradients.append(
            grad_output[0].detach()
        )

    forward_handle = target_layer.register_forward_hook(
        forward_hook
    )

    backward_handle = target_layer.register_full_backward_hook(
        backward_hook
    )

    try:

        # Gradients are required
        model.zero_grad()

        image_tensor = image_tensor.clone().detach()
        image_tensor.requires_grad_(True)

        # Forward pass
        output = model(image_tensor)

        # Select predicted class score
        target_score = output[0, target_class]

        # Backward pass
        target_score.backward()

        # Get activation and gradient
        activation = activations[0]
        gradient = gradients[0]

        # ----------------------------------------------------
        # Global average pooling of gradients
        # ----------------------------------------------------

        weights = gradient.mean(
            dim=(2, 3),
            keepdim=True
        )

        # ----------------------------------------------------
        # Weighted combination
        # ----------------------------------------------------

        cam = (weights * activation).sum(
            dim=1,
            keepdim=True
        )

        # Remove negative values
        cam = F.relu(cam)

        # ----------------------------------------------------
        # Resize CAM to 224x224
        # ----------------------------------------------------

        cam = F.interpolate(
            cam,
            size=(224, 224),
            mode="bilinear",
            align_corners=False
        )

        cam = cam.squeeze().cpu().numpy()

        # ----------------------------------------------------
        # Normalize between 0 and 1
        # ----------------------------------------------------

        cam_min = cam.min()
        cam_max = cam.max()

        if cam_max - cam_min > 1e-8:

            cam = (
                cam - cam_min
            ) / (
                cam_max - cam_min
            )

        else:

            cam = np.zeros_like(cam)

        return cam

    finally:

        forward_handle.remove()
        backward_handle.remove()


# ============================================================
# 8. CREATE GRAD-CAM OVERLAY
# ============================================================

def create_gradcam_overlay(original_image, cam):

    # Convert original image to RGB
    original_image = original_image.convert("RGB")

    # Resize original image
    original_resized = original_image.resize(
        (224, 224)
    )

    original_array = np.array(
        original_resized
    ) / 255.0

    # --------------------------------------------------------
    # Create heatmap
    # --------------------------------------------------------

    cmap = plt.get_cmap("jet")

    heatmap = cmap(cam)

    # Remove alpha channel
    heatmap = heatmap[:, :, :3]

    # --------------------------------------------------------
    # Overlay
    # --------------------------------------------------------

    overlay = (
        0.55 * original_array
        +
        0.45 * heatmap
    )

    overlay = np.clip(
        overlay,
        0,
        1
    )

    return overlay


# ============================================================
# 9. SIDEBAR
# ============================================================

with st.sidebar:

    st.title("Model Information")

    st.write(
        f"**Device:** `{device}`"
    )

    st.write(
        "**Model:** ResNet18"
    )

    st.write(
        f"**Classes:** {len(classes)}"
    )

    st.divider()

    st.subheader("Explainability")

    st.write(
        "Grad-CAM highlights the image regions "
        "that influenced the model's prediction."
    )


# ============================================================
# 10. MAIN TITLE
# ============================================================

st.title("🔍 NEU Surface Defect Detection")

st.write(
    "AI-powered surface defect classification "
    "with Grad-CAM explainability."
)


# ============================================================
# 11. IMAGE UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "Upload a surface image",
    type=["jpg", "jpeg", "png"]
)


# ============================================================
# 12. PROCESS IMAGE
# ============================================================

if uploaded_file is not None:

    # Open image
    image = Image.open(
        uploaded_file
    ).convert("RGB")

    st.subheader("Input Surface Image")

    st.image(
        image,
        width=500
    )

    # --------------------------------------------------------
    # Detect button
    # --------------------------------------------------------

    if st.button(
        "🔎 Detect Defect",
        type="primary"
    ):

        # ====================================================
        # PREPROCESS
        # ====================================================

        image_tensor = transform(
            image
        ).unsqueeze(0).to(device)


        # ====================================================
        # MODEL PREDICTION
        # ====================================================

        with torch.no_grad():

            outputs = model(
                image_tensor
            )

            probabilities = F.softmax(
                outputs,
                dim=1
            )

            confidence, predicted_class = torch.max(
                probabilities,
                dim=1
            )


        predicted_index = predicted_class.item()

        predicted_name = classes[
            predicted_index
        ]

        confidence_value = confidence.item()


        # ====================================================
        # PREDICTION RESULT
        # ====================================================

        st.subheader(
            "Prediction Result"
        )

        st.success(
            f"Defect Detected: {predicted_name}"
        )

        st.metric(
            "Confidence",
            f"{confidence_value * 100:.2f}%"
        )


        # ====================================================
        # CONFIDENCE ASSESSMENT
        # ====================================================

        if confidence_value >= 0.90:

            st.success(
                "🟢 High confidence prediction"
            )

        elif confidence_value >= 0.70:

            st.warning(
                "🟡 Medium confidence prediction"
            )

        else:

            st.error(
                "🔴 Low confidence prediction"
            )


        # ====================================================
        # CLASS PROBABILITIES
        # ====================================================

        st.subheader(
            "Class Probabilities"
        )

        probability_values = (
            probabilities[0]
            .cpu()
            .numpy()
        )

        for class_name, probability in zip(
            classes,
            probability_values
        ):

            st.write(
                f"**{class_name}**"
            )

            st.progress(
                float(probability)
            )

            st.caption(
                f"{probability * 100:.2f}%"
            )


        # ====================================================
        # GRAD-CAM
        # ====================================================

        st.divider()

        st.subheader(
            "🧠 Grad-CAM Explainability"
        )

        st.write(
            "The heatmap shows the regions that "
            "contributed most strongly to the "
            "predicted defect."
        )


        # Generate Grad-CAM
        with st.spinner(
            "Generating Grad-CAM explanation..."
        ):

            cam = generate_gradcam(
                model,
                image_tensor,
                predicted_index
            )

            overlay = create_gradcam_overlay(
                image,
                cam
            )


        # ====================================================
        # DISPLAY ORIGINAL + HEATMAP
        # ====================================================

        col1, col2 = st.columns(2)

        with col1:

            st.image(
                image.resize((224, 224)),
                caption="Original Image",
                width=400
            )

        with col2:

            st.image(
                overlay,
                caption="Grad-CAM Heatmap",
                width=400
            )


        # ====================================================
        # EXPLANATION
        # ====================================================

        st.info(
            f"The highlighted regions represent the "
            f"areas that contributed most to the model's "
            f"prediction of **{predicted_name}**."
        )


        # ====================================================
        # FINAL SUMMARY
        # ====================================================

        st.divider()

        st.subheader(
            "Detection Summary"
        )

        summary_col1, summary_col2, summary_col3 = st.columns(3)

        with summary_col1:

            st.write("**Predicted Defect**")

            st.write(
                predicted_name
            )

        with summary_col2:

            st.write("**Confidence**")

            st.write(
                f"{confidence_value * 100:.2f}%"
            )

        with summary_col3:

            st.write("**Explainability**")

            st.write(
                "Grad-CAM Available ✓"
            )


else:

    st.info(
        "Please upload a surface image to begin detection."
    )