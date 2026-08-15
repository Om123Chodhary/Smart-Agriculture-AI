import streamlit as st
import torch
import torch.nn.functional as F
import numpy as np
import cv2

from PIL import Image
from torchvision.models import mobilenet_v3_large
from torchvision import transforms


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Smart Tomato Plant AI",
    page_icon="🌱",
    layout="wide"
)


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "/content/drive/MyDrive/tomato_mobilenetv3_best.pth"

CLASSES = [
    "Tomato_Bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite",
    "Tomato__Target_Spot",
    "Tomato__Tomato_YellowLeaf__Curl_Virus",
    "Tomato__Tomato_mosaic_virus",
    "Tomato_healthy"
]

DISPLAY_NAMES = {
    "Tomato_Bacterial_spot": "Bacterial Spot",
    "Tomato_Early_blight": "Early Blight",
    "Tomato_Late_blight": "Late Blight",
    "Tomato_Leaf_Mold": "Leaf Mold",
    "Tomato_Septoria_leaf_spot": "Septoria Leaf Spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite":
        "Spider Mites",
    "Tomato__Target_Spot": "Target Spot",
    "Tomato__Tomato_YellowLeaf__Curl_Virus":
        "Yellow Leaf Curl Virus",
    "Tomato__Tomato_mosaic_virus":
        "Tomato Mosaic Virus",
    "Tomato_healthy": "Healthy"
}


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# MODEL
# ============================================================

@st.cache_resource
def load_model():

    model = mobilenet_v3_large(
        weights=None,
        num_classes=len(CLASSES)
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )

    # Support checkpoint containing model_state_dict
    if isinstance(checkpoint, dict) and \
       "model_state_dict" in checkpoint:

        state_dict = checkpoint["model_state_dict"]

    else:

        state_dict = checkpoint

    model.load_state_dict(state_dict)

    model.to(DEVICE)
    model.eval()

    return model


# ============================================================
# TRANSFORM
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
# DISEASE PREDICTION
# ============================================================

def predict_image(image, model):

    tensor = transform(image)
    tensor = tensor.unsqueeze(0)
    tensor = tensor.to(DEVICE)

    with torch.no_grad():

        output = model(tensor)

        probabilities = F.softmax(
            output,
            dim=1
        )

    probabilities = probabilities[0].cpu()

    confidence, index = torch.max(
        probabilities,
        dim=0
    )

    disease = CLASSES[index.item()]

    return (
        disease,
        confidence.item() * 100,
        probabilities
    )


# ============================================================
# AFFECTED AREA ESTIMATION
# ============================================================

def estimate_affected_area(image):

    image_np = np.array(image)

    image_bgr = cv2.cvtColor(
        image_np,
        cv2.COLOR_RGB2BGR
    )

    hsv = cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2HSV
    )

    # --------------------------------------------------------
    # GREEN / LEAF MASK
    # --------------------------------------------------------

    lower_green = np.array(
        [25, 30, 20]
    )

    upper_green = np.array(
        [95, 255, 255]
    )

    leaf_mask = cv2.inRange(
        hsv,
        lower_green,
        upper_green
    )

    kernel = np.ones(
        (5, 5),
        np.uint8
    )

    leaf_mask = cv2.morphologyEx(
        leaf_mask,
        cv2.MORPH_OPEN,
        kernel
    )

    leaf_mask = cv2.morphologyEx(
        leaf_mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    # --------------------------------------------------------
    # POTENTIAL DAMAGE MASK
    # --------------------------------------------------------

    lower_damage = np.array(
        [5, 30, 20]
    )

    upper_damage = np.array(
        [45, 255, 220]
    )

    damage_mask = cv2.inRange(
        hsv,
        lower_damage,
        upper_damage
    )

    damage_mask = cv2.bitwise_and(
        damage_mask,
        leaf_mask
    )

    damage_mask = cv2.morphologyEx(
        damage_mask,
        cv2.MORPH_OPEN,
        kernel
    )

    damage_mask = cv2.morphologyEx(
        damage_mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    leaf_pixels = np.sum(
        leaf_mask > 0
    )

    affected_pixels = np.sum(
        damage_mask > 0
    )

    if leaf_pixels == 0:

        affected_area = 0.0

    else:

        affected_area = (
            affected_pixels /
            leaf_pixels
        ) * 100

    # Safety clamp
    affected_area = float(
        np.clip(
            affected_area,
            0,
            100
        )
    )

    return affected_area


# ============================================================
# ROBUST AREA FUSION
# ============================================================

def robust_affected_area(areas):

    areas = np.array(
        areas,
        dtype=float
    )

    areas = np.clip(
        areas,
        0,
        100
    )

    # Need at least 2 values
    if len(areas) < 2:

        return float(
            np.mean(areas)
        )

    # Median
    median = np.median(areas)

    # Remove extreme values using
    # a robust distance from median
    deviations = np.abs(
        areas - median
    )

    mad = np.median(
        deviations
    )

    # If all values are almost identical
    if mad < 1e-6:

        return float(
            median
        )

    # Robust threshold
    threshold = 3.0 * mad

    valid = areas[
        deviations <= threshold
    ]

    # If filtering removed too many points,
    # fall back to median
    if len(valid) < 2:

        return float(
            median
        )

    return float(
        np.median(valid)
    )


# ============================================================
# SEVERITY
# ============================================================

def calculate_severity(area):

    if area < 5:

        return "Healthy / Very Low"

    elif area < 15:

        return "Low"

    elif area < 30:

        return "Moderate"

    elif area < 50:

        return "High"

    else:

        return "Very High"


# ============================================================
# CONSENSUS
# ============================================================

def calculate_consensus(predictions):

    counts = {}

    for disease in predictions:

        counts[disease] = (
            counts.get(
                disease,
                0
            ) + 1
        )

    best_disease = max(
        counts,
        key=counts.get
    )

    best_count = counts[
        best_disease
    ]

    return (
        best_disease,
        best_count,
        counts
    )


# ============================================================
# RELIABILITY
# ============================================================

def calculate_reliability(
    final_confidence,
    consensus_count,
    probability_std
):

    # HIGH reliability
    if (
        final_confidence >= 85
        and consensus_count >= 3
        and probability_std < 15
    ):

        return "HIGH"

    # MEDIUM reliability
    elif (
        final_confidence >= 65
        and consensus_count >= 2
    ):

        return "MEDIUM"

    # LOW reliability
    else:

        return "LOW"


# ============================================================
# FINAL ACTION
# ============================================================

def calculate_action(
    disease,
    severity,
    reliability
):

    # Healthy plant
    if disease == "Tomato_healthy":

        return "NO INTERVENTION"

    # Uncertain AI result
    if reliability == "LOW":

        return "RE-SCAN REQUIRED"

    if severity == "Healthy / Very Low":

        return "LOW PRIORITY"

    elif severity == "Low":

        return "LOW PRIORITY"

    elif severity == "Moderate":

        return "MEDIUM PRIORITY"

    elif severity in [
        "High",
        "Very High"
    ]:

        return "HIGH PRIORITY"

    return "RE-SCAN REQUIRED"


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("🌱 Smart Agriculture AI")

    st.write(
        "Multi-view tomato plant health "
        "analysis system."
    )

    st.divider()

    st.write("### AI Model")

    st.write(
        "MobileNetV3-Large"
    )

    st.write(
        "10 Tomato Classes"
    )

    st.write(
        "Test Accuracy: 99.00%"
    )

    st.write(
        "Test Macro F1: 98.94%"
    )

    st.divider()

    st.caption(
        "Hackathon prototype / "
        "decision-support system"
    )


# ============================================================
# HEADER
# ============================================================

st.title(
    "🌱 Smart Tomato Plant Health Analyzer"
)

st.subheader(
    "AI-powered 4-image plant scanning"
)

st.write(
    "Upload four different views of the "
    "same tomato plant or leaf."
)


# ============================================================
# LOAD MODEL
# ============================================================

try:

    model = load_model()

except Exception as e:

    st.error(
        "❌ Model could not be loaded."
    )

    st.code(
        str(e)
    )

    st.stop()


# ============================================================
# UPLOAD
# ============================================================

st.markdown(
    "### 📷 Upload 4 Images"
)

uploaded_files = st.file_uploader(
    "Select exactly four images",
    type=[
        "jpg",
        "jpeg",
        "png"
    ],
    accept_multiple_files=True
)


# ============================================================
# PROCESS
# ============================================================

if uploaded_files:

    if len(uploaded_files) != 4:

        st.warning(
            f"Please upload exactly 4 images. "
            f"Currently uploaded: "
            f"{len(uploaded_files)}"
        )

    else:

        st.success(
            "✅ Four images uploaded."
        )

        images = []

        cols = st.columns(4)

        for i, file in enumerate(
            uploaded_files
        ):

            image = Image.open(
                file
            ).convert("RGB")

            images.append(image)

            with cols[i]:

                st.image(
                    image,
                    caption=f"View {i + 1}",
                    use_container_width=True
                )

        st.divider()

        if st.button(
            "🔍 ANALYZE PLANT",
            type="primary",
            use_container_width=True
        ):

            results = []

            progress = st.progress(0)

            # =================================================
            # IMAGE ANALYSIS
            # =================================================

            for i, image in enumerate(
                images
            ):

                disease, confidence, probabilities = (
                    predict_image(
                        image,
                        model
                    )
                )

                affected_area = (
                    estimate_affected_area(
                        image
                    )
                )

                results.append({

                    "disease":
                        disease,

                    "confidence":
                        confidence,

                    "probabilities":
                        probabilities,

                    "affected_area":
                        affected_area
                })

                progress.progress(
                    (i + 1) / 4
                )

            # =================================================
            # DISEASE FUSION
            # =================================================

            all_probabilities = torch.stack(
                [
                    r["probabilities"]
                    for r in results
                ]
            )

            fused_probabilities = torch.mean(
                all_probabilities,
                dim=0
            )

            final_confidence, final_index = (
                torch.max(
                    fused_probabilities,
                    dim=0
                )
            )

            final_disease = CLASSES[
                final_index.item()
            ]

            final_confidence = (
                final_confidence.item()
                * 100
            )

            # =================================================
            # CONSENSUS
            # =================================================

            predictions = [
                r["disease"]
                for r in results
            ]

            (
                consensus_disease,
                consensus_count,
                disease_counts
            ) = calculate_consensus(
                predictions
            )

            # =================================================
            # CONFIDENCE VARIATION
            # =================================================

            confidence_values = [
                r["confidence"]
                for r in results
            ]

            confidence_std = float(
                np.std(
                    confidence_values
                )
            )

            # =================================================
            # ROBUST AFFECTED AREA
            # =================================================

            areas = [
                r["affected_area"]
                for r in results
            ]

            final_area = robust_affected_area(
                areas
            )

            # =================================================
            # SEVERITY
            # =================================================

            final_severity = calculate_severity(
                final_area
            )

            # =================================================
            # RELIABILITY
            # =================================================

            reliability = calculate_reliability(
                final_confidence,
                consensus_count,
                confidence_std
            )

            # =================================================
            # FINAL ACTION
            # =================================================

            action = calculate_action(
                final_disease,
                final_severity,
                reliability
            )

            # =================================================
            # FINAL REPORT
            # =================================================

            st.success(
                "✅ Plant analysis completed."
            )

            st.header(
                "🎯 Final Plant Health Report"
            )

            # -------------------------------------------------
            # Main metrics
            # -------------------------------------------------

            c1, c2, c3, c4 = st.columns(4)

            with c1:

                st.metric(
                    "Disease",
                    DISPLAY_NAMES[
                        final_disease
                    ]
                )

            with c2:

                st.metric(
                    "Confidence",
                    f"{final_confidence:.1f}%"
                )

            with c3:

                st.metric(
                    "Consensus",
                    f"{consensus_count}/4"
                )

            with c4:

                st.metric(
                    "Affected Area",
                    f"{final_area:.1f}%"
                )

            st.divider()

            # -------------------------------------------------
            # Severity
            # -------------------------------------------------

            c1, c2, c3 = st.columns(3)

            with c1:

                st.subheader(
                    "🌿 Severity"
                )

                if final_severity in [
                    "High",
                    "Very High"
                ]:

                    st.error(
                        final_severity
                    )

                elif final_severity == "Moderate":

                    st.warning(
                        final_severity
                    )

                else:

                    st.success(
                        final_severity
                    )

            with c2:

                st.subheader(
                    "🧠 AI Reliability"
                )

                if reliability == "HIGH":

                    st.success(
                        "HIGH"
                    )

                elif reliability == "MEDIUM":

                    st.warning(
                        "MEDIUM"
                    )

                else:

                    st.error(
                        "LOW"
                    )

            with c3:

                st.subheader(
                    "🎯 Recommended Action"
                )

                if action == "HIGH PRIORITY":

                    st.error(
                        action
                    )

                elif action == "MEDIUM PRIORITY":

                    st.warning(
                        action
                    )

                elif action == "LOW PRIORITY":

                    st.success(
                        action
                    )

                elif action == "NO INTERVENTION":

                    st.success(
                        action
                    )

                else:

                    st.info(
                        action
                    )

            # -------------------------------------------------
            # Decision explanation
            # -------------------------------------------------

            st.divider()

            st.subheader(
                "🧠 AI Decision Explanation"
            )

            st.write(
                f"The system analyzed "
                f"**4 views** of the plant."
            )

            st.write(
                f"Most frequent prediction: "
                f"**{DISPLAY_NAMES[consensus_disease]}** "
                f"({consensus_count}/4 views)."
            )

            st.write(
                f"Fused model confidence: "
                f"**{final_confidence:.2f}%**."
            )

            st.write(
                f"Robust estimated affected area: "
                f"**{final_area:.2f}%**."
            )

            st.write(
                f"Estimated severity: "
                f"**{final_severity}**."
            )

            st.write(
                f"AI reliability: "
                f"**{reliability}**."
            )

            # -------------------------------------------------
            # Re-scan warning
            # -------------------------------------------------

            if action == "RE-SCAN REQUIRED":

                st.warning(
                    "⚠️ The four views do not provide "
                    "sufficiently reliable agreement. "
                    "Please capture clearer images of "
                    "the same plant from different angles."
                )

            # -------------------------------------------------
            # Individual results
            # -------------------------------------------------

            st.divider()

            st.subheader(
                "📊 Individual Image Analysis"
            )

            for i, result in enumerate(
                results
            ):

                with st.expander(
                    f"📷 View {i + 1}"
                ):

                    c1, c2, c3 = st.columns(3)

                    with c1:

                        st.write(
                            "**Disease**"
                        )

                        st.write(
                            DISPLAY_NAMES[
                                result["disease"]
                            ]
                        )

                    with c2:

                        st.write(
                            "**Confidence**"
                        )

                        st.write(
                            f"{result['confidence']:.2f}%"
                        )

                    with c3:

                        st.write(
                            "**Affected Area**"
                        )

                        st.write(
                            f"{result['affected_area']:.2f}%"
                        )

            # -------------------------------------------------
            # Disease distribution
            # -------------------------------------------------

            st.divider()

            st.subheader(
                "📈 Disease Consensus"
            )

            for disease, count in sorted(
                disease_counts.items(),
                key=lambda x: x[1],
                reverse=True
            ):

                st.write(
                    f"**{DISPLAY_NAMES[disease]}** "
                    f"→ {count}/4 images"
                )

            # -------------------------------------------------
            # Important prototype note
            # -------------------------------------------------

            st.divider()

            st.warning(
                "⚠️ Prototype decision-support system. "
                "Affected-area estimation is a computer-vision "
                "approximation and should not be treated as a "
                "laboratory measurement. Exact pesticide "
                "dosage or spray duration is NOT determined "
                "by this model."
            )