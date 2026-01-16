import streamlit as st
import torch
import json
import numpy as np
from PIL import Image
from torchvision import transforms

from model.vit_model import SkinCancerViT
from model.ensemble_uncertainty import ensemble_predict
from app.explainability import compute_attention_rollout
from app.safety import validate_image
from app.medical_knowledge import MEDICAL_DATA

# ======================================================
# PAGE CONFIG
# ======================================================
st.set_page_config(
    page_title="Educational Skin Cancer AI",
    layout="wide"
)

st.title("Explainable & Uncertainty-Aware Skin Cancer AI")

st.error(
    "**Important Notice**\n"
    "This system is for **educational purposes only**.\n"
    "It does NOT diagnose disease or provide medical advice.\n"
    "Please consult a qualified healthcare professional."
)

consent = st.checkbox(
    "I understand this system is for educational purposes only and not a medical diagnosis."
)
if not consent:
    st.stop()

# ======================================================
# LOAD CLASS NAMES
# ======================================================
with open("data/class_names.json") as f:
    class_names = json.load(f)

device = "cuda" if torch.cuda.is_available() else "cpu"

# ======================================================
# LOAD ENSEMBLE
# ======================================================
@st.cache_resource
def load_ensemble_models():
    models = []
    for i in range(1, 6):
        model = SkinCancerViT(len(class_names))
        model.load_state_dict(
            torch.load(f"model/ensemble/vit_{i}.pth", map_location=device)
        )
        model.to(device)
        model.eval()
        models.append(model)
    return models

models = load_ensemble_models()

# ======================================================
# IMAGE UPLOAD
# ======================================================
uploaded_file = st.file_uploader(
    "Upload a clear skin image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is None:
    st.stop()

image = Image.open(uploaded_file).convert("RGB")
image_np = np.array(image)

valid, error_msg = validate_image(image_np)
if not valid:
    st.warning(error_msg)
    st.stop()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

input_tensor = transform(image).unsqueeze(0).to(device)

# ======================================================
# ENSEMBLE INFERENCE
# ======================================================
mean_probs, variance, entropy = ensemble_predict(models, input_tensor)

if variance > 0.02 or entropy > 1.2:
    st.warning(
        "The model ensemble shows **high uncertainty**. "
        "A reliable educational interpretation cannot be provided."
    )
    st.stop()

# ======================================================
# TOP-K PREDICTIONS
# ======================================================
topk = torch.topk(mean_probs, k=3)
st.subheader("Possible Conditions (Top-K)")
for idx, score in zip(topk.indices, topk.values):
    st.write(f"- **{class_names[idx]}** — Confidence: {score.item()*100:.1f}%")

# ======================================================
# CONFIDENCE ASSESSMENT
# ======================================================
st.subheader("Model Confidence Assessment")
st.write("Overall confidence: **Moderate**")
st.write("Aggregated from multiple independently trained models. Uncertainty remains.")

# ======================================================
# EXPLAINABILITY
# ======================================================
st.subheader("Visual Explanation (Attention Map)")
attention_map = compute_attention_rollout(models[0], input_tensor)
st.image(
    attention_map,
    caption="Model attention map (educational purposes only)."
)

# ======================================================
# MEDICAL KNOWLEDGE
# ======================================================
st.subheader("Educational Information")
for idx in topk.indices:
    name = class_names[idx]
    if name in MEDICAL_DATA:
        data = MEDICAL_DATA[name]
        st.markdown(f"### {name}")
        st.write("**Overview:**", data["overview"])
        st.write("**Symptoms:**", ", ".join(data["symptoms"]))
        st.write("**Causes:**", ", ".join(data["causes"]))
        st.write("**Risk factors:**", ", ".join(data["risk_factors"]))
        st.write("**General treatment approaches:**", data["treatment"])
        st.write("**When to see a doctor:**", data["doctor"])

# ======================================================
# FINAL SAFETY NOTICE
# ======================================================
st.error(
    "This system is for educational purposes only and is not a medical diagnosis. "
    "Consult a qualified healthcare professional for any health concerns."
)

