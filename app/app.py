import streamlit as st
from PIL import Image
import torch
import numpy as np
import timm, json, cv2
from torchvision import transforms
from torch import nn

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLASSES = ['Acne','Eczema','Psoriasis','Melanoma','Normal']
MODEL_PATHS = [f"model/ensemble/vit_{i}.pth" for i in range(1,6)]

# -----------------------------
# Load ensemble models
# -----------------------------
models = []
for path in MODEL_PATHS:
    model = timm.create_model('vit_base_patch16_224', pretrained=True)
    model.head = nn.Linear(model.head.in_features, len(CLASSES))
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    models.append(model)

# -----------------------------
# Load medical info
# -----------------------------
with open("medical_info/disease_info.json") as f:
    DISEASE_INFO = json.load(f)

# -----------------------------
# Transforms
# -----------------------------
transform = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])

# -----------------------------
# Ensemble prediction function
# -----------------------------
def predict_ensemble(image):
    img = transform(image).unsqueeze(0).to(DEVICE)
    probs_list = []
    with torch.no_grad():
        for model in models:
            outputs = torch.softmax(model(img), dim=1)
            probs_list.append(outputs.cpu().numpy())
    probs = np.mean(probs_list, axis=0)
    top_idx = probs.argsort()[0][::-1][:3]
    top_probs = probs[0][top_idx]
    top_classes = [CLASSES[i] for i in top_idx]
    overall_confidence = "High" if top_probs[0]>0.8 else "Medium" if top_probs[0]>0.6 else "Low"
    return list(zip(top_classes, top_probs)), overall_confidence

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🩺 Explainable & Uncertainty-Aware Skin Disease AI")
uploaded = st.file_uploader("Upload skin image", type=["jpg","png"])

if uploaded:
    image = Image.open(uploaded).convert('RGB')
    st.image(image, caption="Uploaded Image", width=300)
    top_k, conf = predict_ensemble(image)
    st.subheader("Top Predictions (Top-3)")
    for cls, prob in top_k:
        st.write(f"{cls} — Confidence: {prob*100:.2f}%")
    st.write("Overall confidence:", conf)
    
    st.subheader("Educational Medical Info")
    for cls, _ in top_k:
        if cls in DISEASE_INFO:
            st.write(f"**{cls}:**")
            st.write("Symptoms:", DISEASE_INFO[cls]['symptoms'])
            st.write("Causes:", DISEASE_INFO[cls]['causes'])
            st.write("Treatment:", DISEASE_INFO[cls]['treatment'])
    
st.warning("This system is for educational purposes only and is NOT a medical diagnostic tool. Please consult a qualified healthcare professional.")
