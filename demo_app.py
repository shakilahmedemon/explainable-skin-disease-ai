"""
Simple demo to test the application without full model downloads
"""
import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__)))

def create_mock_model_predictions():
    """
    Create mock model predictions for demonstration purposes
    """
    class_labels = [
        "Melanoma", "Basal Cell Carcinoma", "Squamous Cell Carcinoma",
        "Nevus (Mole)", "Dermatofibroma", "Vascular Lesion", "Actinic Keratosis"
    ]
    
    # Generate random but plausible probabilities
    probabilities = np.random.dirichlet([1.0] * len(class_labels), size=1)[0]
    
    # Ensure melanoma, bcc, or scc have higher probability for demo
    high_risk_idx = np.random.choice([0, 1, 2])  # melanoma, bcc, or scc
    probabilities[high_risk_idx] = max(probabilities[high_risk_idx], 0.4)
    
    # Renormalize
    probabilities = probabilities / probabilities.sum()
    
    predicted_class_idx = np.argmax(probabilities)
    predicted_label = class_labels[predicted_class_idx]
    confidence = probabilities[predicted_class_idx]
    
    return {
        'probabilities': probabilities,
        'predicted_label': predicted_label,
        'confidence': confidence,
        'class_labels': class_labels,
        'predicted_class_idx': predicted_class_idx
    }

def generate_mock_explanation(image_array):
    """
    Generate a mock explanation overlay for demonstration
    """
    # Create a simple gradient overlay for demonstration
    h, w = image_array.shape[:2]
    
    # Create a circular gradient centered in the image
    center_x, center_y = w // 2, h // 2
    y, x = np.ogrid[:h, :w]
    mask = ((x - center_x)**2 + (y - center_y)**2) <= (min(w,h)//3)**2
    
    # Create explanation overlay
    explanation = np.zeros((h, w))
    explanation[mask] = 1.0  # Highlight the center region
    
    return explanation

def get_mock_medical_insights(disease_label):
    """
    Return mock medical insights for demo
    """
    insights = {
        "Melanoma": {
            "description": "Melanoma is a serious form of skin cancer that develops from melanocytes.",
            "risk": "HIGH - Requires immediate medical attention"
        },
        "Basal Cell Carcinoma": {
            "description": "Basal cell carcinoma is the most common type of skin cancer.",
            "risk": "MODERATE - Should be evaluated by a dermatologist"
        },
        "Squamous Cell Carcinoma": {
            "description": "Squamous cell carcinoma is the second most common type of skin cancer.",
            "risk": "MODERATE - Requires medical evaluation"
        },
        "Nevus (Mole)": {
            "description": "A benign pigmented lesion commonly known as a mole.",
            "risk": "LOW - Monitor for changes"
        }
    }
    
    info = insights.get(disease_label, {
        "description": "Information about this skin condition.",
        "risk": "MODERATE - Consult a dermatologist"
    })
    
    return f"**Condition:** {disease_label}\n\n**Description:** {info['description']}\n\n**Risk Level:** {info['risk']}"

def main():
    st.title("🩺 Demo: Explainable Skin Disease Classifier")
    st.markdown("Upload a skin lesion image for AI-powered analysis and explanation")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a skin image...", 
        type=["jpg", "jpeg", "png"]
    )
    
    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        if st.button("Analyze & Explain"):
            with st.spinner("Analyzing image and generating explanation..."):
                # Create mock predictions
                results = create_mock_model_predictions()
                
                # Display results
                st.success(f"Predicted: {results['predicted_label']}")
                st.info(f"Confidence: {results['confidence']:.2%}")
                
                # Show all class probabilities
                st.subheader("Class Probabilities")
                prob_chart_data = {label: prob for label, prob in 
                                 zip(results['class_labels'], results['probabilities'])}
                st.bar_chart(prob_chart_data)
                
                # Generate mock explanation
                st.subheader("AI Explanation")
                image_array = np.array(image)
                explanation_map = generate_mock_explanation(image_array)
                
                # Create visualization
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.imshow(image_array)
                ax.imshow(explanation_map, cmap='jet', alpha=0.5)
                ax.axis('off')
                ax.set_title('AI Focus Area (Mock Explanation)')
                st.pyplot(fig)
                plt.close()
                
                # Show mock medical insights
                st.subheader("Medical Insights")
                medical_info = get_mock_medical_insights(results['predicted_label'])
                st.markdown(medical_info)
                
                # Safety warning
                high_risk_conditions = ["Melanoma", "Squamous Cell Carcinoma", "Basal Cell Carcinoma"]
                if results['predicted_label'] in high_risk_conditions:
                    st.warning(f"⚠️ WARNING: {results['predicted_label']} is a serious skin condition. Seek immediate dermatological consultation.")
                else:
                    st.info("⚠️ This is a demo. Always consult with a dermatologist for professional diagnosis.")
    
    # Add information section
    with st.expander("About this demo"):
        st.markdown("""
        This is a demonstration of the Explainable Skin Disease AI application.
        
        **Features:**
        - AI-powered skin lesion classification
        - Visual explanations showing which areas the AI focused on
        - Medical insights and risk assessment
        - Uncertainty quantification
        
        **Important:** This is a demo system. Always consult with a qualified 
        dermatologist for actual medical diagnosis.
        """)

if __name__ == "__main__":
    main()
