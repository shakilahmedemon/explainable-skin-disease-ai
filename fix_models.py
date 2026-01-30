"""
Fix model files directly by creating them programmatically
"""
import sys
import os
# Add the project directory to the path
sys.path.insert(0, 'd:/projects/explainable-skin-disease-ai')

# Import the model classes
from model.vit_model import ViTModel
from model.ensemble.ensemble_uncertainty import EnsembleModel
import torch

def create_models():
    """Create and save basic models"""
    print("Creating model directory...")
    os.makedirs('model', exist_ok=True)
    
    print("Creating ViT model...")
    try:
        vit_model = ViTModel(num_classes=7)
        vit_path = 'model/vit_skin_disease.pth'
        torch.save(vit_model.state_dict(), vit_path)
        print(f"✅ Saved ViT model to {vit_path}")
    except Exception as e:
        print(f"❌ Error creating ViT model: {e}")
    
    print("Creating Ensemble model...")
    try:
        ensemble_model = EnsembleModel(num_classes=7)
        ensemble_path = 'model/ensemble_skin_disease.pth'
        torch.save(ensemble_model.state_dict(), ensemble_path)
        print(f"✅ Saved Ensemble model to {ensemble_path}")
    except Exception as e:
        print(f"❌ Error creating Ensemble model: {e}")
    
    # Verify the files exist
    vit_exists = os.path.exists('model/vit_skin_disease.pth')
    ensemble_exists = os.path.exists('model/ensemble_skin_disease.pth')
    
    print(f"\nVerification:")
    print(f"  ViT model exists: {vit_exists}")
    print(f"  Ensemble model exists: {ensemble_exists}")
    
    if vit_exists and ensemble_exists:
        print("\n🎉 Both models created successfully!")
        print("\nNext steps:")
        print("1. Run training with your dataset: python train.py")
        print("2. Or use the app with these initialized models: streamlit run app/app.py")
    else:
        print("\n⚠️  Some models failed to create. Please check the errors above.")

if __name__ == "__main__":
    create_models()
