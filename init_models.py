"""
Initialize basic models to ensure the application can run
"""
import torch
import os
from model.vit_model import ViTModel
from model.ensemble.ensemble_uncertainty import EnsembleModel

def initialize_models():
    print("Initializing basic models...")
    
    # Create model directory
    os.makedirs('model', exist_ok=True)
    
    # Create and save ViT model
    print("Creating ViT model...")
    vit_model = ViTModel(num_classes=7)
    vit_path = 'model/vit_skin_disease.pth'
    torch.save(vit_model.state_dict(), vit_path)
    print(f"Saved ViT model to {vit_path}")
    
    # Create and save Ensemble model
    print("Creating Ensemble model...")
    ensemble_model = EnsembleModel(num_classes=7)
    ensemble_path = 'model/ensemble_skin_disease.pth'
    torch.save(ensemble_model.state_dict(), ensemble_path)
    print(f"Saved Ensemble model to {ensemble_path}")
    
    print("Basic models initialized successfully!")
    print("\nNext steps:")
    print("1. Run training with your dataset: python train.py")
    print("2. Or use the app with these initialized models: streamlit run app/app.py")

if __name__ == "__main__":
    initialize_models()
