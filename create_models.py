#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script to create initial model files for the skin disease classification system
"""

import torch
import os
import sys
from datetime import datetime

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

def create_models():
    print("Creating initial model files...")
    
    # Create model directory
    os.makedirs('model', exist_ok=True)
    print("Model directory created/verified")
    
    # Create ensemble model
    try:
        from model.ensemble.ensemble_uncertainty import EnsembleModel
        ensemble_model = EnsembleModel(num_classes=7, n_models=3)
        
        # Save ensemble model
        ensemble_path = os.path.join('model', 'ensemble_skin_disease.pth')
        torch.save({
            'model_state_dict': ensemble_model.state_dict(),
            'num_classes': 7,
            'n_models': 3,
            'timestamp': datetime.now().isoformat(),
            'accuracy': 0.0,  # Placeholder accuracy
            'trained': False  # Mark as not properly trained
        }, ensemble_path)
        
        print(f"Ensemble model created: {os.path.exists(ensemble_path)}")
    except Exception as e:
        print(f"Error creating ensemble model: {e}")
    
    # Create ViT model
    try:
        from model.vit_model import ViTModel
        vit_model = ViTModel(num_classes=7)
        
        # Save ViT model
        vit_path = os.path.join('model', 'vit_skin_disease.pth')
        torch.save({
            'model_state_dict': vit_model.state_dict(),
            'num_classes': 7,
            'timestamp': datetime.now().isoformat()
        }, vit_path)
        
        print(f"ViT model created: {os.path.exists(vit_path)}")
    except Exception as e:
        print(f"Error creating ViT model: {e}")
    
    # Verify models exist
    ensemble_exists = os.path.exists(os.path.join('model', 'ensemble_skin_disease.pth'))
    vit_exists = os.path.exists(os.path.join('model', 'vit_skin_disease.pth'))
    
    print(f"\nModel creation summary:")
    print(f"  Ensemble model exists: {ensemble_exists}")
    print(f"  ViT model exists: {vit_exists}")
    
    if ensemble_exists and vit_exists:
        print("\n✅ Both models created successfully!")
        print("The application should now be able to run without import/model errors.")
    else:
        print("\n⚠️ Some models may not have been created properly.")

if __name__ == "__main__":
    create_models()
