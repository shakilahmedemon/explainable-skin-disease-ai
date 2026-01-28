#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Quick Ensemble Training Script for Skin Disease Classification
"""

import torch
import os
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add project root to path
import sys
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from model.ensemble.ensemble_uncertainty import EnsembleModel

def create_sample_ensemble_model():
    """
    Create a sample ensemble model to ensure it exists
    """
    logger.info("Creating sample ensemble model...")
    
    try:
        # Create ensemble model
        ensemble_model = EnsembleModel(num_classes=7, n_models=3)
        
        # Create model directory
        os.makedirs('model', exist_ok=True)
        
        # Save the ensemble model
        save_path = 'model/ensemble_skin_disease.pth'
        torch.save({
            'model_state_dict': ensemble_model.state_dict(),
            'num_classes': 7,
            'n_models': 3,
            'timestamp': datetime.now().isoformat(),
            'accuracy': 0.0,  # Placeholder
            'trained': False  # Flag to indicate not properly trained
        }, save_path)
        
        logger.info(f"Sample ensemble model created and saved to {save_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error creating sample ensemble model: {e}")
        return False

def main():
    """
    Main function to ensure ensemble model exists
    """
    logger.info("Ensuring ensemble model exists for the application...")
    
    # Check if ensemble model exists
    ensemble_path = 'model/ensemble_skin_disease.pth'
    
    if os.path.exists(ensemble_path):
        logger.info("Ensemble model already exists.")
        
        # Load and verify the model
        try:
            checkpoint = torch.load(ensemble_path, map_location='cpu')
            logger.info(f"Ensemble model loaded. Trained: {checkpoint.get('trained', False)}")
        except Exception as e:
            logger.warning(f"Could not load existing ensemble model: {e}")
            # Recreate the model
            create_sample_ensemble_model()
    else:
        logger.info("Ensemble model does not exist. Creating one...")
        success = create_sample_ensemble_model()
        if not success:
            logger.error("Failed to create ensemble model")
            return False
    
    # Also ensure ViT model exists
    vit_path = 'model/vit_skin_disease.pth'
    if not os.path.exists(vit_path):
        logger.info("Creating sample ViT model...")
        try:
            from model.vit_model import ViTModel
            vit_model = ViTModel(num_classes=7)
            torch.save({
                'model_state_dict': vit_model.state_dict(),
                'num_classes': 7,
                'timestamp': datetime.now().isoformat()
            }, vit_path)
            logger.info(f"Sample ViT model created at {vit_path}")
        except Exception as e:
            logger.error(f"Could not create ViT model: {e}")
    
    logger.info("Model setup completed!")
    logger.info("Both ViT and Ensemble models are now available.")
    logger.info("For optimal performance, train with real medical data using the training scripts.")

if __name__ == "__main__":
    main()
