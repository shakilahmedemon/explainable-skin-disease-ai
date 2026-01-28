import os
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
import argparse
from training.train_single_model import main as train_single_model_main
from training.train_ensemble import main as train_ensemble_main
from model.vit_model import ViTModel
from model.ensemble.ensemble_uncertainty import EnsembleModel

def create_directory_structure():
    """Create necessary directories for the project"""
    dirs = [
        'data/raw',
        'data/train',  # Updated to match training script expectation
        'data/processed', 
        'model',
        'results',
        'training_logs',
        'samples'
    ]
    
    for directory in dirs:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")
    
    # Create subdirectories for each class
    classes = [
        "melanoma", 
        "basal_cell_carcinoma", 
        "squamous_cell_carcinoma",
        "nevus", 
        "dermatofibroma", 
        "vascular_lesion", 
        "actinic_keratosis"
    ]
    
    for class_name in classes:
        os.makedirs(f'data/train/{class_name}', exist_ok=True)
        print(f"Created class directory: data/train/{class_name}")

def setup_dataset():
    """
    Check if dataset exists and provide guidance for users
    """
    print("Checking for dataset...")
    
    train_dir = 'data/train'
    if os.path.exists(train_dir):
        class_dirs = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
        if class_dirs:
            total_images = 0
            for class_dir in class_dirs:
                img_count = len([f for f in os.listdir(os.path.join(train_dir, class_dir)) 
                               if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))])
                total_images += img_count
                print(f"  - {class_dir}: {img_count} images")
            
            print(f"Found {len(class_dirs)} classes with {total_images} total images")
            if total_images > 0:
                print("✅ Dataset found!")
                return True
            else:
                print("⚠️  Directories exist but no images found")
        else:
            print("⚠️  Training directory exists but no class subdirectories found")
    else:
        print("❌ Dataset not found. Creating directory structure...")
    
    print("\n📋 To train the model, you need to:")
    print("   1. Place skin disease images in the appropriate class folders:")
    for i, class_name in enumerate([
        "melanoma", 
        "basal_cell_carcinoma", 
        "squamous_cell_carcinoma",
        "nevus", 
        "dermatofibroma", 
        "vascular_lesion", 
        "actinic_keratosis"
    ], 1):
        print(f"      - data/train/{class_name}/")
    print("   2. Run: python train.py")
    print("")
    
    return False

def train_models():
    """Train both single and ensemble models"""
    print("Starting model training process...")
    
    # Train single model (Vision Transformer)
    print("\n" + "="*60)
    print("Training Single Model (Vision Transformer)")
    print("="*60)
    
    try:
        train_single_model_main()
        print("✅ Vision Transformer training completed")
    except Exception as e:
        print(f"❌ Vision Transformer training failed: {e}")
    
    # Train ensemble model
    print("\n" + "="*60)
    print("Training Ensemble Model")
    print("="*60)
    
    try:
        train_ensemble_main()
        print("✅ Ensemble Model training completed")
    except Exception as e:
        print(f"❌ Ensemble Model training failed: {e}")

def validate_setup():
    """Validate that all necessary components are in place"""
    print("Validating setup...")
    
    # Check if required packages are available
    try:
        import torch
        import torchvision
        import timm
        import numpy as np
        import sklearn
        import cv2
        import matplotlib
        import streamlit
        print("✅ All required packages are available")
    except ImportError as e:
        print(f"❌ Missing package: {e}")
        return False
    
    # Check if model files exist
    vit_model_path = 'model/vit_skin_disease.pth'
    ensemble_model_path = 'model/ensemble_skin_disease.pth'
    
    vit_exists = os.path.exists(vit_model_path)
    ensemble_exists = os.path.exists(ensemble_model_path)
    
    if vit_exists:
        print("✅ ViT model found")
    else:
        print("ℹ️  ViT model not found - will be created during training")
    
    if ensemble_exists:
        print("✅ Ensemble model found")
    else:
        print("ℹ️  Ensemble model not found - will be created during training")
    
    return True

def main():
    parser = argparse.ArgumentParser(description='Train Skin Disease Classification Models')
    parser.add_argument('--skip-data-check', action='store_true', 
                       help='Skip dataset validation (useful if using synthetic data)')
    parser.add_argument('--quick-test', action='store_true', 
                       help='Run a quick test with minimal epochs (not implemented in current version)')
    parser.add_argument('--create-dirs-only', action='store_true',
                       help='Only create directory structure without training')
    
    args = parser.parse_args()
    
    print("🚀 Starting Explainable Skin Disease AI Training Process")
    print("="*60)
    
    # Create directory structure
    create_directory_structure()
    
    if args.create_dirs_only:
        print("📁 Directory structure created. Exiting...")
        return
    
    # Validate setup
    if not validate_setup():
        print("❌ Setup validation failed. Please install required packages.")
        return
    
    # Check dataset
    if not args.skip_data_check:
        dataset_ready = setup_dataset()
        if not dataset_ready:
            print("\n❌ Dataset not found. Please prepare your skin disease dataset.")
            print("For training to work properly, add images to the class directories.")
            print("You can find public skin lesion datasets such as ISIC Archive, DermNet, or HAM10000.")
            return
    
    # Train models
    train_models()
    
    print("\n" + "="*60)
    print("✅ Training process completed!")
    print("To run the application, execute: streamlit run app/app.py")
    print("="*60)

if __name__ == "__main__":
    main()
