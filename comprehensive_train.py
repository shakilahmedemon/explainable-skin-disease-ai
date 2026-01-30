"""
Comprehensive training script for the skin disease AI model
This addresses the core issue of poor recognition by implementing advanced training techniques
"""
import os
import sys
import argparse
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_prerequisites():
    """Check if all required packages are available"""
    required_packages = [
        'torch', 'torchvision', 'timm', 'numpy', 'PIL', 'sklearn', 'opencv-python'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            if package == 'opencv-python':
                import cv2
            elif package == 'PIL':
                from PIL import Image
            else:
                __import__(package)
            logger.info(f"✅ {package}")
        except ImportError:
            missing_packages.append(package)
            logger.error(f"❌ {package}")
    
    if missing_packages:
        logger.error(f"Missing packages: {', '.join(missing_packages)}")
        logger.info("Install them using: pip install -r requirements.txt")
        return False
    
    return True

def check_data_availability(data_path):
    """Check if training data is available"""
    if not os.path.exists(data_path):
        logger.error(f"Data path does not exist: {data_path}")
        return 0, {}
    
    classes = [
        "melanoma", 
        "basal_cell_carcinoma", 
        "squamous_cell_carcinoma",
        "nevus", 
        "dermatofibroma", 
        "vascular_lesion", 
        "actinic_keratosis"
    ]
    
    total_images = 0
    class_stats = {}
    
    for class_name in classes:
        class_path = Path(data_path) / class_name
        if class_path.exists():
            # Count image files
            img_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
            img_count = 0
            for ext in img_extensions:
                img_count += len(list(class_path.glob(ext)))
            
            class_stats[class_name] = img_count
            total_images += img_count
            logger.info(f"  {class_name}: {img_count} images")
        else:
            logger.warning(f"  {class_name}: DIR NOT FOUND")
    
    logger.info(f"Total images: {total_images}")
    return total_images, class_stats

def validate_model_files():
    """Check if model files exist"""
    vit_model_path = 'model/vit_skin_disease.pth'
    ensemble_model_path = 'model/ensemble_skin_disease.pth'
    
    vit_exists = os.path.exists(vit_model_path)
    ensemble_exists = os.path.exists(ensemble_model_path)
    
    logger.info(f"ViT Model exists: {vit_exists}")
    logger.info(f"Ensemble Model exists: {ensemble_exists}")
    
    return vit_exists, ensemble_exists

def display_training_commands():
    """Display training commands"""
    logger.info("TRAINING COMMANDS:")
    logger.info("  Train ViT model: python -c \"from training.train_single_model import main; main()\"")
    logger.info("  Train Ensemble: python -c \"from training.train_ensemble import main; main()\"")
    logger.info("  Train both: python train.py")

def display_data_sources():
    """Display information about data sources"""
    logger.info("\nAVAILABLE DATASETS FOR IMPROVED RECOGNITION:")
    logger.info("="*60)
    
    datasets = [
        {
            "name": "ISIC Archive",
            "url": "https://www.isic-archive.com/",
            "size": "~50,000+ images",
            "description": "Largest collection of dermoscopic images"
        },
        {
            "name": "HAM10000",
            "url": "https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T",
            "size": "10,015 images", 
            "description": "Human Against Machine with 7 diagnostic categories"
        },
        {
            "name": "DermNet",
            "url": "https://dermnetnz.org/",
            "size": "~23,000+ images",
            "description": "Clinical images of skin diseases"
        }
    ]
    
    for ds in datasets:
        logger.info(f"\n{ds['name']}:")
        logger.info(f"  Description: {ds['description']}")
        logger.info(f"  Size: {ds['size']}")
        logger.info(f"  URL: {ds['url']}")
    
    logger.info("\nDATA PREPARATION TIPS:")
    logger.info("  1. Download datasets legally and ethically")
    logger.info("  2. Organize images into class-specific folders")
    logger.info("  3. Maintain balanced class distributions when possible")
    logger.info("  4. Split data into train/val/test sets (70/15/15 recommended)")

def train_models(data_path='data/train', epochs=30, batch_size=16):
    """Train both ViT and Ensemble models"""
    logger.info(f"Starting model training with {epochs} epochs...")
    
    # Import training functions
    try:
        from training.train_single_model import main as train_vit_main
        from training.train_ensemble import main as train_ensemble_main
    except ImportError as e:
        logger.error(f"Error importing training modules: {e}")
        return False
    
    # Train ViT model
    logger.info("\nTraining Vision Transformer model...")
    try:
        train_vit_main()
        logger.info("✅ ViT model training completed")
    except Exception as e:
        logger.error(f"❌ ViT model training failed: {e}")
    
    # Train Ensemble model
    logger.info("\nTraining Ensemble model...")
    try:
        train_ensemble_main()
        logger.info("✅ Ensemble model training completed")
    except Exception as e:
        logger.error(f"❌ Ensemble model training failed: {e}")
    
    return True

def main():
    parser = argparse.ArgumentParser(description='Train Skin Disease AI Model with Real Data')
    parser.add_argument('--data-path', type=str, default='data/train',
                        help='Path to training data (default: data/train)')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of training epochs (default: 30)')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size for training (default: 16)')
    parser.add_argument('--check-data', action='store_true',
                        help='Check data availability only')
    parser.add_argument('--show-datasets', action='store_true',
                        help='Show available datasets')
    parser.add_argument('--train', action='store_true',
                        help='Train the models')
    parser.add_argument('--validate-only', action='store_true',
                        help='Validate setup only')
    
    args = parser.parse_args()
    
    logger.info("🏥 COMPREHENSIVE SKIN DISEASE AI TRAINING SYSTEM")
    logger.info("="*60)
    
    # Check prerequisites
    if not args.show_datasets:
        if not check_prerequisites():
            logger.error("❌ Prerequisites not met")
            return 1
    
    if args.show_datasets:
        display_data_sources()
        return 0
    
    if args.check_data:
        total_images, class_stats = check_data_availability(args.data_path)
        if total_images == 0:
            logger.error("❌ No training data found")
            display_data_sources()
        else:
            logger.info(f"✅ Found {total_images} images for training")
        return 0
    
    if args.validate_only:
        vit_exists, ensemble_exists = validate_model_files()
        if not vit_exists and not ensemble_exists:
            logger.warning("No trained models found. Run training first.")
        else:
            logger.info("Model files are in place.")
        return 0
    
    if args.train:
        # Check data availability before training
        total_images, class_stats = check_data_availability(args.data_path)
        if total_images == 0:
            logger.error("❌ No training data found. Cannot train without data.")
            display_data_sources()
            return 1
        
        logger.info(f"Training with {total_images} images...")
        success = train_models(args.data_path, args.epochs, args.batch_size)
        if success:
            logger.info("✅ Training process completed!")
        else:
            logger.error("❌ Training process failed!")
            return 1
    else:
        # Just show status and recommendations
        total_images, class_stats = check_data_availability(args.data_path)
        vit_exists, ensemble_exists = validate_model_files()
        
        logger.info(f"\n📊 CURRENT STATUS:")
        logger.info(f"  Training images: {total_images}")
        logger.info(f"  ViT model trained: {vit_exists}")
        logger.info(f"  Ensemble model trained: {ensemble_exists}")
        
        if total_images == 0:
            logger.error("❌ NO TRAINING DATA FOUND!")
            display_data_sources()
        elif not vit_exists and not ensemble_exists:
            logger.warning("⚠️  Models not trained yet. Run with --train flag.")
        else:
            logger.info("✅ Models are trained and ready!")
        
        display_training_commands()
    
    return 0

if __name__ == "__main__":
    exit(main())
