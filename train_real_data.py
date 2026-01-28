"""
Script to train the skin disease model with real medical data
This addresses the core issue by using actual clinical images
"""
import os
import sys
import argparse
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_directory_structure():
    """Create the proper directory structure for skin disease data"""
    classes = [
        "melanoma", 
        "basal_cell_carcinoma", 
        "squamous_cell_carcinoma",
        "nevus", 
        "dermatofibroma", 
        "vascular_lesion", 
        "actinic_keratosis"
    ]
    
    base_path = Path("data/train")
    base_path.mkdir(parents=True, exist_ok=True)
    
    for class_name in classes:
        (base_path / class_name).mkdir(exist_ok=True)
    
    logger.info(f"Directory structure created at: {base_path}")
    return base_path

def get_public_skin_datasets():
    """Provide information about publicly available skin disease datasets"""
    datasets = {
        "ISIC Archive": {
            "description": "The International Skin Imaging Collaboration: largest collection of dermoscopic images",
            "size": "~50,000+ images",
            "url": "https://www.isic-archive.com/",
            "note": "Gold standard for skin lesion analysis - Requires registration for download"
        },
        "HAM10000": {
            "description": "Human Against Machine with 10,000+ skin lesion images",
            "size": "10,015 images",
            "url": "https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T",
            "note": "7 diagnostic categories with expert annotations"
        },
        "DermNet": {
            "description": "Clinical images of skin diseases",
            "size": "~23,000+ images",
            "url": "https://dermnetnz.org/",
            "note": "Free access with proper attribution"
        },
        "BCN_20000": {
            "description": "20,000 clinical photographs of skin lesions",
            "size": "20,000 images",
            "url": "https://github.com/GIMVI/dataset-bcn_20000",
            "note": "Requires citation if used"
        }
    }
    
    logger.info("AVAILABLE PUBLIC SKIN DISEASE DATASETS:")
    logger.info("="*80)
    
    for name, info in datasets.items():
        logger.info(f"\n{name}:")
        logger.info(f"  Description: {info['description']}")
        logger.info(f"  Size: {info['size']}")
        logger.info(f"  URL: {info['url']}")
        logger.info(f"  Note: {info['note']}")
    
    return datasets

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

def train_models():
    """Train both ViT and Ensemble models with real data"""
    logger.info("Starting model training with real medical data...")
    
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
        logger.info("ViT model training completed")
    except Exception as e:
        logger.error(f"ViT model training failed: {e}")
        return False
    
    # Train Ensemble model
    logger.info("\nTraining Ensemble model...")
    try:
        train_ensemble_main()
        logger.info("Ensemble model training completed")
    except Exception as e:
        logger.error(f"Ensemble model training failed: {e}")
        return False
    
    return True

def display_training_instructions():
    """Display instructions for training with real data"""
    logger.info("\nTRAINING WITH REAL MEDICAL DATA:")
    logger.info("="*80)
    logger.info("\n1. OBTAIN REAL SKIN DISEASE DATASETS FROM:")
    logger.info("   - ISIC Archive: https://www.isic-archive.com/")
    logger.info("   - HAM10000: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T")
    logger.info("   - DermNet: https://dermnetnz.org/")
    logger.info("   Follow all licensing and ethical requirements")
    
    logger.info("\n2. ORGANIZE DATASETS INTO PROPER STRUCTURE:")
    logger.info("   data/train/")
    logger.info("   ├── melanoma/")
    logger.info("   ├── basal_cell_carcinoma/")
    logger.info("   ├── squamous_cell_carcinoma/")
    logger.info("   ├── nevus/")
    logger.info("   ├── dermatofibroma/")
    logger.info("   ├── vascular_lesion/")
    logger.info("   └── actinic_keratosis/")
    
    logger.info("\n3. RUN TRAINING:")
    logger.info("   python train_real_data.py --train")
    
    logger.info("\n4. EVALUATE RESULTS:")
    logger.info("   python -c \"from training.evaluate import main; main()\"")

def main():
    parser = argparse.ArgumentParser(description='Train Skin Disease Model with Real Medical Data')
    parser.add_argument('--data-path', type=str, default='data/train',
                        help='Path to training data (default: data/train)')
    parser.add_argument('--show-datasets', action='store_true',
                        help='Show available public datasets')
    parser.add_argument('--check-data', action='store_true',
                        help='Check data availability only')
    parser.add_argument('--train', action='store_true',
                        help='Train the models with available data')
    parser.add_argument('--create-structure', action='store_true',
                        help='Create directory structure only')
    
    args = parser.parse_args()
    
    logger.info("MEDICAL SKIN DISEASE MODEL TRAINING SYSTEM")
    logger.info("="*80)
    
    if args.show_datasets:
        get_public_skin_datasets()
        return 0
    
    if args.create_structure:
        create_directory_structure()
        display_training_instructions()
        return 0
    
    if args.check_data:
        total_images, class_stats = check_data_availability(args.data_path)
        if total_images == 0:
            logger.error("No training data found")
            get_public_skin_datasets()
        else:
            logger.info(f"Found {total_images} images for training")
        return 0
    
    if args.train:
        # Check data availability before training
        total_images, class_stats = check_data_availability(args.data_path)
        if total_images == 0:
            logger.error("No training data found. Cannot train without data.")
            get_public_skin_datasets()
            return 1
        
        logger.info(f"Training with {total_images} real medical images...")
        success = train_models()
        if success:
            logger.info("Training process completed successfully!")
        else:
            logger.error("Training process failed!")
            return 1
    else:
        # Just show status and recommendations
        create_directory_structure()
        total_images, class_stats = check_data_availability(args.data_path)
        
        logger.info(f"\nCURRENT STATUS:")
        logger.info(f"  Real medical images: {total_images}")
        
        if total_images == 0:
            logger.error("NO TRAINING DATA FOUND!")
            get_public_skin_datasets()
        else:
            logger.info("Data available for training.")
            logger.info("Run with --train flag to start training.")
        
        display_training_instructions()
    
    return 0

if __name__ == "__main__":
    exit(main())
