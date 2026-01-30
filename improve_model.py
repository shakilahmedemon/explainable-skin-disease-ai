"""
Script to improve the skin disease AI model by training it with proper data
This addresses the core issue of poor recognition by implementing proper training
"""
import os
import sys
import shutil
import argparse
from pathlib import Path

def print_header(title):
    """Print a formatted header"""
    print("\n" + "="*60)
    print(f"{title:^60}")
    print("="*60)

def check_prerequisites():
    """Check if all prerequisites are met"""
    print_header("CHECKING PREREQUISITES")
    
    # Check Python version
    if sys.version_info < (3, 7):
        print("❌ Python 3.7+ required")
        return False
    else:
        print(f"✅ Python version: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    
    # Check required packages
    required_packages = [
        ('torch', 'torch'),
        ('torchvision', 'torchvision'),
        ('timm', 'timm'),
        ('numpy', 'numpy'),
        ('PIL', 'PIL.Image'),
        ('sklearn', 'sklearn'),
    ]
    
    missing_packages = []
    for import_name, var_name in required_packages:
        try:
            if '.' in var_name:
                parts = var_name.split('.')
                module = __import__(import_name)
                for part in parts[1:]:
                    module = getattr(module, part)
            else:
                module = __import__(import_name)
            print(f"✅ {import_name}")
        except ImportError:
            missing_packages.append(import_name)
            print(f"❌ {import_name}")
    
    if missing_packages:
        print(f"\n❌ Missing packages: {', '.join(missing_packages)}")
        print("Install them using: pip install -r requirements.txt")
        return False
    
    return True

def setup_data_directories():
    """Create proper data directory structure"""
    print_header("SETTING UP DATA DIRECTORIES")
    
    # Define the directory structure
    base_path = Path("data/train")
    classes = [
        "melanoma", 
        "basal_cell_carcinoma", 
        "squamous_cell_carcinoma",
        "nevus", 
        "dermatofibroma", 
        "vascular_lesion", 
        "actinic_keratosis"
    ]
    
    print(f"Creating directory structure at: {base_path}")
    
    for class_name in classes:
        class_path = base_path / class_name
        class_path.mkdir(parents=True, exist_ok=True)
        img_count = len(list(class_path.glob("*.[jJ][pP][gG]"))) + \
                   len(list(class_path.glob("*.[pP][nN][gG]"))) + \
                   len(list(class_path.glob("*.[jJ][pP][eE][gG]")))
        print(f"  📁 {class_name}: {img_count} images")
    
    print(f"\n✅ Directory structure created at: {base_path}")
    return base_path

def check_data_availability(data_path):
    """Check how much data is available for training"""
    print_header("CHECKING DATA AVAILABILITY")
    
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
        class_path = data_path / class_name
        if class_path.exists():
            # Count image files
            img_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
            img_count = 0
            for ext in img_extensions:
                img_count += len(list(class_path.glob(ext)))
            
            class_stats[class_name] = img_count
            total_images += img_count
            print(f"  {class_name:<25}: {img_count:>5} images")
        else:
            print(f"  {class_name:<25}: {'DIR NOT FOUND':>5}")
    
    print(f"\n📊 Total images: {total_images}")
    
    # Assess data sufficiency
    if total_images == 0:
        print("❌ No images found. Model training will fail without data.")
        print("\n💡 To fix this:")
        print("   1. Obtain a skin disease dataset (e.g., ISIC Archive, HAM10000, DermNet)")
        print("   2. Place images in the appropriate class directories")
        print("   3. Run this script again")
    elif total_images < 1000:
        print("⚠️  Limited data. Model performance may be poor with <1000 images.")
        print("   Aim for at least 1000+ images total, with 100+ per class for good results.")
    else:
        print("✅ Sufficient data for initial training. More data will improve performance.")
    
    return total_images, class_stats

def display_training_command():
    """Display the command to train the model"""
    print_header("TRAINING COMMANDS")
    print("To train the models, run:")
    print("  python train.py")
    print("\nFor more detailed training with logging:")
    print("  python train.py --help")
    print("\nThe training will:")
    print("  • Load your images from data/train/")
    print("  • Split into training and validation sets")
    print("  • Train both ViT and Ensemble models")
    print("  • Save the best models to the model/ directory")
    print("  • Apply data augmentation to improve generalization")

def validate_model_performance():
    """Validate if models have been trained"""
    print_header("MODEL VALIDATION")
    
    vit_model_path = Path("model/vit_skin_disease.pth")
    ensemble_model_path = Path("model/ensemble_skin_disease.pth")
    
    models_trained = 0
    
    if vit_model_path.exists():
        size_mb = vit_model_path.stat().st_size / (1024 * 1024)
        print(f"✅ ViT Model: Found ({size_mb:.1f} MB)")
        models_trained += 1
    else:
        print("❌ ViT Model: Not found - needs training")
    
    if ensemble_model_path.exists():
        size_mb = ensemble_model_path.stat().st_size / (1024 * 1024)
        print(f"✅ Ensemble Model: Found ({size_mb:.1f} MB)")
        models_trained += 1
    else:
        print("❌ Ensemble Model: Not found - needs training")
    
    if models_trained == 2:
        print("\n🎉 Both models are trained and ready!")
        print("You can now run the application: streamlit run app/app.py")
    elif models_trained == 1:
        print("\n⚠️  Only one model is trained. Run training again to get both models.")
    else:
        print("\n❌ No models trained yet. Run training first.")

def provide_dataset_recommendations():
    """Provide recommendations for obtaining skin disease datasets"""
    print_header("DATASET RECOMMENDATIONS")
    print("To improve model accuracy, use these public skin lesion datasets:")
    print()
    print("1. 🔬 ISIC Archive (International Skin Imaging Collaboration)")
    print("   URL: https://www.isic-archive.com/")
    print("   Contains: 50,000+ dermoscopic images")
    print("   License: Free for research")
    print()
    print("2. 🏥 HAM10000 (Human Against Machine with 10000 training images)")
    print("   Contains: 10,015 dermatoscopic images")
    print("   Classes: 7 diagnostic categories")
    print()
    print("3. 🌐 DermNet")
    print("   URL: https://dermnetnz.org/")
    print("   Contains: Clinical images of skin diseases")
    print()
    print("💡 TIPS FOR DATASET PREPARATION:")
    print("   • Ensure balanced classes (similar number of images per class)")
    print("   • Clean and preprocess images consistently")
    print("   • Remove duplicates and low-quality images")
    print("   • Split data into train/val/test sets (70/15/15 recommended)")

def main():
    parser = argparse.ArgumentParser(description='Improve Skin Disease AI Model')
    parser.add_argument('--validate-only', action='store_true',
                        help='Only validate current setup without training')
    parser.add_argument('--setup-data', action='store_true',
                        help='Only setup data directories')
    parser.add_argument('--check-data', action='store_true',
                        help='Only check data availability')
    
    args = parser.parse_args()
    
    print("🏥 SKIN DISEASE AI MODEL IMPROVEMENT TOOL")
    print("This tool helps address poor recognition by training models with proper data")
    
    if not args.setup_data and not args.check_data:
        # Check prerequisites
        if not check_prerequisites():
            print("\n❌ Prerequisites not met. Please install required packages.")
            return 1
    
    # Setup data directories
    if args.setup_data or not args.validate_only:
        data_path = setup_data_directories()
    else:
        data_path = Path("data/train")
    
    # Check data availability
    if args.check_data or not args.validate_only:
        total_images, class_stats = check_data_availability(data_path)
    
    # Validate model status
    validate_model_performance()
    
    # Display training commands
    if not args.check_data and not args.setup_data:
        display_training_command()
    
    # Provide dataset recommendations
    if total_images == 0 or args.check_data:
        provide_dataset_recommendations()
    
    print("\n" + "="*60)
    print("SUMMARY OF NEXT STEPS:")
    print("="*60)
    
    if total_images == 0:
        print("1. 📥 OBTAIN A SKIN DISEASE DATASET (see recommendations above)")
        print("2. 📁 ORGANIZE IMAGES IN data/train/class_name/ DIRECTORIES")
        print("3. 🚀 RUN: python train.py")
    elif total_images < 1000:
        print("1. 📈 CONSIDER ADDING MORE TRAINING DATA")
        print("2. 🚀 RUN: python train.py")
    else:
        print("1. 🚀 RUN: python train.py")
        print("2. 🧪 TEST: streamlit run app/app.py")
    
    return 0

if __name__ == "__main__":
    exit(main())
