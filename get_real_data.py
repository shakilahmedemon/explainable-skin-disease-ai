"""
Download and prepare real skin disease datasets for training
This script helps address the poor recognition by using actual medical images
"""
import os
import requests
import zipfile
import tarfile
from pathlib import Path
import shutil
import argparse
import sys
from urllib.parse import urlparse

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
    
    print(f"✅ Directory structure created at: {base_path}")
    return base_path

def download_file(url, destination, chunk_size=8192):
    """Download a file from URL with progress indication"""
    print(f"📥 Downloading {url}...")
    
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded_size = 0
        
        with open(destination, 'wb') as file:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    file.write(chunk)
                    downloaded_size += len(chunk)
                    
                    if total_size > 0:
                        percent = (downloaded_size / total_size) * 100
                        print(f"\rProgress: {percent:.1f}% ({downloaded_size}/{total_size} bytes)", end='', flush=True)
        
        print(f"\n✅ Download completed: {destination}")
        return True
    except Exception as e:
        print(f"\n❌ Download failed: {e}")
        return False

def extract_archive(archive_path, extract_to):
    """Extract ZIP or TAR archive"""
    print(f"📦 Extracting {archive_path}...")
    
    try:
        if archive_path.endswith('.zip'):
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
        elif archive_path.endswith(('.tar', '.tar.gz', '.tgz')):
            with tarfile.open(archive_path, 'r') as tar_ref:
                tar_ref.extractall(extract_to)
        else:
            print(f"❌ Unsupported archive format: {archive_path}")
            return False
        
        print(f"✅ Extraction completed to: {extract_to}")
        return True
    except Exception as e:
        print(f"❌ Extraction failed: {e}")
        return False

def organize_isic_data(download_path):
    """Organize ISIC dataset into our class structure (conceptual - real implementation would require proper metadata)"""
    print("🔄 Organizing ISIC data (conceptual)...")
    
    # In a real scenario, we would parse the metadata to organize images by diagnosis
    # For demonstration, this shows the concept
    
    # Example of how real ISIC data would be organized:
    # - Parse metadata.csv to get diagnosis for each image
    # - Move images to appropriate class folders based on diagnosis
    
    print("ℹ️  ISIC dataset would be organized by diagnosis from metadata")
    print("ℹ️  Real implementation would require the metadata file to map images to classes")

def get_public_skin_datasets():
    """Provide information about publicly available skin disease datasets"""
    datasets = {
        "ISIC Archive": {
            "url": "https://www.isic-archive.com/",
            "description": "The International Skin Imaging Collaboration: largest collection of dermoscopic images",
            "size": "~50,000+ images",
            "note": "Requires registration for download"
        },
        "HAM10000": {
            "url": "https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T",
            "description": "Human Against Machine with 10,000+ skin lesion images",
            "size": "10,015 images",
            "note": "7 diagnostic categories"
        },
        "DermNet": {
            "url": "https://dermnetnz.org/",
            "description": "Clinical images of skin diseases",
            "size": "~23,000+ images",
            "note": "Free access with proper attribution"
        },
        "BCN_20000": {
            "url": "https://github.com/GIMVI/dataset-bcn_20000",
            "description": "20,000 clinical photographs of skin lesions",
            "size": "20,000 images",
            "note": "Requires citation if used"
        }
    }
    
    print("🌐 AVAILABLE PUBLIC SKIN DISEASE DATASETS:")
    print("="*60)
    
    for name, info in datasets.items():
        print(f"\n{name}:")
        print(f"  📖 Description: {info['description']}")
        print(f"  💾 Size: {info['size']}")
        print(f"  🔗 URL: {info['url']}")
        print(f"  ℹ️  Note: {info['note']}")
    
    return datasets

def download_sample_images():
    """Download sample images to demonstrate the concept (using placeholder URLs)"""
    print("📷 Downloading sample skin lesion images...")
    
    # These are placeholder URLs - in reality, you'd use actual dataset URLs
    sample_urls = [
        # Note: These are illustrative - real URLs would point to actual skin lesion images
        # For actual implementation, you would use official dataset download links
    ]
    
    download_path = Path("downloads")
    download_path.mkdir(exist_ok=True)
    
    print("ℹ️  For actual implementation:")
    print("   1. Visit the dataset websites mentioned above")
    print("   2. Download the datasets legally and ethically")
    print("   3. Organize images into the appropriate class folders")
    print("   4. Run the training script")
    
    return True

def prepare_training_data():
    """Prepare data for training by organizing it properly"""
    print("🧹 Preparing training data...")
    
    train_path = Path("data/train")
    val_path = Path("data/val")
    test_path = Path("data/test")
    
    # Create validation and test directories
    for path in [val_path, test_path]:
        path.mkdir(exist_ok=True)
        for class_name in (train_path).iterdir():
            if class_name.is_dir():
                (path / class_name.name).mkdir(exist_ok=True)
    
    print("✅ Training data structure prepared")
    print(f"   Train: {train_path}")
    print(f"   Val: {val_path}")
    print(f"   Test: {test_path}")

def display_training_instructions():
    """Display instructions for training with real data"""
    print("\n🎯 TRAINING WITH REAL DATA:")
    print("="*60)
    print("\n1. 📥 OBTAIN REAL SKIN DISEASE DATASETS:")
    print("   - Visit: https://www.isic-archive.com/")
    print("   - Or: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T")
    print("   - Download datasets following their terms of use")
    
    print("\n2. 📁 ORGANIZE DATASETS INTO PROPER STRUCTURE:")
    print("   data/train/")
    print("   ├── melanoma/")
    print("   ├── basal_cell_carcinoma/")
    print("   ├── squamous_cell_carcinoma/")
    print("   ├── nevus/")
    print("   ├── dermatofibroma/")
    print("   ├── vascular_lesion/")
    print("   └── actinic_keratosis/")
    
    print("\n3. 🚀 RUN TRAINING:")
    print("   python train.py")
    
    print("\n4. 🧪 TEST IMPROVED MODEL:")
    print("   streamlit run app/app.py")

def main():
    parser = argparse.ArgumentParser(description='Prepare Real Skin Disease Dataset for Training')
    parser.add_argument('--show-datasets', action='store_true',
                        help='Show available public datasets')
    parser.add_argument('--prepare-structure', action='store_true',
                        help='Create directory structure only')
    parser.add_argument('--download-samples', action='store_true',
                        help='Download sample images (conceptual)')
    
    args = parser.parse_args()
    
    print("🏥 PREPARING REAL SKIN DISEASE DATASET FOR IMPROVED RECOGNITION")
    print("="*70)
    
    if args.show_datasets:
        get_public_skin_datasets()
        return
    
    if args.prepare_structure:
        create_directory_structure()
        prepare_training_data()
        display_training_instructions()
        return
    
    if args.download_samples:
        download_sample_images()
        return
    
    # Default: Full preparation
    print("Step 1: Creating directory structure...")
    data_path = create_directory_structure()
    
    print("\nStep 2: Available public datasets...")
    datasets = get_public_datasets()
    
    print("\nStep 3: Preparing data structure...")
    prepare_training_data()
    
    print("\nStep 4: Training instructions...")
    display_training_instructions()
    
    print(f"\n✅ Setup complete! Data path: {data_path}")
    print("💡 Remember: For best results, use real medical-grade skin lesion images")

if __name__ == "__main__":
    main()
