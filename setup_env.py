import subprocess
import sys
import os

def install_packages():
    """Install required packages from requirements.txt"""
    print("Installing required packages...")
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✓ Successfully installed packages from requirements.txt")
    except subprocess.CalledProcessError:
        print("✗ Failed to install packages from requirements.txt")
        return False
    
    # Additional packages that might be needed
    additional_packages = [
        "shap",      # For SHAP explanations
        "lime",      # For LIME explanations
        "scipy",     # For scientific computations
        "tqdm",      # For progress bars
        "seaborn"    # For statistical data visualization
    ]
    
    for package in additional_packages:
        try:
            __import__(package)
            print(f"✓ {package} is already installed")
        except ImportError:
            print(f"Installing {package}...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                print(f"✓ Successfully installed {package}")
            except subprocess.CalledProcessError:
                print(f"✗ Failed to install {package}")
    
    return True

def setup_environment():
    """Set up the Python environment for the project"""
    print("Setting up Python environment...")
    
    # Check Python version
    python_version = sys.version_info
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 7):
        print(f"✗ Python 3.7 or higher is required. Current version: {sys.version}")
        return False
    else:
        print(f"✓ Python version: {sys.version}")
    
    # Install packages
    if not install_packages():
        return False
    
    # Create necessary directories
    dirs = [
        'data/raw',
        'data/processed', 
        'model',
        'results',
        'training_logs',
        'samples'
    ]
    
    for directory in dirs:
        os.makedirs(directory, exist_ok=True)
        print(f"✓ Created directory: {directory}")
    
    print("✓ Environment setup completed!")
    return True

def verify_installation():
    """Verify that all critical packages are installed and working"""
    print("Verifying installation...")
    
    required_packages = [
        ('torch', 'torch'),
        ('torchvision', 'torchvision'), 
        ('timm', 'timm'),
        ('numpy', 'np'),
        ('sklearn', 'sklearn'),
        ('cv2', 'cv2'),
        ('PIL', 'PIL.Image'),
        ('streamlit', 'streamlit'),
        ('matplotlib', 'matplotlib.pyplot')
    ]
    
    missing_packages = []
    
    for import_name, var_name in required_packages:
        try:
            if '.' in var_name:
                module_parts = var_name.split('.')
                module = __import__(import_name)
                for part in module_parts[1:]:
                    module = getattr(module, part)
            else:
                exec(f"import {import_name}")
            print(f"✓ {import_name} is available")
        except ImportError:
            missing_packages.append(import_name)
            print(f"✗ {import_name} is missing")
    
    if missing_packages:
        print(f"\n✗ Missing packages: {', '.join(missing_packages)}")
        return False
    else:
        print("\n✓ All required packages are available")
        return True

def create_sample_config():
    """Create a sample configuration file"""
    config_content = '''# Configuration for Skin Disease Classification AI

# Model settings
MODEL_TYPE = "vit"  # Options: "vit", "ensemble"
NUM_CLASSES = 7

# Training settings
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
NUM_EPOCHS = 20
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Data settings
IMAGE_SIZE = (224, 224)
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# Explainability settings
GRADCAM_LAYER = "vit.blocks[-1].norm1"  # Layer to use for GradCAM
'''
    
    with open('config.py', 'w') as f:
        f.write(config_content)
    
    print("✓ Created sample configuration file: config.py")

def main():
    print("🔧 Setting up Explainable Skin Disease AI Environment")
    print("="*60)
    
    # Setup environment
    if not setup_environment():
        print("✗ Environment setup failed")
        return False
    
    # Verify installation
    if not verify_installation():
        print("✗ Installation verification failed")
        return False
    
    # Create sample config
    create_sample_config()
    
    print("\n" + "="*60)
    print("✅ Environment setup completed successfully!")
    print("\nTo start training, run: python train.py")
    print("To run the application, run: streamlit run app/app.py")
    print("="*60)
    
    return True

if __name__ == "__main__":
    main()
