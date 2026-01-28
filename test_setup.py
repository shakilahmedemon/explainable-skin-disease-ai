"""
Test script to verify the skin disease AI application components
"""
import torch
import sys
import os

def test_imports():
    """Test that all required modules can be imported"""
    print("Testing imports...")
    
    modules_to_test = [
        "torch",
        "torchvision", 
        "timm",
        "numpy",
        "cv2",
        "PIL.Image",
        "sklearn",
        "streamlit",
    ]
    
    failed_imports = []
    
    for module_path in modules_to_test:
        try:
            if '.' in module_path:
                parts = module_path.split('.')
                module = __import__(parts[0])
                for part in parts[1:]:
                    module = getattr(module, part)
            else:
                module = __import__(module_path)
            print(f"✓ {module_path}")
        except ImportError as e:
            print(f"✗ {module_path}: {e}")
            failed_imports.append((module_path, str(e)))
    
    # Test matplotlib separately since it has submodules
    try:
        import matplotlib.pyplot
        print("✓ matplotlib.pyplot")
    except ImportError as e:
        print(f"✗ matplotlib.pyplot: {e}")
        failed_imports.append(("matplotlib.pyplot", str(e)))
    
    return len(failed_imports) == 0

def test_model_creation():
    """Test that we can create model instances"""
    print("\nTesting model creation...")
    
    try:
        from model.vit_model import ViTModel
        from model.ensemble.ensemble_uncertainty import EnsembleModel
        
        # Test ViT model creation
        vit_model = ViTModel(num_classes=7)
        print(f"✓ ViT Model created with {sum(p.numel() for p in vit_model.parameters()):,} parameters")
        
        # Test Ensemble model creation
        ensemble_model = EnsembleModel(num_classes=7)
        print(f"✓ Ensemble Model created with {sum(p.numel() for p in ensemble_model.parameters()):,} parameters")
        
        return True
    except Exception as e:
        print(f"✗ Model creation failed: {e}")
        return False

def test_app_components():
    """Test that app components can be imported"""
    print("\nTesting app components...")
    
    components = [
        "app.explainability",
        "app.medical_knowledge", 
        "app.safety",
        "app.utils"
    ]
    
    failed_components = []
    
    for component in components:
        try:
            __import__(component)
            print(f"✓ {component}")
        except ImportError as e:
            print(f"✗ {component}: {e}")
            failed_components.append((component, str(e)))
    
    return len(failed_components) == 0

def test_training_components():
    """Test that training components can be imported"""
    print("\nTesting training components...")
    
    components = [
        "training.train_single_model",
        "training.train_ensemble",
        "training.evaluate",
        "training.calibration"
    ]
    
    failed_components = []
    
    for component in components:
        try:
            __import__(component)
            print(f"✓ {component}")
        except ImportError as e:
            print(f"✗ {component}: {e}")
            failed_components.append((component, str(e)))
    
    return len(failed_components) == 0

def run_basic_model_test():
    """Run a basic model test with dummy data"""
    print("\nRunning basic model test...")
    
    try:
        from model.vit_model import ViTModel
        import torch
        
        # Create a model
        model = ViTModel(num_classes=7)
        model.eval()
        
        # Create dummy input (batch_size=1, channels=3, height=224, width=224)
        dummy_input = torch.randn(1, 3, 224, 224)
        
        # Run forward pass
        with torch.no_grad():
            output = model(dummy_input)
        
        print(f"✓ Model forward pass successful. Output shape: {output.shape}")
        
        # Check that output has correct number of classes
        assert output.shape[1] == 7, f"Expected 7 classes, got {output.shape[1]}"
        print("✓ Output has correct shape for 7 skin disease classes")
        
        return True
    except Exception as e:
        print(f"✗ Basic model test failed: {e}")
        return False

def main():
    print("🧪 Testing Explainable Skin Disease AI Components")
    print("="*60)
    
    # Change to project directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    all_tests_passed = True
    
    # Run all tests
    all_tests_passed &= test_imports()
    all_tests_passed &= test_model_creation()
    all_tests_passed &= test_app_components()
    all_tests_passed &= test_training_components()
    all_tests_passed &= run_basic_model_test()
    
    print("\n" + "="*60)
    if all_tests_passed:
        print("🎉 All tests passed! The application is ready to use.")
        print("\nTo run the application:")
        print("  streamlit run app/app.py")
        print("\nTo train the models:")
        print("  python train.py")
    else:
        print("❌ Some tests failed. Please check the installation.")
    
    print("="*60)
    return all_tests_passed

if __name__ == "__main__":
    main()
