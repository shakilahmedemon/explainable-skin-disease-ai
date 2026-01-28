"""
Verification Script
Checks if all components are working correctly
"""
import os
import torch
from pathlib import Path

def verify_setup():
    print("=" * 60)
    print("EXPLAINABLE SKIN DISEASE AI - SYSTEM VERIFICATION")
    print("=" * 60)
    
    # Add project root to path
    import sys
    project_root = Path(__file__).parent
    sys.path.insert(0, str(project_root))
    
    checks_passed = 0
    total_checks = 0
    
    # 1. Check model files
    print("\n1. Checking model files...")
    total_checks += 1
    vit_path = project_root / 'model' / 'vit_skin_disease.pth'
    ensemble_path = project_root / 'model' / 'ensemble_skin_disease.pth'
    
    vit_exists = vit_path.exists()
    ensemble_exists = ensemble_path.exists()
    
    if vit_exists:
        print(f"   ✅ ViT Model: Found ({vit_path.stat().st_size / (1024*1024):.1f} MB)")
        checks_passed += 1
    else:
        print("   ❌ ViT Model: Not found")
    
    if ensemble_exists:
        print(f"   ✅ Ensemble Model: Found ({ensemble_path.stat().st_size / (1024*1024):.1f} MB)")
        checks_passed += 1
    else:
        print("   ❌ Ensemble Model: Not found")
    
    # 2. Check model loading
    print("\n2. Testing model loading...")
    total_checks += 1
    try:
        if vit_exists:
            vit_checkpoint = torch.load(vit_path, map_location='cpu')
            print("   ✅ ViT Model loads successfully")
            checks_passed += 1
        if ensemble_exists:
            ensemble_checkpoint = torch.load(ensemble_path, map_location='cpu')
            print("   ✅ Ensemble Model loads successfully")
            checks_passed += 1
    except Exception as e:
        print(f"   ❌ Model loading failed: {e}")
    
    # 3. Check imports
    print("\n3. Testing imports...")
    total_checks += 1
    try:
        from model.vit_model import ViTModel
        from model.ensemble.ensemble_uncertainty import EnsembleModel
        print("   ✅ Model imports successful")
        checks_passed += 1
    except Exception as e:
        print(f"   ❌ Model imports failed: {e}")
    
    # 4. Check app components
    print("\n4. Testing application components...")
    total_checks += 1
    try:
        from app.utils import preprocess_image
        from app.explainability import get_gradcam_explanation
        print("   ✅ App component imports successful")
        checks_passed += 1
    except Exception as e:
        print(f"   ❌ App component imports failed: {e}")
    
    # 5. Check data directory
    print("\n5. Checking data structure...")
    total_checks += 1
    data_dir = project_root / 'data' / 'train'
    if data_dir.exists():
        classes = [d.name for d in data_dir.iterdir() if d.is_dir()]
        if len(classes) >= 7:
            print(f"   ✅ Data directory: Found {len(classes)} classes")
            checks_passed += 1
        else:
            print(f"   ⚠️  Data directory: Only {len(classes)} classes found (expected 7)")
    else:
        print("   ⚠️  Data directory: Not found (will be created automatically)")
    
    # Summary
    print("\n" + "=" * 60)
    print(f"VERIFICATION COMPLETE: {checks_passed}/{total_checks} checks passed")
    
    if checks_passed == total_checks:
        print("🎉 ALL SYSTEMS READY!")
        print("\nYou can now run the application with:")
        print("   streamlit run app/app.py")
    elif checks_passed >= total_checks * 0.8:
        print("✅ SYSTEM FUNCTIONAL WITH MINOR ISSUES")
        print("The application should work, but consider addressing the warnings above")
    else:
        print("❌ SYSTEM NEEDS ATTENTION")
        print("Please fix the issues before running the application")
    
    print("=" * 60)

if __name__ == "__main__":
    verify_setup()
