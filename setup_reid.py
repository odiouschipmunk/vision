#!/usr/bin/env python3
"""
Enhanced Player ReID System Installation Script
===============================================

This script helps set up the enhanced player re-identification system.
"""

import subprocess
import sys
import os
import importlib.util

def check_package(package_name):
    """Check if a package is installed"""
    spec = importlib.util.find_spec(package_name)
    return spec is not None

def install_package(package_name):
    """Install a package using pip"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
        return True
    except subprocess.CalledProcessError:
        return False

def check_gpu_support():
    """Check if GPU support is available"""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False

def main():
    print("🔧 Enhanced Player ReID System Setup")
    print("=" * 40)
    
    # Required packages for ReID system
    required_packages = [
        "torch",
        "torchvision", 
        "opencv-python",
        "numpy",
        "pillow",
        "scikit-learn"
    ]
    
    # Check existing installations
    print("\n📋 Checking existing installations...")
    missing_packages = []
    
    for package in required_packages:
        if check_package(package):
            print(f"✅ {package} - already installed")
        else:
            print(f"❌ {package} - missing")
            missing_packages.append(package)
    
    # Install missing packages
    if missing_packages:
        print(f"\n📦 Installing {len(missing_packages)} missing packages...")
        for package in missing_packages:
            print(f"Installing {package}...")
            if install_package(package):
                print(f"✅ {package} installed successfully")
            else:
                print(f"❌ Failed to install {package}")
                return False
    
    # Check GPU support
    print(f"\n🖥️  GPU Support: {'✅ Available' if check_gpu_support() else '❌ Not available (CPU only)'}")
    
    # Test ReID system
    print("\n🧪 Testing ReID system...")
    try:
        from enhanced_player_reid import EnhancedPlayerReID
        reid_system = EnhancedPlayerReID()
        print("✅ ReID system initialized successfully")
        
        # Test feature extraction
        import cv2
        import numpy as np
        
        test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        features = reid_system.extract_appearance_features(test_image)
        
        if features is not None and len(features) > 0:
            print("✅ Feature extraction working")
        else:
            print("❌ Feature extraction failed")
            return False
            
    except Exception as e:
        print(f"❌ ReID system test failed: {e}")
        return False
    
    # Create necessary directories
    print("\n📁 Creating output directories...")
    directories = ["output", "reid_test_output", "logs"]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ {directory}/ directory ready")
    
    # Installation complete
    print("\n🎉 Installation Complete!")
    print("=" * 40)
    print("✅ Enhanced Player ReID system is ready to use")
    print("✅ All required packages installed")
    print("✅ Output directories created")
    
    if check_gpu_support():
        print("✅ GPU acceleration enabled")
    else:
        print("ℹ️  Using CPU (consider installing CUDA for better performance)")
    
    print("\n🚀 Next steps:")
    print("   1. Run your squash analysis: python ef.py")
    print("   2. Test ReID system: python test_reid_system.py --video your_video.mp4")
    print("   3. Check README_REID.md for detailed usage instructions")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
