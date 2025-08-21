#!/usr/bin/env python3
"""
Squash Coaching System Runner
This script ensures proper virtual environment usage and runs the comprehensive squash coaching analysis
"""

import os
import sys
import subprocess
import time

def check_venv():
    """Check if virtual environment is activated"""
    return hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)

def activate_venv():
    """Try to activate virtual environment if not already active"""
    if not check_venv():
        print("🔧 Virtual environment not detected")
        print("💡 Please activate your virtual environment before running this script")
        print("   Example: source venv/bin/activate (Linux/Mac) or venv\\Scripts\\activate (Windows)")
        return False
    return True

def install_requirements():
    """Install required packages if needed"""
    try:
        import torch
        import ultralytics
        import cv2
        import numpy
        import matplotlib
        print("✅ All required packages are available")
        return True
    except ImportError as e:
        print(f"⚠️ Missing package: {e}")
        print("💡 Installing requirements...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
            print("✅ Requirements installed successfully")
            return True
        except subprocess.CalledProcessError:
            print("❌ Failed to install requirements")
            return False

def run_squash_coaching(video_path="self2.mp4", max_frames=None):
    """Run the comprehensive squash coaching analysis"""
    
    print("🎾 SQUASH COACHING SYSTEM")
    print("=" * 50)
    
    # Check virtual environment
    if not activate_venv():
        return False
    
    # Check requirements
    if not install_requirements():
        return False
    
    print(f"🚀 Starting comprehensive squash coaching analysis...")
    print(f"📹 Video: {video_path}")
    if max_frames:
        print(f"🛑 Max frames: {max_frames}")
    print("=" * 50)
    
    # Import and run the main function
    try:
        from ef import main
        start_time = time.time()
        
        # Run the main analysis
        main(path=video_path, max_frames=max_frames)
        
        end_time = time.time()
        print(f"\n✅ Analysis completed in {end_time - start_time:.2f} seconds")
        
        # Show output summary
        print("\n📁 OUTPUTS GENERATED:")
        output_dirs = [
            "output/graphics", "output/clips", "output/heatmaps", "output/highlights",
            "output/patterns", "output/raw_data", "output/stats", "output/reports",
            "output/trajectories", "output/visualizations"
        ]
        
        total_files = 0
        for dir_path in output_dirs:
            if os.path.exists(dir_path):
                files = [f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))]
                if files:
                    print(f"   📂 {dir_path}: {len(files)} files")
                    total_files += len(files)
        
        print(f"\n🎯 Total outputs: {total_files} files")
        print("📋 Check output/comprehensive_output_summary.txt for detailed summary")
        
        return True
        
    except Exception as e:
        print(f"❌ Error running squash coaching analysis: {e}")
        return False

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run comprehensive squash coaching analysis")
    parser.add_argument("--video", default="self2.mp4", help="Path to video file")
    parser.add_argument("--max-frames", type=int, help="Maximum frames to process")
    
    args = parser.parse_args()
    
    success = run_squash_coaching(args.video, args.max_frames)
    
    if success:
        print("\n🎾 SQUASH COACHING ANALYSIS COMPLETE! 🎾")
        sys.exit(0)
    else:
        print("\n❌ Analysis failed")
        sys.exit(1)
