#!/usr/bin/env python3
"""
Enhanced runner script that ensures virtual environment activation and proper error handling
"""

import os
import sys
import subprocess
import time
import platform

def check_venv():
    """Check if virtual environment is activated"""
    return hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)

def activate_venv_auto():
    """Try to automatically activate virtual environment"""
    print("🔧 Checking virtual environment...")
    
    if check_venv():
        print("✅ Virtual environment is already active")
        return True
    
    # Try to find and activate virtual environment
    venv_paths = [
        "venv",
        "env", 
        ".venv",
        ".env"
    ]
    
    for venv_path in venv_paths:
        if os.path.exists(venv_path):
            print(f"📁 Found virtual environment: {venv_path}")
            
            if platform.system() == "Windows":
                activate_script = os.path.join(venv_path, "Scripts", "activate.bat")
                if os.path.exists(activate_script):
                    print("🔄 Activating virtual environment...")
                    try:
                        # Use subprocess to activate and run
                        cmd = f'"{activate_script}" && python ef.py'
                        subprocess.run(cmd, shell=True, check=True)
                        return True
                    except subprocess.CalledProcessError as e:
                        print(f"❌ Failed to activate virtual environment: {e}")
            else:
                activate_script = os.path.join(venv_path, "bin", "activate")
                if os.path.exists(activate_script):
                    print("🔄 Activating virtual environment...")
                    try:
                        # Use subprocess to activate and run
                        cmd = f'source "{activate_script}" && python ef.py'
                        subprocess.run(cmd, shell=True, check=True)
                        return True
                    except subprocess.CalledProcessError as e:
                        print(f"❌ Failed to activate virtual environment: {e}")
    
    print("❌ No virtual environment found")
    print("💡 Please create a virtual environment first:")
    print("   python -m venv venv")
    print("   Then activate it and install requirements:")
    print("   venv\\Scripts\\activate (Windows)")
    print("   source venv/bin/activate (Linux/Mac)")
    print("   pip install -r requirements.txt")
    return False

def install_requirements():
    """Install required packages"""
    print("📦 Checking and installing requirements...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Requirements installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install requirements: {e}")
        return False

def run_analysis(video_path="self2.mp4", max_frames=None):
    """Run the squash coaching analysis"""
    print("🎾 SQUASH COACHING SYSTEM - ENHANCED RUNNER")
    print("=" * 60)
    
    # Check virtual environment
    if not check_venv():
        print("⚠️ Virtual environment not detected")
        if not activate_venv_auto():
            return False
    
    # Install requirements
    if not install_requirements():
        return False
    
    print(f"🚀 Starting comprehensive squash coaching analysis...")
    print(f"📹 Video: {video_path}")
    if max_frames:
        print(f"🛑 Max frames: {max_frames}")
    print("=" * 60)
    
    try:
        # Import and run the main function
        from ef import main
        
        start_time = time.time()
        
        # Run the main analysis
        outputs = main(path=video_path, max_frames=max_frames)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        print(f"\n✅ Analysis completed in {processing_time:.2f} seconds")
        print(f"📊 Generated {len(outputs)} outputs")
        
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
        
        # Check for specific output files
        important_files = [
            "output/graphics/court_positioning.png",
            "output/graphics/shot_analysis.png", 
            "output/graphics/basic_statistics.png",
            "output/trajectories/shot_trajectories.json",
            "output/visualizations/3d_court_visualization.png",
            "output/visualizations/performance_dashboard.png"
        ]
        
        print("\n🔍 Checking key output files:")
        for file_path in important_files:
            if os.path.exists(file_path):
                file_size = os.path.getsize(file_path)
                print(f"   ✅ {file_path} ({file_size} bytes)")
            else:
                print(f"   ❌ {file_path} (not found)")
        
        print("\n📋 Check output/comprehensive_output_summary.txt for detailed summary")
        
        return True
        
    except KeyboardInterrupt:
        print("\n⚠️ Analysis interrupted by user")
        return False
    except Exception as e:
        print(f"\n❌ Error running squash coaching analysis: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced Squash Coaching System Runner")
    parser.add_argument("--video", default="self2.mp4", help="Path to video file")
    parser.add_argument("--max-frames", type=int, help="Maximum frames to process")
    parser.add_argument("--test", action="store_true", help="Run test mode")
    
    args = parser.parse_args()
    
    if args.test:
        # Run test script
        print("🧪 Running in test mode...")
        from test_fixes import main as test_main
        return test_main()
    else:
        # Run main analysis
        return run_analysis(args.video, args.max_frames)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
