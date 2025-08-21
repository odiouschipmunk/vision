#!/usr/bin/env python3
"""
Simple script to run the squash coaching analysis with proper error handling
"""

import os
import sys
import time

def check_venv():
    """Check if virtual environment is active"""
    return hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)

def main():
    """Main function"""
    print("🎾 SQUASH COACHING SYSTEM")
    print("=" * 50)
    
    # Check virtual environment
    if check_venv():
        print("✅ Virtual environment is active")
    else:
        print("⚠️ Virtual environment not detected")
        print("💡 Please activate your virtual environment before running this script")
        print("   Example: source venv/bin/activate (Linux/Mac) or venv\\Scripts\\activate (Windows)")
        return False
    
    # Check if video file exists
    video_path = "self2.mp4"
    if not os.path.exists(video_path):
        print(f"❌ Video file not found: {video_path}")
        print("💡 Please ensure the video file exists in the current directory")
        return False
    
    print(f"📹 Video file found: {video_path}")
    print("🚀 Starting analysis...")
    print("=" * 50)
    
    try:
        # Import and run the main function
        from ef import main as ef_main
        
        start_time = time.time()
        
        # Run the main analysis
        outputs = ef_main(path=video_path, max_frames=100)  # Limit to 100 frames for testing
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        print(f"\n✅ Analysis completed in {processing_time:.2f} seconds")
        print(f"📊 Generated {len(outputs)} outputs")
        
        # Check for key output files
        print("\n🔍 Checking key output files:")
        key_files = [
            "output/graphics/court_positioning.png",
            "output/graphics/shot_analysis.png",
            "output/graphics/basic_statistics.png",
            "output/trajectories/shot_trajectories.json",
            "output/visualizations/3d_court_visualization.png",
            "output/visualizations/performance_dashboard.png"
        ]
        
        for file_path in key_files:
            if os.path.exists(file_path):
                file_size = os.path.getsize(file_path)
                print(f"   ✅ {file_path} ({file_size} bytes)")
            else:
                print(f"   ❌ {file_path} (not found)")
        
        return True
        
    except KeyboardInterrupt:
        print("\n⚠️ Analysis interrupted by user")
        return False
    except Exception as e:
        print(f"\n❌ Error running analysis: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
