#!/usr/bin/env python3
"""
Ultimate Squash Analysis Launcher
================================

This is the easiest way to run the complete squash coaching analysis.
Simply run this script and provide your video file.

Usage:
    python run_ultimate_analysis.py [video_file]
    
If no video file is provided, it will use the default (self2.mp4)
"""

import sys
import os
import subprocess
import time
from pathlib import Path

def display_startup_banner():
    """Display startup banner"""
    print("🎾" * 50)
    print("🎾" + " " * 48 + "🎾")
    print("🎾         ULTIMATE SQUASH COACHING ANALYSIS       🎾")
    print("🎾              Complete AI-Powered Pipeline       🎾")
    print("🎾" + " " * 48 + "🎾")
    print("🎾" * 50)
    print()

def check_requirements():
    """Check if required files exist"""
    required_files = [
        'ef.py',
        'enhanced_ball_physics.py',
        'enhanced_shot_integration.py',
        'enhanced_coaching_config.py',
        'autonomous_coaching.py',
        'ultimate_squash_insights.py'
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print("❌ Missing required files:")
        for file in missing_files:
            print(f"   • {file}")
        print("\nPlease ensure all files are in the current directory.")
        return False
    
    print("✅ All required files found!")
    return True

def check_video_file(video_path):
    """Check if video file exists"""
    if not os.path.exists(video_path):
        print(f"❌ Video file not found: {video_path}")
        return False
    
    file_size = os.path.getsize(video_path)
    print(f"✅ Video file found: {video_path} ({file_size:,} bytes)")
    return True

def run_analysis(video_path):
    """Run the complete analysis"""
    print(f"\n🚀 Starting Ultimate Squash Analysis...")
    print(f"📹 Video: {video_path}")
    print(f"⏰ Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n" + "=" * 60)
    
    try:
        # Import and run ef.py main function
        print("🔥 Importing analysis modules...")
        import ef
        
        print("🎾 Running complete analysis pipeline...")
        start_time = time.time()
        
        # Run the main analysis
        ef.main(path=video_path)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        print("\n" + "=" * 60)
        print(f"✅ Analysis completed successfully!")
        print(f"⏱️  Total processing time: {processing_time/60:.1f} minutes")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")
        print("Please check the error messages above for details.")
        return False

def display_results():
    """Display analysis results"""
    print(f"\n🎯 Generating comprehensive insights summary...")
    
    try:
        # Run the insights generator
        import ultimate_squash_insights
        ultimate_squash_insights.main()
        
    except Exception as e:
        print(f"⚠️ Error generating insights summary: {e}")
        print("Analysis completed, but summary generation failed.")
        print("You can still check the output/ directory for results.")

def main():
    """Main launcher function"""
    display_startup_banner()
    
    # Check for required files
    if not check_requirements():
        return
    
    # Get video file path
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
    else:
        # Try default files
        default_files = ['self2.mp4', 'self1.mp4', 'squash_video.mp4']
        video_path = None
        
        for default_file in default_files:
            if os.path.exists(default_file):
                video_path = default_file
                break
        
        if not video_path:
            print("❌ No video file specified and no default file found.")
            print("\nUsage: python run_ultimate_analysis.py [video_file]")
            print("Or place a video file named 'self2.mp4' in the current directory.")
            return
    
    # Check video file
    if not check_video_file(video_path):
        return
    
    print(f"\n🎾 ULTIMATE ANALYSIS FEATURES:")
    print("   🧠 Enhanced Ball Physics Detection")
    print("   🎯 Advanced Shot Classification") 
    print("   🤖 AI-Powered Coaching Analysis")
    print("   📊 Comprehensive Visualizations")
    print("   👥 Player Re-identification")
    print("   📈 Performance Benchmarking")
    print("   🎨 Interactive Analytics")
    print("   📋 Personalized Training Plans")
    
    # Confirm start
    response = input(f"\n🚀 Ready to analyze {video_path}? [Y/n]: ").strip().lower()
    if response and response != 'y' and response != 'yes':
        print("Analysis cancelled.")
        return
    
    # Run the analysis
    success = run_analysis(video_path)
    
    if success:
        # Display results
        display_results()
        
        print(f"\n🎾 ULTIMATE ANALYSIS COMPLETE! 🎾")
        print("=" * 60)
        print("📁 Check the output/ directory for all results:")
        print("   • enhanced_autonomous_coaching_report.txt - AI insights")
        print("   • ultimate_coaching_analysis.json - Deep analysis")
        print("   • graphics/ - Visual analytics")
        print("   • enhanced_shots/ - Shot detection data")
        print("   • final.csv - Complete match data")
        print("=" * 60)
        print("🎯 Your personalized squash improvement plan is ready!")
        
    else:
        print("\n❌ Analysis failed. Please check error messages above.")

if __name__ == "__main__":
    main()
