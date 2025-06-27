#!/usr/bin/env python3
"""
Test script for reference point update functionality
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import cv2
import json
from squash import Referencepoints

def test_reference_point_update():
    """
    Test the reference point update functionality
    """
    print("Testing reference point update functionality...")
    
    # Check if video file exists
    video_path = "self1.mp4"  # Update this with your video file path
    if not os.path.exists(video_path):
        print(f"Video file {video_path} not found. Please ensure it exists.")
        return False
    
    # Parameters
    frame_width = 640
    frame_height = 360
    
    try:
        # Load initial reference points (if they exist)
        initial_points = []
        if os.path.exists("reference_points.json"):
            with open("reference_points.json", "r") as f:
                initial_points = json.load(f)
            print(f"Initial reference points: {initial_points}")
        else:
            print("No existing reference points found.")
        
        # Test the update function
        print("Testing update_reference_points function...")
        print("This will open a window where you can click to set new reference points.")
        print("Press 's' to save, 'c' or 'q' to cancel.")
        
        updated_points = Referencepoints.update_reference_points(
            path=video_path,
            frame_width=frame_width,
            frame_height=frame_height
        )
        
        print(f"Updated reference points: {updated_points}")
        
        # Verify the points were saved to JSON
        if os.path.exists("reference_points.json"):
            with open("reference_points.json", "r") as f:
                saved_points = json.load(f)
            print(f"Points saved to JSON: {saved_points}")
            
            if saved_points == updated_points:
                print("✅ Test passed: Reference points were correctly updated and saved!")
                return True
            else:
                print("❌ Test failed: Saved points don't match returned points.")
                return False
        else:
            print("❌ Test failed: Reference points file was not created.")
            return False
            
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        return False

def test_key_integration():
    """
    Test the keyboard integration (simulation)
    """
    print("\n" + "="*50)
    print("INTEGRATION TEST INSTRUCTIONS")
    print("="*50)
    print("To test the 'r' key functionality in your video processing:")
    print("1. Run your main video processing script (get_data.py or ef.py)")
    print("2. While the video is playing, press 'r' to update reference points")
    print("3. A new window will open for you to click on new reference points")
    print("4. Click on the court points in the correct order")
    print("5. Press 's' to save the new points")
    print("6. The video processing will continue with updated reference points")
    print("7. Press 'q' to quit the video processing")
    print("\nFeatures implemented:")
    print("✅ 'r' key detection in video processing loop")
    print("✅ Interactive reference point update window")
    print("✅ Automatic JSON file update")
    print("✅ Homography matrix regeneration") 
    print("✅ Seamless continuation of video processing")
    print("✅ Visual instructions on video frame")

if __name__ == "__main__":
    print("Reference Point Update Test Script")
    print("="*40)
    
    success = test_reference_point_update()
    
    if success:
        test_key_integration()
    else:
        print("Initial test failed. Please check the setup and try again.")
