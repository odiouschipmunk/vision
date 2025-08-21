#!/usr/bin/env python3
"""
Test script to verify fixes for trajectory error and graphics generation
"""

import os
import sys
import time
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Check if virtual environment is active
def check_venv():
    """Check if virtual environment is activated"""
    return hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)

def test_trajectory_fix():
    """Test that the trajectory generation function works without errors"""
    print("🧪 Testing trajectory generation fix...")
    
    # Mock data
    past_ball_pos = [(100, 200), (110, 190), (120, 180), (130, 170), (140, 160)]
    
    class MockShotTracker:
        def __init__(self):
            self.completed_shots = [
                {
                    'id': 1,
                    'trajectory': [(100, 200), (110, 190), (120, 180)],
                    'shot_type': 'drive',
                    'player_who_hit': 'player1',
                    'duration': 0.5
                },
                {
                    'id': 2,
                    'trajectory': [(150, 150), (160, 140), (170, 130)],
                    'shot_type': 'drop',
                    'player_who_hit': 'player2',
                    'duration': 0.3
                }
            ]
            self.active_shots = []
    
    shot_tracker = MockShotTracker()
    
    # Test the fixed function
    try:
        from ef import generate_trajectory_outputs
        
        # Create output directories if they don't exist
        os.makedirs('output/trajectories', exist_ok=True)
        
        outputs = generate_trajectory_outputs(past_ball_pos, shot_tracker)
        
        print(f"✅ Trajectory generation successful! Generated {len(outputs)} outputs")
        for output in outputs:
            print(f"   📄 {output}")
        
        # Verify the JSON file was created
        if os.path.exists('output/trajectories/shot_trajectories.json'):
            with open('output/trajectories/shot_trajectories.json', 'r') as f:
                data = json.load(f)
            print(f"✅ Shot trajectories JSON created with {len(data)} shots")
        else:
            print("❌ Shot trajectories JSON not created")
            
        return True
        
    except Exception as e:
        print(f"❌ Trajectory generation failed: {e}")
        return False

def test_graphics_generation():
    """Test that graphics are being generated properly"""
    print("\n🎨 Testing graphics generation...")
    
    # Mock data
    frame_count = 1000
    
    class MockPlayer:
        def get_latest_pose(self):
            class MockPose:
                def __init__(self):
                    self.xyn = [[[(0.5, 0.5) for _ in range(17)]]]
            return MockPose()
    
    players = {1: MockPlayer(), 2: MockPlayer()}
    past_ball_pos = [(100, 200), (110, 190), (120, 180), (130, 170), (140, 160)]
    
    class MockShotTracker:
        def __init__(self):
            self.completed_shots = [
                {'shot_type': 'drive'},
                {'shot_type': 'drop'},
                {'shot_type': 'drive'}
            ]
            self.active_shots = []
    
    shot_tracker = MockShotTracker()
    
    try:
        from ef import generate_graphics_outputs
        
        # Create output directories if they don't exist
        os.makedirs('output/graphics', exist_ok=True)
        
        outputs = generate_graphics_outputs(frame_count, players, past_ball_pos, shot_tracker)
        
        print(f"✅ Graphics generation successful! Generated {len(outputs)} outputs")
        for output in outputs:
            print(f"   🖼️ {output}")
            if os.path.exists(output):
                file_size = os.path.getsize(output)
                print(f"      📏 File size: {file_size} bytes")
            else:
                print(f"      ❌ File not found: {output}")
        
        return True
        
    except Exception as e:
        print(f"❌ Graphics generation failed: {e}")
        return False

def test_enhanced_visualizations():
    """Test enhanced visualizations generation"""
    print("\n🚀 Testing enhanced visualizations...")
    
    # Mock data
    frame_count = 1000
    
    class MockPlayer:
        def get_latest_pose(self):
            class MockPose:
                def __init__(self):
                    self.xyn = [[[(0.5, 0.5) for _ in range(17)]]]
            return MockPose()
    
    players = {1: MockPlayer(), 2: MockPlayer()}
    past_ball_pos = [(100, 200), (110, 190), (120, 180), (130, 170), (140, 160)]
    
    class MockShotTracker:
        def __init__(self):
            self.completed_shots = [
                {'shot_type': 'drive'},
                {'shot_type': 'drop'},
                {'shot_type': 'drive'}
            ]
            self.active_shots = []
    
    shot_tracker = MockShotTracker()
    
    try:
        from ef import generate_enhanced_visualizations
        
        # Create output directories if they don't exist
        os.makedirs('output/visualizations', exist_ok=True)
        
        outputs = generate_enhanced_visualizations(frame_count, players, past_ball_pos, shot_tracker)
        
        print(f"✅ Enhanced visualizations successful! Generated {len(outputs)} outputs")
        for output in outputs:
            print(f"   🎯 {output}")
            if os.path.exists(output):
                file_size = os.path.getsize(output)
                print(f"      📏 File size: {file_size} bytes")
            else:
                print(f"      ❌ File not found: {output}")
        
        return True
        
    except Exception as e:
        print(f"❌ Enhanced visualizations failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 SQUASH COACHING SYSTEM - FIX VERIFICATION")
    print("=" * 60)
    
    # Check virtual environment
    if check_venv():
        print("✅ Virtual environment is active")
    else:
        print("⚠️ Virtual environment not detected")
        print("💡 Please activate your virtual environment before running tests")
        return False
    
    # Run tests
    trajectory_success = test_trajectory_fix()
    graphics_success = test_graphics_generation()
    enhanced_success = test_enhanced_visualizations()
    
    print("\n" + "=" * 60)
    print("📊 TEST RESULTS SUMMARY:")
    print(f"   🎯 Trajectory Fix: {'✅ PASS' if trajectory_success else '❌ FAIL'}")
    print(f"   🎨 Graphics Generation: {'✅ PASS' if graphics_success else '❌ FAIL'}")
    print(f"   🚀 Enhanced Visualizations: {'✅ PASS' if enhanced_success else '❌ FAIL'}")
    
    if trajectory_success and graphics_success and enhanced_success:
        print("\n🎉 All tests passed! The fixes are working correctly.")
        return True
    else:
        print("\n⚠️ Some tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    main()
