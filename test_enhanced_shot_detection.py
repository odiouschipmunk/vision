#!/usr/bin/env python3
"""
Test script for enhanced shot detection system
"""

import cv2
import numpy as np
import sys
import os

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_enhanced_shot_detection():
    """Test the enhanced shot detection with a simple video"""
    
    print("🎾 Testing Enhanced Shot Detection System")
    print("=" * 50)
    
    try:
        # Import the enhanced ef module
        from ef import main
        
        print("✅ Enhanced ef module imported successfully")
        print("📊 Available enhanced features:")
        print("   • ShotClassificationModel: Advanced shot type detection")
        print("   • PlayerHitDetector: Multi-algorithm player identification") 
        print("   • ShotPhaseDetector: Start/Middle/End phase tracking")
        print("   • Enhanced visualization with phase-based colors")
        print()
        
        # Test with a small frame limit for demonstration
        print("🎬 Testing with video file (limited to 100 frames for demo)")
        print("   Starting enhanced shot detection...")
        
        # Run the enhanced detection on a sample video
        main(path="self2.mp4", input_frame_width=640, input_frame_height=360, max_frames=100)
        
        print("✅ Enhanced shot detection test completed!")
        print()
        print("📈 Check the following outputs:")
        print("   • output/shots_log.jsonl - Detailed shot data")
        print("   • output/enhanced_autonomous_coaching_report.txt - Analysis report")
        print("   • Real-time visualization with phase markers")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("   Make sure all dependencies are installed")
        
    except FileNotFoundError as e:
        print(f"❌ File not found: {e}")
        print("   Make sure video files are in the correct location")
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        print("   Check the error log for details")

def test_shot_classification_features():
    """Test individual shot classification features"""
    
    print("\n🔬 Testing Shot Classification Features")
    print("=" * 40)
    
    try:
        from ef import ShotClassificationModel
        
        # Create test trajectory
        test_trajectory = [
            (100, 200, 1),   # Start position
            (150, 180, 2),   # Moving right and up
            (200, 160, 3),   # Continuing
            (250, 140, 4),   # Still moving
            (300, 120, 5),   # End position
        ]
        
        classifier = ShotClassificationModel()
        result = classifier.classify_shot(test_trajectory, (640, 360))
        
        print(f"✅ Test trajectory classified as: {result['type']}")
        print(f"   Confidence: {result['confidence']:.2f}")
        print(f"   Features extracted: {len(result['features'])} metrics")
        
    except Exception as e:
        print(f"❌ Classification test error: {e}")

def test_player_hit_detection():
    """Test player hit detection features"""
    
    print("\n👥 Testing Player Hit Detection")
    print("=" * 35)
    
    try:
        from ef import PlayerHitDetector
        
        detector = PlayerHitDetector()
        print(f"✅ PlayerHitDetector initialized with {len(detector.hit_detection_methods)} detection methods:")
        print("   • Proximity detection")
        print("   • Trajectory analysis") 
        print("   • Racket position analysis")
        print("   • Movement pattern analysis")
        
    except Exception as e:
        print(f"❌ Player detection test error: {e}")

def test_phase_detection():
    """Test shot phase detection"""
    
    print("\n🎯 Testing Shot Phase Detection")
    print("=" * 35)
    
    try:
        from ef import ShotPhaseDetector
        
        detector = ShotPhaseDetector()
        
        # Test trajectory that should show phase transitions
        test_trajectory = [
            (100, 300, 1),   # Start: back court
            (120, 280, 2),   # Moving toward front
            (140, 260, 3),   # Continuing
            (30, 200, 4),    # Near wall (should trigger middle phase)
            (50, 220, 5),    # Bounced off wall
            (100, 300, 6),   # Back to floor (should trigger end phase)
        ]
        
        phase_result = detector.detect_shot_phases(test_trajectory, (640, 360), 'start')
        
        print(f"✅ Phase detection result: {phase_result['phase']}")
        print(f"   Confidence: {phase_result['confidence']:.2f}")
        print(f"   Transition: {phase_result.get('transition', 'none')}")
        
    except Exception as e:
        print(f"❌ Phase detection test error: {e}")

if __name__ == "__main__":
    print("🚀 Enhanced Shot Detection System - Test Suite")
    print("=" * 60)
    
    # Test individual components
    test_shot_classification_features()
    test_player_hit_detection() 
    test_phase_detection()
    
    # Test full system (commented out by default to avoid long runs)
    # Uncomment the next line to test with actual video
    # test_enhanced_shot_detection()
    
    print("\n✅ All component tests completed!")
    print("\n💡 To test with actual video, uncomment the test_enhanced_shot_detection() call")
    print("   and ensure you have a video file named 'self2.mp4' in the current directory.")
