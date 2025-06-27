#!/usr/bin/env python3
"""
Test script for enhanced squash coaching pipeline with GPU optimization and bounce detection
"""

import sys
import torch
import cv2
import numpy as np

def test_gpu_availability():
    """Test GPU availability and optimization"""
    print("🔧 Testing GPU availability...")
    
    if torch.cuda.is_available():
        print(f"✅ GPU available: {torch.cuda.get_device_name(0)}")
        print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        device = torch.device('cuda')
        
        # Test GPU tensor operations
        try:
            test_tensor = torch.randn(1000, 1000, device=device)
            result = torch.matmul(test_tensor, test_tensor.T)
            print("✅ GPU tensor operations working")
            return True
        except Exception as e:
            print(f"❌ GPU tensor operations failed: {e}")
            return False
    else:
        print("❌ GPU not available, will use CPU")
        return False

def test_bounce_detection():
    """Test enhanced bounce detection algorithm"""
    print("\n🏐 Testing enhanced bounce detection...")
    
    try:
        # Import our enhanced functions
        from ef import detect_ball_bounces_gpu, detect_ball_bounces_cpu
        
        # Create test trajectory with simulated bounces
        test_trajectory = [
            [100, 100, 1], [110, 95, 2], [120, 90, 3], [130, 85, 4],  # Smooth trajectory
            [140, 95, 5], [150, 105, 6], [160, 115, 7],              # Bounce 1
            [170, 110, 8], [180, 105, 9], [190, 100, 10],            # Continue
            [200, 120, 11], [210, 140, 12], [220, 160, 13]           # Bounce 2
        ]
        
        # Test GPU bounce detection
        gpu_bounces = detect_ball_bounces_gpu(test_trajectory)
        print(f"✅ GPU bounce detection found {len(gpu_bounces)} bounces")
        
        # Test CPU bounce detection
        cpu_bounces = detect_ball_bounces_cpu(test_trajectory)
        print(f"✅ CPU bounce detection found {len(cpu_bounces)} bounces")
        
        return True
        
    except Exception as e:
        print(f"❌ Bounce detection test failed: {e}")
        return False

def test_ball_tracker():
    """Test enhanced ball tracker"""
    print("\n🎾 Testing enhanced ball tracker...")
    
    try:
        from balltrack import EnhancedBallTracker
        
        # Initialize tracker
        tracker = EnhancedBallTracker()
        print(f"✅ Ball tracker initialized on {tracker.device}")
        
        # Test prediction functionality
        if hasattr(tracker, 'predict_position_gpu') and tracker.use_gpu:
            print("✅ GPU prediction available")
        else:
            print("📝 Using CPU prediction")
        
        return True
        
    except Exception as e:
        print(f"❌ Ball tracker test failed: {e}")
        return False

def test_autonomous_coaching():
    """Test autonomous coaching system"""
    print("\n🤖 Testing autonomous coaching system...")
    
    try:
        from autonomous_coaching import AutonomousSquashCoach
        
        # Initialize coach
        coach = AutonomousSquashCoach()
        print(f"✅ Autonomous coach initialized")
        
        # Test with dummy data
        dummy_data = [
            {
                'frame': 1,
                'ball_position': [100, 100],
                'wall_bounce_count': 2,
                'match_active': True
            },
            {
                'frame': 2,
                'ball_position': [110, 105],
                'wall_bounce_count': 1,
                'match_active': True
            }
        ]
        
        # Test bounce pattern analysis
        bounce_patterns = coach.analyze_bounce_patterns(dummy_data)
        print(f"✅ Bounce analysis: {bounce_patterns['total_bounces']} bounces detected")
        
        return True
        
    except Exception as e:
        print(f"❌ Autonomous coaching test failed: {e}")
        return False

def test_visualization():
    """Test enhanced visualization components"""
    print("\n🎨 Testing enhanced visualization...")
    
    try:
        # Create test image
        test_image = np.zeros((360, 640, 3), dtype=np.uint8)
        
        # Test bounce visualization
        bounce_positions = [(100, 100), (200, 150), (300, 200)]
        
        for i, bounce_pos in enumerate(bounce_positions):
            # Multi-layer circle visualization
            cv2.circle(test_image, bounce_pos, 15, (0, 255, 255), 3)  # Outer ring
            cv2.circle(test_image, bounce_pos, 8, (0, 255, 255), -1)  # Filled center
            cv2.circle(test_image, bounce_pos, 5, (255, 255, 255), -1)  # White core
            
            # Add label
            cv2.putText(test_image, f"B{i+1}", 
                       (bounce_pos[0] - 10, bounce_pos[1] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 2)
        
        print("✅ Enhanced bounce visualization working")
        return True
        
    except Exception as e:
        print(f"❌ Visualization test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 TESTING ENHANCED SQUASH COACHING PIPELINE")
    print("=" * 60)
    
    tests = [
        test_gpu_availability,
        test_bounce_detection,
        test_ball_tracker,
        test_autonomous_coaching,
        test_visualization
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
    
    print("\n" + "=" * 60)
    print(f"🎯 TEST RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! Enhanced pipeline is ready to use.")
        print("\nTo run the enhanced pipeline:")
        print("python ef.py")
    else:
        print("⚠️  Some tests failed. Check the error messages above.")
        print("The pipeline may still work with reduced functionality.")
    
    print("=" * 60)

if __name__ == "__main__":
    main()
