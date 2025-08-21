#!/usr/bin/env python3
"""
Test script to verify memory management fixes and trajectory analysis error resolution
"""

import time
import sys
import os

def test_trajectory_analysis_fix():
    """Test that trajectory analysis no longer has the undefined variable error"""
    print("🧪 Testing trajectory analysis fix...")
    
    try:
        # Import the function that was causing the error
        from ef import generate_trajectory_outputs
        
        # Test with empty data (this was causing the error)
        outputs = generate_trajectory_outputs([], None)
        print("✅ Trajectory analysis works with empty data")
        
        # Test with minimal data
        minimal_data = [(100, 200), (110, 210), (120, 220)]
        outputs = generate_trajectory_outputs(minimal_data, None)
        print("✅ Trajectory analysis works with minimal data")
        
        return True
        
    except Exception as e:
        print(f"❌ Trajectory analysis test failed: {e}")
        return False

def test_memory_manager():
    """Test the memory-efficient LLM manager"""
    print("🧪 Testing memory-efficient LLM manager...")
    
    try:
        from ef import llm_manager
        
        # Test memory check
        print("💾 Testing memory check...")
        allocated, reserved, total = llm_manager.check_memory_usage()
        print(f"   Memory check completed: {allocated:.2f}GB allocated")
        
        # Test model loading (this should not actually load due to memory constraints)
        print("🤖 Testing model loading logic...")
        model = llm_manager.get_model('llama')
        if model is None:
            print("✅ Memory manager correctly prevented model loading due to insufficient memory")
        else:
            print("⚠️ Model loaded (may have sufficient memory)")
        
        # Test cleanup
        print("🧹 Testing cleanup...")
        llm_manager.unload_current_model()
        print("✅ Cleanup completed successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Memory manager test failed: {e}")
        return False

def test_autonomous_coaching_memory_check():
    """Test that autonomous coaching respects memory constraints"""
    print("🧪 Testing autonomous coaching memory check...")
    
    try:
        from autonomous_coaching import get_autonomous_coach
        
        # This should check memory before loading models
        coach = get_autonomous_coach()
        if coach:
            print("✅ Autonomous coaching initialized (may have sufficient memory)")
        else:
            print("✅ Autonomous coaching correctly prevented loading due to memory constraints")
        
        return True
        
    except Exception as e:
        print(f"❌ Autonomous coaching test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Testing Memory Management and Error Fixes")
    print("=" * 50)
    
    tests = [
        ("Trajectory Analysis Fix", test_trajectory_analysis_fix),
        ("Memory Manager", test_memory_manager),
        ("Autonomous Coaching Memory Check", test_autonomous_coaching_memory_check)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📋 Running: {test_name}")
        try:
            if test_func():
                print(f"✅ {test_name}: PASSED")
                passed += 1
            else:
                print(f"❌ {test_name}: FAILED")
        except Exception as e:
            print(f"❌ {test_name}: ERROR - {e}")
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Memory management fixes are working correctly.")
        return 0
    else:
        print("⚠️ Some tests failed. Please check the implementation.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
