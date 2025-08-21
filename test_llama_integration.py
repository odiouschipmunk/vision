#!/usr/bin/env python3
"""
Test script to verify Llama 3.1-8B-Instruct integration
"""

import os
import sys
import time
import json

def check_venv():
    """Check if virtual environment is active"""
    return hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)

def test_llama_import():
    """Test if transformers can be imported"""
    print("🧪 Testing transformers import...")
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        print("✅ Transformers library imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Failed to import transformers: {e}")
        return False

def test_llama_model_loading():
    """Test if Llama model can be loaded"""
    print("\n🤖 Testing Llama model loading...")
    try:
        from llama_coaching_enhancement import LlamaCoachingEnhancer
        
        # Create enhancer instance
        enhancer = LlamaCoachingEnhancer()
        print("✅ LlamaCoachingEnhancer created successfully")
        
        # Try to initialize model (this might take time)
        print("⏳ Initializing Llama model (this may take a few minutes)...")
        enhancer.initialize_model()
        
        if enhancer.is_initialized:
            print("✅ Llama model initialized successfully")
            return True
        else:
            print("❌ Llama model initialization failed")
            return False
            
    except Exception as e:
        print(f"❌ Error testing Llama model: {e}")
        return False

def test_coaching_insight_generation():
    """Test coaching insight generation"""
    print("\n🧠 Testing coaching insight generation...")
    try:
        from llama_coaching_enhancement import get_llama_enhancer
        
        enhancer = get_llama_enhancer()
        
        if not enhancer.is_initialized:
            print("⚠️ Model not initialized, skipping insight test")
            return False
        
        # Test simple prompt
        test_prompt = "As a squash coach, provide one tip for improving serve accuracy."
        print(f"📝 Testing prompt: {test_prompt}")
        
        insight = enhancer.generate_coaching_insight(test_prompt, max_new_tokens=50)
        print(f"🤖 Generated insight: {insight}")
        
        if insight and len(insight) > 10:
            print("✅ Coaching insight generation successful")
            return True
        else:
            print("❌ Generated insight too short or empty")
            return False
            
    except Exception as e:
        print(f"❌ Error generating coaching insight: {e}")
        return False

def test_shot_analysis():
    """Test shot pattern analysis"""
    print("\n🎯 Testing shot pattern analysis...")
    try:
        from llama_coaching_enhancement import get_llama_enhancer
        
        enhancer = get_llama_enhancer()
        
        if not enhancer.is_initialized:
            print("⚠️ Model not initialized, skipping shot analysis test")
            return False
        
        # Mock shot data
        mock_shot_data = [
            {
                'shot_type': 'drive',
                'trajectory': [(100, 200), (110, 190), (120, 180)],
                'player_who_hit': 'player1',
                'duration': 0.5
            },
            {
                'shot_type': 'drop',
                'trajectory': [(150, 150), (160, 140), (170, 130)],
                'player_who_hit': 'player2',
                'duration': 0.3
            },
            {
                'shot_type': 'drive',
                'trajectory': [(200, 100), (210, 90), (220, 80)],
                'player_who_hit': 'player1',
                'duration': 0.4
            }
        ]
        
        print(f"📊 Analyzing {len(mock_shot_data)} shots...")
        analysis = enhancer.analyze_shot_patterns(mock_shot_data)
        
        print(f"📈 Shot distribution: {analysis.get('shot_distribution', {})}")
        print(f"🧠 AI analysis preview: {analysis.get('ai_analysis', '')[:100]}...")
        
        if analysis.get('ai_analysis'):
            print("✅ Shot pattern analysis successful")
            return True
        else:
            print("❌ Shot pattern analysis failed")
            return False
            
    except Exception as e:
        print(f"❌ Error in shot analysis: {e}")
        return False

def test_integration_with_main_system():
    """Test integration with main system"""
    print("\n🔗 Testing integration with main system...")
    try:
        # Test if the import works
        from ef import generate_llama_enhanced_analysis
        print("✅ Llama integration function imported successfully")
        
        # Test with mock data
        mock_frame_count = 1000
        mock_players = {}
        mock_past_ball_pos = [(100, 200), (110, 190), (120, 180)]
        
        class MockShotTracker:
            def __init__(self):
                self.completed_shots = [
                    {'shot_type': 'drive', 'id': 1},
                    {'shot_type': 'drop', 'id': 2}
                ]
                self.active_shots = []
        
        mock_shot_tracker = MockShotTracker()
        
        class MockCoachingData:
            def get_summary(self):
                return {"sessions": 1, "total_time": "10 minutes"}
        
        mock_coaching_data = MockCoachingData()
        
        print("📊 Testing enhanced analysis generation...")
        outputs = generate_llama_enhanced_analysis(
            mock_frame_count, 
            mock_players, 
            mock_past_ball_pos, 
            mock_shot_tracker, 
            mock_coaching_data
        )
        
        print(f"📁 Generated {len(outputs)} outputs")
        for output in outputs:
            print(f"   📄 {output}")
        
        return len(outputs) > 0
        
    except Exception as e:
        print(f"❌ Error testing integration: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all Llama integration tests"""
    print("🤖 LLAMA 3.1-8B-INSTRUCT INTEGRATION TEST")
    print("=" * 60)
    
    # Check virtual environment
    if check_venv():
        print("✅ Virtual environment is active")
    else:
        print("⚠️ Virtual environment not detected")
        print("💡 Please activate your virtual environment before running tests")
        return False
    
    # Run tests
    tests = [
        ("Transformers Import", test_llama_import),
        ("Model Loading", test_llama_model_loading),
        ("Insight Generation", test_coaching_insight_generation),
        ("Shot Analysis", test_shot_analysis),
        ("System Integration", test_integration_with_main_system)
    ]
    
    results = {}
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST RESULTS SUMMARY:")
    
    passed = 0
    total = len(tests)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Llama integration is working correctly.")
        return True
    elif passed >= 3:
        print("\n⚠️ Most tests passed. Llama integration is partially working.")
        return True
    else:
        print("\n❌ Many tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
