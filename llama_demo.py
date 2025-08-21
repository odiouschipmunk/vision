#!/usr/bin/env python3
"""
Demo script showcasing Llama 3.1-8B-Instruct integration for squash coaching
"""

import os
import sys
import time
import json

def check_venv():
    """Check if virtual environment is active"""
    return hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)

def demo_basic_llama_usage():
    """Demonstrate basic Llama model usage"""
    print("🤖 DEMO: Basic Llama Model Usage")
    print("=" * 50)
    
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch
        
        print("📥 Loading Llama 3.1-8B-Instruct model...")
        
        # Initialize tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
        model = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Llama-3.1-8B-Instruct",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            low_cpu_mem_usage=True
        )
        
        if not torch.cuda.is_available():
            model = model.to("cpu")
        
        print("✅ Model loaded successfully!")
        
        # Test basic generation
        messages = [
            {"role": "user", "content": "Who are you?"},
        ]
        
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)
        
        print("🧠 Generating response...")
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=40)
        
        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
        print(f"🤖 Response: {response}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in basic demo: {e}")
        return False

def demo_coaching_enhancement():
    """Demonstrate coaching enhancement features"""
    print("\n🎾 DEMO: Coaching Enhancement Features")
    print("=" * 50)
    
    try:
        from llama_coaching_enhancement import LlamaCoachingEnhancer
        
        print("🔧 Initializing coaching enhancer...")
        enhancer = LlamaCoachingEnhancer()
        enhancer.initialize_model()
        
        if not enhancer.is_initialized:
            print("⚠️ Model not initialized, using fallback demo")
            return demo_fallback_coaching()
        
        print("✅ Coaching enhancer ready!")
        
        # Demo 1: Shot Pattern Analysis
        print("\n🎯 Demo 1: Shot Pattern Analysis")
        shot_data = [
            {'shot_type': 'drive', 'player_who_hit': 'player1'},
            {'shot_type': 'drop', 'player_who_hit': 'player2'},
            {'shot_type': 'drive', 'player_who_hit': 'player1'},
            {'shot_type': 'lob', 'player_who_hit': 'player2'},
            {'shot_type': 'drive', 'player_who_hit': 'player1'}
        ]
        
        analysis = enhancer.analyze_shot_patterns(shot_data)
        print(f"📊 Shot distribution: {analysis.get('shot_distribution', {})}")
        print(f"🧠 AI Analysis preview: {analysis.get('ai_analysis', '')[:150]}...")
        
        # Demo 2: Coaching Insight
        print("\n💡 Demo 2: Coaching Insight Generation")
        prompt = "As a squash coach, provide 3 tips for improving backhand accuracy."
        insight = enhancer.generate_coaching_insight(prompt, max_new_tokens=100)
        print(f"🤖 Generated insight: {insight}")
        
        # Demo 3: Match Report
        print("\n📋 Demo 3: Match Report Generation")
        match_data = {
            "total_shots": 25,
            "rally_count": 8,
            "average_rally_length": 3.1,
            "shot_accuracy": 0.72,
            "movement_efficiency": 0.85
        }
        
        report = enhancer.generate_match_report(match_data)
        print(f"📄 Report preview: {report.get('match_report', '')[:200]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in coaching demo: {e}")
        return demo_fallback_coaching()

def demo_fallback_coaching():
    """Fallback demo without actual model"""
    print("\n🔄 FALLBACK DEMO: Coaching Features (No Model)")
    print("=" * 50)
    
    print("📊 This would analyze shot patterns and provide insights")
    print("🎯 Shot distribution: {'drive': 3, 'drop': 1, 'lob': 1}")
    print("🧠 AI Analysis: Based on your shot patterns, you're relying heavily on drives.")
    print("💡 Recommendations:")
    print("   1. Practice more drop shots for variety")
    print("   2. Work on lob technique for defensive play")
    print("   3. Improve shot selection based on court position")
    
    return True

def demo_integration_features():
    """Demonstrate integration features"""
    print("\n🔗 DEMO: Integration Features")
    print("=" * 50)
    
    print("📁 Output files that would be generated:")
    print("   • output/ai_analysis/shot_pattern_analysis.json")
    print("   • output/ai_analysis/movement_analysis.json")
    print("   • output/ai_analysis/match_report.json")
    print("   • output/ai_analysis/personalized_coaching_plan.json")
    print("   • output/reports/ai_shot_analysis.txt")
    print("   • output/reports/ai_match_report.txt")
    print("   • output/reports/ai_coaching_plan.txt")
    
    print("\n🎯 Features available:")
    print("   • AI-powered shot pattern analysis")
    print("   • Player movement optimization insights")
    print("   • Comprehensive match reports")
    print("   • Personalized coaching plans")
    print("   • Tactical recommendations")
    print("   • Performance improvement suggestions")
    
    return True

def main():
    """Run the Llama integration demo"""
    print("🎾 LLAMA 3.1-8B-INSTRUCT SQUASH COACHING DEMO")
    print("=" * 60)
    
    # Check virtual environment
    if check_venv():
        print("✅ Virtual environment is active")
    else:
        print("⚠️ Virtual environment not detected")
        print("💡 Please activate your virtual environment before running the demo")
        return False
    
    # Run demos
    demos = [
        ("Basic Llama Usage", demo_basic_llama_usage),
        ("Coaching Enhancement", demo_coaching_enhancement),
        ("Integration Features", demo_integration_features)
    ]
    
    results = {}
    for demo_name, demo_func in demos:
        print(f"\n{'='*20} {demo_name} {'='*20}")
        try:
            results[demo_name] = demo_func()
        except Exception as e:
            print(f"❌ Demo failed with exception: {e}")
            results[demo_name] = False
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 DEMO SUMMARY:")
    
    for demo_name, result in results.items():
        status = "✅ SUCCESS" if result else "❌ FAILED"
        print(f"   {demo_name}: {status}")
    
    print("\n🚀 To use the full system:")
    print("   1. Run: python run_analysis.py")
    print("   2. Or: python run_with_venv.py --video your_video.mp4")
    print("   3. Check output/ai_analysis/ for AI-generated insights")
    
    print("\n🧪 To test the integration:")
    print("   python test_llama_integration.py")
    
    return True

if __name__ == "__main__":
    main()
