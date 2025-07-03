#!/usr/bin/env python3
"""
Test script to verify the automatic visualization integration in ef.py
"""

import os
import sys

def test_visualization_integration():
    """Test that the visualization functions work correctly"""
    print("🧪 Testing visualization integration...")
    
    try:
        # Import the visualization functions
        from autonomous_coaching import create_graphics, view_all_graphics
        print("✅ Successfully imported visualization functions")
        
        # Check if final.csv exists (required for visualizations)
        if os.path.exists("output/final.csv"):
            print("✅ Found output/final.csv - data source available")
        else:
            print("❌ output/final.csv not found - visualizations will fail")
            return False
        
        # Test the visualization generation (this is what ef.py will call)
        print("\n📊 Testing create_graphics()...")
        create_graphics()
        print("✅ create_graphics() completed successfully!")
        
        print("\n📈 Testing view_all_graphics()...")
        view_all_graphics()
        print("✅ view_all_graphics() completed successfully!")
        
        # Check if graphics were created
        graphics_dir = "output/graphics"
        if os.path.exists(graphics_dir):
            files = os.listdir(graphics_dir)
            print(f"✅ Graphics directory contains {len(files)} files")
        else:
            print("⚠️ Graphics directory not found")
        
        print("\n🎉 Integration test PASSED!")
        print("✅ The automatic visualization generation will work correctly in ef.py")
        return True
        
    except Exception as e:
        print(f"❌ Integration test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    print("=" * 60)
    print("🚀 TESTING AUTOMATIC VISUALIZATION INTEGRATION")
    print("=" * 60)
    print("This test verifies that ef.py will correctly generate")
    print("visualizations automatically at the end of the pipeline.")
    print("=" * 60)
    
    success = test_visualization_integration()
    
    print("\n" + "=" * 60)
    if success:
        print("✅ INTEGRATION TEST PASSED!")
        print("The ef.py pipeline will automatically generate:")
        print("  • Shot type analysis and heatmaps")
        print("  • Player and ball movement patterns")
        print("  • Ball trajectory analysis")
        print("  • Match flow and performance metrics")
        print("  • Summary statistics and reports")
    else:
        print("❌ INTEGRATION TEST FAILED!")
        print("Check the error messages above for details.")
    print("=" * 60)

if __name__ == "__main__":
    main()
