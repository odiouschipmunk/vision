#!/usr/bin/env python3
"""
Test script to demonstrate the enhanced ball-player detection and shot tracking features
"""

import cv2
import numpy as np
import json
import time
import math

class MockShotTracker:
    """Mock version of the ShotTracker for demonstration"""
    
    def __init__(self):
        self.active_shots = []
        self.completed_shots = []
        self.shot_id_counter = 0
        
    def get_shot_color(self, shot_type):
        """Get color for shot based on type"""
        if isinstance(shot_type, list) and len(shot_type) > 0:
            shot_str = str(shot_type[0]).lower()
        else:
            shot_str = str(shot_type).lower()
            
        color_map = {
            'crosscourt': (0, 255, 0),      # Green
            'wide_crosscourt': (0, 200, 0), # Dark green
            'straight': (255, 255, 0),       # Yellow
            'boast': (255, 0, 255),         # Magenta
            'drop': (0, 165, 255),          # Orange
            'lob': (255, 0, 0),             # Blue
            'drive': (0, 255, 255),         # Cyan
            'default': (255, 255, 255)      # White
        }
        
        for shot_name, color in color_map.items():
            if shot_name in shot_str:
                return color
                
        return color_map['default']
    
    def simulate_shot_tracking(self):
        """Simulate shot tracking with sample data"""
        
        # Create a demo frame
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Add court-like background
        cv2.rectangle(frame, (50, 50), (590, 430), (100, 100, 100), 2)
        cv2.line(frame, (50, 240), (590, 240), (100, 100, 100), 1)
        cv2.line(frame, (320, 50), (320, 430), (100, 100, 100), 1)
        
        # Sample shot trajectories
        sample_shots = [
            {
                'type': 'crosscourt',
                'trajectory': [(100, 200), (200, 250), (300, 300), (400, 350), (500, 400)],
                'player': 1
            },
            {
                'type': 'straight', 
                'trajectory': [(150, 100), (150, 150), (150, 200), (150, 250), (150, 300)],
                'player': 2
            },
            {
                'type': 'boast',
                'trajectory': [(200, 200), (250, 180), (300, 160), (350, 140), (400, 120)],
                'player': 1
            }
        ]
        
        # Draw shot trajectories
        for shot in sample_shots:
            color = self.get_shot_color(shot['type'])
            trajectory = shot['trajectory']
            
            # Draw trajectory line
            for i in range(1, len(trajectory)):
                pt1 = trajectory[i-1]
                pt2 = trajectory[i]
                cv2.line(frame, pt1, pt2, color, 3)
            
            # Draw shot info
            start_pos = trajectory[0]
            text = f"P{shot['player']}: {shot['type']}"
            cv2.putText(frame, text, (start_pos[0], start_pos[1] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Draw enhanced ball at end of trajectory
            end_pos = trajectory[-1]
            cv2.circle(frame, end_pos, 8, color, -1)
            cv2.circle(frame, end_pos, 12, color, 2)
        
        # Add enhancement information
        cv2.putText(frame, "ENHANCED SHOT TRACKING DEMO", (150, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Color legend
        legend_y = 450
        legend_items = [
            ("Crosscourt: GREEN", (0, 255, 0)),
            ("Straight: YELLOW", (255, 255, 0)),
            ("Boast: MAGENTA", (255, 0, 255))
        ]
        
        for i, (text, color) in enumerate(legend_items):
            x_pos = 50 + i * 180
            cv2.putText(frame, text, (x_pos, legend_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        return frame

def demonstrate_enhanced_detection():
    """Demonstrate the enhanced ball-player detection"""
    
    print("🎾 ENHANCED SQUASH ANALYSIS FEATURES DEMONSTRATION")
    print("=" * 60)
    print()
    
    print("1. ENHANCED BALL-PLAYER HIT DETECTION:")
    print("   ✓ Weighted keypoint analysis (racket hand priority)")
    print("   ✓ Multi-factor scoring system:")
    print("     - Proximity to ball")
    print("     - Player movement towards ball") 
    print("     - Trajectory change detection")
    print("     - Velocity change analysis")
    print()
    
    print("2. REAL-TIME SHOT TRACKING:")
    print("   ✓ Color-coded trajectory visualization")
    print("   ✓ Shot type classification in real-time")
    print("   ✓ Automatic shot completion detection")
    print("   ✓ Complete shot data logging")
    print()
    
    print("3. VISUAL IMPROVEMENTS:")
    print("   ✓ Enhanced ball highlighting during shots")
    print("   ✓ Real-time shot statistics display")
    print("   ✓ Trajectory lines with shot identification")
    print()
    
    # Create shot tracker demo
    tracker = MockShotTracker()
    
    print("Generating demonstration frame...")
    demo_frame = tracker.simulate_shot_tracking()
    
    # Save demonstration frame
    cv2.imwrite("output/shot_tracking_demo.png", demo_frame)
    print("✅ Demo frame saved to: output/shot_tracking_demo.png")
    
    # Create enhancement summary
    print("\nCreating enhancement summary...")
    create_enhancement_summary()
    
    print("\n🎯 KEY IMPROVEMENTS SUMMARY:")
    print("-" * 40)
    print("• More accurate player-ball hit detection")
    print("• Real-time shot visualization with color coding")
    print("• Complete shot tracking and data logging")
    print("• Enhanced ball highlighting during active shots")
    print("• Comprehensive shot analysis reports")
    print()
    print("📁 Check the 'output/' directory for:")
    print("  - shot_tracking_demo.png (visualization demo)")
    print("  - enhancement_summary.png (features overview)")
    print("  - shots_log.jsonl (shot data logging - created during analysis)")
    print("  - shot_analysis_report.txt (comprehensive analysis - created during analysis)")

def create_enhancement_summary():
    """Create a visual summary of the enhancements"""
    
    summary_img = np.ones((600, 800, 3), dtype=np.uint8) * 50  # Dark background
    
    # Title
    cv2.putText(summary_img, "SQUASH ANALYSIS ENHANCEMENTS", (150, 50), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    
    # Enhancement sections
    sections = [
        ("1. Enhanced Ball-Player Hit Detection:", (0, 255, 0)),
        ("   - Weighted keypoint analysis", (200, 200, 200)),
        ("   - Multi-factor scoring system", (200, 200, 200)),
        ("   - Velocity & trajectory change detection", (200, 200, 200)),
        ("", (255, 255, 255)),
        ("2. Real-time Shot Tracking:", (0, 255, 0)),
        ("   - Color-coded trajectory visualization", (200, 200, 200)),
        ("   - Crosscourt shots: GREEN lines", (0, 255, 0)),
        ("   - Straight shots: YELLOW lines", (255, 255, 0)),
        ("   - Boast shots: MAGENTA lines", (255, 0, 255)),
        ("   - Drop shots: ORANGE lines", (0, 165, 255)),
        ("   - Lob shots: BLUE lines", (255, 0, 0)),
        ("", (255, 255, 255)),
        ("3. Shot Data Logging:", (0, 255, 0)),
        ("   - Complete shot trajectories saved", (200, 200, 200)),
        ("   - Shot classification and duration", (200, 200, 200)),
        ("   - Player identification per shot", (200, 200, 200)),
        ("   - Comprehensive analysis reports", (200, 200, 200)),
        ("", (255, 255, 255)),
        ("4. Visual Improvements:", (0, 255, 0)),
        ("   - Enhanced ball highlighting during shots", (200, 200, 200)),
        ("   - Real-time shot statistics display", (200, 200, 200)),
        ("   - Trajectory lines with shot ID", (200, 200, 200))
    ]
    
    y_pos = 90
    for text, color in sections:
        font_size = 0.5 if text and text[0].isdigit() else 0.4
        cv2.putText(summary_img, text, (50, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, font_size, color, 1)
        y_pos += 22
    
    # Save summary
    cv2.imwrite("output/enhancement_summary.png", summary_img)
    print("✅ Enhancement summary saved to: output/enhancement_summary.png")

if __name__ == "__main__":
    # Create output directory if it doesn't exist
    import os
    os.makedirs("output", exist_ok=True)
    
    # Run demonstration
    demonstrate_enhanced_detection()
