#!/usr/bin/env python3
"""
Test script for enhanced shot detection system
"""

import math

def calculate_vector_angle(v1, v2):
    """Calculate angle between two vectors"""
    try:
        dot_product = v1[0] * v2[0] + v1[1] * v2[1]
        mag1 = math.sqrt(v1[0]**2 + v1[1]**2)
        mag2 = math.sqrt(v2[0]**2 + v2[1]**2)
        
        if mag1 == 0 or mag2 == 0:
            return 0
        
        cos_angle = dot_product / (mag1 * mag2)
        cos_angle = max(-1, min(1, cos_angle))
        return math.degrees(math.acos(cos_angle))
        
    except Exception:
        return 0

def check_velocity_direction_change(positions):
    """Check for significant velocity/direction change indicating collision"""
    if len(positions) < 3:
        return 0.0
    
    # Calculate velocity before and after middle point
    p1, p2, p3 = positions[0], positions[1], positions[2]
    
    velocity_before = math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
    velocity_after = math.sqrt((p3[0] - p2[0])**2 + (p3[1] - p2[1])**2)
    
    # Direction vectors
    dir_before = [(p2[0] - p1[0]), (p2[1] - p1[1])]
    dir_after = [(p3[0] - p2[0]), (p3[1] - p2[1])]
    
    # Calculate direction change
    if velocity_before > 0 and velocity_after > 0:
        # Normalize direction vectors
        dir_before = [dir_before[0]/velocity_before, dir_before[1]/velocity_before]
        dir_after = [dir_after[0]/velocity_after, dir_after[1]/velocity_after]
        
        # Dot product to find direction similarity
        dot_product = dir_before[0]*dir_after[0] + dir_before[1]*dir_after[1]
        direction_similarity = max(-1, min(1, dot_product))
        
        # Lower similarity means more direction change
        direction_change = 1 - (direction_similarity + 1) / 2
        
        return direction_change
    
    return 0.0

def detect_wall_hit_with_type(positions, court_width=640, court_height=360):
    """Detect wall hits and classify wall type (front, side_left, side_right, back)"""
    result = {'confidence': 0.0, 'wall_type': 'none', 'player_proximity': 0}
    
    if len(positions) < 3:
        return result
    
    current_pos = positions[-1]
    x, y = current_pos[0], current_pos[1]
    
    # Define wall zones (pixels from edge)
    wall_threshold = 30
    front_wall_threshold = 30  # Same as wall threshold for consistency
    
    # Distance to each wall
    dist_to_left = x
    dist_to_right = court_width - x
    dist_to_front = y  # Assuming front wall is at top (y=0)
    dist_to_back = court_height - y
    
    min_wall_distance = min(dist_to_left, dist_to_right, dist_to_front, dist_to_back)
    
    # Check if ball is near any wall
    if min_wall_distance < wall_threshold:
        # Determine wall type and set base confidence
        if min_wall_distance == dist_to_front:
            result['wall_type'] = 'front'
            result['confidence'] = 0.8  # High base confidence for front wall
        elif min_wall_distance == dist_to_left:
            result['wall_type'] = 'side_left'
            result['confidence'] = 0.6  # Good confidence for side walls
        elif min_wall_distance == dist_to_right:
            result['wall_type'] = 'side_right'
            result['confidence'] = 0.6
        elif min_wall_distance == dist_to_back:
            result['wall_type'] = 'back'
            result['confidence'] = 0.5  # Lower confidence for back wall
        
        # If wall type was set, enhance confidence with velocity/direction change
        if result['wall_type'] != 'none' and len(positions) >= 3:
            velocity_change = check_velocity_direction_change(positions[-3:])
            if velocity_change > 0.3:  # Significant direction change
                result['confidence'] = min(0.95, result['confidence'] + velocity_change * 0.3)
            
            # Proximity bonus - closer to wall = higher confidence
            proximity_factor = 1 - (min_wall_distance / wall_threshold)
            result['confidence'] = min(0.95, result['confidence'] * (1 + proximity_factor * 0.2))
    
    return result

def test_shot_detection():
    """Test the enhanced shot detection functions"""
    print("Testing Enhanced Shot Detection System")
    print("=" * 50)
    
    # Test case 1: Ball approaching front wall
    print("\nTest 1: Ball approaching front wall")
    front_wall_trajectory = [
        [320, 50, 1],   # Near front wall
        [320, 40, 2],   # Closer to front wall
        [320, 20, 3],   # Very close to front wall
        [320, 15, 4],   # At front wall
        [320, 25, 5]    # Bouncing back
    ]
    
    wall_result = detect_wall_hit_with_type(front_wall_trajectory)
    print(f"  Position: {front_wall_trajectory[-1]}")
    print(f"  Distance to front: {front_wall_trajectory[-1][1]}")
    print(f"  Wall type detected: {wall_result['wall_type']}")
    print(f"  Confidence: {wall_result['confidence']:.2f}")
    print(f"  Expected: front wall with high confidence")
    
    # Test case 2: Ball hitting side wall
    print("\nTest 2: Ball hitting left side wall")
    side_wall_trajectory = [
        [35, 180, 1],   # Near left wall
        [25, 180, 2],   # Closer to left wall
        [15, 180, 3],   # At left wall
        [25, 180, 4],   # Bouncing back
        [20, 180, 5]    # Still near wall
    ]
    
    wall_result = detect_wall_hit_with_type(side_wall_trajectory)
    print(f"  Position: {side_wall_trajectory[-1]}")
    print(f"  Distance to left: {side_wall_trajectory[-1][0]}")
    print(f"  Wall type detected: {wall_result['wall_type']}")
    print(f"  Confidence: {wall_result['confidence']:.2f}")
    print(f"  Expected: side_left wall")
    
    # Test case 3: Ball in center court (no wall hit)
    print("\nTest 3: Ball in center court (no wall hit)")
    center_trajectory = [
        [320, 180, 1],
        [330, 190, 2],
        [340, 200, 3],
        [350, 210, 4],
        [360, 220, 5]
    ]
    
    wall_result = detect_wall_hit_with_type(center_trajectory)
    print(f"  Position: {center_trajectory[-1]}")
    print(f"  Wall type detected: {wall_result['wall_type']}")
    print(f"  Confidence: {wall_result['confidence']:.2f}")
    print(f"  Expected: no wall hit detected")
    
    # Test case 4: Debug wall detection logic
    print("\nTest 4: Debug wall detection")
    test_pos = [320, 15, 4]  # Close to front wall
    print(f"  Test position: {test_pos}")
    
    court_width, court_height = 640, 360
    x, y = test_pos[0], test_pos[1]
    
    dist_to_left = x
    dist_to_right = court_width - x
    dist_to_front = y
    dist_to_back = court_height - y
    
    print(f"  Distance to left: {dist_to_left}")
    print(f"  Distance to right: {dist_to_right}")
    print(f"  Distance to front: {dist_to_front}")
    print(f"  Distance to back: {dist_to_back}")
    
    min_wall_distance = min(dist_to_left, dist_to_right, dist_to_front, dist_to_back)
    print(f"  Minimum wall distance: {min_wall_distance}")
    print(f"  Wall threshold: 30")
    print(f"  Is near wall: {min_wall_distance < 30}")
    
    if min_wall_distance == dist_to_front:
        print(f"  Closest to front wall")
    elif min_wall_distance == dist_to_left:
        print(f"  Closest to left wall")
    elif min_wall_distance == dist_to_right:
        print(f"  Closest to right wall")
    elif min_wall_distance == dist_to_back:
        print(f"  Closest to back wall")
    
    # Test case 5: Direction change detection
    print("\nTest 5: Direction change detection")
    sharp_turn_trajectory = [
        [100, 100, 1],
        [110, 100, 2],  # Moving right
        [105, 100, 3],  # Sharp turn, moving left
    ]
    
    direction_change = check_velocity_direction_change(sharp_turn_trajectory)
    print(f"  Direction change score: {direction_change:.2f}")
    print(f"  Expected: high score (>0.8) for sharp direction change")
    
    print("\n" + "=" * 50)
    print("Enhanced Shot Detection Test Complete!")
    
    return True

if __name__ == "__main__":
    test_shot_detection()