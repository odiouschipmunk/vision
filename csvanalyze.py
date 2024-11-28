import pandas as pd
import numpy as np
import re

def parse_coordinates(coord_str):
    """Convert string representation of coordinates to numpy array"""
    try:
        # Extract numbers using regex
        numbers = re.findall(r'[\d.]+', coord_str)
        # Convert to floats and reshape into pairs
        coords = np.array([float(x) for x in numbers]).reshape(-1, 2)
        return coords
    except Exception as e:
        print(f"Error parsing coordinates: {e}")
        return None

def get_court_position(keypoints):
    """Determine player's position on court based on hip points (11,12)
    Returns x_avg, y_avg, position_label
    """
    if keypoints is None or len(keypoints) < 17:
        return None, None, "unknown"
    
    # Get hip points (indices 11,12)
    hip_points = keypoints[11:13]
    # Filter zero points
    valid_points = hip_points[~np.all(hip_points == 0, axis=1)]
    
    if len(valid_points) == 0:
        return None, None, "unknown"
    
    x_avg = np.mean(valid_points[:, 0])
    y_avg = np.mean(valid_points[:, 1])
    
    x_pos = "center"
    if x_avg < 0.45:
        x_pos = "left"
    elif x_avg > 0.55:
        x_pos = "right"
        
    y_pos = "middle"
    if y_avg < 0.6:
        y_pos = "front"
    elif y_avg > 0.75:
        y_pos = "back"
    
    position_label = f"{y_pos} {x_pos}"
    return x_avg, y_avg, position_label

def is_lunging(keypoints):
    """Detect if player is lunging based on knee and ankle positions"""
    if keypoints is None or len(keypoints) < 17:
        return False
    
    knees = keypoints[13:15]
    ankles = keypoints[15:17]
    
    valid_knees = knees[~np.all(knees == 0, axis=1)]
    valid_ankles = ankles[~np.all(ankles == 0, axis=1)]
    
    if len(valid_knees) == 0 or len(valid_ankles) == 0:
        return False
    
    knee_y = np.mean(valid_knees[:, 1])
    ankle_y = np.mean(valid_ankles[:, 1])
    
    return abs(knee_y - ankle_y) > 0.15

def detect_leaning(keypoints):
    """Detect if player is leaning forward/backward or side to side"""
    if keypoints is None or len(keypoints) < 17:
        return "unknown leaning"
    
    shoulders = keypoints[5:7]  # Left and Right Shoulders
    hips = keypoints[11:13]     # Left and Right Hips
    
    valid_shoulders = shoulders[~np.all(shoulders == 0, axis=1)]
    valid_hips = hips[~np.all(hips == 0, axis=1)]
    
    if len(valid_shoulders) < 2 or len(valid_hips) < 2:
        return "unknown leaning"
    
    # Average positions
    shoulder_avg = np.mean(valid_shoulders, axis=0)
    hip_avg = np.mean(valid_hips, axis=0)
    
    lean = ""
    
    # Forward or Backward Lean
    if shoulder_avg[1] < hip_avg[1] - 0.05:
        lean += "forward"
    elif shoulder_avg[1] > hip_avg[1] + 0.05:
        lean += "backward"
    else:
        lean += "upright"
    
    # Side to Side Lean
    if shoulder_avg[0] < hip_avg[0] - 0.05:
        lean += " and leaning left"
    elif shoulder_avg[0] > hip_avg[0] + 0.05:
        lean += " and leaning right"
    else:
        lean += " and not leaning sideways"
    
    return lean

def analyze_movement(current_pos, previous_pos):
    """Analyze movement direction and speed based on current and previous positions"""
    if previous_pos is None or current_pos is None or previous_pos is None:
        return "stationary"
    
    delta = current_pos - previous_pos
    distance = np.linalg.norm(delta)
    
    if distance < 0.01:
        return "stationary"
    elif distance < 0.05:
        direction = ""
        if delta[0] > 0.02:
            direction += "right "
        elif delta[0] < -0.02:
            direction += "left "
        
        if delta[1] > 0.02:
            direction += "forward"
        elif delta[1] < -0.02:
            direction += "backward"
        
        return direction.strip()
    else:
        return "moving rapidly"

def analyze_frame(row, prev_positions):
    """Generate description for a single frame"""
    frame = row['Frame count']
    p1_keypoints = parse_coordinates(row['Player 1 Keypoints'])
    p2_keypoints = parse_coordinates(row['Player 2 Keypoints'])
    
    # Parse ball position - handle both string and list formats
    if isinstance(row['Ball Position'], str):
        ball_pos = [int(x) for x in re.findall(r'\d+', row['Ball Position'])]
    else:
        ball_pos = row['Ball Position']
    
    shot_type = row['Shot Type']
    
    description = []
    
    # Player 1 analysis
    p1_x_avg, p1_y_avg, p1_pos_label = get_court_position(p1_keypoints)
    p1_lunging = is_lunging(p1_keypoints)
    p1_lean = detect_leaning(p1_keypoints)
    
    # Create a numpy array for current position
    p1_current_pos = np.array([p1_x_avg, p1_y_avg]) if p1_x_avg is not None and p1_y_avg is not None else None
    p1_previous_pos = prev_positions['Player 1 Numerical']
    
    p1_movement = analyze_movement(p1_current_pos, p1_previous_pos)
    p1_desc = f"Player 1 is in the {p1_pos_label} area, {p1_lean}, and is {p1_movement}"
    if p1_lunging:
        p1_desc += " and is lunging"
    description.append(p1_desc)
    prev_positions['Player 1 Numerical'] = p1_current_pos
    
    # Player 2 analysis
    p2_x_avg, p2_y_avg, p2_pos_label = get_court_position(p2_keypoints)
    p2_lunging = is_lunging(p2_keypoints)
    p2_lean = detect_leaning(p2_keypoints)
    
    # Create a numpy array for current position
    p2_current_pos = np.array([p2_x_avg, p2_y_avg]) if p2_x_avg is not None and p2_y_avg is not None else None
    p2_previous_pos = prev_positions['Player 2 Numerical']
    
    p2_movement = analyze_movement(p2_current_pos, p2_previous_pos)
    p2_desc = f"Player 2 is in the {p2_pos_label} area, {p2_lean}, and is {p2_movement}"
    if p2_lunging:
        p2_desc += " and is lunging"
    description.append(p2_desc)
    prev_positions['Player 2 Numerical'] = p2_current_pos
    
    # Ball analysis
    if ball_pos and not (ball_pos == [0, 0]):
        description.append(f"Ball position: {ball_pos}")
        if shot_type:
            description.append(f"Shot type: {shot_type}")
    
    return "\n".join(description)

def main():
    prev_positions = {
        'Player 1 Numerical': None,
        'Player 2 Numerical': None
    }
    df = pd.read_csv('output/final.csv')
    analyses=[]
    for index, row in df.iterrows():
        frame = row['Frame count']
        print(f"\nFrame {frame}:")
        try:
            frame_analysis = analyze_frame(row, prev_positions)
            print(frame_analysis)
        except Exception as e:
            print(f"Error processing frame {frame}: {e}")
        print("-" * 50)
        analyses.append(frame_analysis)
        if len(analyses)>2:
            if analyses[-1]==analyses[-2]==frame_analysis:
                with open('output/frame_analysis.txt', 'a') as f:
                    f.write(f"\nFrame {frame}:\nSame as last frame.\n{'-' * 50}\n")
            else:
                with open('output/frame_analysis.txt', 'a') as f:
                    f.write(f"\nFrame {frame}:\n{frame_analysis}\n{'-' * 50}\n")
        else:
            with open('output/frame_analysis.txt', 'a') as f:
                f.write(f"\nFrame {frame}:\n{frame_analysis}\n{'-' * 50}\n")

if __name__ == "__main__":
    main()