import pandas as pd
import numpy as np
import re
import ast, csv
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

def parse_through(start,end,filename):
    prev_positions = {
        'Player 1 Numerical': None,
        'Player 2 Numerical': None
    }
    df = pd.read_csv(filename)
    bigstring=''
    analyses=[]
    for index, row in df.iterrows():
        frame = row['Frame count']
        if index>=start and index<=end:
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
                    bigstring+=f"\nFrame {frame}:\nSame as last frame.\n{'-' * 50}\n"
                else:
                    bigstring+=f"\nFrame {frame}:\n{frame_analysis}\n{'-' * 50}\n"
            else:
                bigstring+=f"\nFrame {frame}:\n{frame_analysis}\n{'-' * 50}\n"
    return bigstring


def human_readable(filename):
    import csv
    import ast

    def analyze_posture(keypoints):
        try:
            points = ast.literal_eval(keypoints)
            description = []

            # Head and neck analysis (points 0 and 1)
            if points[0][0] != 0 and points[1][0] != 0:
                head_tilt = points[0][0] - points[1][0]
                if abs(head_tilt) > 0.05:
                    description.append(f"head tilted {'right' if head_tilt > 0 else 'left'}")
                else:
                    description.append("head facing forward")

            # Shoulder analysis (points 5 and 6)
            if points[5][1] != 0 and points[6][1] != 0:
                shoulder_level = abs(points[5][1] - points[6][1])
                if shoulder_level > 0.05:
                    description.append("shoulders tilted")
                else:
                    description.append("shoulders level")

            # Arm positions (points 7, 8, 9, 10)
            if points[7][1] != 0 and points[9][1] != 0 and points[5][1] != 0 and points[6][1] != 0:
                left_arm_up = points[7][1] < points[5][1]
                right_arm_up = points[9][1] < points[6][1]
                if left_arm_up and right_arm_up:
                    description.append("both arms raised")
                elif left_arm_up:
                    description.append("left arm raised")
                elif right_arm_up:
                    description.append("right arm raised")
                else:
                    description.append("arms down")

            # Hip analysis (points 11 and 12)
            if points[11][1] != 0 and points[12][1] != 0:
                hip_level = abs(points[11][1] - points[12][1])
                if hip_level > 0.05:
                    description.append("hips tilted")
                else:
                    description.append("hips level")

            # Leg positions (points 13 and 14)
            if points[13][0] != 0 and points[14][0] != 0:
                leg_spread = abs(points[13][0] - points[14][0])
                if leg_spread > 0.2:
                    description.append("legs widely spread")
                elif leg_spread > 0.1:
                    description.append("legs moderately spread")
                else:
                    description.append("legs close together")

            # Knee bend (points 13-15 and 14-16)
            if points[13][1] != 0 and points[15][1] != 0:
                left_knee_bend = points[15][1] - points[13][1]
                if left_knee_bend > 0.1:
                    description.append("left knee bent")
            if points[14][1] != 0 and points[16][1] != 0:
                right_knee_bend = points[16][1] - points[14][1]
                if right_knee_bend > 0.1:
                    description.append("right knee bent")

            # Foot positions (points 15 and 16)
            if points[15][0] != 0 and points[16][0] != 0:
                feet_distance = abs(points[15][0] - points[16][0])
                if feet_distance > 0.2:
                    description.append("feet wide apart")
                else:
                    description.append("feet close together")

            return ", ".join(description) if description else "in neutral position"
        except:
            return "position unclear"

    def relative_to_ball(player_points, ball_pos):
        try:
            points = ast.literal_eval(player_points)
            ball = ast.literal_eval(ball_pos)

            valid_points = [p[0] for p in points if p[0] != 0]
            if not valid_points:
                return "position unknown"

            center_x = sum(valid_points) / len(valid_points)
            ball_x = ball[0] / 1000 if isinstance(ball[0], (int, float)) else 0

            distance = abs(center_x - ball_x)
            if distance < 0.2:
                return "very close to"
            elif distance < 0.4:
                return "near"
            else:
                return "far from"
        except:
            return "position unknown"

    try:
        with open(filename, 'r') as file:
            reader = csv.reader(file)
            next(reader)  # Skip header

            for row in reader:
                if len(row) != 5:
                    continue

                frame, p1_points, p2_points, ball_pos, shot = row

                p1_description = analyze_posture(p1_points)
                p2_description = analyze_posture(p2_points)

                p1_ball_pos = relative_to_ball(p1_points, ball_pos)
                p2_ball_pos = relative_to_ball(p2_points, ball_pos)

                description = f"Frame {frame}: Player 1 is {p1_description} and is {p1_ball_pos} the ball. "
                description += f"Player 2 is {p2_description} and is {p2_ball_pos} the ball. "
                description += f"The shot being played is a {shot}."

                print(description)
    except FileNotFoundError:
        print(f"Error: Could not find file {filename}")
    except Exception as e:
        print(f"Error processing file: {str(e)}")

if __name__ == '__main__':
    human_readable('output/final.csv')