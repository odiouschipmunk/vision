import numpy as np
import re
import json

def parse_frame_data(frame_text):
    """
    Parse a single frame's data.
    
    Args:
        frame_text (str): Text containing a single frame's data
        
    Returns:
        dict: Dictionary containing frame data or None if invalid
    """
    # Keypoint mapping
    keypoint_names = {
        0: "Nose", 1: "Left Eye", 2: "Right Eye", 3: "Left Ear", 4: "Right Ear",
        5: "Left Shoulder", 6: "Right Shoulder", 7: "Left Elbow", 8: "Right Elbow",
        9: "Left Wrist", 10: "Right Wrist", 11: "Left Hip", 12: "Right Hip",
        13: "Left Knee", 14: "Right Knee", 15: "Left Ankle", 16: "Right Ankle"
    }
    
    def process_player_keypoints(keypoints_str):
        # Clean up the string and convert to proper format
        rows = keypoints_str.strip().split('\n')
        keypoints = []
        
        for row in rows:
            # Skip empty rows or brackets
            row = row.strip(' []')
            if not row:
                continue
            # Clean up the row and convert to numbers
            clean_row = row.split()
            if len(clean_row) == 2:
                keypoints.append([float(clean_row[0]), float(clean_row[1])])
        
        # Convert to numpy array
        keypoints = np.array(keypoints)
        
        # Create dictionary of valid keypoints
        processed_keypoints = {}
        for i, coords in enumerate(keypoints):
            # Only include keypoints that are not [0, 0]
            if not np.array_equal(coords, [0, 0]):
                processed_keypoints[keypoint_names[i]] = {
                    'x': float(coords[0]),
                    'y': float(coords[1])
                }
        
        return processed_keypoints

    # Extract frame number
    frame_match = re.search(r'Frame: (\d+)', frame_text)
    if not frame_match:
        return None
    
    frame_number = int(frame_match.group(1))
    
    # Extract player keypoints using regular expressions
    player1_match = re.search(r'Player 1: \[(.*?)\]\s*\n\s*Player 2:', frame_text, re.DOTALL)
    player2_match = re.search(r'Player 2: \[(.*?)\]\s*\n\s*Ball:', frame_text, re.DOTALL)
    ball_match = re.search(r'Ball: \[(\d+), (\d+)\]', frame_text)
    
    if not all([player1_match, player2_match, ball_match]):
        return None
    
    player1_keypoints = process_player_keypoints(player1_match.group(1))
    player2_keypoints = process_player_keypoints(player2_match.group(1))
    ball_position = [int(ball_match.group(1)), int(ball_match.group(2))]
    
    return {
        'frame_number': frame_number,
        'player1_keypoints': player1_keypoints,
        'player2_keypoints': player2_keypoints,
        'ball_position': ball_position
    }

def parse_file(filename):
    """
    Parse entire file containing multiple frames of pose data.
    
    Args:
        filename (str): Path to the input file
        
    Returns:
        list: List of dictionaries containing parsed data for each frame
    """
    frames_data = []
    
    with open(filename, 'r') as file:
        content = file.read()
        
        # Split content into frames
        # Find all text between "Frame: X{" and the next frame or end of file
        frame_pattern = r'Frame: \d+\{.*?(?=Frame: \d+\{|$)'
        frames = re.findall(frame_pattern, content, re.DOTALL)
        
        # Parse each frame
        for frame_text in frames:
            frame_data = parse_frame_data(frame_text)
            if frame_data:
                frames_data.append(frame_data)
    #print(frames_data)
    return frames_data

def print_pose_data(parsed_data):
    """Helper function to print parsed pose data in a readable format"""
    print(f"Frame: {parsed_data['frame_number']}")
    
    print("\nPlayer 1 Detected Keypoints:")
    for point_name, coords in parsed_data['player1_keypoints'].items():
        print(f"{point_name}: x={coords['x']:.3f}, y={coords['y']:.3f}")
    
    print("\nPlayer 2 Detected Keypoints:")
    for point_name, coords in parsed_data['player2_keypoints'].items():
        print(f"{point_name}: x={coords['x']:.3f}, y={coords['y']:.3f}")
    
    print(f"\nBall Position: {parsed_data['ball_position']}")
    print("\n" + "="*50 + "\n")

# Example usage
def main():
    filename = "output(25k)/output/final.txt" 
    frames_data = parse_file(filename)
    
    print(f"Processed {len(frames_data)} frames\n")
    for frame_data in frames_data:
        print_pose_data(frame_data)
    print(frames_data)
    # Dump all parsed data into a JSON file
    output_filename = "parsed_frames_data.json"
    with open(output_filename, 'w') as json_file:
        json.dump(frames_data, json_file, indent=4)

    print(f"Data has been dumped into {output_filename}")
    #return frames_data  # Return the data for further processing if needed


if __name__ == "__main__":
    print(main())
