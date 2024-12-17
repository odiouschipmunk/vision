import csv
import ast
import numpy as np
import cv2
import seaborn as sns
import matplotlib.pyplot as plt
import json
def read_player_positions(csv_path='output/final.csv'):
    """
    Read player positions from CSV file and return processed positions for both players
    """
    frames = []
    player1_pos = []
    player2_pos = []
    
    with open(csv_path, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        next(csvreader)  # Skip header row if it exists
        for row in csvreader:
            if len(row) >= 3:  # Make sure we have enough columns
                try:
                    pos1 = ast.literal_eval(row[1].strip())  # Remove any whitespace
                    pos2 = ast.literal_eval(row[2].strip())
                    player1_pos.append(pos1)
                    player2_pos.append(pos2)
                except Exception as e:
                    print(f"Error processing row: {row}")
                    print(f"Error details: {str(e)}")
                    continue
    
    if not player1_pos or not player2_pos:
        raise ValueError("No valid player positions were found in the CSV file")
        
    return player1_pos, player2_pos

def read_reference_points(path='reference_points.json'):
    with open(path, 'r') as f:
        return json.load(f)

def load_reference_points():
    reference_points_3d = [
        [0, 9.75, 0],  # Top-left corner, 1
        [6.4, 9.75, 0],  # Top-right corner, 2
        [6.4, 0, 0],  # Bottom-right corner, 3
        [0, 0, 0],  # Bottom-left corner, 4
        [3.2, 0, 4.31],  # "T" point, 5
        [0, 2.71, 0],  # Left bottom of the service box, 6
        [6.4, 2.71, 0],  # Right bottom of the service box, 7
        [0, 9.75, 0.48],  # left of tin, 8
        [6.4, 9.75, 0.48],  # right of tin, 9
        [0, 9.75, 1.78],  # Left of the service line, 10
        [6.4, 9.75, 1.78],  # Right of the service line, 11
            [0, 9.75, 4.57],  # Left of the top line of the front court, 12
        [6.4, 9.75, 4.57],  # Right of the top line of the front court, 13
    ]
    return reference_points_3d

def generate_camera_projection(pixel_points, real_world_points):
    """
    Generate camera projection parameters from corresponding pixel and 3D real-world points.
    
    Args:
        pixel_points (list): List of [x, y] coordinates in pixel space
        real_world_points (list): List of [x, y, z] coordinates in real-world space
        
    Returns:
        tuple: (rotation_vector, translation_vector, camera_matrix)
    """
    if len(pixel_points) < 6 or len(real_world_points) < 6:
        raise ValueError("At least 6 corresponding points are recommended for accurate 3D projection")
    
    # Convert points to numpy arrays
    pixel_points = np.array(pixel_points, dtype=np.float32)
    real_world_points = np.array(real_world_points, dtype=np.float32)
    
    # Estimate initial camera parameters
    camera_matrix = np.array([[1000, 0, 500],
                            [0, 1000, 500],
                            [0, 0, 1]], dtype=np.float32)  # Initial guess
    dist_coeffs = np.zeros((4,1))  # Assume no lens distortion
    
    # Find the rotation and translation vectors
    success, rotation_vector, translation_vector = cv2.solvePnP(
        real_world_points,
        pixel_points,
        camera_matrix,
        dist_coeffs
    )
    
    if not success:
        raise ValueError("Could not compute camera projection parameters")
        
    return rotation_vector, translation_vector, camera_matrix