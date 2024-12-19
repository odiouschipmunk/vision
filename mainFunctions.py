import csv
import ast
import numpy as np
import cv2
import seaborn as sns
import matplotlib.pyplot as plt
import json

def read_player_positions(csv_path='final.csv'):
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

def read_ball_positions(path='final.csv'):
    ball_positions = []
    with open(path, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        next(csvreader)  # Skip header row if it exists
        for row in csvreader:
            if len(row) >= 3:  # Make sure we have enough columns
                ball_positions.append(ast.literal_eval(row[3].strip()))
    return ball_positions

def read_reference_points(path='reference_points.json'):
    with open(path, 'r') as f:
        return json.load(f)
    
def load_reference_points():
    reference_points_3d = [
        [0, 9.75, 0],  # Top-left corner, 1
        [6.4, 9.75, 0],  # Top-right corner, 2
        [6.4, 0, 0],  # Bottom-right corner, 3
        [0, 0, 0],  # Bottom-left corner, 4
        [3.2, 4.31, 0],  # "T" point, 5
        [0, 2.71, 0],  # Left bottom of the service box, 6
        [6.4, 2.71, 0],  # Right bottom of the service box, 7
        [0, 4.31, 0], #left top of service box, 8
        [6.4, 4.31, 0], #right top of service box, 9
        [0, 9.75, 0.48],  # left of tin, 10
        [6.4, 9.75, 0.48],  # right of tin, 11
        [0, 9.75, 1.78],  # Left of the service line, 12
        [6.4, 9.75, 1.78],  # Right of the service line, 13
        [0, 9.75, 4.57],  # Left of the top line of the front court, 14
        [6.4, 9.75, 4.57],  # Right of the top line of the front court, 15
    ]
    return reference_points_3d
def read_rl_positions(path='final.csv'):
    rlplayer1pos=[]
    rlplayer2pos=[]
    rlballpos=[]
    with open(path, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        next(csvreader)  # Skip header row if it exists
        for row in csvreader:
            if len(row) >= 8:
                rlplayer1pos.append(ast.literal_eval(row[5].strip()))
                rlplayer2pos.append(ast.literal_eval(row[6].strip()))
                rlballpos.append(ast.literal_eval(row[7].strip()))
    return rlplayer1pos, rlplayer2pos, rlballpos
def generate_homography_matrices(reference_points_2d, reference_points_3d):
    # Convert inputs to numpy arrays
    reference_points_2d = np.array(reference_points_2d)
    reference_points_3d = np.array(reference_points_3d)
    
    H_xy = cv2.findHomography(reference_points_2d, reference_points_3d[:, :2])[0]
    # Reshape Z coordinates to be 2D points
    z_coords = np.column_stack((reference_points_2d[:, 0], reference_points_3d[:, 2]))
    H_z = cv2.findHomography(reference_points_2d, z_coords)[0]
    return H_xy, H_z
def convert_2d_to_3d(points_2d, H_xy, H_z):
    """
    Convert 2D points to 3D coordinates using homography transformation
    
    Args:
        points_2d: List of 2D points to convert [[x,y], ...]
        reference_points_2d: List of 2D reference points from camera view
        reference_points_3d: List of corresponding 3D reference points in real-world coordinates
        
    Returns:
        points_3d: List of converted 3D points
    """
    # Convert inputs to numpy arrays
    points_2d = np.array(points_2d, dtype=np.float32)
    
    # Ensure points are in correct shape
    if len(points_2d.shape) == 1:
        points_2d = points_2d.reshape(-1, 2)
    
    # Convert points to homogeneous coordinates
    points_homogeneous = np.hstack((points_2d, np.ones((len(points_2d), 1))))
    
    # Transform points
    xy_transformed = np.dot(H_xy, points_homogeneous.T)
    z_transformed = np.dot(H_z, points_homogeneous.T)
    
    # Convert back from homogeneous coordinates
    xy_transformed = xy_transformed / xy_transformed[2]
    z_transformed = z_transformed / z_transformed[2]
    
    # Combine XY and Z coordinates
    points_3d = np.column_stack((xy_transformed[:2].T, z_transformed[0].T))
    
    return points_3d

def visualize_3d_ball_position(reference_points_3d, ball_positions):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    ref_points = np.array(reference_points_3d)
    ball_pos = np.array(ball_positions)
    ax.scatter(ref_points[:, 0], ref_points[:, 1], ref_points[:, 2], c='green', s=100, marker='o', label='Reference Points')
    ax.scatter(ball_pos[:, 0], ball_pos[:, 1], ball_pos[:, 2], c='red', s=100, marker='o', label='Ball Position')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.legend()
    plt.show()


def visualize_3d_animation(reference_points_3d, player1_positions, player2_positions):
    """
    Create an interactive 3D animation of player movements with reference points
    
    Args:
        reference_points_3d: List of 3D reference points
        player1_positions: List of 3D positions for player 1
        player2_positions: List of 3D positions for player 2
    """
    # Create figure and 3D axes
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Convert inputs to numpy arrays for easier handling
    ref_points = np.array(reference_points_3d)
    p1_pos = np.array(player1_positions)
    p2_pos = np.array(player2_positions)
    
    # Plot reference points (green and bold)
    ax.scatter(ref_points[:, 0], ref_points[:, 1], ref_points[:, 2], 
              c='green', s=100, marker='o', label='Reference Points')
    
    # Initialize player positions (will be updated in animation)
    p1_scatter = ax.scatter([], [], [], c='blue', s=100, label='Player 1')
    p2_scatter = ax.scatter([], [], [], c='red', s=100, label='Player 2')
    
    # Add trail effect (last 10 positions)
    trail_length = 10
    p1_trail = ax.scatter([], [], [], c='lightblue', s=30, alpha=0.5)
    p2_trail = ax.scatter([], [], [], c='pink', s=30, alpha=0.5)
    
    # Set axis labels and limits
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    
    # Set consistent axis limits based on court dimensions
    ax.set_xlim(0, 6.4)
    ax.set_ylim(0, 9.75)
    ax.set_zlim(0, 4.57)
    
    # Add legend
    ax.legend()
    
    def update(frame):
        # Update current positions
        p1_scatter._offsets3d = (p1_pos[frame:frame+1, 0], 
                                p1_pos[frame:frame+1, 1], 
                                p1_pos[frame:frame+1, 2])
        p2_scatter._offsets3d = (p2_pos[frame:frame+1, 0], 
                                p2_pos[frame:frame+1, 1], 
                                p2_pos[frame:frame+1, 2])
        
        # Update trails
        start_idx = max(0, frame - trail_length)
        p1_trail._offsets3d = (p1_pos[start_idx:frame, 0],
                              p1_pos[start_idx:frame, 1],
                              p1_pos[start_idx:frame, 2])
        p2_trail._offsets3d = (p2_pos[start_idx:frame, 0],
                              p2_pos[start_idx:frame, 1],
                              p2_pos[start_idx:frame, 2])
        return p1_scatter, p2_scatter, p1_trail, p2_trail
    
    # Create animation
    from matplotlib.animation import FuncAnimation
    anim = FuncAnimation(fig, update, frames=len(p1_pos),
                        interval=5, blit=True)
    
    # Add interactive rotation
    plt.show()
    
    return anim