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

def create_heatmap(player1_positions, player2_positions, output_path='heatmap.png', size=(640, 360)):
    """
    Create a heatmap visualization of player positions
    Args:
        player1_positions: list of [player1_pos, player2_pos] for each frame
        player2_positions: list of [player1_pos, player2_pos] for each frame
        output_path: where to save the resulting heatmap
        size: tuple of (width, height) for the output image
    """
    # Convert input lists to numpy arrays and extract x,y coordinates
    # Let's say we use the first keypoint (index 0) for each player
    player1_positions = np.array(player1_positions)
    player2_positions = np.array(player2_positions)
    
    # Extract x and y coordinates from the first keypoint of each frame
    player1_x = [pos[0][0] * size[0] for pos in player1_positions]  # Scale x coordinates
    player1_y = [pos[0][1] * size[1] for pos in player1_positions]  # Scale y coordinates
    player2_x = [pos[0][0] * size[0] for pos in player2_positions]  # Scale x coordinates
    player2_y = [pos[0][1] * size[1] for pos in player2_positions]  # Scale y coordinates
    
    # Create a black background for better contrast
    heatmap = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    
    # Create separate heatmaps for each player
    for positions_x, positions_y, color, alpha in [
        (player1_x, player1_y, 'red', 0.7),
        (player2_x, player2_y, 'blue', 0.7)
    ]:
        if len(positions_x) == 0:
            continue
            
        # Create 2D histogram
        heatmap_data, xedges, yedges = np.histogram2d(
            positions_x,  # x coordinates
            positions_y,  # y coordinates
            bins=[80, 45],  # Increased number of bins
            range=[[0, size[0]], [0, size[1]]]
        )
        
        # Apply stronger Gaussian blur for smoother visualization
        heatmap_data = cv2.GaussianBlur(heatmap_data, (21, 21), 0)
        
        # Normalize the heatmap
        heatmap_data = heatmap_data / np.max(heatmap_data)
        
        # Create a colored heatmap using seaborn with custom settings
        plt.figure(figsize=(12, 7))
        sns.heatmap(
            heatmap_data.T,
            cmap=f'{color.capitalize()}s',
            alpha=alpha,
            cbar=True,
            cbar_kws={'label': f'Player Density ({color})'}
        )
        plt.axis('off')
        
        # Save temporary heatmap with higher DPI
        plt.savefig('temp_heatmap.png', 
                   bbox_inches='tight', 
                   pad_inches=0, 
                   dpi=300)
        plt.close()
        
        # Read and resize the temporary heatmap
        temp_heatmap = cv2.imread('temp_heatmap.png')
        temp_heatmap = cv2.resize(temp_heatmap, size)
        
        # Blend the heatmap with the background
        heatmap = cv2.addWeighted(heatmap, 1, temp_heatmap, alpha, 0)
    
    # Add a title and legend to the final heatmap
    cv2.putText(heatmap, 'Player Movement Heatmap', 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                1, (255, 255, 255), 2)
    
    # Save the final heatmap
    cv2.imwrite(output_path, heatmap)
    return heatmap

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

def pixel_to_3d_points(pixel_points, reference_points_2d, reference_points_3d):
    """
    Convert pixel coordinates to 3D world coordinates
    
    Args:
        pixel_points: List of [x,y] pixel coordinates to convert
        reference_points_2d: List of reference points in pixel coordinates
        reference_points_3d: List of reference points in 3D world coordinates
    
    Returns:
        List of [x,y,z] world coordinates
    """
    # Generate homography matrix
    H = generate_homography(reference_points_2d, reference_points_3d)
    
    # Convert pixel points to numpy array
    points = np.array(pixel_points, dtype=np.float32)
    if points.ndim == 1:
        points = points.reshape(1, -1)
    
    # Add ones to make homogeneous coordinates
    ones = np.ones((points.shape[0], 1))
    points_homogeneous = np.hstack((points, ones))
    
    # Apply homography transformation
    transformed_points = []
    for point in points_homogeneous:
        # Apply perspective transformation
        transformed = H @ point
        # Normalize by dividing by the last coordinate
        transformed = transformed / transformed[2]
        # Add y-coordinate (height) based on court position
        # You might want to adjust this based on your needs
        transformed_3d = np.array([transformed[0], 0, transformed[1]])
        transformed_points.append(transformed_3d)
    
    return transformed_points
def convert_game_positions_to_3d(player_positions, reference_points_file='reference_points.json'):
    # Load reference points
    reference_points_2d = read_reference_points(reference_points_file)
    reference_points_3d = load_reference_points()
    
    # Convert each position
    converted_positions = []
    for pos in player_positions:
        # Assuming pos is in [x,y] format
        world_pos = pixel_to_3d_points([pos], reference_points_2d, reference_points_3d)
        converted_positions.append(world_pos[0])  # Take first element as we're converting one point
        
    return converted_positions
def main():
    try:
        # Read player positions from CSV
        player1_pos, player2_pos = read_player_positions()
        
        # Create heatmap visualization
        create_heatmap(player1_pos, player2_pos)
    except Exception as e:
        print(f"Main error: {str(e)}")

if __name__ == "__main__":
    main()
