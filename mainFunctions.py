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
