import numpy as np

class player:
    def __init__(self, player_id, initial_position, initial_pose):
        self.player_id = player_id  # A unique identifier for each player
        self.positions = [initial_position]  # List to store past positions
        self.poses = [initial_pose]  # List to store past poses (e.g., keypoints)
    
    def update(self, new_position, new_pose):
        # Append the new position and pose to the history
        self.positions.append(new_position)
        self.poses.append(new_pose)
    
    def get_last_position(self):
        # Get the most recent position
        return self.positions[-1] if self.positions else None
    
    def get_last_pose(self):
        # Get the most recent pose
        return self.poses[-1] if self.poses else None
    
    def get_movement_history(self):
        # Return all stored movements (positions) over time
        return self.positions

