class Player:
    def __init__(self, player_id):
        self.player_id = player_id  # Unique track ID
        self.poses = []  # List to store poses (keypoints) over time

    def add_pose(self, keypoints):
        # Add the latest pose (keypoints) to the history
        self.poses.append(keypoints)

    def get_latest_pose(self):
        # Return the latest pose data
        return self.poses[-1] if self.poses else None
