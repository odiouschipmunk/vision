import cv2
from collections import defaultdict
import numpy as np
from ultralytics import YOLO
from Player import Player
# Load the YOLOv11 model for pose detection
model = YOLO("models/yolo11m-pose.pt")  # Ensure this is a pose model

# Open the video file
video_path = "main-video (1).mp4"
cap = cv2.VideoCapture(video_path)

# Store Player objects based on track ID
players = {}

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLO11 pose tracking on the frame, persisting tracks between frames
        results = model.track(frame, persist=True)

        # Ensure results are not empty
# Maximum number of players
        MAX_PLAYERS = 2
        player_last_positions = {}  # To track the last known positions of players

        # Track the results
        track_results = pose_model.track(frame, persist=True)

        if track_results and hasattr(track_results[0], 'keypoints') and track_results[0].keypoints is not None:
            # Extract boxes, track IDs, and keypoints from pose results
            boxes = track_results[0].boxes.xywh.cpu()
            track_ids = track_results[0].boxes.id.int().cpu().tolist()
            keypoints = track_results[0].keypoints.cpu().numpy()

            for box, track_id, kp in zip(boxes, track_ids, keypoints):
                x, y, w, h = box

                # If there are fewer than MAX_PLAYERS, add new players as necessary
                if track_id not in players and len(players) < MAX_PLAYERS:
                    players[track_id] = Player(player_id=track_id)
                    player_last_positions[track_id] = (x, y)  # Store the player's last known position

                # If the ID is not new but the player was temporarily occluded
                elif track_id in players:
                    # Update the player's position and pose
                    players[track_id].add_pose(kp)
                    player_last_positions[track_id] = (x, y)

            # Handle reappearance of occluded players
            for player_id, last_position in list(player_last_positions.items()):
                if player_id not in track_ids:
                    # The player is temporarily occluded, we will search for them in future frames
                    print(f"Player {player_id} is occluded, keeping track.")

                elif len(track_ids) < MAX_PLAYERS and len(players) == MAX_PLAYERS:
                    # In case of occlusion and then reappearance, we can map the ID back
                    if player_id not in track_ids:
                        # Find the closest box to the last known position and reassign the ID
                        distances = [np.linalg.norm(np.array(last_position) - np.array([box[0], box[1]])) for box in boxes]
                        min_distance_index = np.argmin(distances)
                        closest_box = boxes[min_distance_index]

                        # Reassign the closest box to the occluded player
                        players[player_id] = players[track_ids[min_distance_index]]
                        track_ids[min_distance_index] = player_id

            # Display the annotated frame
            cv2.imshow("YOLO11 Pose Tracking", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
