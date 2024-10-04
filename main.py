import cv2
from ultralytics import YOLO
import numpy as np
import math
# Load models
pose_model = YOLO('models/yolo11m-pose.pt')
ballmodel = YOLO('trained-models/g-ball2.pt')
racketmodel=YOLO('trained-models/squash-racket.pt')
courtmodel=YOLO('trained-models/court-key!.pt')
# Video file path
video_file = 'Squash Farag v Hesham - Houston Open 2022 - Final Highlights.mp4'
video_folder = 'full-games'
path = 'main-video (1).mp4'

cap = cv2.VideoCapture(path)
frame_width = 1920
frame_height = 1080
players={}
from Ball import Ball
# Get video dimensions
import logging
from Player import Player
max_players = 2
player_last_positions = {}
occluded_players = set()    # To keep track of occluded players

logging.getLogger('ultralytics').setLevel(logging.ERROR)
# Create a blank canvas for heatmap based on video resolution
heatmap = np.zeros((frame_height, frame_width), dtype=np.float32)
mainball=Ball(0,0,0,0)
ballmap=np.zeros((frame_height, frame_width), dtype=np.float32)
def drawmap(lx,ly,rx,ry, map):

    # Update heatmap at the ankle positions
    lx = min(max(lx, 0), map.shape[1] - 1)  # Bound lx to [0, width-1]
    ly = min(max(ly, 0), map.shape[0] - 1)  # Bound ly to [0, height-1]
    rx = min(max(rx, 0), map.shape[1] - 1)  # Bound rx to [0, width-1]
    ry = min(max(ry, 0), map.shape[0] - 1)
    map[ly, lx] += 1
    map[ry, rx] += 1
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break
    frame = cv2.resize(frame, (frame_width, frame_height))

    # Pose and ball detection
    ball = ballmodel(frame)
    pose_results = pose_model(frame)
    #only plot the top 2 confs
    annotated_frame=pose_results[0].plot()
    court_results=courtmodel(frame)
    # Check if keypoints exist and are not empty
    if pose_results[0].keypoints.xyn is not None and len(pose_results[0].keypoints.xyn[0]) > 0:
        for person in pose_results[0].keypoints.xyn:
            #print(f'Length of person keypoints: {len(person)}')
            #print(f'Person keypoints: {person}')
        
            if len(person) >= 17:  # Ensure at least 17 keypoints are present

                left_ankle_x = int(person[16][0] * frame_width)  # Scale the X coordinate
                left_ankle_y = int(person[16][1] * frame_height)  # Scale the Y coordinate
                right_ankle_x = int(person[15][0] * frame_width)  # Scale the X coordinate
                right_ankle_y = int(person[15][1] * frame_height)  # Scale the Y coordinate
                if left_ankle_x > 0 or left_ankle_y > 0 or right_ankle_x > 0 or right_ankle_y > 0:
                    drawmap(left_ankle_x, left_ankle_y, right_ankle_x, right_ankle_y, heatmap)
    else:
        #print("No keypoints detected in this frame.")
        continue
    highestconf=0
    x1=x2=y1=y2=0
    # Ball detection
    #make it so that if it detects the ball in the same place multiple times it takes that out
    for box in ball[0].boxes:
        coords = box.xyxy[0] if len(box.xyxy) == 1 else box.xyxy
        x1temp, y1temp, x2temp, y2temp = coords
        label = ballmodel.names[int(box.cls)]
        confidence = float(box.conf)  # Convert tensor to float
        avgxtemp=int((x1temp+x2temp)/2)
        avgytemp=int((y1temp+y2temp)/2)
        if abs(avgxtemp-363)<10 and abs(avgytemp-72)<10:
            #false positive near the "V"
            #TODO find out how to check for false positives for general videos
            continue
        if confidence>highestconf:
            highestconf=confidence
            x1=x1temp
            y1=y1temp
            x2=x2temp
            y2=y2temp
    cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
    cv2.putText(annotated_frame, f'{label} {highestconf:.2f}', (int(x1), int(y1) - 10), 
    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    avg_x=int((x1+x2)/2)
    avg_y=int((y1+y2)/2)
    distance=0
    size=avg_x*avg_y
    if avg_x>0 or avg_y>0:
        if mainball.getlastpos()[0]!=avg_x or mainball.getlastpos()[1]!=avg_y:
            #print(mainball.getlastpos())
            #print(mainball.getloc())
            mainball.update(avg_x, avg_y, size)
            #print(mainball.getlastpos())
            #print(mainball.getloc())
            distance=math.hypot(avg_x-mainball.getlastpos()[0], avg_y-mainball.getlastpos()[1])
            
            with open('ball.txt', 'a') as f:
                f.write(f'Position(in pixels): {mainball.getloc()}\nDistance: {distance}\n')
                #print(f'Position(in pixels): {mainball.getloc()}\nDistance: {distance}\n')
                drawmap(mainball.getloc()[0], mainball.getloc()[1], mainball.getlastpos()[0], mainball.getlastpos()[1], ballmap)

    
    # Blur and normalize the heatmap for display
    #heatmap_blurred = cv2.GaussianBlur(heatmap, (15, 15), 0)
    heatmap_normalized = cv2.normalize(heatmap, None, 100, 255, cv2.NORM_MINMAX)
    heatmap_colored = cv2.applyColorMap(heatmap_normalized.astype(np.uint8), cv2.COLORMAP_BONE)
    ball_normalized = cv2.normalize(ballmap, None, 100, 255, cv2.NORM_MINMAX)
    ballmap_colorized = cv2.applyColorMap(ball_normalized.astype(np.uint8), cv2.COLORMAP_BONE)

    track_results=pose_model.track(frame, persist=True)
    if track_results and hasattr(track_results[0], 'keypoints') and track_results[0].keypoints is not None:
        # Extract boxes, track IDs, and keypoints from pose results
        boxes = track_results[0].boxes.xywh.cpu()
        track_ids = track_results[0].boxes.id.int().cpu().tolist()
        keypoints = track_results[0].keypoints.cpu().numpy()

        current_ids = set(track_ids)

        # Update or add players for currently visible track IDs
        for box, track_id, kp in zip(boxes, track_ids, keypoints):
            x, y, w, h = box

            # If player is already tracked, update their info
            if track_id in players:
                players[track_id].add_pose(kp)
                player_last_positions[track_id] = (x, y)  # Update position
                if track_id in occluded_players:
                    occluded_players.remove(track_id)  # Player is no longer occluded

            # If the player is new and fewer than MAX_PLAYERS are being tracked
            elif len(players) < max_players:
                players[track_id] = Player(player_id=track_id)
                player_last_positions[track_id] = (x, y)
                print(f"Player {track_id} added.")

        # Handle occluded players
        for player_id in list(player_last_positions.keys()):
            if player_id not in current_ids:
                # The player is temporarily occluded
                occluded_players.add(player_id)
                print(f"Player {player_id} is occluded, keeping track.")

        # Reassign occluded players if they reappear
        for player_id in occluded_players.copy():  # Use copy to modify set inside loop
            # Only reassign if there are fewer than MAX_PLAYERS
            if len(players) <= max_players:
                # Find the closest detected box to the occluded player's last known position
                distances = [np.linalg.norm(np.array(player_last_positions[player_id]) - np.array([box[0], box[1]])) for box in boxes]
                min_distance_index = np.argmin(distances)
                closest_box = boxes[min_distance_index]

                # Ensure the distance is within a reasonable threshold to reassign the ID
                if distances[min_distance_index] < frame_width/3:  # Adjust threshold if needed
                    reassigned_id = track_ids[min_distance_index]

                    # Only reassign the ID if the new ID (reassigned_id) is not already tracked
                    if reassigned_id not in players:
                        # Reassign the closest box to the occluded player
                        players[player_id] = players.pop(reassigned_id)  # Transfer player data
                        track_ids[min_distance_index] = player_id  # Reassign the ID to the occluded player
                        print(f"Player {player_id} reappeared and reassigned from ID {reassigned_id}.")

                        occluded_players.remove(player_id)  # Remove from occluded list

            else:
                print(f"Player {player_id} reappeared, but too many players are tracked.")
    highestconf=0
    x1c=x2c=y1c=y2c=0

    for box in court_results[0].boxes:
        coords = box.xyxy[0] if len(box.xyxy) == 1 else box.xyxy
        x1temp, y1temp, x2temp, y2temp = coords
        label = courtmodel.names[int(box.cls)]
        confidence = float(box.conf)
        cv2.rectangle(annotated_frame, (int(x1temp), int(y1temp)), (int(x2temp), int(y2temp)), (255, 0, 0), 2)
        cv2.putText(annotated_frame, f'{label} {confidence:.2f}', (int(x1temp), int(y1temp) - 10), 
        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        #print(f'{label} {confidence:.2f} GOT COURT')
    # Save the heatmap
    cv2.imwrite('foot_placement_heatmap2.png', heatmap_colored)
    cv2.imwrite('ball_heatmap.png', ballmap_colorized)
    # Display the annotated frame
    cv2.imshow('Annotated Frame', annotated_frame)
    '''
    COURT DETECTION
    for box in court[0].boxes:
        coords = box.xyxy[0] if len(box.xyxy) == 1 else box.xyxy
        x1temp, y1temp, x2temp, y2temp = coords
        label = courtmodel.names[int(box.cls)]
        confidence = float(box.conf)
        cv2.rectangle(annotated_frame, (int(x1temp), int(y1temp)), (int(x2temp), int(y2temp)), (0, 255, 0), 2)
        cv2.putText(annotated_frame, f'{label} {confidence:.2f}', (int(x1temp), int(y1temp) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

'''
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()