import cv2
import numpy as np
import torch
from deep_sort_realtime.deepsort_tracker import DeepSort
from torchvision import models, transforms
from PIL import Image
from squash.Player import Player
from squash.Functions import Functions

# Initialize DeepSORT tracker with optimized parameters for squash
tracker = DeepSort(
    max_age=30,              # Reduced to handle fast movements better
    n_init=15,                # Reduced to initialize tracks faster
    max_cosine_distance=0.3, # Increased to be more lenient with appearance changes
    nn_budget=500,           # Added budget to maintain reliable tracking
    override_track_class=None,
    embedder="clip_ViT-B/16",
    half=True,
    bgr=True,
    embedder_gpu=True
)

# Initialize ResNet for appearance features with more robust feature extraction
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
feature_extractor = models.resnet50(pretrained=True).to(device)  # Using ResNet50 for better features
feature_extractor.eval()
feature_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Track ID to Player ID mapping with confidence scores
track_to_player = {}
player_positions_history = {1: [], 2: []}
MIN_BBOX_SIZE = 50  # Minimum bounding box size in pixels

def validate_bbox(bbox, frame_width, frame_height):
    """Validate and adjust bounding box dimensions"""
    x, y, w, h = map(int, bbox)
    
    # Ensure minimum size
    w = max(w, MIN_BBOX_SIZE)
    h = max(h, MIN_BBOX_SIZE)
    
    # Ensure aspect ratio is reasonable for a person (height should be greater than width)
    if w > h:
        h = int(w * 1.5)
    
    # Ensure box stays within frame
    x = max(0, min(x, frame_width - w))
    y = max(0, min(y, frame_height - h))
    
    return [x, y, w, h]

def extract_features(frame, bbox):
    """Extract appearance features with additional checks"""
    x, y, w, h = map(int, bbox)
    
    # Validate crop region
    if w <= 0 or h <= 0 or x < 0 or y < 0 or x + w > frame.shape[1] or y + h > frame.shape[0]:
        return np.zeros(1000)  # Return zero feature vector for invalid crops
        
    crop = frame[y:y+h, x:x+w]
    if crop.size == 0:
        return np.zeros(1000)
        
    try:
        img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        img = feature_transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            features = feature_extractor(img)
        return features.cpu().numpy().flatten()
    except Exception:
        return np.zeros(1000)

def update_player_position_history(player_id, position):
    """Keep track of player positions for trajectory analysis"""
    player_positions_history[player_id].append(position)
    if len(player_positions_history[player_id]) > 30:  # Keep last 30 frames
        player_positions_history[player_id].pop(0)

def get_player_velocity(player_id):
    """Calculate player velocity from position history"""
    positions = player_positions_history[player_id]
    if len(positions) < 2:
        return 0, 0
    
    last_pos = positions[-1]
    prev_pos = positions[-2]
    return last_pos[0] - prev_pos[0], last_pos[1] - prev_pos[1]

def framepose(
    pose_model,
    frame,
    otherTrackIds,
    updated,
    references1,
    references2,
    pixdiffs,
    players,
    frame_count,
    player_last_positions,
    frame_width,
    frame_height,
    annotated_frame,
    max_players=2,
):
    try:
        track_results = pose_model.track(frame, persist=True, show=False)
        
        if (track_results and hasattr(track_results[0], "keypoints") 
            and track_results[0].keypoints is not None):
            
            boxes = track_results[0].boxes.xywh.cpu()
            track_ids = track_results[0].boxes.id.int().cpu().tolist()
            keypoints = track_results[0].keypoints.cpu().numpy()

            # Validate and adjust bounding boxes
            valid_detections = []
            for i, (box, kp) in enumerate(zip(boxes, keypoints)):
                # Use keypoints to improve bounding box
                print(f'kp: {kp[0][:, 2]}')
                valid_points = kp[0][kp[0][:, 2][0] > 0.5]  # Only use keypoints with high confidence
                if len(valid_points) > 0:
                    x_min = valid_points[:, 0].min() * frame_width
                    x_max = valid_points[:, 0].max() * frame_width
                    y_min = valid_points[:, 1].min() * frame_height
                    y_max = valid_points[:, 1].max() * frame_height
                    
                    # Add padding
                    width = (x_max - x_min) * 1.2
                    height = (y_max - y_min) * 1.2
                    
                    bbox = validate_bbox([x_min, y_min, width, height], frame_width, frame_height)
                    feature = extract_features(frame, bbox)
                    valid_detections.append([bbox, 0.9, feature])

            # Update tracks
            tracks = tracker.update_tracks(valid_detections, frame=frame)

            # Process each track
            for track, kp in zip(tracks, keypoints):
                if not track.is_confirmed():
                    continue

                track_id = track.track_id
                bbox = track.to_tlwh()
                bbox = validate_bbox(bbox, frame_width, frame_height)
                x, y, w, h = map(int, bbox)

                # Determine player ID with enhanced logic
                if track_id not in track_to_player:
                    if len(track_to_player) == 0:
                        track_to_player[track_id] = 1
                    elif len(track_to_player) == 1:
                        # Assign player 2 based on position relative to player 1
                        other_track_id = list(track_to_player.keys())[0]
                        other_player_pos = player_last_positions[track_to_player[other_track_id]]
                        if x < other_player_pos[0]:  # Left player is player 1
                            track_to_player[track_id] = 2 if track_to_player[other_track_id] == 1 else 1
                        else:
                            track_to_player[track_id] = 1 if track_to_player[other_track_id] == 2 else 2
                    else:
                        # Use appearance and motion features for matching
                        feature = extract_features(frame, [x, y, w, h])
                        best_match = None
                        best_score = float('-inf')
                        
                        for pid in [1, 2]:
                            if pid in player_last_positions:
                                px, py = player_last_positions[pid]
                                vx, vy = get_player_velocity(pid)
                                
                                # Predict position based on velocity
                                predicted_x = px + vx
                                predicted_y = py + vy
                                
                                # Calculate position and appearance scores
                                distance_score = -np.sqrt((predicted_x - x)**2 + (predicted_y - y)**2)
                                appearance_score = np.dot(feature, extract_features(frame, [px, py, w, h]))
                                
                                # Combined score
                                total_score = distance_score * 0.7 + appearance_score * 0.3
                                
                                if total_score > best_score:
                                    best_score = total_score
                                    best_match = pid
                        
                        track_to_player[track_id] = best_match if best_match else (1 if len(players) == 0 else 2)

                playerid = track_to_player[track_id]
                update_player_position_history(playerid, (x, y))

                # Update player info
                if playerid in players:
                    players[playerid].add_pose(kp)
                    player_last_positions[playerid] = (x, y)
                    updated[playerid-1][0] = True
                    updated[playerid-1][1] = frame_count
                elif len(players) < max_players:
                    players[playerid] = Player(player_id=playerid)
                    player_last_positions[playerid] = (x, y)
                    updated[playerid-1][0] = True
                    updated[playerid-1][1] = frame_count

                # Draw visualizations
                color = (0, 0, 255) if playerid == 1 else (255, 0, 0)
                cv2.rectangle(annotated_frame, (x, y), (x+w, y+h), color, 2)
                
                # Draw keypoints
                for keypoint in kp:
                    for i, k in enumerate(keypoint.xyn[0]):
                        if keypoint.conf[0][i] > 0.5:  # Only draw high-confidence keypoints
                            kx, ky = int(k[0] * frame_width), int(k[1] * frame_height)
                            cv2.circle(annotated_frame, (kx, ky), 3, color, 5)
                            if i == 16:  # Head keypoint
                                cv2.putText(
                                    annotated_frame,
                                    f"P{playerid}",
                                    (kx, ky),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    1.0,
                                    color,
                                    2,
                                )

        return [
            pose_model, frame, otherTrackIds, updated, references1, references2,
            pixdiffs, players, frame_count, player_last_positions,
            frame_width, frame_height, annotated_frame,
        ]

    except Exception as e:
        print(f"Error in framepose: {e}")
        print(f'line was {e.__traceback__.tb_lineno}')
        print(f'all other info: {e.__traceback__}')
        return [
            pose_model, frame, otherTrackIds, updated, references1, references2,
            pixdiffs, players, frame_count, player_last_positions,
            frame_width, frame_height, annotated_frame,
        ]