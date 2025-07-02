import cv2
import numpy as np
import torch
from deep_sort_realtime.deepsort_tracker import DeepSort
from torchvision import models, transforms
from PIL import Image
from squash.Player import Player

# Global tracker variable to prevent multiple initializations
tracker = None
device = None
device_name = None
feature_extractor = None
feature_transform = None

def initialize_tracker():
    """Initialize DeepSORT tracker and feature extractor only once"""
    global tracker, device, device_name, feature_extractor, feature_transform
    
    if tracker is None:
        print("Initializing DeepSORT tracker (one-time setup)...")
        
        # Initialize DeepSORT tracker with optimized parameters for squash
        tracker = DeepSort(
            max_age=30,  # Reduced to handle fast movements better
            n_init=15,  # Reduced to initialize tracks faster
            max_cosine_distance=0.3,  # Increased to be more lenient with appearance changes
            nn_budget=500,  # Added budget to maintain reliable tracking
            override_track_class=None,
            embedder="clip_ViT-B/16",
            half=True,
            bgr=True,
            embedder_gpu=True if torch.cuda.is_available() else False,  # Enable GPU for embedder
        )

        # Initialize ResNet for appearance features with more robust feature extraction
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device_name = "GPU" if torch.cuda.is_available() else "CPU"
        print(f"DeepSORT using {device_name} for feature extraction")

        feature_extractor = models.resnet50(pretrained=True).to(device)  # Using ResNet50 for better features
        feature_extractor.eval()
        
        feature_transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        
        print(" DeepSORT tracker initialized successfully")

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
    if (
        w <= 0
        or h <= 0
        or x < 0
        or y < 0
        or x + w > frame.shape[1]
        or y + h > frame.shape[0]
    ):
        return np.zeros(1000)  # Return zero feature vector for invalid crops

    crop = frame[y : y + h, x : x + w]
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
    occluded=False,
    importantdata=[],
    embeddings=[[], []],
    plast=[[], []]
):
    try:
        # Initialize tracker only once
        initialize_tracker()
        
        track_results = pose_model.track(frame, persist=True, show=False)

        if (
            track_results
            and hasattr(track_results[0], "keypoints")
            and track_results[0].keypoints is not None
            and len(track_results) > 0
        ):
            boxes = track_results[0].boxes.xywh.cpu()
            track_ids = track_results[0].boxes.id.int().cpu().tolist() if track_results[0].boxes.id is not None else []
            keypoints = track_results[0].keypoints.cpu().numpy()

            # Ensure we have matching numbers of boxes, track_ids, and keypoints
            num_detections = min(len(boxes), len(keypoints))
            if len(track_ids) > 0:
                num_detections = min(num_detections, len(track_ids))

            # Validate and adjust bounding boxes
            valid_detections = []
            for i in range(num_detections):
                if i >= len(boxes) or i >= len(keypoints):
                    continue
                    
                box = boxes[i]
                kp = keypoints[i]
                # Use keypoints to improve bounding box
                # print(f'kp: {kp[0].data[:, :, 2]}')  # Access the confidence scores
                valid_points = kp[0].data[
                    kp[0].data[:, :, 2] > 0.5
                ]  # Filter keypoints with confidence > 0.5
                # print(f'Valid points: {valid_points}')
                if len(valid_points) > 0:
                    x_min = valid_points[:, 0].min() * frame_width
                    x_max = valid_points[:, 0].max() * frame_width
                    y_min = valid_points[:, 1].min() * frame_height
                    y_max = valid_points[:, 1].max() * frame_height

                    # Add padding
                    width = (x_max - x_min) * 1.2
                    height = (y_max - y_min) * 1.2

                    bbox = validate_bbox(
                        [x_min, y_min, width, height], frame_width, frame_height
                    )
                    feature = extract_features(frame, bbox)
                    valid_detections.append([bbox, 0.9, feature])

            # Update tracks
            tracks = tracker.update_tracks(valid_detections, frame=frame)

            # Process each track with proper bounds checking
            for track_idx, track in enumerate(tracks):
                if not track.is_confirmed():
                    continue
                
                # Make sure we have corresponding keypoints
                if track_idx >= len(keypoints) or track_idx >= num_detections:
                    continue
                    
                kp = keypoints[track_idx]
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
                        other_player_id = track_to_player[other_track_id]
                        if other_player_id in player_last_positions and len(player_last_positions[other_player_id]) >= 2:
                            other_player_pos = player_last_positions[other_player_id]
                            if x < other_player_pos[0]:  # Left player is player 1
                                track_to_player[track_id] = (
                                    2 if track_to_player[other_track_id] == 1 else 1
                                )
                            else:
                                track_to_player[track_id] = (
                                    1 if track_to_player[other_track_id] == 2 else 2
                                )
                        else:
                            # Default assignment if no position history
                            track_to_player[track_id] = 2
                    else:
                        # Use appearance and motion features for matching
                        feature = extract_features(frame, [x, y, w, h])
                        best_match = None
                        best_score = float("-inf")

                        for pid in [1, 2]:
                            if pid in player_last_positions and len(player_last_positions[pid]) >= 2:
                                px, py = player_last_positions[pid][:2]  # Safe access to first two elements
                                vx, vy = get_player_velocity(pid)

                                # Predict position based on velocity
                                predicted_x = px + vx
                                predicted_y = py + vy

                                # Calculate position and appearance scores
                                distance_score = -np.sqrt(
                                    (predicted_x - x) ** 2 + (predicted_y - y) ** 2
                                )
                                appearance_score = np.dot(
                                    feature, extract_features(frame, [px, py, w, h])
                                )

                                # Combined score
                                total_score = (
                                    distance_score * 0.7 + appearance_score * 0.3
                                )

                                if total_score > best_score:
                                    best_score = total_score
                                    best_match = pid

                        track_to_player[track_id] = (
                            best_match
                            if best_match
                            else (1 if len(players) == 0 else 2)
                        )

                playerid = track_to_player[track_id]
                update_player_position_history(playerid, (x, y))

                # Update player info with proper error handling
                if playerid in players:
                    try:
                        players[playerid].add_pose(kp)
                    except (IndexError, AttributeError) as e:
                        print(f"Error adding pose for player {playerid}: {e}")
                    player_last_positions[playerid] = (x, y)
                    if playerid - 1 < len(updated):
                        updated[playerid - 1][0] = True
                        updated[playerid - 1][1] = frame_count
                elif len(players) < max_players:
                    players[playerid] = Player(player_id=playerid)
                    try:
                        players[playerid].add_pose(kp)
                    except (IndexError, AttributeError) as e:
                        print(f"Error adding pose for new player {playerid}: {e}")
                    player_last_positions[playerid] = (x, y)
                    if playerid - 1 < len(updated):
                        updated[playerid - 1][0] = True
                        updated[playerid - 1][1] = frame_count

                # Draw visualizations
                color = (0, 0, 255) if playerid == 1 else (255, 0, 0)
                cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), color, 2)

                # Draw keypoints with error handling
                try:
                    if hasattr(kp, '__len__') and len(kp) > 0:
                        for keypoint in kp:
                            if hasattr(keypoint, 'xyn') and len(keypoint.xyn) > 0 and hasattr(keypoint, 'conf') and len(keypoint.conf) > 0:
                                for i, k in enumerate(keypoint.xyn[0]):
                                    if (
                                        i < len(keypoint.conf[0]) and
                                        keypoint.conf[0][i] > 0.5
                                    ):  # Only draw high-confidence keypoints
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
                except (IndexError, AttributeError, TypeError) as e:
                    print(f"Error drawing keypoints for player {playerid}: {e}")
                    continue

        return [
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
            occluded,
            importantdata,
            embeddings,
            plast
        ]

    except Exception as e:
        print(f"Error in framepose: {e}")
        print(f"line was {e.__traceback__.tb_lineno}")
        print(f"all other info: {e.__traceback__}")
        return [
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
            occluded,
            importantdata,
            embeddings,
            plast
        ]
