import cv2
import numpy as np
import torch
from deep_sort_realtime.deepsort_tracker import DeepSort
from torchvision import models, transforms
from PIL import Image
from squash.Player import Player
from squash.Functions import Functions

# Initialize DeepSORT tracker with stricter parameters
tracker = DeepSort(
    max_age=30,
    n_init=2,
    max_cosine_distance=0.2,  # Stricter appearance matching
    nn_budget=100,
    override_track_class=None,
    embedder="mobilenet",
    half=True,
    bgr=True,
    embedder_gpu=True
)

# Initialize ResNet for appearance features
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
feature_extractor = models.resnet18(pretrained=True).to(device)
feature_extractor.eval()
feature_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Track ID to Player ID mapping
track_to_player = {}

def extract_features(frame, bbox):
    """Extract appearance features from player crop"""
    x, y, w, h = map(int, bbox)
    crop = frame[y:y+h, x:x+w]
    img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    img = feature_transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        features = feature_extractor(img)
    return features.cpu().numpy().flatten()

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

            # Prepare detections with features
            detections = []
            for box in boxes:
                x, y, w, h = map(int, box)
                feature = extract_features(frame, [x, y, w, h])
                detections.append([
                    [x, y, w, h], 
                    0.9,  # High confidence for pose detections
                    feature
                ])

            # Update tracks
            tracks = tracker.update_tracks(detections, frame=frame)

            # Process each track
            for track, kp in zip(tracks, keypoints):
                if not track.is_confirmed():
                    continue

                track_id = track.track_id
                bbox = track.to_tlwh()
                x, y, w, h = map(int, bbox)

                # Determine player ID
                if track_id not in track_to_player:
                    if len(track_to_player) == 0:
                        track_to_player[track_id] = 1
                    elif len(track_to_player) == 1:
                        track_to_player[track_id] = 2
                    else:
                        # Compare features with existing players
                        feature = extract_features(frame, [x, y, w, h])
                        best_match = None
                        best_similarity = -1
                        
                        for tid, pid in track_to_player.items():
                            if pid in player_last_positions:
                                px, py = player_last_positions[pid]
                                player_feature = extract_features(frame, [px, py, w, h])
                                similarity = np.dot(feature, player_feature)
                                if similarity > best_similarity:
                                    best_similarity = similarity
                                    best_match = pid
                        
                        track_to_player[track_id] = best_match if best_match else (1 if len(players) == 0 else 2)

                playerid = track_to_player[track_id]

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
                    print(f"Player {playerid} added with track ID {track_id}")

                # Update references
                if playerid == 1:
                    references1.append(Functions.sum_pixels_in_bbox(frame, [x, y, w, h]))
                else:
                    references2.append(Functions.sum_pixels_in_bbox(frame, [x, y, w, h]))

                if len(references1) > 1 and len(references2) > 1 and len(pixdiffs) < 5:
                    pixdiffs.append(abs(references1[-1] - references2[-1]))

                # Draw bounding box and keypoints with consistent colors
                color = (0, 0, 255) if playerid == 1 else (255, 0, 0)
                cv2.rectangle(annotated_frame, (x, y), (x+w, y+h), color, 2)
                
                # Draw keypoints with same color as bounding box
                for keypoint in kp:
                    i = 0
                    for k in keypoint.xyn[0]:
                        kx, ky = k
                        kx = int(kx * frame_width)
                        ky = int(ky * frame_height)
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
                        i += 1

        return [
            pose_model, frame, otherTrackIds, updated, references1, references2,
            pixdiffs, players, frame_count, player_last_positions,
            frame_width, frame_height, annotated_frame,
        ]

    except Exception as e:
        print(f"Error in framepose: {e}")
        return [
            pose_model, frame, otherTrackIds, updated, references1, references2,
            pixdiffs, players, frame_count, player_last_positions,
            frame_width, frame_height, annotated_frame,
        ]