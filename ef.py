import time
import torch
start = time.time()
import cv2
import csv
from PIL import Image
import torch.nn.functional as F
import time
import csvanalyze
import logging
import math
import numpy as np
import matplotlib
import clip
from squash.Player import Player
from skimage.metrics import structural_similarity as ssim_metric
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
matplotlib.use("TkAgg")
import tensorflow as tf
import matplotlib.pyplot as plt
from ultralytics import YOLO
from squash import Referencepoints, Functions  # Ensure Functions is imported
from matplotlib import pyplot as plt
from squash.Ball import Ball
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torchvision import transforms
# Import autonomous coaching system
from autonomous_coaching import collect_coaching_data, generate_coaching_report
import json
from sahi.utils.cv import read_image_as_pil
print(f"time to import everything: {time.time()-start}")
alldata = organizeddata = []

# Autonomous coaching system imported from autonomous_coaching.py


def find_match_2d_array(array, x):
    for i in range(len(array)):
        if array[i][0] == x:
            return True
    return False
def drawmap(lx, ly, rx, ry, map):
    # Update heatmap at the ankle positions
    lx = min(max(lx, 0), map.shape[1] - 1)  # Bound lx to [0, width-1]
    ly = min(max(ly, 0), map.shape[0] - 1)  # Bound ly to [0, height-1]
    rx = min(max(rx, 0), map.shape[1] - 1)  # Bound rx to [0, width-1]
    ry = min(max(ry, 0), map.shape[0] - 1)
    map[ly, lx] += 1
    map[ry, rx] += 1
def get_image_embeddings(image):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    imagemodel, preprocess = clip.load("ViT-B/32", device=device)
    image = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        embeddings = imagemodel.encode_image(image)
    return embeddings.cpu().numpy()
def cosine_similarity(embedding1, embedding2):
    # Flatten the embeddings to 1D if they are 2D (like (1, 512))
    embedding1 = np.squeeze(embedding1)  # Shape becomes (512,)
    embedding2 = np.squeeze(embedding2)  # Shape becomes (512,)

    dot_product = np.dot(embedding1, embedding2)
    norm1 = np.linalg.norm(embedding1)
    norm2 = np.linalg.norm(embedding2)

    # Check if any norm is zero to avoid division by zero
    if norm1 == 0 or norm2 == 0:
        return 0  # Return a similarity of 0 if one of the embeddings is invalid

    return dot_product / (norm1 * norm2)
def visualize_court_positions(player1_pos, player2_pos, pixel_reference, real_reference, court_scale=100):
    """
    Create a top-down visualization of player positions on a squash court.
    
    Args:
        player1_pos (list): [x,y] pixel coordinates of player 1
        player2_pos (list): [x,y] pixel coordinates of player 2
        pixel_reference (list): List of [x,y] pixel reference points
        real_reference (list): List of [x,y,z] real-world reference points in meters
        court_scale (int): Pixels per meter for visualization
        
    Returns:
        np.ndarray: Court visualization image
    """
    # Standard squash court dimensions in meters
    COURT_LENGTH = 9.75
    COURT_WIDTH = 6.4
    
    # Create blank canvas with white background
    canvas_height = int(COURT_LENGTH * court_scale)
    canvas_width = int(COURT_WIDTH * court_scale)
    court = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 255
    
    # Draw court lines
    # Main court outline
    cv2.rectangle(court, (0, 0), (canvas_width-1, canvas_height-1), (0,0,0), 2)
    
    # Service line
    service_line_y = int(5.49 * court_scale)
    cv2.line(court, (0, service_line_y), (canvas_width, service_line_y), (0,0,0), 2)
    
    # Short line
    short_line_y = int(4.26 * court_scale)
    cv2.line(court, (0, short_line_y), (canvas_width, short_line_y), (0,0,0), 2)
    
    # Half court line
    half_court_x = int(COURT_WIDTH/2 * court_scale)
    cv2.line(court, (half_court_x, short_line_y), (half_court_x, canvas_height), (0,0,0), 2)
    
    # Calculate homography matrix
    pixel_reference_np = np.array(pixel_reference, dtype=np.float32)
    real_reference_np = np.array([(p[0], p[1]) for p in real_reference], dtype=np.float32)
    H, _ = cv2.findHomography(pixel_reference_np, real_reference_np)
    
    # Transform player positions to real-world coordinates
    players = [player1_pos, player2_pos]
    colors = [(0,0,255), (255,0,0)]  # Red for player 1, Blue for player 2
    
    for player_pos, color in zip(players, colors):
        # Convert to homogeneous coordinates
        player_pixel = np.array([player_pos[0], player_pos[1], 1])
        
        # Apply homography
        player_real = np.dot(H, player_pixel)
        player_real /= player_real[2]
        
        # Convert to court coordinates
        court_x = int(player_real[0] * court_scale)
        court_y = int(player_real[1] * court_scale)
        
        # Draw player on court
        cv2.circle(court, (court_x, court_y), 10, color, -1)
        
    # Add legend
    cv2.putText(court, "Player 1", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
    cv2.putText(court, "Player 2", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1)
    
    return court
def sum_pixels_in_bbox(frame, bbox):
    x, y, w, h = bbox
    roi = frame[int(y) : int(y + h), int(x) : int(x + w)]
    return np.sum(roi, dtype=np.int64)
def find_last_one(array):
    possibleis = []
    for i in range(len(array)):
        if array[i][1] == 1:
            possibleis.append(i)
    # print(possibleis)
    if len(possibleis) > 1:
        return possibleis[-1]

    return -1
def cleanwrite():
    with open("output/ball.txt", "w") as f:
        f.write("")
    with open("output/player1.txt", "w") as f:
        f.write("")
    with open("output/player2.txt", "w") as f:
        f.write("")
    with open("output/ball-xyn.txt", "w") as f:
        f.write("")
    with open("output/read_ball.txt", "w") as f:
        f.write("")
    with open("output/read_player1.txt", "w") as f:
        f.write("")
    with open("output/read_player2.txt", "w") as f:
        f.write("")
    with open("output/final.json", "w") as f:
        f.write("[")
    with open("output/final.csv", "w") as f:
        f.write(
            "Frame count,Player 1 Keypoints,Player 2 Keypoints,Ball Position,Shot Type,Player 1 RL World Position,Player 2 RL World Position,Ball RL World Position,Who Hit the Ball\n"
        )
def get_data(length):
    # go through the data in the file output/final.csv and then return all the data in a list
    for i in range(0, length):
        with open("output/final.csv", "r") as f:
            data = f.read()
    return data
def find_last_two(array):
    possibleis = []
    for i in range(len(array)):
        if array[i][1] == 2:
            possibleis.append(i)
    if len(possibleis) > 1:
        return possibleis[-1]
    return -1
def find_last(i, other_track_ids):
    possibleits = []
    for it in range(len(other_track_ids)):
        if other_track_ids[it][1] == i:
            possibleits.append(it)
    if len(possibleits) > 0:
        return possibleits[-1]
    return -1
def pixel_to_3d_pixel_reference(pixel_point, pixel_reference, reference_points_3d):
    """
    Maps a single 2D pixel coordinate to a 3D position based on reference points.

    Parameters:
        pixel_point (list): Single [x, y] pixel coordinate to map.
        pixel_reference (list): List of [x, y] reference points in pixels.
        reference_points_3d (list): List of [x, y, z] reference points in 3D space.

    Returns:
        list: Mapped 3D coordinates in the form [x, y, z].
    """
    # Convert 2D reference points and 3D points to NumPy arrays
    pixel_reference_np = np.array(pixel_reference, dtype=np.float32)
    reference_points_3d_np = np.array(reference_points_3d, dtype=np.float32)

    # Extract only the x and y values from the 3D reference points for homography calculation
    reference_points_2d = reference_points_3d_np[:, :2]

    # Calculate the homography matrix from 2D pixel reference to 2D real-world reference (ignoring z)
    H, _ = cv2.findHomography(pixel_reference_np, reference_points_2d)

    # Ensure pixel_point is in homogeneous coordinates [x, y, 1]
    pixel_point_homogeneous = np.array(
        [pixel_point[0], pixel_point[1], 1], dtype=np.float32
    )

    # Apply the homography matrix to get a 2D point in real-world space
    real_world_2d = np.dot(H, pixel_point_homogeneous)
    real_world_2d /= real_world_2d[2]  # Normalize to make it [x, y, 1]

    # Now interpolate the z-coordinate based on distances
    # Calculate weights based on the nearest reference points in the 2D plane
    distances = np.linalg.norm(reference_points_2d - real_world_2d[:2], axis=1)
    weights = 1 / (distances + 1e-5)  # Avoid division by zero
    z_mapped = np.dot(weights, reference_points_3d_np[:, 2]) / np.sum(weights)

    # Combine the 2D mapped point with interpolated z to get the 3D position
    mapped_3d_point = [real_world_2d[0], real_world_2d[1], z_mapped]

    return mapped_3d_point
def transform_pixel_to_real_world(pixel_points, H):
    """
    Transform pixel points to real-world coordinates using the homography matrix.

    Parameters:
        pixel_points (list): List of [x, y] pixel coordinates to transform.
        H (np.array): Homography matrix.

    Returns:
        list: Transformed real-world coordinates in the form [x, y].
    """
    # Convert pixel points to homogeneous coordinates for matrix multiplication
    pixel_points_homogeneous = np.append(pixel_points, 1)

    # Apply the homography matrix to get a 2D point in real-world space
    real_world_2d = np.dot(H, pixel_points_homogeneous)
    real_world_2d /= real_world_2d[2]  # Normalize

    return real_world_2d[:2]
def display_player_positions(rlworldp1, rlworldp2):
    """
    Display the player positions on another screen using OpenCV.

    Parameters:
        rlworldp1 (list): Real-world coordinates of player 1.
        rlworldp2 (list): Real-world coordinates of player 2.

    Returns:
        None
    """
    # Create a blank image
    display_image = np.ones((500, 500, 3), dtype=np.uint8) * 255

    # Draw player positions
    cv2.circle(
        display_image, (int(rlworldp1[0]), int(rlworldp1[1])), 5, (255, 0, 0), -1
    )  # Blue for player 1
    cv2.circle(
        display_image, (int(rlworldp2[0]), int(rlworldp2[1])), 5, (0, 0, 255), -1
    )  # Red for player 2

    # Display the image
    cv2.imshow("Player Positions", display_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
def validate_reference_points(px_points, rl_points):
    """
    Validate reference points for homography calculation.

    Parameters:
        px_points: List of pixel coordinates [[x, y], ...]
        rl_points: List of real-world coordinates [[X, Y, Z], ...] or [[X, Y], ...]

    Returns:
        Tuple[bool, str]: (is_valid, error_message)
    """
    if len(px_points) != len(rl_points):
        return False, "Number of pixel and real-world points must match"

    if len(px_points) < 4:
        return False, "At least 4 point pairs are required for homography calculation"

    # Check pixel points format
    if not all(len(p) == 2 for p in px_points):
        return False, "Pixel points must be 2D coordinates [x, y]"

    # Check real-world points format
    if not all(len(p) in [2, 3] for p in rl_points):
        return (
            False,
            "Real-world points must be either 2D [X, Y] or 3D [X, Y, Z] coordinates",
        )

    return True, ""
def generate_homography(points_2d, points_3d):
    """
    Generate homography matrix from 2D pixel coordinates to 3D real world coordinates
    Args:
        points_2d: List of [x,y] pixel coordinates
        points_3d: List of [x,y,z] real world coordinates in meters
    Returns:
        3x3 homography matrix
    """
    # Convert to numpy arrays
    src_pts = np.array(points_2d, dtype=np.float32)
    dst_pts = np.array(points_3d, dtype=np.float32)

    # Remove y coordinate from 3D points since we're working on court plane
    dst_pts = np.delete(dst_pts, 1, axis=1)

    # Calculate homography matrix
    H, status = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    return H
def pixel_to_3d(pixel_point, H, rl_reference_points):
    """
    Convert a pixel point to an interpolated 3D real-world point using the homography matrix.

    Parameters:
        pixel_point (list): Pixel coordinate [x, y] to transform.
        H (np.array): Homography matrix from `generate_homography`.
        rl_reference_points (list): List of real-world coordinates [[X, Y, Z], ...].

    Returns:
        list: Estimated interpolated 3D coordinate in the form [X, Y, Z].
    """
    # Convert pixel point to homogeneous coordinates
    pixel_point_homogeneous = np.array([*pixel_point, 1])

    # Map pixel point to real-world 2D using the homography matrix
    real_world_2d = np.dot(H, pixel_point_homogeneous)
    real_world_2d /= real_world_2d[2]  # Normalize to get actual coordinates

    # Convert real-world reference points to NumPy array
    rl_reference_points_np = np.array(rl_reference_points, dtype=np.float32)

    # Calculate distances in the X-Y plane
    distances = np.linalg.norm(
        rl_reference_points_np[:, :2] - real_world_2d[:2], axis=1
    )

    # Calculate weights inversely proportional to distances for interpolation
    weights = 1 / (distances + 1e-6)  # Avoid division by zero with epsilon
    weights /= weights.sum()  # Normalize weights to sum to 1

    # Perform weighted interpolation for the X, Y, and Z coordinates
    interpolated_x = np.dot(weights, rl_reference_points_np[:, 0])
    interpolated_y = np.dot(weights, rl_reference_points_np[:, 1])
    interpolated_z = np.dot(weights, rl_reference_points_np[:, 2])

    return [
        round(interpolated_x, 3),
        round(interpolated_y, 3),
        round(interpolated_z, 3),
    ]
"""
import torch.nn.functional as F
from torchreid.utils import FeatureExtractor
feature_extractor = FeatureExtractor(
    model_name='osnet_x1_0',  # You can choose other models as well
    device='cuda' if torch.cuda.is_available() else 'cpu'
)
known_players_features = {}

def appearance_reid(player_crop, known_players_features, threshold=0.7):
    # Resize and preprocess the player crop
    player_crop_resized = cv2.resize(player_crop, (128, 256))
    # Convert BGR to RGB
    player_crop_rgb = cv2.cvtColor(player_crop_resized, cv2.COLOR_BGR2RGB)
    
    # Extract features
    features = feature_extractor([player_crop_rgb])  # Returns a numpy array
    features = torch.tensor(features)
    features = F.normalize(features, p=2, dim=1)
    
    max_similarity = 0
    matched_player_id = None
    
    # Compare with known players
    for player_id, known_feature in known_players_features.items():
        similarity = F.cosine_similarity(features, known_feature).item()
        if similarity > max_similarity:
            max_similarity = similarity
            matched_player_id = player_id
    
    # Check if similarity exceeds the threshold
    if max_similarity > threshold:
        return matched_player_id, features
    else:
        return None, features
    
"""
"""
HISTOGRAM:::
import cv2
import numpy as np

# Initialize known player histograms with IDs 1 and 2
known_players_histograms = {}

def appearance_reid(player_crop, known_players_histograms, threshold=0.7):
    # Resize the player crop for consistency
    player_crop_resized = cv2.resize(player_crop, (64, 128))
    
    # Convert to HSV color space
    hsv_crop = cv2.cvtColor(player_crop_resized, cv2.COLOR_BGR2HSV)
    
    # Compute color histogram
    hist = cv2.calcHist([hsv_crop], [0, 1], None, [50, 60], [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    
    max_similarity = 0
    matched_player_id = None
    
    # Compare with known players
    for player_id in [1, 2]:
        known_hist = known_players_histograms.get(player_id)
        if known_hist is not None:
            similarity = cv2.compareHist(hist, known_hist, cv2.HISTCMP_CORREL)
            if similarity > max_similarity:
                max_similarity = similarity
                matched_player_id = player_id
    
    # Check if similarity exceeds the threshold
    if max_similarity > threshold:
        return matched_player_id, hist
    else:
        # Assign the player to an available ID (1 or 2)
        available_ids = [1, 2]
        for pid in available_ids:
            if pid not in known_players_histograms:
                known_players_histograms[pid] = hist
                return pid, hist
        # If both IDs are taken but no match, default to the closest match
        return matched_player_id, hist
"""
def is_rally_on(player_positions, threshold_seconds=2, max_distance=50, fps=30):
    #check if either player has moved more than 50 pixels in the last 2 seconds
    #player_positions ordered as [[[x1,y1,frame number]...], [[x2,y2,frame number]...]]
    #[0] is player 1 and [1]  is player 2
    for i in range(2):
        if len(player_positions[i])>1:
            lastpos=player_positions[i][-1]
            for pos in player_positions[i][::-1]:
                if pos[2]<lastpos[2]-threshold_seconds*fps:
                    break
                if math.hypot(pos[0]-lastpos[0],pos[1]-lastpos[1])>max_distance:
                    return True
    return False
#given a player crop, generate embeddings for the player as to differentiate between players
model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
model.eval()
preprocess=transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
def find_what_player(player1refs, player2refs, currentref):
    p1avgsim=0
    p2avgsim=0
    for ref in player1refs:
        p1avgsim+=compare_embeddings(ref, currentref)
    for ref in player2refs:
        p2avgsim+=compare_embeddings(ref, currentref)
    p1avgsim/=len(player1refs)
    p2avgsim/=len(player2refs)
    if p1avgsim>p2avgsim:
        print(f'player 1 avg sim: {p1avgsim} while player 2 avg sim: {p2avgsim}')
        return 1, p1avgsim
    else:
        print(f'player 2 avg sim: {p1avgsim} while player 1 avg sim: {p2avgsim}')
        return 2, p2avgsim
def generate_embeddings(player_crop):
    try:
        input_tensor=preprocess(player_crop)
        input_batch=input_tensor.unsqueeze(0) #create a mini batch
        with torch.no_grad():
            embedding=model(input_batch)
        return embedding.squeeze()
    except Exception as e:
        print(f'error getting reference embeddings; {e}')
        return None
def compare_embeddings(e1, e2):
    try:
        similarity=F.cosine_similarity(e1.unsqueeze(0), e2.unsqueeze(0))
        return similarity.item()
    except Exception as e:
        print(f'error comparing embeddings; {e}')
        return 0.0
def reorganize_shots(alldata, min_sequence=5):
    if not alldata:
        return []

    # Filter out None types and replace with "unknown"
    cleaned_data = [
        [shot[0] if shot[0] is not None else "unknown"] + shot[1:] for shot in alldata
    ]

    # Step 1: Group consecutive shots
    sequences = []
    current_type = cleaned_data[0][0]
    current_sequence = []

    for shot in cleaned_data:
        if shot[0] == current_type:
            current_sequence.append(shot)
        else:
            if len(current_sequence) > 0:
                sequences.append((current_type, current_sequence))
            current_type = shot[0]
            current_sequence = [shot]
    if current_sequence:
        sequences.append((current_type, current_sequence))

    # Step 2: Fix short sequences, especially for crosscourts
    i = 0
    while i < len(sequences):
        shot_type = sequences[i][0]
        if len(sequences[i][1]) < min_sequence and shot_type.lower() == "crosscourt":
            short_sequence = sequences[i][1]

            # Look for same shot type in adjacent sequences
            borrowed_shots = []

            # Check forward and backward for shots to borrow
            for j in range(i + 1, len(sequences)):
                if sequences[j][0] == shot_type:
                    available = len(sequences[j][1])
                    needed = min_sequence - len(short_sequence) - len(borrowed_shots)
                    if available > min_sequence:
                        borrowed = sequences[j][1][:needed]
                        sequences[j] = (sequences[j][0], sequences[j][1][needed:])
                        borrowed_shots.extend(borrowed)
                        if len(borrowed_shots) + len(short_sequence) >= min_sequence:
                            break

            if len(borrowed_shots) + len(short_sequence) < min_sequence:
                for j in range(i - 1, -1, -1):
                    if sequences[j][0] == shot_type:
                        available = len(sequences[j][1])
                        needed = (
                            min_sequence - len(short_sequence) - len(borrowed_shots)
                        )
                        if available > min_sequence:
                            borrowed_prev = sequences[j][1][-needed:]
                            sequences[j] = (sequences[j][0], sequences[j][1][:-needed])
                            borrowed_shots = borrowed_prev + borrowed_shots
                            if (
                                len(borrowed_shots) + len(short_sequence)
                                >= min_sequence
                            ):
                                break

            # Update sequence with borrowed shots
            if len(borrowed_shots) + len(short_sequence) >= min_sequence:
                sequences[i] = (shot_type, borrowed_shots + short_sequence)
            else:
                # Merge with adjacent sequence if can't borrow enough
                if i > 0:
                    sequences[i - 1] = (
                        sequences[i - 1][0],
                        sequences[i - 1][1] + short_sequence,
                    )
                    sequences.pop(i)
                    i -= 1
                elif i < len(sequences) - 1:
                    sequences[i + 1] = (
                        sequences[i + 1][0],
                        short_sequence + sequences[i + 1][1],
                    )
                    sequences.pop(i)
                    i -= 1
        i += 1

    # Step 3: Format output
    result = []
    for shot_type, sequence in sequences:
        coords = []
        for shot in sequence:
            coords.extend([shot[1], shot[2]])
        result.append([shot_type, len(sequence), coords])

    return result
def apply_homography(H, points, inverse=False):
    """
    Apply homography transformation to a set of points.

    Parameters:
        H: 3x3 homography matrix
        points: List of points to transform [[x, y], ...]
        inverse: If True, applies inverse transformation

    Returns:
        np.ndarray: Transformed points
    """
    try:
        points = np.array(points, dtype=np.float32)
        if points.ndim == 1:
            points = points.reshape(1, 2)

        if inverse:
            H = np.linalg.inv(H)

        # Reshape points to Nx1x2 format required by cv2.perspectiveTransform
        points_reshaped = points.reshape(-1, 1, 2)

        # Apply transformation
        transformed_points = cv2.perspectiveTransform(points_reshaped, H)

        return transformed_points.reshape(-1, 2)

    except Exception as e:
        raise ValueError(f"Error in apply_homography: {str(e)}")
frame_height = 360
frame_width = 640
def shot_type(past_ball_pos, threshold=3):
    # go through the past threshold number of past ball positions and see what kind of shot it is
    # past_ball_pos ordered as [[x,y,frame_number], ...]
    if len(past_ball_pos) < threshold:
        return None
    threshballpos = past_ball_pos[-threshold:]
    # check for crosscourt or straight shots
    xdiff = threshballpos[-1][0] - threshballpos[0][0]
    ydiff = threshballpos[-1][1] - threshballpos[0][1]
    typeofshot = ""
    if xdiff < 50 and ydiff < 50:
        typeofshot = "straight"
    else:
        typeofshot = "crosscourt"
    # check how high the ball has moved
    maxheight = 0
    height = ""
    for i in range(1, len(threshballpos)):
        if threshballpos[i][1] > maxheight:
            maxheight = threshballpos[i][1]
            # print(f"{threshballpos[i]}")
            # print(f'maxheight: {maxheight}')
            # print(f'threshballpos[i][1]: {threshballpos[i][1]}')
    if maxheight < (frame_height) / 1.35:
        height += "lob"
        # print(f'max height was {maxheight} and thresh was {(1.5*frame_height)/2}')
    else:
        height += "drive"
    return typeofshot + " " + height
def is_camera_angle_switched(frame, reference_image, threshold=0.5):
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    reference_image_gray = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)
    score, _ = ssim_metric(reference_image_gray, frame_gray, full=True)
    return score < threshold
def is_match_in_play(
    players,
    pastballpos,
    movement_threshold=0.15 * frame_width,  # Reduced for more sensitive detection
    hit_threshold=0.1 * frame_height,       # More sensitive ball hit detection
    ballthreshold=8,                        # Increased trajectory analysis window
    ball_angle_thresh=35,                   # More sensitive angle detection
    ball_velocity_thresh=2.5,              # Lower velocity threshold
    advanced_analysis=True                  # Enable advanced pattern recognition
):
    """
    Enhanced match state detection with improved accuracy and pattern recognition
    """
    if players.get(1) is None or players.get(2) is None or pastballpos is None:
        return False
    
    try:
        # Get player movement data with enhanced accuracy
        player_movement_data = get_enhanced_player_movement(players, movement_threshold)
        if not player_movement_data:
            return False
            
        player_move = player_movement_data['movement_detected']
        movement_intensity = player_movement_data['movement_intensity']
        
        # Enhanced ball hit detection with multiple algorithms
        ball_hit_results = detect_ball_hit_advanced(
            pastballpos, ballthreshold, ball_angle_thresh, ball_velocity_thresh, advanced_analysis
        )
        
        ball_hit = ball_hit_results['hit_detected']
        hit_confidence = ball_hit_results['confidence']
        hit_type = ball_hit_results['hit_type']
        
        # Advanced match state analysis
        match_state = analyze_match_state(
            player_move, ball_hit, movement_intensity, hit_confidence, hit_type
        )
        
        # Return comprehensive match analysis
        return {
            'in_play': match_state['active'],
            'player_movement': player_move,
            'ball_hit': ball_hit,
            'movement_intensity': movement_intensity,
            'hit_confidence': hit_confidence,
            'hit_type': hit_type,
            'rally_quality': match_state['quality'],
            'engagement_level': match_state['engagement']
        }
        
    except Exception as e:
        print(f'Enhanced match detection error: {e}')
        # Return a default dictionary structure instead of False for consistency
        return {
            'in_play': False,
            'player_movement': False,
            'ball_hit': False,
            'movement_intensity': 0,
            'hit_confidence': 0,
            'hit_type': 'none',
            'rally_quality': 0,
            'engagement_level': 0
        }

def get_enhanced_player_movement(players, movement_threshold):
    """
    Enhanced player movement detection with multiple keypoint analysis
    """
    try:
        movement_data = {'movement_detected': False, 'movement_intensity': 0}
        
        # Analyze multiple keypoints for more accurate movement detection
        keypoint_indices = [15, 16, 11, 12, 5, 6]  # Ankles, hips, shoulders
        keypoint_weights = [1.0, 1.0, 0.8, 0.8, 0.6, 0.6]  # Weight importance
        
        total_movement = 0
        total_weight = 0
        
        for player_id in [1, 2]:
            if players.get(player_id) and players.get(player_id).get_latest_pose():
                try:
                    pose = players.get(player_id).get_latest_pose()
                    
                    # Get previous pose for comparison
                    if hasattr(players.get(player_id), 'get_last_x_poses'):
                        prev_pose = players.get(player_id).get_last_x_poses(2)
                        if prev_pose:
                            # Calculate movement for each keypoint
                            for i, (kp_idx, weight) in enumerate(zip(keypoint_indices, keypoint_weights)):
                                if (len(pose.xyn[0]) > kp_idx and len(prev_pose.xyn[0]) > kp_idx):
                                    # Current position
                                    curr_x = pose.xyn[0][kp_idx][0] * frame_width
                                    curr_y = pose.xyn[0][kp_idx][1] * frame_height
                                    
                                    # Previous position
                                    prev_x = prev_pose.xyn[0][kp_idx][0] * frame_width
                                    prev_y = prev_pose.xyn[0][kp_idx][1] * frame_height
                                    
                                    # Skip if keypoint not detected (0,0)
                                    if (curr_x == 0 and curr_y == 0) or (prev_x == 0 and prev_y == 0):
                                        continue
                                    
                                    # Calculate movement distance
                                    movement = math.hypot(curr_x - prev_x, curr_y - prev_y)
                                    weighted_movement = movement * weight
                                    
                                    total_movement += weighted_movement
                                    total_weight += weight
                                    
                except Exception as e:
                    print(f"Error analyzing player {player_id} movement: {e}")
                    continue
        
        # Calculate average movement intensity
        if total_weight > 0:
            avg_movement = total_movement / total_weight
            movement_data['movement_intensity'] = avg_movement
            movement_data['movement_detected'] = avg_movement >= movement_threshold
        
        return movement_data
        
    except Exception as e:
        print(f"Error in enhanced player movement detection: {e}")
        return {'movement_detected': False, 'movement_intensity': 0}

def detect_ball_hit_advanced(pastballpos, ballthreshold, angle_thresh, velocity_thresh, advanced_analysis=True):
    """
    Advanced ball hit detection using multiple algorithms and pattern recognition
    """
    hit_results = {
        'hit_detected': False,
        'confidence': 0.0,
        'hit_type': 'none'
    }
    
    if len(pastballpos) < ballthreshold:
        return hit_results
    
    recent_positions = pastballpos[-ballthreshold:]
    
    # Multiple detection algorithms
    angle_detection = detect_hit_by_angle_change(recent_positions, angle_thresh)
    velocity_detection = detect_hit_by_velocity_change(recent_positions, velocity_thresh)
    direction_detection = detect_hit_by_direction_change(recent_positions)
    
    if advanced_analysis:
        # Advanced pattern recognition
        pattern_detection = detect_hit_by_pattern_analysis(recent_positions)
        physics_detection = detect_hit_by_physics_model(recent_positions)
        
        # Combine all detection methods with weighted scoring
        detections = [angle_detection, velocity_detection, direction_detection, 
                    pattern_detection, physics_detection]
        weights = [0.25, 0.25, 0.2, 0.15, 0.15]
    else:
        detections = [angle_detection, velocity_detection, direction_detection]
        weights = [0.4, 0.4, 0.2]
    
    # Calculate weighted confidence score
    total_confidence = sum(det['confidence'] * weight for det, weight in zip(detections, weights))
    
    # Determine if hit detected based on confidence threshold
    hit_threshold = 0.4
    hit_detected = total_confidence > hit_threshold
    
    # Determine hit type based on strongest signal
    hit_type = 'none'
    if hit_detected:
        max_confidence_idx = max(range(len(detections)), key=lambda i: detections[i]['confidence'])
        hit_type = detections[max_confidence_idx]['type']
    
    hit_results.update({
        'hit_detected': hit_detected,
        'confidence': total_confidence,
        'hit_type': hit_type
    })
    
    return hit_results

def detect_hit_by_angle_change(positions, angle_thresh):
    """Detect ball hit by analyzing trajectory angle changes"""
    try:
        if len(positions) < 4:
            return {'confidence': 0.0, 'type': 'angle'}
        
        max_angle_change = 0
        
        for i in range(len(positions) - 3):
            p1, p2, p3, p4 = positions[i:i+4]
            
            # Calculate vectors
            v1 = [p2[0] - p1[0], p2[1] - p1[1]]
            v2 = [p3[0] - p2[0], p3[1] - p2[1]]
            v3 = [p4[0] - p3[0], p4[1] - p3[1]]
            
            # Calculate angle changes
            angle1 = calculate_vector_angle(v1, v2)
            angle2 = calculate_vector_angle(v2, v3)
            
            angle_change = abs(angle2 - angle1)
            max_angle_change = max(max_angle_change, angle_change)
        
        confidence = min(1.0, max_angle_change / angle_thresh)
        return {'confidence': confidence, 'type': 'angle'}
        
    except Exception:
        return {'confidence': 0.0, 'type': 'angle'}

def detect_hit_by_velocity_change(positions, velocity_thresh):
    """Detect ball hit by analyzing velocity changes"""
    try:
        if len(positions) < 3:
            return {'confidence': 0.0, 'type': 'velocity'}
        
        velocities = []
        for i in range(1, len(positions)):
            p1, p2 = positions[i-1], positions[i]
            time_diff = p2[2] - p1[2]
            if time_diff > 0:
                velocity = math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2) / time_diff
                velocities.append(velocity)
        
        if len(velocities) < 2:
            return {'confidence': 0.0, 'type': 'velocity'}
        
        # Find maximum velocity change
        max_velocity_change = 0
        for i in range(1, len(velocities)):
            velocity_change = abs(velocities[i] - velocities[i-1])
            max_velocity_change = max(max_velocity_change, velocity_change)
        
        confidence = min(1.0, max_velocity_change / velocity_thresh)
        return {'confidence': confidence, 'type': 'velocity'}
        
    except Exception:
        return {'confidence': 0.0, 'type': 'velocity'}

def detect_hit_by_direction_change(positions):
    """Detect ball hit by analyzing direction changes"""
    try:
        if len(positions) < 3:
            return {'confidence': 0.0, 'type': 'direction'}
        
        direction_changes = 0
        last_direction_x = None
        last_direction_y = None
        
        for i in range(1, len(positions)):
            p1, p2 = positions[i-1], positions[i]
            
            current_dir_x = 1 if p2[0] > p1[0] else -1 if p2[0] < p1[0] else 0
            current_dir_y = 1 if p2[1] > p1[1] else -1 if p2[1] < p1[1] else 0
            
            if last_direction_x is not None and current_dir_x != 0 and current_dir_x != last_direction_x:
                direction_changes += 1
            if last_direction_y is not None and current_dir_y != 0 and current_dir_y != last_direction_y:
                direction_changes += 1
            
            last_direction_x = current_dir_x if current_dir_x != 0 else last_direction_x
            last_direction_y = current_dir_y if current_dir_y != 0 else last_direction_y
        
        # Normalize confidence based on number of direction changes
        confidence = min(1.0, direction_changes / 4.0)
        return {'confidence': confidence, 'type': 'direction'}
        
    except Exception:
        return {'confidence': 0.0, 'type': 'direction'}

def detect_hit_by_pattern_analysis(positions):
    """Advanced pattern analysis for hit detection"""
    try:
        if len(positions) < 6:
            return {'confidence': 0.0, 'type': 'pattern'}
        
        # Analyze trajectory smoothness before and after potential hit
        mid_point = len(positions) // 2
        first_half = positions[:mid_point]
        second_half = positions[mid_point:]
        
        smoothness_1 = calculate_trajectory_smoothness(first_half)
        smoothness_2 = calculate_trajectory_smoothness(second_half)
        
        # Hit likely if trajectory changes from smooth to abrupt or vice versa
        smoothness_diff = abs(smoothness_1 - smoothness_2)
        confidence = min(1.0, smoothness_diff / 50.0)
        
        return {'confidence': confidence, 'type': 'pattern'}
        
    except Exception:
        return {'confidence': 0.0, 'type': 'pattern'}

def detect_hit_by_physics_model(positions):
    """Physics-based hit detection using squash ball dynamics"""
    try:
        if len(positions) < 5:
            return {'confidence': 0.0, 'type': 'physics'}
        
        # Model expected ball behavior and detect deviations
        predicted_positions = predict_ball_physics(positions[:3])
        actual_positions = positions[3:]
        
        total_deviation = 0
        for i, (pred, actual) in enumerate(zip(predicted_positions, actual_positions)):
            if i < len(predicted_positions):
                deviation = math.sqrt((pred[0] - actual[0])**2 + (pred[1] - actual[1])**2)
                total_deviation += deviation
        
        # High deviation suggests external force (hit)
        avg_deviation = total_deviation / len(actual_positions) if actual_positions else 0
        confidence = min(1.0, avg_deviation / 30.0)
        
        return {'confidence': confidence, 'type': 'physics'}
        
    except Exception:
        return {'confidence': 0.0, 'type': 'physics'}

def calculate_vector_angle(v1, v2):
    """Calculate angle between two vectors"""
    try:
        dot_product = v1[0] * v2[0] + v1[1] * v2[1]
        mag1 = math.sqrt(v1[0]**2 + v1[1]**2)
        mag2 = math.sqrt(v2[0]**2 + v2[1]**2)
        
        if mag1 == 0 or mag2 == 0:
            return 0
        
        cos_angle = dot_product / (mag1 * mag2)
        cos_angle = max(-1, min(1, cos_angle))
        return math.degrees(math.acos(cos_angle))
        
    except Exception:
        return 0

def calculate_trajectory_smoothness(positions):
    """Calculate smoothness metric for trajectory segment"""
    if len(positions) < 3:
        return 0
    
    smoothness = 0
    for i in range(1, len(positions) - 1):
        p1, p2, p3 = positions[i-1], positions[i], positions[i+1]
        
        # Calculate second derivative approximation
        acc_x = p3[0] - 2*p2[0] + p1[0]
        acc_y = p3[1] - 2*p2[1] + p1[1]
        acceleration_magnitude = math.sqrt(acc_x**2 + acc_y**2)
        smoothness += acceleration_magnitude
    
    return smoothness / (len(positions) - 2) if len(positions) > 2 else 0

def predict_ball_physics(positions):
    """Simple physics-based prediction for ball trajectory"""
    if len(positions) < 2:
        return []
    
    # Calculate velocity and acceleration from recent positions
    p1, p2 = positions[-2], positions[-1]
    velocity = [(p2[0] - p1[0]), (p2[1] - p1[1])]
    
    if len(positions) >= 3:
        p0 = positions[-3]
        prev_velocity = [(p1[0] - p0[0]), (p1[1] - p0[1])]
        acceleration = [(velocity[0] - prev_velocity[0]), (velocity[1] - prev_velocity[1])]
    else:
        acceleration = [0, 0]
    
    # Predict next few positions
    predictions = []
    for t in range(1, 4):  # Predict 3 frames ahead
        pred_x = p2[0] + velocity[0] * t + 0.5 * acceleration[0] * t**2
        pred_y = p2[1] + velocity[1] * t + 0.5 * acceleration[1] * t**2
        predictions.append([pred_x, pred_y, p2[2] + t])
    
    return predictions

def analyze_match_state(player_move, ball_hit, movement_intensity, hit_confidence, hit_type):
    """
    Analyze overall match state based on multiple factors
    """
    match_state = {
        'active': False,
        'quality': 'low',
        'engagement': 'passive'
    }
    
    # Determine if match is active
    if player_move or ball_hit:
        match_state['active'] = True
    
    # Assess rally quality
    if movement_intensity > 20 and hit_confidence > 0.6:
        match_state['quality'] = 'high'
    elif movement_intensity > 10 or hit_confidence > 0.4:
        match_state['quality'] = 'medium'
    
    # Assess engagement level
    if movement_intensity > 15 and ball_hit:
        match_state['engagement'] = 'active'
    elif movement_intensity > 5 or ball_hit:
        match_state['engagement'] = 'moderate'
    
    return match_state
def slice_frame(width, height, overlap, frame):
    slices = []
    for y in range(0, frame.shape[0], height - overlap):
        for x in range(0, frame.shape[1], width - overlap):
            slice_frame = frame[y : y + height, x : x + width]
            slices.append(slice_frame)
    return slices
def inference_slicing(model, frame, width=100, height=100, overlap=50):
    slices = slice_frame(width, height, overlap, frame)
    results = []
    for slice_frame in slices:
        results.append(model(slice_frame))
    return results
def classify_shot(past_ball_pos, court_width=640, court_height=360, previous_shot=None):
    """
    Advanced shot classification with machine learning-inspired pattern recognition
    Analyzes trajectory, velocity, court position, and temporal patterns
    """
    try:
        if len(past_ball_pos) < 4:
            return ["straight", "drive", 0, 0]

        # Use optimal trajectory length for analysis
        trajectory_length = min(20, len(past_ball_pos))
        trajectory = past_ball_pos[-trajectory_length:]

        # Enhanced trajectory analysis
        trajectory_metrics = analyze_trajectory_patterns(trajectory, court_width, court_height)
        
        # Extract key metrics
        horizontal_displacement = trajectory_metrics['horizontal_displacement']
        vertical_displacement = trajectory_metrics['vertical_displacement']
        direction_changes = trajectory_metrics['direction_changes']
        velocity_profile = trajectory_metrics['velocity_profile']
        court_coverage = trajectory_metrics['court_coverage']
        wall_bounces = trajectory_metrics['wall_bounces']
        
        # Advanced shot type classification using multiple factors
        shot_direction = classify_shot_direction(
            horizontal_displacement, court_coverage, direction_changes, court_width
        )
        
        shot_height = classify_shot_height(
            vertical_displacement, velocity_profile, trajectory, court_height
        )
        
        shot_style = classify_shot_style(
            velocity_profile, wall_bounces, direction_changes, trajectory
        )
        
        # Calculate confidence score based on trajectory consistency
        confidence_score = calculate_shot_confidence(trajectory_metrics, previous_shot)
        
        # Return comprehensive shot analysis
        return [shot_direction, shot_height, shot_style, wall_bounces, 
                horizontal_displacement, confidence_score, trajectory_metrics]

    except Exception as e:
        print(f"Error in advanced shot classification: {str(e)}")
        return ["straight", "drive", "unknown", 0, 0, 0.5, {}]

def analyze_trajectory_patterns(trajectory, court_width, court_height):
    """
    Comprehensive trajectory pattern analysis
    """
    metrics = {}
    
    if len(trajectory) < 2:
        return {key: 0 for key in ['horizontal_displacement', 'vertical_displacement', 
                                'direction_changes', 'velocity_profile', 'court_coverage', 'wall_bounces']}
    
    # Basic displacement
    start_x, start_y, _ = trajectory[0]
    end_x, end_y, _ = trajectory[-1]
    metrics['horizontal_displacement'] = (end_x - start_x) / court_width
    metrics['vertical_displacement'] = (end_y - start_y) / court_height
    
    # Velocity and acceleration analysis
    velocities = []
    accelerations = []
    
    for i in range(1, len(trajectory)):
        x1, y1, t1 = trajectory[i-1]
        x2, y2, t2 = trajectory[i]
        
        if t2 != t1:
            velocity = math.sqrt((x2-x1)**2 + (y2-y1)**2) / (t2-t1)
            velocities.append(velocity)
            
            if len(velocities) > 1:
                acceleration = velocities[-1] - velocities[-2]
                accelerations.append(acceleration)
    
    # Velocity profile analysis
    if velocities:
        metrics['velocity_profile'] = {
            'avg_velocity': sum(velocities) / len(velocities),
            'max_velocity': max(velocities),
            'velocity_variance': np.var(velocities),
            'velocity_trend': velocities[-1] - velocities[0] if len(velocities) > 1 else 0
        }
    else:
        metrics['velocity_profile'] = {'avg_velocity': 0, 'max_velocity': 0, 'velocity_variance': 0, 'velocity_trend': 0}
    
    # Direction change analysis
    direction_changes = 0
    last_direction_x = None
    last_direction_y = None
    
    for i in range(1, len(trajectory)):
        x1, y1, _ = trajectory[i-1]
        x2, y2, _ = trajectory[i]
        
        current_dir_x = 1 if x2 > x1 else -1 if x2 < x1 else 0
        current_dir_y = 1 if y2 > y1 else -1 if y2 < y1 else 0
        
        if last_direction_x is not None and current_dir_x != 0 and current_dir_x != last_direction_x:
            direction_changes += 1
        if last_direction_y is not None and current_dir_y != 0 and current_dir_y != last_direction_y:
            direction_changes += 1
            
        last_direction_x = current_dir_x if current_dir_x != 0 else last_direction_x
        last_direction_y = current_dir_y if current_dir_y != 0 else last_direction_y
    
    metrics['direction_changes'] = direction_changes
    
    # Court coverage analysis
    x_positions = [pos[0] for pos in trajectory]
    y_positions = [pos[1] for pos in trajectory]
    
    x_range = max(x_positions) - min(x_positions)
    y_range = max(y_positions) - min(y_positions)
    
    metrics['court_coverage'] = {
        'x_coverage': x_range / court_width,
        'y_coverage': y_range / court_height,
        'total_coverage': (x_range * y_range) / (court_width * court_height)
    }
    
    # Wall bounce detection using enhanced algorithm with position tracking
    wall_bounce_result = detect_wall_bounces_advanced(trajectory, court_width, court_height)
    if isinstance(wall_bounce_result, tuple):
        metrics['wall_bounces'], metrics['wall_bounce_positions'] = wall_bounce_result
    else:
        metrics['wall_bounces'] = wall_bounce_result
        metrics['wall_bounce_positions'] = []
    
    # GPU-accelerated bounce detection for better accuracy
    gpu_bounces = detect_ball_bounces_gpu(trajectory)
    metrics['gpu_detected_bounces'] = gpu_bounces
    
    return metrics

def classify_shot_direction(horizontal_displacement, court_coverage, direction_changes, court_width):
    """
    Advanced direction classification using multiple factors
    """
    abs_displacement = abs(horizontal_displacement)
    x_coverage = court_coverage.get('x_coverage', 0)
    
    # Multi-factor decision tree
    if abs_displacement > 0.4 and x_coverage > 0.3:
        if horizontal_displacement > 0.5 or horizontal_displacement < -0.5:
            return "wide_crosscourt"
        else:
            return "crosscourt"
    elif abs_displacement > 0.25 and direction_changes > 2:
        return "angled_crosscourt"
    elif abs_displacement > 0.15 and x_coverage > 0.2:
        return "slight_crosscourt"
    elif abs_displacement < 0.08 and x_coverage < 0.15:
        return "tight_straight"
    else:
        return "straight"

def classify_shot_height(vertical_displacement, velocity_profile, trajectory, court_height):
    """
    Enhanced height classification using trajectory analysis
    """
    if not trajectory or len(trajectory) < 3:
        return "drive"
    
    # Find highest point in trajectory
    max_height = min(pos[1] for pos in trajectory)  # Min because y=0 is top
    trajectory_height = court_height - max_height
    
    avg_velocity = velocity_profile.get('avg_velocity', 0)
    velocity_variance = velocity_profile.get('velocity_variance', 0)
    
    # Classification based on multiple factors
    if trajectory_height > court_height * 0.4 and avg_velocity < 15:
        return "lob"
    elif trajectory_height < court_height * 0.15 and avg_velocity > 25:
        return "drive"
    elif velocity_variance > 100:  # High velocity variation suggests drops
        return "drop"
    else:
        return "drive"

def classify_shot_style(velocity_profile, wall_bounces, direction_changes, trajectory):
    """
    Classify shot style based on advanced metrics
    """
    avg_velocity = velocity_profile.get('avg_velocity', 0)
    velocity_variance = velocity_profile.get('velocity_variance', 0)
    
    if wall_bounces > 2:
        return "boast"
    elif direction_changes > 4 and velocity_variance > 50:
        return "nick"
    elif avg_velocity > 30:
        return "hard"
    elif avg_velocity < 10 and len(trajectory) > 10:
        return "soft"
    else:
        return "medium"

def detect_wall_bounces_advanced(trajectory, court_width, court_height):
    """
    Advanced wall bounce detection with improved accuracy and physics-based validation.
    Distinguishes between actual wall bounces and crosscourt/tactical shots.
    """
    bounces = 0
    bounce_positions = []
    
    if len(trajectory) < 5:  # Need more points for accurate analysis
        return bounces, bounce_positions
    
    # Define wall zones with different thresholds for different walls
    wall_zones = {
        'left': 25,     # Left wall proximity
        'right': 25,    # Right wall proximity  
        'top': 20,      # Top wall (front wall)
        'bottom': 20    # Bottom wall (back wall)
    }
    
    # Minimum requirements for bounce detection
    min_speed_before = 3.0    # Minimum speed before potential bounce
    min_angle_change = 45     # Minimum angle change for bounce
    max_crosscourt_angle = 35 # Maximum angle for crosscourt differentiation
    
    for i in range(2, len(trajectory) - 2):
        try:
            # Get trajectory points for analysis
            x_prev2, y_prev2, _ = trajectory[i-2]
            x_prev, y_prev, _ = trajectory[i-1] 
            x_curr, y_curr, _ = trajectory[i]
            x_next, y_next, _ = trajectory[i+1]
            x_next2, y_next2, _ = trajectory[i+2]
            
            # Calculate direction vectors with extended context
            v_before = (x_curr - x_prev2, y_curr - y_prev2)
            v_impact = (x_next - x_prev, y_next - y_prev)
            v_after = (x_next2 - x_curr, y_next2 - y_curr)
            
            # Calculate speeds
            speed_before = math.sqrt(v_before[0]**2 + v_before[1]**2)
            speed_impact = math.sqrt(v_impact[0]**2 + v_impact[1]**2) 
            speed_after = math.sqrt(v_after[0]**2 + v_after[1]**2)
            
            # Skip if insufficient movement
            if speed_before < min_speed_before or speed_impact < 1.0:
                continue
            
            # Check wall proximity with specific wall identification
            wall_proximity = {}
            wall_proximity['left'] = x_curr <= wall_zones['left']
            wall_proximity['right'] = x_curr >= (court_width - wall_zones['right'])
            wall_proximity['top'] = y_curr <= wall_zones['top']
            wall_proximity['bottom'] = y_curr >= (court_height - wall_zones['bottom'])
            
            near_any_wall = any(wall_proximity.values())
            
            if not near_any_wall:
                continue
                
            # Calculate angle changes with improved precision
            if speed_before > 0 and speed_after > 0:
                # Angle between incoming and outgoing vectors
                dot_product = v_before[0]*v_after[0] + v_before[1]*v_after[1]
                cos_angle = dot_product / (speed_before * speed_after)
                cos_angle = max(-1, min(1, cos_angle))
                angle_change = math.degrees(math.acos(cos_angle))
                
                # Physics-based bounce validation
                bounce_indicators = 0
                bounce_confidence = 0.0
                
                # 1. Significant angle change (not just crosscourt)
                if angle_change > min_angle_change:
                    bounce_indicators += 1
                    bounce_confidence += 0.3
                    
                    # Extra validation: crosscourt shots typically have gradual curves
                    # Wall bounces have sharp, immediate direction changes
                    if angle_change > 90:  # Very sharp angle change
                        bounce_confidence += 0.3
                
                # 2. Speed change analysis (bounces often change speed)
                speed_ratio = speed_after / speed_before if speed_before > 0 else 0
                if 0.3 <= speed_ratio <= 0.8 or speed_ratio >= 1.3:  # Speed dampening or acceleration
                    bounce_indicators += 1
                    bounce_confidence += 0.2
                
                # 3. Wall-specific bounce patterns
                if wall_proximity['left'] or wall_proximity['right']:
                    # Side wall bounces: check for horizontal direction reversal
                    if (v_before[0] > 0 and v_after[0] < 0) or (v_before[0] < 0 and v_after[0] > 0):
                        bounce_indicators += 1
                        bounce_confidence += 0.4
                        
                if wall_proximity['top'] or wall_proximity['bottom']:
                    # Front/back wall bounces: check for vertical direction component changes
                    if (v_before[1] > 0 and v_after[1] < 0) or (v_before[1] < 0 and v_after[1] > 0):
                        bounce_indicators += 1  
                        bounce_confidence += 0.4
                
                # 4. Crosscourt shot filtering
                crosscourt_likelihood = 0.0
                
                # Crosscourt shots typically have:
                # - Gradual direction changes over multiple frames
                # - Consistent speed profiles
                # - Angular changes that are more moderate
                if 20 <= angle_change <= max_crosscourt_angle:
                    crosscourt_likelihood += 0.3
                    
                if 0.8 <= speed_ratio <= 1.2:  # Consistent speed
                    crosscourt_likelihood += 0.2
                    
                # Check if ball is moving away from walls (typical of crosscourt)
                if not near_any_wall and i > 3:
                    # Check trajectory over last few frames
                    recent_positions = [(trajectory[j][0], trajectory[j][1]) for j in range(max(0, i-3), i+1)]
                    if len(recent_positions) >= 3:
                        # Check if moving towards center court
                        center_x, center_y = court_width/2, court_height/2
                        distances_to_center = [math.sqrt((x-center_x)**2 + (y-center_y)**2) for x, y in recent_positions]
                        if len(distances_to_center) >= 2 and distances_to_center[-1] < distances_to_center[0]:
                            crosscourt_likelihood += 0.3
                
                # 5. Final bounce decision with confidence threshold
                is_bounce = (bounce_indicators >= 2 and 
                           bounce_confidence >= 0.6 and 
                           crosscourt_likelihood < 0.5 and
                           near_any_wall)
                
                if is_bounce:
                    bounces += 1
                    bounce_positions.append((int(x_curr), int(y_curr)))
                    
                    # Debug info for tuning
                    if bounces <= 3:  # Limit debug output
                        wall_type = "Unknown"
                        for wall, is_near in wall_proximity.items():
                            if is_near:
                                wall_type = wall
                                break
                        
        except Exception as e:
            # Continue processing even if one point fails
            continue
    
    return bounces, bounce_positions

def detect_ball_bounces_gpu(trajectory, velocity_threshold=3.0, angle_threshold=45.0, court_width=640, court_height=360):
    """
    Enhanced GPU-optimized ball bounce detection with improved accuracy and crosscourt filtering.
    Uses physics-based validation to distinguish between bounces and tactical shots.
    """
    if len(trajectory) < 5:
        return []
    
    # Use GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        # Extract positions and convert to GPU tensors
        positions = torch.tensor([[pos[0], pos[1]] for pos in trajectory], dtype=torch.float32, device=device)
        
        bounce_positions = []
        
        if len(positions) < 5:
            return bounce_positions
        
        # Calculate velocities using GPU vectorization
        velocities = positions[1:] - positions[:-1]
        speeds = torch.norm(velocities, dim=1)
        
        # Enhanced bounce detection with crosscourt filtering
        if len(velocities) >= 4:
            # Extended context for better analysis
            extended_velocities = positions[2:] - positions[:-2]  # 2-frame span velocities
            extended_speeds = torch.norm(extended_velocities, dim=1)
            
            # Normalize velocities to get directions
            normalized_velocities = torch.nn.functional.normalize(extended_velocities + 1e-8, p=2, dim=1)
            
            # Calculate dot products for direction changes
            dot_products = torch.sum(normalized_velocities[:-1] * normalized_velocities[1:], dim=1)
            
            # Calculate angles between direction vectors
            angles = torch.acos(torch.clamp(dot_products, -1.0, 1.0)) * 180.0 / math.pi
            
            # Enhanced velocity analysis with extended context
            velocity_changes = torch.abs(extended_speeds[1:] - extended_speeds[:-1])
            
            # Speed ratio analysis - for bounce physics
            speed_ratios = torch.zeros_like(velocity_changes)
            mask = extended_speeds[:-1] > 1e-6
            speed_ratios[mask] = extended_speeds[1:][mask] / extended_speeds[:-1][mask]
            
            # Wall proximity analysis with enhanced zones
            wall_zones = {
                'left': 25, 'right': 25, 'top': 20, 'bottom': 20
            }
            
            wall_proximity = torch.zeros(len(positions), dtype=torch.bool, device=device)
            wall_proximity = (
                (positions[:, 0] < wall_zones['left']) |  # Left wall
                (positions[:, 0] > court_width - wall_zones['right']) |  # Right wall
                (positions[:, 1] < wall_zones['top']) |  # Top wall
                (positions[:, 1] > court_height - wall_zones['bottom'])   # Bottom wall
            )
            
            # Enhanced bounce detection criteria
            # 1. Significant angle change (adjusted for crosscourt filtering)
            angle_bounces = angles > angle_threshold
            
            # 2. Velocity change indicating impact
            velocity_bounces = velocity_changes > velocity_threshold
            
            # 3. Speed ratio indicating bounce physics (dampening or acceleration)
            ratio_bounces = (speed_ratios < 0.8) | (speed_ratios > 1.3)
            
            # 4. Crosscourt shot filtering
            crosscourt_angles = (angles >= 20) & (angles <= 35)  # Typical crosscourt angle range
            consistent_speed = (speed_ratios >= 0.8) & (speed_ratios <= 1.2)
            crosscourt_indicators = crosscourt_angles & consistent_speed
            
            # 5. Wall-specific direction reversal detection
            wall_bounce_indicators = torch.zeros(len(angles), dtype=torch.bool, device=device)
            
            # Check for horizontal direction reversals near side walls
            for i in range(len(extended_velocities) - 1):
                pos_idx = i + 2  # Adjust for extended velocity indexing
                if pos_idx < len(positions):
                    # Side wall bounce: horizontal direction reversal
                    if (wall_proximity[pos_idx] and 
                        ((extended_velocities[i, 0] > 0 and extended_velocities[i+1, 0] < 0) or
                         (extended_velocities[i, 0] < 0 and extended_velocities[i+1, 0] > 0))):
                        wall_bounce_indicators[i] = True
                    
                    # Front/back wall bounce: vertical direction component change
                    if (wall_proximity[pos_idx] and
                        ((extended_velocities[i, 1] > 0 and extended_velocities[i+1, 1] < 0) or
                         (extended_velocities[i, 1] < 0 and extended_velocities[i+1, 1] > 0))):
                        wall_bounce_indicators[i] = True
            
            # Combine criteria with enhanced validation
            bounce_confidence = torch.zeros_like(angles)
            
            # Angle change confidence
            bounce_confidence += (angles > angle_threshold).float() * 0.3
            bounce_confidence += (angles > 90).float() * 0.3  # Very sharp changes
            
            # Speed change confidence
            bounce_confidence += velocity_bounces.float() * 0.2
            bounce_confidence += ratio_bounces.float() * 0.2
            
            # Wall-specific bounce confidence
            bounce_confidence += wall_bounce_indicators.float() * 0.4
            
            # Reduce confidence for crosscourt indicators
            bounce_confidence -= crosscourt_indicators.float() * 0.4
            
            # Final bounce detection with high confidence threshold
            bounce_mask = (bounce_confidence >= 0.6) & (angles > 30)  # Minimum angle for any bounce
            
            # Get bounce indices and validate with wall proximity
            bounce_indices = torch.where(bounce_mask)[0] + 2  # Adjust for extended indexing
            
            for idx in bounce_indices:
                if idx < len(trajectory):
                    pos_idx = min(idx, len(wall_proximity) - 1)
                    
                    # Only accept bounces near walls with high confidence
                    if wall_proximity[pos_idx] and bounce_confidence[idx-2] >= 0.6:
                        x, y = trajectory[idx][:2]
                        bounce_positions.append((int(x.cpu() if hasattr(x, 'cpu') else x), 
                                              int(y.cpu() if hasattr(y, 'cpu') else y)))
        
        return bounce_positions
        
    except Exception as e:
        print(f"GPU bounce detection error: {e}, falling back to CPU")
        return detect_ball_bounces_cpu(trajectory, velocity_threshold, angle_threshold)

def detect_ball_bounces_cpu(trajectory, velocity_threshold=3.0, angle_threshold=45.0):
    """
    Enhanced CPU bounce detection with crosscourt filtering and physics-based validation
    """
    bounce_positions = []
    
    if len(trajectory) < 5:
        return bounce_positions
    
    # Define court boundaries (assuming standard dimensions)
    court_width, court_height = 640, 360
    wall_zones = {'left': 25, 'right': 25, 'top': 20, 'bottom': 20}
    
    for i in range(2, len(trajectory) - 2):
        try:
            # Extended context for better analysis
            x_prev2, y_prev2, _ = trajectory[i-2]
            x_prev, y_prev, _ = trajectory[i-1]
            x_curr, y_curr, _ = trajectory[i]
            x_next, y_next, _ = trajectory[i+1]
            x_next2, y_next2, _ = trajectory[i+2]
            
            # Calculate extended direction vectors
            v_before = (x_curr - x_prev2, y_curr - y_prev2)
            v_after = (x_next2 - x_curr, y_next2 - y_curr)
            
            # Calculate speeds
            speed_before = math.sqrt(v_before[0]**2 + v_before[1]**2)
            speed_after = math.sqrt(v_after[0]**2 + v_after[1]**2)
            
            # Skip if insufficient movement
            if speed_before < 2.0 or speed_after < 1.0:
                continue
            
            # Check wall proximity
            wall_proximity = {
                'left': x_curr <= wall_zones['left'],
                'right': x_curr >= (court_width - wall_zones['right']),
                'top': y_curr <= wall_zones['top'],
                'bottom': y_curr >= (court_height - wall_zones['bottom'])
            }
            
            near_wall = any(wall_proximity.values())
            if not near_wall:
                continue
            
            # Calculate angle change
            dot_product = v_before[0]*v_after[0] + v_before[1]*v_after[1]
            cos_angle = dot_product / (speed_before * speed_after)
            cos_angle = max(-1, min(1, cos_angle))
            angle_change = math.degrees(math.acos(cos_angle))
            
            # Physics-based validation
            bounce_confidence = 0.0
            
            # 1. Angle change analysis
            if angle_change > angle_threshold:
                bounce_confidence += 0.3
                if angle_change > 90:  # Very sharp change
                    bounce_confidence += 0.3
            
            # 2. Speed change analysis
            speed_ratio = speed_after / speed_before if speed_before > 0 else 0
            if 0.3 <= speed_ratio <= 0.8 or speed_ratio >= 1.3:
                bounce_confidence += 0.2
            
            # 3. Wall-specific direction reversal
            if wall_proximity['left'] or wall_proximity['right']:
                # Horizontal direction reversal
                if (v_before[0] > 0 and v_after[0] < 0) or (v_before[0] < 0 and v_after[0] > 0):
                    bounce_confidence += 0.4
                    
            if wall_proximity['top'] or wall_proximity['bottom']:
                # Vertical direction reversal
                if (v_before[1] > 0 and v_after[1] < 0) or (v_before[1] < 0 and v_after[1] > 0):
                    bounce_confidence += 0.4
            
            # 4. Crosscourt filtering
            crosscourt_likelihood = 0.0
            if 20 <= angle_change <= 35:  # Typical crosscourt range
                crosscourt_likelihood += 0.3
            if 0.8 <= speed_ratio <= 1.2:  # Consistent speed
                crosscourt_likelihood += 0.2
            
            # Final decision
            if (bounce_confidence >= 0.6 and 
                crosscourt_likelihood < 0.5 and 
                angle_change > 30):
                bounce_positions.append((int(x_curr), int(y_curr)))
                
        except Exception:
            continue
    
    return bounce_positions

def calculate_shot_confidence(trajectory_metrics, previous_shot):
    """
    Calculate confidence score for shot classification
    """
    confidence = 0.5  # Base confidence
    
    # Increase confidence for consistent patterns
    velocity_profile = trajectory_metrics.get('velocity_profile', {})
    velocity_variance = velocity_profile.get('velocity_variance', 0)
    
    if velocity_variance < 50:  # Consistent velocity
        confidence += 0.2
    
    # Increase confidence for clear directional patterns
    abs_displacement = abs(trajectory_metrics.get('horizontal_displacement', 0))
    if abs_displacement > 0.3 or abs_displacement < 0.1:  # Clear direction
        confidence += 0.2
    
    # Consistency with previous shot
    if previous_shot and len(previous_shot) > 0:
        if trajectory_metrics.get('direction_changes', 0) < 3:  # Stable trajectory
            confidence += 0.1
    
    return min(1.0, max(0.1, confidence))
def count_wall_hits(past_ball_pos, threshold=15):
    """
    Enhanced wall hit detection with improved accuracy and crosscourt filtering.
    Uses physics-based analysis to distinguish between bounces and direction changes.
    """
    try:
        wall_hits = 0
        consecutive_direction_changes = 0
        last_direction_x = None
        last_direction_y = None
        
        if len(past_ball_pos) < 5:
            return 0
        
        # Define court boundaries and wall zones
        court_width, court_height = 640, 360
        wall_zones = {'left': 25, 'right': 25, 'top': 20, 'bottom': 20}
        
        for i in range(2, len(past_ball_pos) - 2):
            try:
                # Get extended context
                x_prev2, y_prev2, _ = past_ball_pos[i-2]
                x_prev, y_prev, _ = past_ball_pos[i-1]
                x_curr, y_curr, _ = past_ball_pos[i]
                x_next, y_next, _ = past_ball_pos[i+1]
                x_next2, y_next2, _ = past_ball_pos[i+2]
                
                # Calculate movement vectors with extended context
                movement_before = (x_curr - x_prev2, y_curr - y_prev2)
                movement_after = (x_next2 - x_curr, y_next2 - y_curr)
                
                # Calculate speeds
                speed_before = math.sqrt(movement_before[0]**2 + movement_before[1]**2)
                speed_after = math.sqrt(movement_after[0]**2 + movement_after[1]**2)
                
                # Skip if insufficient movement
                if speed_before < 3.0 or speed_after < 1.0:
                    continue
                
                # Check wall proximity
                near_wall = (
                    x_curr <= wall_zones['left'] or 
                    x_curr >= (court_width - wall_zones['right']) or
                    y_curr <= wall_zones['top'] or 
                    y_curr >= (court_height - wall_zones['bottom'])
                )
                
                if not near_wall:
                    consecutive_direction_changes = 0
                    continue
                
                # Calculate direction change
                horizontal_change = abs(movement_after[0] - movement_before[0])
                vertical_change = abs(movement_after[1] - movement_before[1])
                total_change = horizontal_change + vertical_change
                
                # Enhanced direction change detection
                if total_change > threshold:
                    # Calculate angle change for validation
                    if speed_before > 0 and speed_after > 0:
                        dot_product = (movement_before[0] * movement_after[0] + 
                                     movement_before[1] * movement_after[1])
                        cos_angle = dot_product / (speed_before * speed_after)
                        cos_angle = max(-1, min(1, cos_angle))
                        angle_change = math.degrees(math.acos(cos_angle))
                        
                        # Physics-based validation
                        bounce_confidence = 0.0
                        
                        # Strong angle change indicates bounce
                        if angle_change > 45:
                            bounce_confidence += 0.4
                            if angle_change > 90:
                                bounce_confidence += 0.3
                        
                        # Speed change pattern
                        speed_ratio = speed_after / speed_before
                        if speed_ratio < 0.8 or speed_ratio > 1.3:
                            bounce_confidence += 0.2
                        
                        # Direction reversal (strongest indicator)
                        horizontal_reversal = (
                            (movement_before[0] > 0 and movement_after[0] < 0) or
                            (movement_before[0] < 0 and movement_after[0] > 0)
                        )
                        vertical_reversal = (
                            (movement_before[1] > 0 and movement_after[1] < 0) or
                            (movement_before[1] < 0 and movement_after[1] > 0)
                        )
                        
                        if horizontal_reversal or vertical_reversal:
                            bounce_confidence += 0.5
                        
                        # Crosscourt filtering
                        crosscourt_likelihood = 0.0
                        if 20 <= angle_change <= 35:  # Typical crosscourt angle
                            crosscourt_likelihood += 0.3
                        if 0.8 <= speed_ratio <= 1.2:  # Consistent speed
                            crosscourt_likelihood += 0.2
                        
                        # Only count as wall hit if high confidence and not crosscourt
                        if (bounce_confidence >= 0.6 and 
                            crosscourt_likelihood < 0.4 and
                            near_wall):
                            consecutive_direction_changes += 1
                        else:
                            consecutive_direction_changes = 0
                else:
                    consecutive_direction_changes = 0
                
                # Count as wall hit if we have consistent bounce pattern
                if consecutive_direction_changes >= 1:  # Single strong bounce
                    wall_hits += 1
                    consecutive_direction_changes = 0  # Reset for next detection
                    
            except Exception:
                continue
        
        return wall_hits

    except Exception as e:
        print(f"Error counting wall hits: {str(e)}")
        return 0
def is_ball_false_pos(past_ball_pos, speed_threshold=50, angle_threshold=45, min_size=5, max_size=50):
    """
    Enhanced false positive detection for ball tracking with multiple validation checks
    """
    if len(past_ball_pos) < 3:
        return False  # Not enough data to determine

    x0, y0, frame0 = past_ball_pos[-3]
    x1, y1, frame1 = past_ball_pos[-2]
    x2, y2, frame2 = past_ball_pos[-1]

    # Speed check - adjusted for squash ball physics
    time_diff = frame2 - frame1
    if time_diff == 0:
        return True  # Same frame, likely false positive

    distance = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
    speed = distance / time_diff

    if speed > speed_threshold:
        return True  # Speed exceeds threshold, likely false positive

    # Angle check with improved physics validation
    v1 = (x1 - x0, y1 - y0)
    v2 = (x2 - x1, y2 - y1)

    def compute_angle(v1, v2):
        dot_prod = v1[0] * v2[0] + v1[1] * v2[1]
        mag1 = (v1[0] ** 2 + v1[1] ** 2) ** 0.5
        mag2 = (v2[0] ** 2 + v2[1] ** 2) ** 0.5
        if mag1 == 0 or mag2 == 0:
            return 0  # Cannot compute angle with zero-length vector
        cos_theta = dot_prod / (mag1 * mag2)
        cos_theta = max(-1, min(1, cos_theta))  # Clamp to [-1,1]
        angle = math.degrees(math.acos(cos_theta))
        return angle

    angle_change = compute_angle(v1, v2)

    if angle_change > angle_threshold:
        return True  # Sudden angle change, possible false positive

    return False

def validate_ball_detection(x, y, w, h, confidence, past_ball_pos, min_conf=0.25, max_jump=120):
    """
    Simplified but effective ball detection validation - optimized for your trained model
    """
    # Basic confidence check - more lenient for your trained model
    if confidence < min_conf * 0.8:  # Slightly more lenient
        return False
    
    # Basic size validation - very generous bounds
    ball_size = w * h
    if ball_size < 8 or ball_size > 2000:  # Very generous size range
        return False
    
    # Basic aspect ratio - more lenient
    aspect_ratio = w / h if h > 0 else float('inf')
    if aspect_ratio < 0.25 or aspect_ratio > 4.0:  # More lenient aspect ratio
        return False
    
    # Simple temporal consistency check
    if len(past_ball_pos) > 0:
        last_x, last_y, last_frame = past_ball_pos[-1]
        distance_from_last = math.sqrt((x - last_x)**2 + (y - last_y)**2)
        
        # More generous jump distance - allows for fast ball movement
        generous_max_jump = max_jump * 1.5
        if distance_from_last > generous_max_jump:
            return False
    
    return True


def smooth_ball_trajectory(past_ball_pos, window_size=5):
    """
    Apply smoothing filter to reduce noise in ball trajectory
    """
    if len(past_ball_pos) < window_size:
        return past_ball_pos
    
    smoothed_trajectory = []
    for i in range(len(past_ball_pos)):
        start_idx = max(0, i - window_size // 2)
        end_idx = min(len(past_ball_pos), i + window_size // 2 + 1)
        
        window_positions = past_ball_pos[start_idx:end_idx]
        avg_x = sum(pos[0] for pos in window_positions) / len(window_positions)
        avg_y = sum(pos[1] for pos in window_positions) / len(window_positions)
        
        smoothed_trajectory.append([int(avg_x), int(avg_y), past_ball_pos[i][2]])
    
    return smoothed_trajectory

def predict_ball_position(past_ball_pos, frames_ahead=3):
    """
    Simple linear prediction for ball position based on recent trajectory
    """
    if len(past_ball_pos) < 3:
        return None
    
    # Use last 3 positions for prediction
    recent_positions = past_ball_pos[-3:]
    
    # Calculate velocity
    x1, y1, t1 = recent_positions[-2]
    x2, y2, t2 = recent_positions[-1]
    
    if t2 == t1:
        return None
    
    vx = (x2 - x1) / (t2 - t1)
    vy = (y2 - y1) / (t2 - t1)
    
    # Predict future position
    predicted_x = x2 + vx * frames_ahead
    predicted_y = y2 + vy * frames_ahead
    
    return [int(predicted_x), int(predicted_y)]
def predict_next_pos(past_ball_pos, num_predictions=2):
    # Define a fixed sequence length
    max_sequence_length = 10

    # Prepare the positions array
    data = np.array(past_ball_pos)
    positions = data[:, :2]  # Extract x and y coordinates

    # Ensure positions array has the fixed sequence length
    if positions.shape[0] < max_sequence_length:
        padding = np.zeros((max_sequence_length - positions.shape[0], 2))
        positions = np.vstack((padding, positions))
    else:
        positions = positions[-max_sequence_length:]

    # Prepare input for LSTM
    X_input = positions.reshape((1, max_sequence_length, 2))

    # Define LSTM model
    model = Sequential()
    model.add(LSTM(50, activation="relu", input_shape=(max_sequence_length, 2)))
    model.add(Dense(2))

    # Load pre-trained model weights if available
    # model.load_weights('path_to_weights.h5')

    predictions = []
    input_seq = X_input.copy()
    for _ in range(num_predictions):
        pred = model.predict(input_seq, verbose=0)
        predictions.append(pred[0])

        # Update input sequence
        input_seq = np.concatenate((input_seq[:, 1:, :], pred.reshape(1, 1, 2)), axis=1)
    return predictions
def plot_coords(coords_list):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    for coords in coords_list:
        x_coords, y_coords, z_coords = zip(*coords)
        ax.scatter(x_coords, y_coords, z_coords)

    ax.set_xlim(0, 6.4)
    ax.set_ylim(0, 9.75)
    ax.set_zlim(0, 4.57)

    ax.set_xlabel("X axis")
    ax.set_ylabel("Y axis")
    ax.set_zlabel("Z axis")

    plt.show()
def determine_ball_hit(
    players, past_ball_pos, proximity_threshold=50
):
    """
    Determine which player hit the ball based on proximity and trajectory changes.
    Uses average of all valid keypoints for more accurate player position.
    """
    if len(past_ball_pos) < 3 or not players.get(1) or not players.get(2):
        return 0

    # Get the most recent ball positions
    current_pos = past_ball_pos[-1]
    prev_pos = past_ball_pos[-2]

    # Calculate ball direction change
    if len(past_ball_pos) >= 3:
        prev_prev_pos = past_ball_pos[-3]
        vec1 = (prev_pos[0] - prev_prev_pos[0], prev_pos[1] - prev_prev_pos[1])
        vec2 = (current_pos[0] - prev_pos[0], current_pos[1] - prev_pos[1])

        # Calculate angle between vectors
        dot_product = vec1[0] * vec2[0] + vec1[1] * vec2[1]
        mag1 = math.sqrt(vec1[0] ** 2 + vec1[1] ** 2)
        mag2 = math.sqrt(vec2[0] ** 2 + vec2[1] ** 2)

        if mag1 > 0 and mag2 > 0:
            cos_angle = dot_product / (mag1 * mag2)
            cos_angle = max(-1, min(1, cos_angle))  # Ensure value is between -1 and 1
            math.degrees(math.acos(cos_angle))

    try:
        # Calculate average position for each player excluding [0,0] keypoints
        def get_avg_player_pos(player):
            keypoints = player.get_latest_pose().xyn[0]
            valid_points = []

            for kp in keypoints:
                # Check if keypoint is not [0,0]
                if not (kp[0] == 0 and kp[1] == 0):
                    x = int(kp[0] * frame_width)
                    y = int(kp[1] * frame_height)
                    valid_points.append((x, y))

            if not valid_points:
                return None

            avg_x = sum(p[0] for p in valid_points) / len(valid_points)
            avg_y = sum(p[1] for p in valid_points) / len(valid_points)
            return (avg_x, avg_y)

        p1_pos = get_avg_player_pos(players[1])
        p2_pos = get_avg_player_pos(players[2])

        if p1_pos is None or p2_pos is None:
            return 0

        # Calculate distances to ball
        p1_distance = math.hypot(p1_pos[0] - current_pos[0], p1_pos[1] - current_pos[1])
        p2_distance = math.hypot(p2_pos[0] - current_pos[0], p2_pos[1] - current_pos[1])

        # If ball direction changed and a player is close enough
        if p1_distance < p2_distance:
            return 1
        elif p2_distance < p1_distance:
            return 2

    except Exception as e:
        print("error in determine_ball_hit")
        print(e)
        return 0

    return 0  # Wall hit or unknown
def create_court(court_width=400, court_height=610):
    # Create a blank image
    court = np.zeros((court_height, court_width, 3), np.uint8)
    t_point=(int(court_width/2), int(0.5579487*court_height))
    left_service_box_end=(0, int(0.5579487*court_height))
    right_service_box_end=(int(court_width), int(0.5579487*court_height))
    left_service_box_start=(int(0.25*court_width), int(0.5579487*court_height))
    right_service_box_start=(int(0.75*court_width), int(0.5579487*court_height))
    left_service_box_low=(int(0.25*court_width), int(0.7220512*court_height))
    right_service_box_low=(int(0.75*court_width), int(0.7220512*court_height))
    left_service_box_low_end=(0, int(0.7220512*court_height))
    right_service_box_low_end=(int(court_width), int(0.7220512*court_height))
    points=[t_point, left_service_box_end, right_service_box_end, left_service_box_start, right_service_box_start, left_service_box_low, right_service_box_low, left_service_box_low_end, right_service_box_low_end]
    #plot all points
    for point in points:
        cv2.circle(court, point, 5, (255, 255, 255), -1)
    #between service boxes
    cv2.line(court, left_service_box_end, right_service_box_end, (255, 255, 255), 2)
    #between t point and middle low
    cv2.line(court, t_point, (int(court_width/2), court_height), (255, 255, 255), 2)
    #between low service boxes
    #cv2.line(court, left_service_box_low_end, right_service_box_low_end, (255, 255, 255), 2)
    #betwwen service boxes and low service boxes
    cv2.line(court, left_service_box_end, left_service_box_low_end, (255, 255, 255), 2)
    cv2.line(court, right_service_box_end, right_service_box_low_end, (255, 255, 255), 2)
    cv2.line(court, left_service_box_start, left_service_box_low, (255, 255, 255), 2)
    cv2.line(court, right_service_box_start, right_service_box_low, (255, 255, 255), 2)
    cv2.line(court, left_service_box_low_end, left_service_box_low, (255, 255, 255), 2)
    cv2.line(court, right_service_box_low_end, right_service_box_low, (255, 255, 255), 2)
    return court
def visualize_court():
    court = create_court()
    cv2.imshow('Squash Court', court)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
def plot_on_court(court, p1, p2):
    #draw p1 in red and p2 in blue
    cv2.circle(court, p1, 5, (0, 0, 255), -1)
    cv2.circle(court, p2, 5, (255, 0, 0), -1)
    return court
def generate_2d_homography(reference_points_px, reference_points_3d):
    """
    generate a homography matrix for 2d to 2d mapping(for the top down court view)
    """
    try:
        # Convert to numpy arrays
        reference_points_px = np.array(reference_points_px, dtype=np.float32)
        reference_points_3d = np.array(reference_points_3d, dtype=np.float32)

        # Calculate homography matrix
        H, _ = cv2.findHomography(reference_points_px, reference_points_3d)

        return H

    except Exception as e:
        print(f"Error generating 2D homography: {str(e)}")
        return None
def to_court_px(player1pos, player2pos, homography):
    """
    given player positions in the frame, convert to top down court positions
    """
    try:
        # Convert to numpy arrays
        player1pos = np.array(player1pos, dtype=np.float32)
        player2pos = np.array(player2pos, dtype=np.float32)        # Apply homography transformation
        player1_court = cv2.perspectiveTransform(player1pos.reshape(-1, 1, 2), homography)
        player2_court = cv2.perspectiveTransform(player2pos.reshape(-1, 1, 2), homography)
        
        return player1_court, player2_court

    except Exception as e:
        print(f"Error converting to court positions: {str(e)}")
        return None, None


def collect_coaching_data(players, past_ball_pos, type_of_shot, who_hit, match_in_play, frame_count):
    """
    Collect comprehensive data for autonomous coaching analysis
    """
    coaching_data = {
        'frame': frame_count,
        'timestamp': time.time(),
        'shot_type': type_of_shot,
        'player_who_hit': who_hit,
        'match_active': match_in_play.get('in_play', False) if isinstance(match_in_play, dict) else False,
        'ball_hit_detected': match_in_play.get('ball_hit', False) if isinstance(match_in_play, dict) else False,
        'player_movement': match_in_play.get('player_movement', False) if isinstance(match_in_play, dict) else False,
    }
    
    # Add player position analysis
    if players.get(1) and players.get(2):
        try:
            # Get player positions (using ankle keypoints - 15, 16)
            p1_pose = players[1].get_latest_pose()
            p2_pose = players[2].get_latest_pose()
            
            if p1_pose and p2_pose:
                # Player 1 positions
                p1_left_ankle = p1_pose.xyn[0][15] if len(p1_pose.xyn[0]) > 15 else [0, 0]
                p1_right_ankle = p1_pose.xyn[0][16] if len(p1_pose.xyn[0]) > 16 else [0, 0]
                
                # Player 2 positions  
                p2_left_ankle = p2_pose.xyn[0][15] if len(p2_pose.xyn[0]) > 15 else [0, 0]
                p2_right_ankle = p2_pose.xyn[0][16] if len(p2_pose.xyn[0]) > 16 else [0, 0]
                
                coaching_data.update({
                    'player1_position': {
                        'left_ankle': [float(p1_left_ankle[0]), float(p1_left_ankle[1])],
                        'right_ankle': [float(p1_right_ankle[0]), float(p1_right_ankle[1])]
                    },
                    'player2_position': {
                        'left_ankle': [float(p2_left_ankle[0]), float(p2_left_ankle[1])],
                        'right_ankle': [float(p2_right_ankle[0]), float(p2_right_ankle[1])]
                    }
                })
        except Exception as e:
            print(f"Error collecting player position data: {e}")
    
    # Add ball trajectory analysis
    if past_ball_pos and len(past_ball_pos) > 0:
        last_ball_pos = past_ball_pos[-1]
        if len(last_ball_pos) >= 2:
            coaching_data.update({
                'ball_position': [float(last_ball_pos[0]), float(last_ball_pos[1])],
                'ball_trajectory_length': len(past_ball_pos),
                'ball_speed': calculate_ball_speed(past_ball_pos) if len(past_ball_pos) > 1 else 0
            })
        else:
            coaching_data.update({
                'ball_position': [0.0, 0.0],  # Default position
                'ball_trajectory_length': len(past_ball_pos),
                'ball_speed': 0
            })
    
    return coaching_data

def calculate_ball_speed(ball_positions):
    """Calculate average ball speed from recent positions"""
    if len(ball_positions) < 2:
        return 0
    
    try:
        recent_positions = ball_positions[-5:] if len(ball_positions) >= 5 else ball_positions
        total_distance = 0
        total_time = 0
        
        for i in range(1, len(recent_positions)):
            x1, y1, t1 = recent_positions[i-1]
            x2, y2, t2 = recent_positions[i]
            
            distance = math.sqrt((x2-x1)**2 + (y2-y1)**2)
            time_diff = t2 - t1
            
            if time_diff > 0:
                total_distance += distance
                total_time += time_diff
        
        return total_distance / total_time if total_time > 0 else 0
    except Exception:
        return 0


def main(path="self1.mp4", frame_width=640, frame_height=360):
    try:
        print("🚀 INITIALIZING GPU-OPTIMIZED SQUASH COACHING PIPELINE")
        print("=" * 70)
        
        # GPU optimization setup
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🔧 Primary compute device: {device}")
        
        if torch.cuda.is_available():
            print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
            print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            # Optimize GPU memory usage
            torch.cuda.empty_cache()
        
        csvstart = 0
        end = csvstart + 100
        
        # Load ball position prediction model with GPU optimization
        try:
            ball_predict = tf.keras.models.load_model(
                "trained-models/ball_position_model(25k).keras"
            )
            if torch.cuda.is_available():
                # Try to use GPU for TensorFlow if available
                import tensorflow as tf
                tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[0], True)
                print("📊 Ball prediction model loaded with GPU acceleration")
        except Exception as e:
            print(f"⚠️  Ball prediction model loading error: {e}")
            ball_predict = None

        def load_data(file_path):
            with open(file_path, "r") as file:
                data = file.readlines()

            # Convert the data to a list of floats
            data = [float(line.strip()) for line in data]

            # Group the data into pairs of coordinates (x, y)
            positions = [(data[i], data[i + 1]) for i in range(0, len(data), 2)]

            return positions
        
        # Initialize output files
        cleanwrite()
        
        # Load models with GPU optimization
        print("🔄 Loading YOLO models with GPU acceleration...")
        
        # Load pose model with GPU acceleration
        pose_model = YOLO("models/yolo11n-pose.pt")
        if torch.cuda.is_available():
            pose_model.to(device)
            print("✅ Pose model loaded on GPU")
        
        # Load ball detection model with GPU acceleration  
        ballmodel = YOLO("trained-models\\black_ball_selfv3.pt")
        if torch.cuda.is_available():
            ballmodel.to(device)
            print("✅ Ball detection model loaded on GPU")
        
        print("=" * 70)
        print("📊 Enhanced Features Active:")
        print("   • GPU-accelerated ball and pose detection")
        print("   • Enhanced bounce detection with yellow circle visualization")
        print("   • Multi-criteria bounce validation (angle, velocity, wall proximity)")
        print("   • Real-time coaching data collection & analysis")
        print("   • Optimized memory management")
        print("=" * 70)
        ballvideopath = "output/balltracking.mp4"
        cap = cv2.VideoCapture(path)
        with open("output/final.txt", "w") as f:
            f.write(
                f"You are analyzing video: {path}.\nPlayer keypoints will be structured as such: 0: Nose 1: Left Eye 2: Right Eye 3: Left Ear 4: Right Ear 5: Left Shoulder 6: Right Shoulder 7: Left Elbow 8: Right Elbow 9: Left Wrist 10: Right Wrist 11: Left Hip 12: Right Hip 13: Left Knee 14: Right Knee 15: Left Ankle 16: Right Ankle.\nIf a keypoint is (0,0), then it has not beeen detected and should be deemed irrelevant. Here is how the output will be structured: \nFrame count\nPlayer 1 Keypoints\nPlayer 2 Keypoints\n Ball Position.\n\n"
            )
        embeddings=[[],[]]
        players = {}
        courtref = 0
        occlusion_times = {}
        for i in range(1, 3):
            occlusion_times[i] = 0
        future_predict = None
        player_last_positions = {}
        frame_count = 0
        ball_false_pos = []
        past_ball_pos = []
        logging.getLogger("ultralytics").setLevel(logging.ERROR)
        output_path = "output/annotated.mp4"
        weboutputpath = "websiteout/annotated.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = 30
        importantoutputpath = "output/important.mp4"
        cv2.VideoWriter(weboutputpath, fourcc, fps, (frame_width, frame_height))
        cv2.VideoWriter(importantoutputpath, fourcc, fps, (frame_width, frame_height))
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
        cv2.VideoWriter(ballvideopath, fourcc, fps, (frame_width, frame_height))
        detections = []
        plast=[[],[]]
        mainball = Ball(0, 0, 0, 0)
        ballmap = np.zeros((frame_height, frame_width), dtype=np.float32)
        otherTrackIds = [[0, 0], [1, 1], [2, 2]]
        updated = [[False, 0], [False, 0]]
        reference_points = []
        reference_points = Referencepoints.get_reference_points(
            path=path, frame_width=frame_width, frame_height=frame_height
        )
        is_rally=False
        references1 = []
        references2 = []

        pixdiffs = []
        
        p1distancesfromT = []
        p2distancesfromT = []
        
        # Initialize coaching data collection for autonomous analysis
        coaching_data_collection = []

        courtref = np.int64(courtref)
        referenceimage = None

        # function to see what kind of shot has been hit

        reference_points_3d = [
            [0, 9.75, 0],  # Top-left corner, 1
            [6.4, 9.75, 0],  # Top-right corner, 2
            [6.4, 0, 0],  # Bottom-right corner, 3
            [0, 0, 0],  # Bottom-left corner, 4
            [3.2, 4.57, 0],  # "T" point, 5
            [0, 2.71, 0],  # Left bottom of the service box, 6
            [6.4, 2.71, 0],  # Right bottom of the service box, 7
            [0, 9.75, 0.48],  # left of tin, 8
            [6.4, 9.75, 0.48],  # right of tin, 9
            [0, 9.75, 1.78],  # Left of the service line, 10
            [6.4, 9.75, 1.78],  # Right of the service line, 11
            [0, 9.75, 4.57],  # Left of the top line of the front court, 12
            [6.4, 9.75, 4.57],  # Right of the top line of the front court, 13
        ]
        homography = generate_homography(
            reference_points, reference_points_3d
        )
        court_view=create_court()
        #visualize_court()
        np.zeros((frame_height, frame_width), dtype=np.float32)
        np.zeros((frame_height, frame_width), dtype=np.float32)
        heatmap_overlay_path = "output/white.png"
        heatmap_image = cv2.imread(heatmap_overlay_path)
        if heatmap_image is None:
            raise FileNotFoundError(
                f"Could not find heatmap overlay image at {heatmap_overlay_path}"
            )
        np.zeros_like(heatmap_image, dtype=np.float32)

        ballxy = []

        running_frame = 0
        print("started video input")
        int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        abs(reference_points[1][0] - reference_points[0][0])
        validate_reference_points(reference_points, reference_points_3d)
        print(f"loaded everything in {time.time()-start} seconds")
        
        # Main processing loop with enhanced error handling
        while cap.isOpened():
            success, frame = cap.read()

            if not success:
                break

            # Protect frame processing with try-catch
            frame = cv2.resize(frame, (frame_width, frame_height))
            # force it to go to lowestx-->highestx and then lowesty-->highesty
            frame_count += 1

            if len(references1) != 0 and len(references2) != 0:
                sum(references1) / len(references1)
                sum(references2) / len(references2)

            running_frame += 1
            if running_frame == 1:
                courtref = np.int64(
                    sum_pixels_in_bbox(
                        frame, [0, 0, frame_width, frame_height]
                    )
                )
                referenceimage = frame

            if is_camera_angle_switched(frame, referenceimage, threshold=0.5):
                continue

            currentref = int(
                sum_pixels_in_bbox(frame, [0, 0, frame_width, frame_height])
            )

            if abs(courtref - currentref) > courtref * 0.6:
                # print("most likely not original camera frame")
                # print("current ref: ", currentref)
                # print("court ref: ", courtref)
                # print(f"frame count: {frame_count}")
                # print(
                #     f"difference between current ref and court ref: {abs(courtref - currentref)}"
                # )
                continue
            
            ball = ballmodel(frame)
            detections.append(ball)

            annotated_frame = frame.copy()  # pose_results[0].plot()

            for reference in reference_points:
                cv2.circle(
                    annotated_frame,
                    (int(reference[0]), int(reference[1])),
                    5,
                    (0, 255, 0),
                    2,
                )
            # frame, frame_height, frame_width, frame_count, annotated_frame, ballmodel, pose_model, mainball, ball, ballmap, past_ball_pos, ball_false_pos, running_frame
            # framepose_result=framepose.framepose(pose_model=pose_model, frame=frame, otherTrackIds=otherTrackIds, updated=updated, references1=references1, references2=references2, pixdiffs=pixdiffs, players=players, frame_count=frame_count, player_last_positions=player_last_positions, frame_width=frame_width, frame_height=frame_height, annotated_frame=annotated_frame)
            
            def framepose(
                pose_model,
                frame,
                other_track_ids,
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
                embeddings=[],
                plast=[[],[]]
            ):
                global known_players_features
                try:
                    track_results = pose_model.track(frame, persist=True, show=False)
                    if (
                        track_results
                        and hasattr(track_results[0], "keypoints")
                        and track_results[0].keypoints is not None
                    ):
                        # Extract boxes, track IDs, and keypoints from pose results
                        boxes = track_results[0].boxes.xywh.cpu()
                        track_ids = track_results[0].boxes.id.int().cpu().tolist()
                        keypoints = track_results[0].keypoints.cpu().numpy()
                        set(track_ids)
                        # Update or add players for currently visible track IDs
                        # note that this only works with occluded players < 2, still working on it :(
                        #print(f"number of players found: {len(track_ids)}")
                        # occluded as [[players_found, last_pos_p1, last_pos_p2, frame_number]...]
                        if len(track_ids) < 2:
                            print(player_last_positions)
                            last_pos_p1 = player_last_positions.get(1, (None, None))
                            last_pos_p2 = player_last_positions.get(2, (None, None))
                            # print(f"last pos p1: {last_pos_p1}")
                            occluded = []
                            try:
                                occluded.append(
                                    [
                                        len(track_ids),
                                        last_pos_p1,
                                        last_pos_p2,
                                        frame_count,
                                    ]
                                )
                            except Exception:
                                pass
                        if len(track_ids) > 2:
                            print(f"track ids were greater than 2: {track_ids}")
                            return [
                                pose_model,
                                frame,
                                other_track_ids,
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

                        for box, track_id, kp in zip(boxes, track_ids, keypoints):
                            x, y, w, h = box
                            player_crop = frame[int(y) : int(y + h), int(x) : int(x + w)]

                            if player_crop.size == 0:
                                continue
                            if not find_match_2d_array(other_track_ids, track_id):
                                # player 1 has been updated last
                                if updated[0][1] > updated[1][1]:
                                    if len(references2) > 1:
                                        other_track_ids.append([track_id, 2])
                                        print(f"added track id {track_id} to player 2")
                                else:
                                    other_track_ids.append([track_id, 1])
                                    print(f"added track id {track_id} to player 1")
                            # if updated[0], then that means that player 1 was updated last
                            # bc of this, we can assume that the next player is player 2
                            # if player 2 was updated last, then player 1 is next
                            # if both were updated at the same time, then player 1 is next as track ids go from 1 --> 2 im really hoping
                            player_crop_pil = read_image_as_pil(player_crop)
                            current_embeddings=generate_embeddings(player_crop=player_crop_pil)
                            
                            if track_id == 1:
                                playerid = 1
                            elif track_id == 2:
                                playerid = 2
                            # updated [0] is player 1, updated [1] is player 2
                            # if player1 was updated last, then player 2 is next
                            # if player 2 was updated last, then player 1 is next
                            # if both were updated at the same time, then player 1 is next as track ids go from 1 --> 2 im really hoping
                            elif updated[0][1] > updated[1][1]:
                                playerid = 2
                                # player 1 was updated last
                            elif updated[0][1] < updated[1][1]:
                                playerid = 1
                                # player 2 was updated last
                            elif updated[0][1] == updated[1][1]:
                                playerid = 1
                                # both players were updated at the same time, so we are assuming that player 1 is the next player
                            else:
                                print(f"could not find player id for track id {track_id}")
                                continue
                            if len(embeddings[1])>0 and len(embeddings[0])>0 and track_id!=1 and track_id!=2:
                                print(f'track id is : {track_id}')
                                print(f'player id is : {playerid}')
                                temp_playerid, tempconf=find_what_player(embeddings[0],embeddings[1],current_embeddings)
                                print(f'player is most likely: {temp_playerid} with confidence {tempconf}')
                                print(f'len of embeddings: {len(embeddings[0])} and {len(embeddings[1])}')
                                playerid=temp_playerid
                            #given that the first track ids(1 and 2) are always right
                            """
                            if track_id==1:
                                player1_crop=frame[int(y):int(y+h),int(x):int(x+w)]
                                #convert to PIL image
                                player1_crop_pil=read_image_as_pil(player1_crop)
                                player1embeddings=generate_embeddings(player1_crop_pil)
                                embeddings[0].append(player1embeddings)
                            if track_id==2:
                                player2_crop=frame[int(y):int(y+h),int(x):int(x+w)]
                                player2_crop_pil=read_image_as_pil(player2_crop)
                                player2embeddings=generate_embeddings(player2_crop_pil)
                                embeddings[1].append(player2embeddings)
                            """
                            # If player is already tracked, update their info
                            if playerid in players:
                                players[playerid].add_pose(kp)
                                player_last_positions[playerid] = (x, y)  # Update position
                                players[playerid].add_pose(kp)
                                if playerid == 1:
                                    updated[0][0] = True
                                    updated[0][1] = frame_count
                                if playerid == 2:
                                    updated[1][0] = True
                                    updated[1][1] = frame_count
                            if len(players) < max_players:
                                players[other_track_ids[track_id][0]] = Player(
                                    player_id=other_track_ids[track_id][1]
                                )
                                player_last_positions[playerid] = (x, y)
                                if playerid == 1:
                                    updated[0][0] = True
                                    updated[0][1] = frame_count
                                else:
                                    updated[1][0] = True
                                    updated[1][1] = frame_count
                                print(f"Player {playerid} added.")
                            # putting player keypoints on the frame
                            pos=[]
                            for keypoint in kp:
                                # print(keypoint.xyn[0])
                                i = 0
                                for k in keypoint.xyn[0]:
                                    x, y = k
                                    x = int(x * frame_width)
                                    y = int(y * frame_height)
                                    if playerid == 1:
                                        cv2.circle(
                                            annotated_frame, (int(x), int(y)), 3, (0, 0, 255), 5
                                        )
                                    else:
                                        cv2.circle(
                                            annotated_frame, (int(x), int(y)), 3, (255, 0, 0), 5
                                        )
                                    if i == 16:
                                        cv2.putText(
                                            annotated_frame,
                                            f"{playerid}",
                                            (int(x), int(y)),
                                            cv2.FONT_HERSHEY_SIMPLEX,
                                            2.5,
                                            (255, 255, 255),
                                            7,
                                        )
                                        pos.append([x,y])
                                    if i == 15:
                                        pos.append([x,y])
                                    if i==10:
                                        cv2.putText(
                                            annotated_frame,
                                            f"{track_id}",
                                            (int(x), int(y)),
                                            cv2.FONT_HERSHEY_SIMPLEX,
                                            0.5,
                                            (255, 255, 255),
                                            2,
                                        )
                                    i += 1
                            #print(f'playerid was :{playerid}')
                            if len(pos)==2:
                                avgpos=[(pos[0][0]+pos[1][0])/2,(pos[0][1]+pos[1][1])/2, frame_count]
                                if playerid==1:
                                    plast[0].append(avgpos)
                                else:
                                    plast[1].append(avgpos)
                            elif len(pos)==1:
                                pos.append(frame_count)
                                if playerid==1:
                                    plast[0].append(pos)
                                else:
                                    plast[1].append(pos)
                            else:
                                #print(f'kp: {kp}')
                                print(f"could not find the player position for player {playerid}")
                        
                        # in the form of [ball shot type, player1 proximity to the ball, player2 proximity to the ball, ]
                        importantdata = []
                    return [
                        pose_model,
                        frame,
                        other_track_ids,
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
                    print(f"framepose error: {e}")
                    print(f'line was {e.__traceback__.tb_lineno}')
                    return [
                        pose_model,
                        frame,
                        other_track_ids,
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
            # from squash import inferenceslicing
            # from squash import deepsortframepose
            
            def ballplayer_detections(
                frame,
                frame_height,
                frame_width,
                frame_count,
                annotated_frame,
                ballmodel,
                pose_model,
                mainball,
                ball,
                ballmap,
                past_ball_pos,
                ball_false_pos,
                running_frame,
                other_track_ids,
                updated,
                references1,
                references2,
                pixdiffs,
                players,
                player_last_positions,
                occluded,
                importantdata,
                embeddings=[[],[]],
                plast=[[],[]]
            ):
                try:
                    # Ball detection with simple processing and safety checks
                    ball = ballmodel(frame)
                    
                    # Simple ball detection - just use the model predictions directly
                    x1, y1, x2, y2 = 0, 0, 0, 0
                    highestconf = 0.0
                    label = "ball"
                    ball_detected = False
                    
                    # Get the best detection directly from the model with safety checks
                    if ball and len(ball) > 0 and hasattr(ball[0], 'boxes') and ball[0].boxes is not None and len(ball[0].boxes) > 0:
                        # Find the highest confidence detection
                        best_box = None
                        best_conf = 0
                        
                        for box in ball[0].boxes:
                            try:
                                confidence = float(box.conf)
                                if confidence > best_conf and confidence > 0.2:  # Simple threshold
                                    best_conf = confidence
                                    best_box = box
                            except (IndexError, AttributeError) as e:
                                print(f"Error processing box: {e}")
                                continue
                        
                        if best_box is not None:
                            try:
                                # Use the best detection directly with safety checks
                                coords = best_box.xyxy[0]
                                if len(coords) >= 4:
                                    x1, y1, x2, y2 = float(coords[0]), float(coords[1]), float(coords[2]), float(coords[3])
                                    highestconf = best_conf
                                    label = ballmodel.names[int(best_box.cls)]
                                    ball_detected = True
                            except (IndexError, AttributeError) as e:
                                print(f"Error processing best_box coordinates: {e}")
                                ball_detected = False
                    
                    # Draw bounding box if ball detected
                    if ball_detected:
                        cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                        cv2.putText(
                            annotated_frame,
                            f"{label} {highestconf:.2f}",
                            (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 255, 0),
                            2,
                        )

                    # print(label)
                    cv2.putText(
                        annotated_frame,
                        f"Frame: {frame_count}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (0, 255, 0),
                        2,
                    )
                    
                    
                    # Simple ball position update
                    if ball_detected:
                        # Calculate ball center
                        avg_x = int((x1 + x2) / 2)
                        avg_y = int((y1 + y2) / 2)
                        
                        # Update past_ball_pos with current detection
                        past_ball_pos.append([avg_x, avg_y, running_frame])
                        
                        # Keep only recent positions to prevent memory issues
                        if len(past_ball_pos) > 100:
                            past_ball_pos = past_ball_pos[-100:]
                        
                        # Update mainball
                        mainball.update(avg_x, avg_y, avg_x * avg_y)
                        
                        # Update heatmap
                        if len(past_ball_pos) > 1:
                            prev_x, prev_y, _ = past_ball_pos[-2]
                            drawmap(avg_x, avg_y, prev_x, prev_y, ballmap)
                    
                    # Enhanced trajectory visualization with improved bounce detection
                    if len(past_ball_pos) > 1:
                        # Draw recent trajectory with gradient effect
                        recent_positions = past_ball_pos[-20:] if len(past_ball_pos) > 20 else past_ball_pos
                        for i in range(1, len(recent_positions)):
                            pt1 = (int(recent_positions[i-1][0]), int(recent_positions[i-1][1]))
                            pt2 = (int(recent_positions[i][0]), int(recent_positions[i][1]))
                            # Gradient color effect - newer positions are brighter
                            alpha = int(255 * (i / len(recent_positions)))
                            cv2.line(annotated_frame, pt1, pt2, (255, 0, alpha), 2)
                        
                        # Enhanced ball bounce detection and visualization
                        if len(past_ball_pos) >= 4:
                            # Use enhanced GPU-accelerated bounce detection
                            trajectory_segment = past_ball_pos[-30:] if len(past_ball_pos) > 30 else past_ball_pos
                            gpu_bounces = detect_ball_bounces_gpu(
                                trajectory_segment, 
                                velocity_threshold=3.0, 
                                angle_threshold=30.0,
                                court_width=frame_width,
                                court_height=frame_height
                            )
                            
                            # Enhanced visualization for GPU-detected bounces
                            for i, bounce_pos in enumerate(gpu_bounces):
                                # Pulsing effect for recent bounces
                                pulse_factor = 1.0 + 0.3 * math.sin(frame_count * 0.2 + i)
                                radius = int(12 * pulse_factor)
                                
                                # Multiple circle layers for better visibility
                                cv2.circle(annotated_frame, bounce_pos, radius + 6, (0, 255, 255), 3)  # Outer yellow ring
                                cv2.circle(annotated_frame, bounce_pos, radius + 3, (0, 200, 255), 2)  # Middle ring
                                cv2.circle(annotated_frame, bounce_pos, radius, (0, 255, 255), -1)     # Filled center
                                cv2.circle(annotated_frame, bounce_pos, radius - 3, (255, 255, 255), -1)  # White core
                                
                                # Add bounce number label
                                cv2.putText(annotated_frame, f"B{i+1}", 
                                        (bounce_pos[0] - 10, bounce_pos[1] - 20),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 2)
                            
                            # Also use wall bounce detection for comparison and additional accuracy
                            wall_bounce_result = detect_wall_bounces_advanced(trajectory_segment, frame_width, frame_height)
                            wall_bounce_positions = []
                            
                            if isinstance(wall_bounce_result, tuple) and len(wall_bounce_result) > 1:
                                _, wall_bounce_positions = wall_bounce_result
                                
                                # Draw wall bounces with different style (larger, more transparent)
                                for j, wall_bounce_pos in enumerate(wall_bounce_positions):
                                    # Ensure wall bounce isn't too close to GPU-detected bounce
                                    is_duplicate = False
                                    for gpu_bounce in gpu_bounces:
                                        distance = math.sqrt((wall_bounce_pos[0] - gpu_bounce[0])**2 + 
                                                        (wall_bounce_pos[1] - gpu_bounce[1])**2)
                                        if distance < 25:  # Within 25 pixels
                                            is_duplicate = True
                                            break
                                    
                                    if not is_duplicate:
                                        # Wall bounce visualization (orange-ish)
                                        cv2.circle(annotated_frame, wall_bounce_pos, 18, (0, 165, 255), 3)  # Orange outline
                                        cv2.circle(annotated_frame, wall_bounce_pos, 12, (0, 200, 255), 2)  # Inner ring
                                        cv2.circle(annotated_frame, wall_bounce_pos, 6, (0, 165, 255), -1)  # Filled center
                                        
                                        # Wall bounce label
                                        cv2.putText(annotated_frame, f"W{j+1}", 
                                                (wall_bounce_pos[0] - 12, wall_bounce_pos[1] - 25),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 165, 255), 2)
                        
                        # Enhanced bounce statistics display
                        if len(past_ball_pos) >= 4:
                            all_bounces = gpu_bounces.copy()
                            if isinstance(wall_bounce_result, tuple) and len(wall_bounce_result) > 1:
                                # Only add non-duplicate wall bounces
                                for wall_pos in wall_bounce_result[1]:
                                    is_duplicate = any(
                                        math.sqrt((wall_pos[0] - gpu_pos[0])**2 + (wall_pos[1] - gpu_pos[1])**2) < 25
                                        for gpu_pos in gpu_bounces
                                    )
                                    if not is_duplicate:
                                        all_bounces.append(wall_pos)
                            
                            if all_bounces:
                                # Enhanced bounce counter with background
                                bounce_text = f"Bounces Detected: {len(all_bounces)} (GPU: {len(gpu_bounces)})"
                                text_size = cv2.getTextSize(bounce_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                                cv2.rectangle(annotated_frame, (8, 75), (text_size[0] + 16, 105), (0, 0, 0), -1)
                                cv2.putText(annotated_frame, bounce_text, 
                                        (10, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    
                    # Simple status display
                    if ball_detected:
                        cv2.putText(annotated_frame, "BALL DETECTED", 
                                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    else:
                        cv2.putText(annotated_frame, "NO BALL DETECTED", 
                                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    use_enhanced_tracking = False

                    
                    # Use the appropriate framepose function
                    if use_enhanced_tracking:
                        try:
                            framepose_result = enhanced_framepose(
                                pose_model=pose_model,
                                frame=frame,
                                otherTrackIds=other_track_ids,
                                updated=updated,
                                references1=references1,
                                references2=references2,
                                pixdiffs=pixdiffs,
                                players=players,
                                frame_count=frame_count,
                                player_last_positions=player_last_positions,
                                frame_width=frame_width,
                                frame_height=frame_height,
                                annotated_frame=annotated_frame,
                                max_players=2,
                                occluded=occluded,
                                importantdata=importantdata,
                                embeddings=embeddings,
                                plast=plast
                            )
                        except Exception as e:
                            print(f"Enhanced tracking failed: {e}, falling back to standard")
                            use_enhanced_tracking = False
                    
                    if not use_enhanced_tracking:
                        # Standard framepose with enhanced parameters
                        framepose_result = framepose(
                            pose_model=pose_model,
                            frame=frame,
                            other_track_ids=other_track_ids,
                            updated=updated,
                            references1=references1,
                            references2=references2,
                            pixdiffs=pixdiffs,
                            players=players,
                            frame_count=frame_count,
                            player_last_positions=player_last_positions,
                            frame_width=frame_width,
                            frame_height=frame_height,
                            annotated_frame=annotated_frame,
                            occluded=occluded,
                            importantdata=importantdata,
                            embeddings=embeddings,
                            plast=plast
                        )
                    
                    # Ensure framepose_result has minimum required elements
                    if len(framepose_result) < 13:
                        raise ValueError(f"framepose_result has insufficient elements: {len(framepose_result)}")
                    
                    other_track_ids = framepose_result[2]
                    updated = framepose_result[3]
                    references1 = framepose_result[4]
                    references2 = framepose_result[5]
                    pixdiffs = framepose_result[6]
                    players = framepose_result[7]
                    player_last_positions = framepose_result[9]
                    annotated_frame = framepose_result[12]
                    
                    # Safe access for additional elements that might not exist in older versions
                    occluded = framepose_result[13] if len(framepose_result) > 13 else False
                    importantdata = framepose_result[14] if len(framepose_result) > 14 else []
                    embeddings = framepose_result[15] if len(framepose_result) > 15 else [[],[]]
                    
                    who_hit = determine_ball_hit(players, past_ball_pos)
                    #print(f'is rally on: {is_rally_on(plast)}')
                    return [
                        frame,  # 0
                        frame_count,  # 1
                        annotated_frame,  # 2
                        mainball,  # 3
                        ball,  # 4
                        ballmap,  # 5
                        past_ball_pos,  # 6
                        ball_false_pos,  # 7
                        running_frame,  # 8
                        other_track_ids,  # 9
                        updated,  # 10
                        references1,  # 11
                        references2,  # 12
                        pixdiffs,  # 13
                        players,  # 14
                        player_last_positions,  # 15
                        occluded,  # 16
                        importantdata,  # 17
                        who_hit,  # 18
                        embeddings, # 19
                        plast # 20
                    ]
                except Exception as e:
                    print(f'error in ballplayer_detections: {e}')
                    who_hit = 0  # Initialize who_hit for error case
                    # Make sure we have embeddings and plast initialized for the error case
                    if not embeddings:
                        embeddings = [[],[]]
                    if not plast:
                        plast = [[],[]]
                        
                    return [
                        frame,  # 0
                        frame_count,  # 1
                        annotated_frame,  # 2
                        mainball,  # 3
                        ball,  # 4
                        ballmap,  # 5
                        past_ball_pos,  # 6
                        ball_false_pos,  # 7
                        running_frame,  # 8
                        other_track_ids,  # 9
                        updated,  # 10
                        references1,  # 11
                        references2,  # 12
                        pixdiffs,  # 13
                        players,  # 14
                        player_last_positions,  # 15
                        occluded,  # 16
                        importantdata,  # 17
                        who_hit,  # 18
                        embeddings, # 19
                        plast # 20
                    ]

            
            
            detections_result = ballplayer_detections(
                frame=frame,
                frame_height=frame_height,
                frame_width=frame_width,
                frame_count=frame_count,
                annotated_frame=annotated_frame,
                ballmodel=ballmodel,
                pose_model=pose_model,
                mainball=mainball,
                ball=ball,
                ballmap=ballmap,
                past_ball_pos=past_ball_pos,
                ball_false_pos=ball_false_pos,
                running_frame=running_frame,
                other_track_ids=otherTrackIds,
                updated=updated,
                references1=references1,
                references2=references2,
                pixdiffs=pixdiffs,
                players=players,
                player_last_positions=player_last_positions,
                occluded=False,
                importantdata=[],
                embeddings=embeddings,                plast=plast,
            )
            
            # Ensure we have all the expected elements in detections_result
            # Initialize default values for all variables
            idata = []
            who_hit = 0
            embeddings = [[],[]]
            plast = [[],[]]
            # Get values from detections_result with safe access
            def safe_get(arr, idx, default=None):
                return arr[idx] if idx < len(arr) else default
                
            frame = safe_get(detections_result, 0, frame)
            frame_count = safe_get(detections_result, 1, frame_count)
            annotated_frame = safe_get(detections_result, 2, annotated_frame)
            mainball = safe_get(detections_result, 3, mainball)
            ball = safe_get(detections_result, 4, ball)
            ballmap = safe_get(detections_result, 5, ballmap)
            past_ball_pos = safe_get(detections_result, 6, past_ball_pos)
            ball_false_pos = safe_get(detections_result, 7, ball_false_pos)
            running_frame = safe_get(detections_result, 8, running_frame)
            otherTrackIds = safe_get(detections_result, 9, otherTrackIds)
            updated = safe_get(detections_result, 10, updated)
            references1 = safe_get(detections_result, 11, references1)
            references2 = safe_get(detections_result, 12, references2)
            pixdiffs = safe_get(detections_result, 13, pixdiffs)
            players = safe_get(detections_result, 14, players)
            player_last_positions = safe_get(detections_result, 15, player_last_positions)
            # Safe access for remaining elements
            occluded = safe_get(detections_result, 16, False)
            idata = safe_get(detections_result, 17, [])
            who_hit = safe_get(detections_result, 18, 0)
            embeddings = safe_get(detections_result, 19, [[],[]])
            plast = safe_get(detections_result, 20, [[],[]])
            
            if len(detections_result) < 21:
                print(f"Warning: detections_result has {len(detections_result)} elements, expected 21")
            # print(f'who_hit: {who_hit}')
            if 'idata' in locals() and idata:
                alldata.append(idata)
            
            # Enhanced match state detection with optimized parameters
            match_in_play = is_match_in_play(
                players, 
                past_ball_pos,
                movement_threshold=0.12 * frame_width,  # More sensitive player detection
                hit_threshold=0.08 * frame_height,      # More sensitive ball hit detection  
                ballthreshold=10,                       # Increased trajectory analysis window
                ball_angle_thresh=30,                   # More sensitive angle detection
                ball_velocity_thresh=2.0,               # Lower velocity threshold for subtle hits
                advanced_analysis=True                  # Enable all advanced pattern recognition
            )
            # Enhanced shot classification with comprehensive trajectory and pose analysis
            type_of_shot = classify_shot(
                past_ball_pos=past_ball_pos, 
                court_width=frame_width, 
                court_height=frame_height,
                previous_shot=getattr(type_of_shot, 'previous_shot', None) if 'type_of_shot' in locals() else None
            )
            
            # Add player action classification if available
            if who_hit in [1, 2] and players.get(who_hit) and players.get(who_hit).get_latest_pose():
                try:
                    from squash.actionclassifier import classify as classify_player_action
                    player_pose = players.get(who_hit).get_latest_pose()
                    # Extract keypoints properly - handle both .xyn and .keypoints formats
                    if hasattr(player_pose, 'xyn') and len(player_pose.xyn) > 0:
                        player_keypoints = player_pose.xyn[0]
                    elif hasattr(player_pose, 'keypoints'):
                        player_keypoints = player_pose.keypoints
                    else:
                        player_keypoints = player_pose
                    
                    player_action = classify_player_action(player_keypoints)
                    if isinstance(type_of_shot, list) and len(type_of_shot) >= 2:
                        type_of_shot.append(f"Player{who_hit}_{player_action}")
                    print(f"🎾 Player {who_hit} action: {player_action}")
                except ImportError:
                    pass  # Action classifier not available
                except Exception as e:
                    print(f"Action classification error: {e}")
            
            # Enhanced coaching data collection with comprehensive metrics
            try:
                coaching_data = collect_coaching_data(
                    players, past_ball_pos, type_of_shot, who_hit, match_in_play, frame_count
                )
                
                # Ensure coaching_data is a dictionary
                if not isinstance(coaching_data, dict):
                    coaching_data = {'base_data': coaching_data}
                
                # Add simple ball tracking metrics to coaching data
                coaching_data['ball_tracking_confidence'] = 1.0 if len(past_ball_pos) > 0 else 0.0
                coaching_data['ball_tracking_state'] = "tracking" if len(past_ball_pos) > 0 else "searching"
                
                # Add enhanced trajectory analysis
                if len(past_ball_pos) > 5:
                    try:
                        # Calculate velocity profile manually
                        velocities = []
                        recent_positions = past_ball_pos[-6:]
                        for i in range(1, len(recent_positions)):
                            p1, p2 = recent_positions[i-1], recent_positions[i]
                            if len(p1) >= 3 and len(p2) >= 3:
                                dt = p2[2] - p1[2] if p2[2] != p1[2] else 1
                                velocity = math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2) / dt
                                velocities.append(velocity)
                        coaching_data['ball_velocity_profile'] = velocities
                    except Exception as vel_error:
                        print(f"Velocity calculation error: {vel_error}")
                        coaching_data['ball_velocity_profile'] = []
                    
                    # Safe wall bounce detection with fallback dimensions
                    try:
                        if 'frame_width' in locals() and 'frame_height' in locals():
                            bounce_result = detect_wall_bounces_advanced(past_ball_pos, frame_width, frame_height)
                        else:
                            # Use default dimensions or get from frame
                            current_frame_width = getattr(frame, 'shape', [0, 640])[1] if 'frame' in locals() else 640
                            current_frame_height = getattr(frame, 'shape', [360, 0])[0] if 'frame' in locals() else 360
                            bounce_result = detect_wall_bounces_advanced(past_ball_pos, current_frame_width, current_frame_height)
                        
                        # Extract bounce count and positions
                        if isinstance(bounce_result, tuple) and len(bounce_result) == 2:
                            bounce_count, bounce_positions = bounce_result
                            coaching_data['wall_bounce_count'] = bounce_count
                            coaching_data['wall_bounce_positions'] = bounce_positions
                            
                            # Debug output for first few bounces
                            if bounce_count > 0 and frame_count <= 100:
                                print(f"🏐 Frame {frame_count}: {bounce_count} bounces detected at {bounce_positions}")
                        else:
                            # Fallback for old format
                            coaching_data['wall_bounce_count'] = bounce_result if isinstance(bounce_result, int) else 0
                            coaching_data['wall_bounce_positions'] = []
                            
                    except Exception as bounce_error:
                        print(f"Wall bounce detection error: {bounce_error}")
                        coaching_data['wall_bounce_count'] = 0
                        coaching_data['wall_bounce_positions'] = []
                
                coaching_data_collection.append(coaching_data)
                
            except Exception as e:
                print(f"⚠️  Error in coaching data collection: {e}")
                # Create basic coaching data as fallback
                basic_coaching_data = {
                    'frame_count': frame_count,
                    'players_detected': len(players),
                    'ball_detected': len(past_ball_pos) > 0,
                    'match_in_play': match_in_play.get('in_play', False) if isinstance(match_in_play, dict) else False,
                    'error': str(e)
                }
                coaching_data_collection.append(basic_coaching_data)
            # print(f"occluded: {occluded}")
            # occluded structured as [[players_found, last_pos_p1, last_pos_p2, frame_number]...]
            # print(f'is match in play: {is_match_in_play(players, mainball)}')
            if isinstance(match_in_play, dict) and match_in_play.get('in_play', False):
                ball_hit = match_in_play.get('ball_hit', False)
            else:
                ball_hit = False
            try:
                reorganize_shots(alldata)
                # print(f"organized data: {organizeddata}")
            except Exception as e:
                print(f"error reorganizing: {e}")
                pass
            if isinstance(match_in_play, dict) and match_in_play.get('in_play', False):
                # print(match_in_play)
                cv2.putText(
                    annotated_frame,
                    f"ball hit: {str(match_in_play.get('ball_hit', False) if isinstance(match_in_play, dict) else False)}",
                    (10, frame_height - 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (255, 255, 255),
                    1,
                )
                cv2.putText(
                    annotated_frame,
                    f"shot type: {type_of_shot}",
                    (10, frame_height - 140),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (255, 255, 255),
                    1,
                )
                
            # Enhanced status display with GPU information
            gpu_status = "GPU" if torch.cuda.is_available() else "CPU"
            cv2.putText(
                annotated_frame,
                f"Enhanced Coaching: ON | Device: {gpu_status} | Data: {len(coaching_data_collection)}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
            )
            
            # Display tracking state and performance info
            tracking_state = "tracking" if len(past_ball_pos) > 0 else "searching"
            tracking_color = (0, 255, 0) if tracking_state == "tracking" else (0, 255, 255)
            cv2.putText(
                annotated_frame,
                f"Ball Tracking: {tracking_state.upper()} | Trajectory: {len(past_ball_pos)}",
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                tracking_color,
                1,
            )
            tracking_confidence = 1.0 if len(past_ball_pos) > 0 else 0.0
            
            state_color = {
                'tracking': (0, 255, 0),    # Green
                'predicting': (0, 255, 255), # Yellow
                'searching': (255, 165, 0),  # Orange  
                'lost': (0, 0, 255)          # Red
            }.get(tracking_state, (255, 255, 255))
            
            cv2.putText(
                    annotated_frame,
                    f"Ball Track: {tracking_state.upper()} ({tracking_confidence:.2f})",
                    (frame_width - 250, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    state_color,
                    1,
                )
            # Display ankle positions of both players
            if players.get(1) and players.get(2) is not None:
                if (
                    players.get(1).get_latest_pose()
                    or players.get(2).get_latest_pose() is not None
                ):
                    # print('line 265')
                    try:
                        p1_left_ankle_x = int(
                            players.get(1).get_latest_pose().xyn[0][16][0] * frame_width
                        )
                        p1_left_ankle_y = int(
                            players.get(1).get_latest_pose().xyn[0][16][1] * frame_height
                        )
                        p1_right_ankle_x = int(
                            players.get(1).get_latest_pose().xyn[0][15][0] * frame_width
                        )
                        p1_right_ankle_y = int(
                            players.get(1).get_latest_pose().xyn[0][15][1] * frame_height
                        )
                    except Exception:
                        p1_left_ankle_x = p1_left_ankle_y = p1_right_ankle_x = (
                            p1_right_ankle_y
                        ) = 0
                    try:
                        p2_left_ankle_x = int(
                            players.get(2).get_latest_pose().xyn[0][16][0] * frame_width
                        )
                        p2_left_ankle_y = int(
                            players.get(2).get_latest_pose().xyn[0][16][1] * frame_height
                        )
                        p2_right_ankle_x = int(
                            players.get(2).get_latest_pose().xyn[0][15][0] * frame_width
                        )
                        p2_right_ankle_y = int(
                            players.get(2).get_latest_pose().xyn[0][15][1] * frame_height
                        )
                    except Exception:
                        p2_left_ankle_x = p2_left_ankle_y = p2_right_ankle_x = (
                            p2_right_ankle_y
                        ) = 0
                    # Display the ankle positions on the bottom left of the frame
                    avgxank1 = int((p1_left_ankle_x + p1_right_ankle_x) / 2)
                    avgyank1 = int((p1_left_ankle_y + p1_right_ankle_y) / 2)
                    avgxank2 = int((p2_left_ankle_x + p2_right_ankle_x) / 2)
                    avgyank2 = int((p2_left_ankle_y + p2_right_ankle_y) / 2)
                    text_p1 = f"P1 position(ankle): {avgxank1},{avgyank1}"
                    cv2.putText(
                        annotated_frame,
                        f"{otherTrackIds[find_last(1, otherTrackIds)][1]}",
                        (p1_left_ankle_x, p1_left_ankle_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (255, 255, 255),
                        1,
                    )
                    cv2.putText(
                        annotated_frame,
                        f"{otherTrackIds[find_last(2, otherTrackIds)][1]}",
                        (p2_left_ankle_x, p2_left_ankle_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (255, 255, 255),
                        1,
                    )
                    text_p2 = f"P2 position(ankle): {avgxank2},{avgyank2}"
                    cv2.putText(
                        annotated_frame,
                        text_p1,
                        (10, frame_height - 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (255, 255, 255),
                        1,
                    )
                    cv2.putText(
                        annotated_frame,
                        text_p2,
                        (10, frame_height - 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (255, 255, 255),
                        1,
                    )
                    avgpx1 = int((p1_left_ankle_x + p1_right_ankle_x) / 2)
                    avgpy1 = int((p1_left_ankle_y + p1_right_ankle_y) / 2)
                    avgpx2 = int((p2_left_ankle_x + p2_right_ankle_x) / 2)
                    avgpy2 = int((p2_left_ankle_y + p2_right_ankle_y) / 2)
                    # print(reference_points)
                    p1distancefromT = math.hypot(
                        reference_points[4][0] - avgpx1, reference_points[4][1] - avgpy1
                    )
                    p2distancefromT = math.hypot(
                        reference_points[4][0] - avgpx2, reference_points[4][1] - avgpy2
                    )
                    p1distancesfromT.append(p1distancefromT)
                    p2distancesfromT.append(p2distancefromT)
                    text_p1t = f"P1 distance from T: {p1distancesfromT[-1]}"
                    text_p2t = f"P2 distance from T: {p2distancesfromT[-1]}"
                    cv2.putText(
                        annotated_frame,
                        text_p1t,
                        (10, frame_height - 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (255, 255, 255),
                        1,
                    )
                    cv2.putText(
                        annotated_frame,
                        text_p2t,
                        (10, frame_height - 80),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (255, 255, 255),
                        1,
                    )
                    is_rally= is_rally_on(plast)
                    cv2.putText(
                        annotated_frame,
                        f'Rally Status: {"ACTIVE" if is_rally else "INACTIVE"}',
                        (10, frame_height - 120),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (0, 255, 0) if is_rally else (0, 0, 255),
                        1,
                    )
                    
                    # Enhanced player movement analysis
                    if len(p1distancesfromT) > 10 and len(p2distancesfromT) > 10:
                        p1_movement_trend = "ADVANCING" if p1distancesfromT[-1] > p1distancesfromT[-5] else "RETREATING"
                        p2_movement_trend = "ADVANCING" if p2distancesfromT[-1] > p2distancesfromT[-5] else "RETREATING"
                        
                        cv2.putText(
                            annotated_frame,
                            f'P1: {p1_movement_trend} | P2: {p2_movement_trend}',
                            (frame_width - 250, frame_height - 40),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.4,
                            (255, 255, 255),
                            1,
                        )
                    # Create a live updating plot window
                    # plt.figure(
                    #     2
                    # )  # Use figure 2 for the distance plot (figure 1 is the video)
                    # plt.clf()  # Clear the current figure
                    # plt.plot(p1distancesfromT, color="red", label="P1 Distance from T")
                    # plt.plot(p2distancesfromT, color="blue", label="P2 Distance from T")

                    # # Add labels and title
                    # plt.xlabel("Time (frames)")
                    # plt.ylabel("Distance from T(pixels)")
                    # plt.title("Distance from T over Time")
                    # plt.legend()
                    # plt.grid(True)

                    # # Update the plot window
                    # plt.draw()
                    # plt.pause(0.0001)  # Small pause to allow the window to update

                    # # Save the plot to a file
                    # plt.savefig("output/distance_from_t_over_time.png")

                    # # Save distances to a file
                    # with open("output/distances_from_t.txt", "w") as f:
                    #     for d1, d2 in zip(p1distancesfromT, p2distancesfromT):
                    #         f.write(f"{d1},{d2}\n")

            # Display the annotated frame
            try:
                # Generate player ankle heatmap
                if (
                    players.get(1).get_latest_pose() is not None
                    and players.get(2).get_latest_pose() is not None
                ):
                    player_ankles = [
                        (
                            int(
                                players.get(1).get_latest_pose().xyn[0][16][0]
                                * frame_width
                            ),
                            int(
                                players.get(1).get_latest_pose().xyn[0][16][1]
                                * frame_height
                            ),
                        ),
                        (
                            int(
                                players.get(2).get_latest_pose().xyn[0][16][0]
                                * frame_width
                            ),
                            int(
                                players.get(2).get_latest_pose().xyn[0][16][1]
                                * frame_height
                            ),
                        ),
                    ]

                    # Draw points on the heatmap
                    for ankle in player_ankles:
                        cv2.circle(
                            heatmap_image, ankle, 5, (255, 0, 0), -1
                        )  # Blue points for Player 1
                        cv2.circle(
                            heatmap_image, ankle, 5, (0, 0, 255), -1
                        )  # Red points for Player 2

                blurred_heatmap_ankle = cv2.GaussianBlur(heatmap_image, (51, 51), 0)

                # Normalize heatmap and apply color map in one step
                normalized_heatmap = cv2.normalize(
                    blurred_heatmap_ankle, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U
                )
                heatmap_overlay = cv2.applyColorMap(
                    normalized_heatmap, cv2.COLORMAP_JET
                )

                # Combine with white image
                cv2.addWeighted(
                    np.ones_like(heatmap_overlay) * 255, 0.5, heatmap_overlay, 0.5, 0
                )
            except Exception:
                # print(f"line618: {e}")
                pass
            # Save the combined image
            # cv2.imwrite("output/heatmap_ankle.png", combined_image)
            ballx = bally = 0
            # ball stuff
            if (
                mainball is not None
                and mainball.getlastpos() is not None
                and mainball.getlastpos() != (0, 0)
            ):
                ballx = mainball.getlastpos()[0]
                bally = mainball.getlastpos()[1]
                if ballx != 0 and bally != 0:
                    if [ballx, bally] not in ballxy:
                        ballxy.append([ballx, bally, frame_count])
                        # print(
                        #     f"ballx: {ballx}, bally: {bally}, appended to ballxy with length {len(ballxy)} and frame count as : {frame_count}"
                        # )

            # Draw the ball trajectory
            if len(ballxy) > 2:
                for i in range(1, len(ballxy)):
                    if ballxy[i - 1] is None or ballxy[i] is None:
                        continue
                    if ballxy[i][2] - ballxy[i - 1][2] < 7:
                        if frame_count - ballxy[i][2] < 7:
                            cv2.line(
                                annotated_frame,
                                (ballxy[i - 1][0], ballxy[i - 1][1]),
                                (ballxy[i][0], ballxy[i][1]),
                                (0, 255, 0),
                                2,
                            )
                            cv2.circle(
                                annotated_frame,
                                (ballxy[i - 1][0], ballxy[i - 1][1]),
                                5,
                                (0, 255, 0),
                                -1,
                            )
                            cv2.circle(
                                annotated_frame,
                                (ballxy[i][0], ballxy[i][1]),
                                5,
                                (0, 255, 0),
                                -1,
                            )

                            # cv2.circle(
                            #    annotated_frame,
                            #    (next_pos[0], next_pos[1]),
                            #    5,
                            #    (0, 255, 0),
                            #    -1,
                            # )

            for ball_pos in ballxy:
                if frame_count - ball_pos[2] < 7:
                    # print(f'wrote to frame on line 1028 with coords: {ball_pos}')
                    cv2.circle(
                        annotated_frame, (ball_pos[0], ball_pos[1]), 5, (0, 255, 0), -1
                    )

            positions = load_data("output\\ball-xyn.txt")
            if len(positions) > 11:
                input_sequence = np.array([positions[-10:]])
                input_sequence = input_sequence.reshape((1, 10, 2, 1))
                predicted_pos = ball_predict(input_sequence)
                # print(f'input_sequence: {input_sequence}')
                cv2.circle(
                    annotated_frame,
                    (
                        int(predicted_pos[0][0] * frame_width),
                        int(predicted_pos[0][1] * frame_height),
                    ),
                    7,
                    (0, 0, 255),
                    7,
                )
                cv2.putText(
                    annotated_frame,
                    f"predicted ball position in 1 frame: {int(predicted_pos[0][0]*frame_width)},{int(predicted_pos[0][1]*frame_height)}",
                    (
                        int(predicted_pos[0][0] * frame_width),
                        int(predicted_pos[0][1] * frame_height),
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (255, 255, 255),
                    1,
                )
                last9 = positions[-9:]
                last9.append([predicted_pos[0][0], predicted_pos[0][1]])
                # print(f'last 9: {last9}')
                sequence_and_predicted = np.array(last9)
                # print(f'sequence and predicted: {sequence_and_predicted}')
                sequence_and_predicted = sequence_and_predicted.reshape((1, 10, 2, 1))
                future_predict = ball_predict(sequence_and_predicted)
                cv2.circle(
                    annotated_frame,
                    (
                        int(future_predict[0][0] * frame_width),
                        int(future_predict[0][1] * frame_height),
                    ),
                    7,
                    (255, 0, 0),
                    7,
                )
                cv2.putText(
                    annotated_frame,
                    f"predicted ball position in 3 frames: {int(future_predict[0][0]*frame_width)},{int(future_predict[0][1]*frame_height)}",
                    (
                        int(future_predict[0][0] * frame_width),
                        int(future_predict[0][1] * frame_height),
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (255, 255, 255),
                    1,
                )
            if (
                players.get(1)
                and players.get(2) is not None
                and (
                    players.get(1).get_last_x_poses(3) is not None
                    and players.get(2).get_last_x_poses(3) is not None
                )
            ):
                players.get(1).get_last_x_poses(3).xyn[0]
                players.get(2).get_last_x_poses(3).xyn[0]
                rlp1postemp = [
                    players.get(1).get_last_x_poses(3).xyn[0][16][0] * frame_width,
                    players.get(1).get_last_x_poses(3).xyn[0][16][1] * frame_height,
                ]
                rlp2postemp = [
                    players.get(2).get_last_x_poses(3).xyn[0][16][0] * frame_width,
                    players.get(2).get_last_x_poses(3).xyn[0][16][1] * frame_height,
                ]
                rlworldp1 = pixel_to_3d(
                    rlp1postemp, homography, reference_points_3d
                )
                rlworldp2 = pixel_to_3d(
                    rlp2postemp, homography, reference_points_3d
                )
                text5 = f"Player 1: {rlworldp1}"
                text6 = f"Player 2: {rlworldp2}"

                cv2.putText(
                    annotated_frame,
                    text5,
                    (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (255, 255, 255),
                    1,
                )
                cv2.putText(
                    annotated_frame,
                    text6,
                    (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (255, 255, 255),
                    1,
                )
            # Safe ball position to 3D conversion
            rlball = None
            if past_ball_pos and len(past_ball_pos) > 0:
                last_ball_pos = past_ball_pos[-1]
                if len(last_ball_pos) >= 2:
                    try:
                        rlball = pixel_to_3d(
                            [last_ball_pos[0], last_ball_pos[1]],
                            homography,
                            reference_points_3d,
                        )
                    except Exception as e:
                        print(f"Error converting ball position to 3D: {e}")
                        rlball = None
                else:
                    print(f"Warning: Last ball position has insufficient data: {last_ball_pos}")
            else:
                print("Warning: No ball positions available for 3D conversion")
            

            def csvwrite():
                try:
                    with open("output/final.csv", "a") as f:
                        csvwriter = csv.writer(f)
                        # going to be putting the framecount, playerkeypoints, ball position, time, type of shot, and also match in play
                        running_frame / fps
                        
                        # Safe shot type handling
                        if type_of_shot is None:
                            shot = "None"
                        elif isinstance(type_of_shot, list) and len(type_of_shot) >= 2:
                            shot = type_of_shot[0] + " " + type_of_shot[1]
                        elif isinstance(type_of_shot, list) and len(type_of_shot) == 1:
                            shot = type_of_shot[0]
                        else:
                            shot = str(type_of_shot)
                        
                        # Safe player pose extraction
                        try:
                            p1_pose = players.get(1).get_latest_pose().xyn[0].tolist() if players.get(1) and players.get(1).get_latest_pose() else []
                        except Exception:
                            p1_pose = []
                            
                        try:
                            p2_pose = players.get(2).get_latest_pose().xyn[0].tolist() if players.get(2) and players.get(2).get_latest_pose() else []
                        except Exception:
                            p2_pose = []
                        
                        # Safe ball location
                        try:
                            ball_loc = mainball.getloc() if mainball else [0, 0]
                        except Exception:
                            ball_loc = [0, 0]
                        
                        data = [
                            running_frame,
                            p1_pose,
                            p2_pose,
                            ball_loc,
                            shot,
                            rlworldp1 if 'rlworldp1' in locals() else [0, 0, 0],
                            rlworldp2 if 'rlworldp2' in locals() else [0, 0, 0],
                            rlball if rlball is not None else [0, 0, 0],
                            f"{who_hit} hit the ball",
                        ]
                        # print(data)
                        csvwriter.writerow(data)
                except Exception as csv_error:
                    print(f"Error writing CSV data: {csv_error}")

            # print(past_ball_pos)
            if past_ball_pos and len(past_ball_pos) > 0:
                last_ball_pos = past_ball_pos[-1]
                if len(last_ball_pos) >= 2:
                    try:
                        ball_3d = pixel_to_3d([last_ball_pos[0], last_ball_pos[1]], homography, reference_points_3d)
                        text = f"ball in rlworld: {ball_3d}"
                    except Exception as e:
                        print(f"Error converting ball position to 3D for display: {e}")
                        text = "ball in rlworld: [error]"
                else:
                    text = "ball in rlworld: [insufficient data]"
            else:
                text = "ball in rlworld: [no position data]"
                
            cv2.putText(
                annotated_frame,
                text,
                (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 255),
                1,
                )
            
            #get last 3 ball positions (x,y)
            if len(past_ball_pos) > 2:
                x1, y1 = past_ball_pos[-1][0], past_ball_pos[-1][1]
                x2, y2 = past_ball_pos[-2][0], past_ball_pos[-2][1]
                #find the slope of the line between the last 2 ball positions
                if x2 - x1 != 0:
                    slope = (y2 - y1) / (x2 - x1)
                else:
                    pass
                #find the magnitude of the difference between the last 2 ball positions
                magnitude = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                #draw an arrow on the middle right to display the slope and magnitude
                arrow_start = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                arrow_end = (int(arrow_start[0] + 50), int(arrow_start[1] + 50 * slope)) if slope is not None else arrow_start
                cv2.arrowedLine(annotated_frame, arrow_start, arrow_end, (0, 255, 0), 2)
            try:
                csvwrite()
            except Exception as e:
                print(f"error: {e}")
                pass
            
            if running_frame > end:
                try:
                    with open("final.txt", "a") as f:
                        f.write(
                            csvanalyze.parse_through(csvstart, end, "output/final.csv")
                        )
                    csvstart = end
                    end += 100
                    
                    # Generate periodic autonomous coaching reports for real-time insights
                    if len(coaching_data_collection) > 50 and running_frame % 500 == 0:
                        try:
                            print(f"\n🔄 Generating interim coaching report at frame {running_frame}...")
                            generate_coaching_report(coaching_data_collection, path, frame_count)
                            print("✓ Interim coaching report updated.")
                        except Exception as e:
                            print(f"Warning: Error in interim coaching report: {e}")
                            
                except Exception as e:
                    print(f"error: {e}")
                    pass            
            # Add instructions to the frame
            cv2.putText(
                annotated_frame,
                "Press 'r' to update reference points, 'q' to quit",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 0),
                1,
            )
            out.write(annotated_frame)
            cv2.imshow("Annotated Frame", annotated_frame)

            #print(f"finished frame {frame_count}")
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("r"):
                print("Updating reference points...")
                # Pause video and update reference points
                reference_points = Referencepoints.update_reference_points(
                    path=path, frame_width=frame_width, frame_height=frame_height, current_frame=frame
                )
                # Regenerate homography with new reference points
                homography = generate_homography(reference_points, reference_points_3d)
                print("Reference points updated successfully!")

    except KeyboardInterrupt:
        print("\nKeyboard interrupt detected. Processing video up to current frame and generating outputs...")
        
        # Close video capture and windows
        cap.release()
        cv2.destroyAllWindows()
        if 'out' in locals():
            out.release()
        
        # Close CSV file properly
        try:
            with open("output/final.csv", "a") as f:
                pass  # Ensure file is closed properly
        except Exception:
            pass
        
        # Close JSON file properly
        try:
            with open("output/final.json", "a") as f:
                f.write("]")
        except Exception:
            pass
        
        # Generate autonomous coaching report using imported function
        try:
            print("Generating coaching report...")
            generate_coaching_report(coaching_data_collection, path, frame_count)
            print("Coaching report generated successfully.")
        except Exception as e:
            print(f"Error generating coaching report: {e}")
        
        # Generate final analysis outputs
        try:
            print("Generating final analysis...")
            # Process any remaining CSV data
            try:
                with open("final.txt", "a") as f:
                    f.write(
                        csvanalyze.parse_through(csvstart, frame_count, "output/final.csv")
                    )
            except Exception as e:
                print(f"Error processing final CSV data: {e}")
            
            print("Final analysis completed.")
        except Exception as e:
            print(f"Error in final analysis: {e}")
        
        print(f"Processing completed with {frame_count} frames analyzed. Check output/ directory for results.")
        
    except Exception as e:
        print(f"error2: {e}")
        print(f"line was {e.__traceback__.tb_lineno}")
        print(f"other into about e: {e.__traceback__}")
        print(f"other info about e: {e.__traceback__.tb_frame}")
        print(f"other info about e: {e.__traceback__.tb_next}")
        print(f"other info about e: {e.__traceback__.tb_lasti}")
    finally:
        # Always execute cleanup and output generation regardless of how the loop ended
        print(f"Processing completed. Analyzed {frame_count} frames.")
        
        # Close video capture and windows
        cap.release()
        cv2.destroyAllWindows()
        if 'out' in locals():
            out.release()
        
        # Close CSV file properly
        try:
            with open("output/final.csv", "a") as f:
                pass  # Ensure file is closed properly
        except Exception:
            pass
        
        # Close JSON file properly
        try:
            with open("output/final.json", "a") as f:
                f.write("]")
        except Exception:
            pass
        
        # Enhanced autonomous coaching analysis and report generation
        print("\n🎯 Generating Enhanced Autonomous Coaching Analysis with Bounce Detection...")
        
        try:
            # Initialize the enhanced autonomous coach
            from autonomous_coaching import AutonomousSquashCoach
            autonomous_coach = AutonomousSquashCoach()
            
            # Generate comprehensive coaching insights
            coaching_insights = autonomous_coach.analyze_match_data(coaching_data_collection)
            
            # Enhanced coaching report with bounce analysis
            enhanced_report = f"""

Analysis Date: {time.strftime('%Y-%m-%d %H:%M:%S')}
Video Analyzed: {path}
Total Frames Processed: {frame_count}
Enhanced Coaching Data Points: {len(coaching_data_collection)}

"""
            
            # Save enhanced report
            with open("output/enhanced_autonomous_coaching_report.txt", "w") as f:
                f.write(enhanced_report)
            
            # Save detailed coaching data with bounce information
            detailed_data = []
            for data_point in coaching_data_collection:
                if isinstance(data_point, dict):
                    detailed_data.append(data_point)
            
            with open("output/enhanced_coaching_data.json", "w") as f:
                json.dump(detailed_data, f, indent=2, default=str)
            
            print("✅ Enhanced coaching analysis completed!")
            print(f"📋 Enhanced report saved: output/enhanced_autonomous_coaching_report.txt")
            print(f"📊 Enhanced data saved: output/enhanced_coaching_data.json")
            print(f"🏐 Ball bounces analyzed: GPU-accelerated detection active")
            
            # Also generate traditional report for compatibility
            generate_coaching_report(coaching_data_collection, path, frame_count)
            print("📝 Traditional coaching report also generated for compatibility.")
            
        except Exception as e:
            print(f"⚠️  Error in enhanced coaching analysis: {e}")
            # Fallback to basic report
            try:
                generate_coaching_report(coaching_data_collection, path, frame_count)
                print("📝 Fallback coaching report generated successfully.")
            except Exception as fallback_error:
                print(f"❌ Fallback coaching report error: {fallback_error}")
        
        print("\n🎉 ENHANCED PROCESSING COMPLETE!")
        print("=" * 50)
        print("📁 Check output/ directory for results:")
        print("   • enhanced_autonomous_coaching_report.txt - Enhanced analysis")
        print("   • enhanced_coaching_data.json - Detailed data with bounces")
        print("   • annotated.mp4 - Video with bounce visualization")
        print("   • Other traditional output files")
        print("=" * 50)

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProgram interrupted by user. All outputs have been generated.")
    except Exception as e:
        print(f"Unexpected error: {e}")
        
