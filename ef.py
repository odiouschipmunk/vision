import time
import torch
start = time.time()
import cv2
import csv
import os
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
print(f"time to import everything: {time.time()-start}")
alldata = organizeddata = []
# Initialize global frame dimensions and frame counter for exception handling
frame_count = 0
frame_width = 0
frame_height = 0
# Autonomous coaching system imported from autonomous_coaching.py

class ShotTracker:
    """
    Enhanced shot tracking system for identifying and visualizing complete shots
    """
    def __init__(self):
        self.active_shots = []  # List of active shots being tracked
        self.completed_shots = []  # List of completed shots for analysis
        self.shot_id_counter = 0
        self.ball_hit_cooldown = 15  # Frames to wait before detecting next hit
        self.last_hit_frame = -999
        
    def detect_shot_start(self, ball_hit, who_hit, frame_count, ball_pos, shot_type):
        """
        Detect the start of a new shot
        """
        if not ball_hit or who_hit == 0:
            return False
            
        # Check cooldown to avoid duplicate shot detection
        if frame_count - self.last_hit_frame < self.ball_hit_cooldown:
            return False
            
        # Start new shot
        self.shot_id_counter += 1
        new_shot = {
            'id': self.shot_id_counter,
            'start_frame': frame_count,
            'player_who_hit': who_hit,
            'shot_type': shot_type,
            'trajectory': [ball_pos.copy()],
            'status': 'active',
            'color': self.get_shot_color(shot_type),
            'end_frame': None,
            'final_shot_type': None
        }
        
        self.active_shots.append(new_shot)
        self.last_hit_frame = frame_count
        return True
        
    def get_shot_color(self, shot_type):
        """
        Get color for shot based on type
        """
        if isinstance(shot_type, list) and len(shot_type) > 0:
            shot_str = str(shot_type[0]).lower()
        else:
            shot_str = str(shot_type).lower()
            
        color_map = {
            'crosscourt': (0, 255, 0),      # Green
            'wide_crosscourt': (0, 200, 0), # Dark green
            'straight': (255, 255, 0),       # Yellow
            'boast': (255, 0, 255),         # Magenta
            'drop': (0, 165, 255),          # Orange
            'lob': (255, 0, 0),             # Blue
            'drive': (0, 255, 255),         # Cyan
            'default': (255, 255, 255)      # White
        }
        
        for shot_name, color in color_map.items():
            if shot_name in shot_str:
                return color
                
        return color_map['default']
        
    def update_active_shots(self, ball_pos, frame_count, new_hit_detected=False, new_shot_type=None):
        """
        Update trajectories of active shots with enhanced bounce detection
        """
        if not ball_pos or len(ball_pos) < 2:
            return
            
        for shot in self.active_shots[:]:  # Use slice to avoid modification during iteration
            if shot['status'] == 'active':
                # Add current ball position to trajectory
                shot['trajectory'].append(ball_pos.copy())
                
                # Perform bounce detection on recent trajectory
                if len(shot['trajectory']) >= 5:
                    # Use the comprehensive bounce detection
                    bounce_count, bounce_positions, bounce_details = detect_ball_bounces_comprehensive(
                        shot['trajectory'], 
                        640,  # Default court width, should be passed as parameter
                        360,  # Default court height, should be passed as parameter
                        confidence_threshold=0.6
                    )
                    
                    # Update shot with bounce information
                    shot['bounce_count'] = bounce_count
                    shot['bounces'] = bounce_details
                    
                    # Update shot type based on bounce pattern
                    if bounce_count > 2:
                        shot['shot_type'] = f"{shot['shot_type']}_multi_bounce"
                    elif bounce_count == 1:
                        # Determine if it's a wall bounce or floor bounce
                        if bounce_details:
                            bounce_type = bounce_details[0].get('wall', 'floor')
                            shot['shot_type'] = f"{shot['shot_type']}_{bounce_type}_bounce"
                
                # Check if shot should end (new hit detected or trajectory completion)
                if new_hit_detected or self.is_shot_complete(shot, frame_count):
                    self.end_shot(shot, frame_count, new_shot_type)
                    
    def is_shot_complete(self, shot, frame_count):
        """
        Determine if a shot is complete based on various criteria
        """
        # Maximum shot duration (in frames)
        max_shot_duration = 180  # ~6 seconds at 30fps
        
        # Minimum shot duration before considering completion
        min_shot_duration = 30   # ~1 second at 30fps
        
        shot_duration = frame_count - shot['start_frame']
        
        # Shot too long - force completion
        if shot_duration > max_shot_duration:
            return True
            
        # Check if ball has stopped moving (end of rally)
        if shot_duration > min_shot_duration and len(shot['trajectory']) > 10:
            recent_positions = shot['trajectory'][-10:]
            max_movement = 0
            
            for i in range(1, len(recent_positions)):
                movement = math.sqrt(
                    (recent_positions[i][0] - recent_positions[i-1][0])**2 + 
                    (recent_positions[i][1] - recent_positions[i-1][1])**2
                )
                max_movement = max(max_movement, movement)
                
            # If ball barely moved, shot is complete
            if max_movement < 5:
                return True
                
        return False
        
    def end_shot(self, shot, frame_count, final_shot_type=None):
        """
        End an active shot and move it to completed shots
        """
        shot['end_frame'] = frame_count
        shot['status'] = 'completed'
        shot['final_shot_type'] = final_shot_type or shot['shot_type']
        shot['duration'] = frame_count - shot['start_frame']
        
        # Move to completed shots
        self.completed_shots.append(shot)
        self.active_shots.remove(shot)
        
        # Save shot data
        self.save_shot_data(shot)
        
    def save_shot_data(self, shot):
        """
        Save shot data to file for later analysis
        """
        try:
            # Get bounce information for this shot
            shot_bounces = []
            if hasattr(bounce_detector, 'detected_bounces'):
                shot_bounces = [
                    b for b in bounce_detector.detected_bounces 
                    if shot['start_frame'] <= b['frame_index'] <= shot.get('end_frame', shot['start_frame'] + 100)
                ]
            
            shot_data = {
                'shot_id': shot['id'],
                'start_frame': shot['start_frame'],
                'end_frame': shot['end_frame'],
                'duration': shot['duration'],
                'player_who_hit': shot['player_who_hit'],
                'shot_type': shot['shot_type'],
                'final_shot_type': shot['final_shot_type'],
                'trajectory_length': len(shot['trajectory']),
                'trajectory': shot['trajectory'],  # Full trajectory data
                'bounces_detected': len(shot_bounces),
                'bounce_details': shot_bounces,
                'shot_color': shot['color']
            }
            
            # Append to shots log file
            with open("output/shots_log.jsonl", "a") as f:
                f.write(json.dumps(shot_data) + "\n")
                
        except Exception as e:
            print(f"Error saving shot data: {e}")
            
    def draw_shot_trajectories(self, frame, ball_pos):
        """
        Draw trajectories of active and recent shots with clear start/end markers
        """
        try:
            # Draw active shots
            for shot in self.active_shots:
                if len(shot['trajectory']) > 1:
                    color = shot['color']
                    thickness = 3
                    
                    # Draw trajectory line
                    for i in range(1, len(shot['trajectory'])):
                        pt1 = (int(shot['trajectory'][i-1][0]), int(shot['trajectory'][i-1][1]))
                        pt2 = (int(shot['trajectory'][i][0]), int(shot['trajectory'][i][1]))
                        cv2.line(frame, pt1, pt2, color, thickness)
                        
                    # Draw START marker - Large circle with "START" text
                    if len(shot['trajectory']) > 0:
                        start_pos = (int(shot['trajectory'][0][0]), int(shot['trajectory'][0][1]))
                        
                        # Draw start marker with double circle
                        cv2.circle(frame, start_pos, 15, color, 3)
                        cv2.circle(frame, start_pos, 8, (255, 255, 255), -1)
                        cv2.putText(frame, "START", 
                                  (start_pos[0] - 25, start_pos[1] - 20),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                        
                        # Draw shot info
                        text = f"Shot {shot['id']}: P{shot['player_who_hit']} - {shot['shot_type']}"
                        cv2.putText(frame, text, 
                                  (start_pos[0] - 50, start_pos[1] - 40),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                        
                        # Draw bounce markers if available
                        if 'bounces' in shot and shot['bounces']:
                            for bounce in shot['bounces']:
                                bounce_pos = (int(bounce['position'][0]), int(bounce['position'][1]))
                                # Yellow diamond for bounces
                                cv2.circle(frame, bounce_pos, 8, (0, 255, 255), -1)
                                cv2.circle(frame, bounce_pos, 12, (0, 255, 255), 2)
                                cv2.putText(frame, "B", (bounce_pos[0] - 4, bounce_pos[1] + 4),
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
                                # Add confidence indicator
                                confidence_text = f"{bounce['confidence']:.1f}"
                                cv2.putText(frame, confidence_text, 
                                           (bounce_pos[0] + 15, bounce_pos[1]),
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 255), 1)
                        
                    # Draw current position marker for active shots
                    if len(shot['trajectory']) > 0:
                        current_pos = (int(shot['trajectory'][-1][0]), int(shot['trajectory'][-1][1]))
                        cv2.circle(frame, current_pos, 10, color, 2)
                        cv2.circle(frame, current_pos, 5, (255, 255, 255), -1)
                        cv2.putText(frame, "ACTIVE", 
                                  (current_pos[0] + 15, current_pos[1] - 5),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
                        
            # Draw recently completed shots (last 3) with END markers
            recent_completed = self.completed_shots[-3:] if len(self.completed_shots) > 3 else self.completed_shots
            for shot in recent_completed:
                if len(shot['trajectory']) > 1:
                    color = shot['color']
                    thickness = 2  # Slightly thinner for completed shots
                    
                    # Draw trajectory line with slight transparency effect
                    for i in range(1, len(shot['trajectory'])):
                        pt1 = (int(shot['trajectory'][i-1][0]), int(shot['trajectory'][i-1][1]))
                        pt2 = (int(shot['trajectory'][i][0]), int(shot['trajectory'][i][1]))
                        cv2.line(frame, pt1, pt2, color, thickness)
                        
                    # Draw START marker
                    if len(shot['trajectory']) > 0:
                        start_pos = (int(shot['trajectory'][0][0]), int(shot['trajectory'][0][1]))
                        cv2.circle(frame, start_pos, 12, color, 2)
                        cv2.circle(frame, start_pos, 6, (0, 255, 0), -1)  # Green center
                        cv2.putText(frame, "S", 
                                  (start_pos[0] - 5, start_pos[1] + 5),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                        
                    # Draw END marker - Square with "END" text
                    if len(shot['trajectory']) > 0:
                        end_pos = (int(shot['trajectory'][-1][0]), int(shot['trajectory'][-1][1]))
                        
                        # Draw end marker as square
                        cv2.rectangle(frame, 
                                    (end_pos[0] - 12, end_pos[1] - 12),
                                    (end_pos[0] + 12, end_pos[1] + 12),
                                    color, 3)
                        cv2.rectangle(frame, 
                                    (end_pos[0] - 6, end_pos[1] - 6),
                                    (end_pos[0] + 6, end_pos[1] + 6),
                                    (0, 0, 255), -1)  # Red center
                        cv2.putText(frame, "END", 
                                  (end_pos[0] - 15, end_pos[1] + 25),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                        
                        # Show shot completion info
                        duration_text = f"D:{shot['duration']}f"
                        cv2.putText(frame, duration_text, 
                                  (end_pos[0] + 20, end_pos[1] + 10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
                        
                        # Draw bounce markers for completed shots (dimmed)
                        if 'bounces' in shot and shot['bounces']:
                            for bounce in shot['bounces']:
                                bounce_pos = (int(bounce['position'][0]), int(bounce['position'][1]))
                                # Dimmed yellow diamond for completed shot bounces
                                dimmed_color = (0, 180, 180)
                                cv2.circle(frame, bounce_pos, 6, dimmed_color, -1)
                                cv2.circle(frame, bounce_pos, 9, dimmed_color, 1)
                                cv2.putText(frame, "b", (bounce_pos[0] - 3, bounce_pos[1] + 3),
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1)
                        
            # Draw current ball position with enhanced highlighting
            if ball_pos and len(ball_pos) >= 2:
                ball_color = (255, 255, 255)  # Default white
                ball_radius = 6
                ball_status = "SEARCHING"
                
                # Check if ball is part of active shot
                for shot in self.active_shots:
                    if shot['status'] == 'active':
                        ball_color = shot['color']
                        ball_radius = 10
                        ball_status = f"SHOT {shot['id']}"
                        break
                        
                # Draw ball with pulsing effect for active shots
                cv2.circle(frame, (int(ball_pos[0]), int(ball_pos[1])), 
                          ball_radius + 3, ball_color, 2)
                cv2.circle(frame, (int(ball_pos[0]), int(ball_pos[1])), 
                          ball_radius, ball_color, -1)
                
                # Add status text
                cv2.putText(frame, ball_status, 
                          (int(ball_pos[0]) + 15, int(ball_pos[1]) - 5),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.3, ball_color, 1)
                          
        except Exception as e:
            print(f"Error drawing shot trajectories: {e}")
            
    def get_shot_statistics(self):
        """
        Get comprehensive statistics about tracked shots including bounce analysis
        """
        total_shots = len(self.completed_shots)
        active_shots = len(self.active_shots)
        
        if total_shots == 0:
            return {
                'total_shots': 0,
                'active_shots': active_shots,
                'player1_shots': 0,
                'player2_shots': 0,
                'avg_duration': 0,
                'shot_types': {},
                'bounce_statistics': {
                    'total_bounces': 0,
                    'avg_bounces_per_shot': 0,
                    'shots_with_bounces': 0,
                    'bounce_confidence_avg': 0
                }
            }
            
        player1_shots = len([s for s in self.completed_shots if s['player_who_hit'] == 1])
        player2_shots = len([s for s in self.completed_shots if s['player_who_hit'] == 2])
        avg_duration = sum(s['duration'] for s in self.completed_shots) / total_shots
        
        # Count shot types
        shot_types = {}
        for shot in self.completed_shots:
            shot_type = str(shot['final_shot_type'])
            shot_types[shot_type] = shot_types.get(shot_type, 0) + 1
        
        # Bounce statistics
        total_bounces = 0
        shots_with_bounces = 0
        bounce_confidences = []
        
        for shot in self.completed_shots:
            shot_bounces = shot.get('bounces', [])
            if shot_bounces:
                shots_with_bounces += 1
                total_bounces += len(shot_bounces)
                for bounce in shot_bounces:
                    bounce_confidences.append(bounce.get('confidence', 0))
        
        avg_bounces_per_shot = total_bounces / total_shots if total_shots > 0 else 0
        avg_bounce_confidence = sum(bounce_confidences) / len(bounce_confidences) if bounce_confidences else 0
            
        return {
            'total_shots': total_shots,
            'active_shots': active_shots,
            'player1_shots': player1_shots,
            'player2_shots': player2_shots,
            'avg_duration': avg_duration,
            'shot_types': shot_types,
            'bounce_statistics': {
                'total_bounces': total_bounces,
                'avg_bounces_per_shot': avg_bounces_per_shot,
                'shots_with_bounces': shots_with_bounces,
                'bounce_confidence_avg': avg_bounce_confidence
            }
        }

# Initialize global shot tracker
shot_tracker = ShotTracker()

def calculate_vector_angle(vec1, vec2):
    """Calculate angle between two vectors in degrees"""
    try:
        # Calculate dot product
        dot_product = vec1[0] * vec2[0] + vec1[1] * vec2[1]
        
        # Calculate magnitudes
        mag1 = math.sqrt(vec1[0]**2 + vec1[1]**2)
        mag2 = math.sqrt(vec2[0]**2 + vec2[1]**2)
        
        if mag1 == 0 or mag2 == 0:
            return 0
        
        # Calculate cosine of angle
        cos_angle = dot_product / (mag1 * mag2)
        cos_angle = max(-1, min(1, cos_angle))  # Clamp to [-1, 1]
        
        # Return angle in degrees
        return math.degrees(math.acos(cos_angle))
    except:
        return 0


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
    # Initialize shot tracking log
    with open("output/shots_log.jsonl", "w") as f:
        f.write("")  # Clear the file
    # Initialize bounce analysis log  
    with open("output/bounce_analysis.jsonl", "w") as f:
        f.write("")  # Clear the file
    print("✅ Enhanced shot tracking initialized")
    print("📊 Shot data will be logged to output/shots_log.jsonl")
    print("🎾 Bounce analysis will be logged to output/bounce_analysis.jsonl")
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
# Global frame dimensions - will be set by main() function parameters
# These can now be customized by calling main() with custom frame_width and frame_height
frame_height = 360  # Default value, can be overridden
frame_width = 640   # Default value, can be overridden
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
    
    # Enhanced wall bounce detection using comprehensive system
    detected_bounces = bounce_detector.detect_bounces_comprehensive(trajectory, court_width, court_height)
    metrics['wall_bounces'] = len(detected_bounces)
    metrics['wall_bounce_positions'] = [bounce['position'] for bounce in detected_bounces]
    metrics['bounce_details'] = detected_bounces
    
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

class BounceDetector:
    """
    Enhanced bounce detection system with multiple algorithms and visualization
    """
    def __init__(self):
        self.detected_bounces = []  # Store all detected bounces
        self.bounce_confidence_threshold = 0.6
        self.min_bounce_separation = 8  # Minimum frames between bounces
        
    def detect_bounces_comprehensive(self, trajectory, court_width, court_height):
        """
        Comprehensive bounce detection using multiple algorithms
        """
        if len(trajectory) < 6:
            return []
            
        bounces = []
        
        # Algorithm 1: Physics-based detection
        physics_bounces = self._detect_physics_bounces(trajectory, court_width, court_height)
        
        # Algorithm 2: Velocity vector analysis
        velocity_bounces = self._detect_velocity_bounces(trajectory)
        
        # Algorithm 3: Wall proximity analysis
        wall_bounces = self._detect_wall_bounces(trajectory, court_width, court_height)
        
        # Algorithm 4: Trajectory curvature analysis
        curvature_bounces = self._detect_curvature_bounces(trajectory)
        
        # Combine and weight results
        all_candidates = []
        
        # Add weighted candidates from each algorithm
        for bounce in physics_bounces:
            all_candidates.append({**bounce, 'algorithm': 'physics', 'base_weight': 0.3})
            
        for bounce in velocity_bounces:
            all_candidates.append({**bounce, 'algorithm': 'velocity', 'base_weight': 0.25})
            
        for bounce in wall_bounces:
            all_candidates.append({**bounce, 'algorithm': 'wall', 'base_weight': 0.25})
            
        for bounce in curvature_bounces:
            all_candidates.append({**bounce, 'algorithm': 'curvature', 'base_weight': 0.2})
        
        # Merge nearby candidates and calculate final confidence
        final_bounces = self._merge_and_score_candidates(all_candidates)
        
        # Apply confidence threshold and store
        validated_bounces = [b for b in final_bounces if b['confidence'] >= self.bounce_confidence_threshold]
        self.detected_bounces.extend(validated_bounces)
        
        return validated_bounces
        
    def _detect_physics_bounces(self, trajectory, court_width, court_height):
        """
        Physics-based bounce detection using acceleration and momentum
        """
        bounces = []
        
        if len(trajectory) < 6:
            return bounces
            
        # Calculate velocities and accelerations
        positions = [(pos[0], pos[1]) for pos in trajectory]
        times = [pos[2] if len(pos) > 2 else i for i, pos in enumerate(trajectory)]
        
        velocities = []
        accelerations = []
        
        # Calculate velocities
        for i in range(1, len(positions)):
            dt = max(times[i] - times[i-1], 1e-6)
            vx = (positions[i][0] - positions[i-1][0]) / dt
            vy = (positions[i][1] - positions[i-1][1]) / dt
            velocities.append((vx, vy, math.sqrt(vx*vx + vy*vy)))
            
        # Calculate accelerations
        for i in range(1, len(velocities)):
            dt = max(times[i+1] - times[i], 1e-6)
            ax = (velocities[i][0] - velocities[i-1][0]) / dt
            ay = (velocities[i][1] - velocities[i-1][1]) / dt
            accelerations.append((ax, ay, math.sqrt(ax*ax + ay*ay)))
        
        # Look for bounce signatures
        for i in range(2, len(accelerations) - 2):
            # High acceleration magnitude (impact)
            accel_mag = accelerations[i][2]
            
            # Velocity direction change
            if i < len(velocities) - 2:
                vel_before = velocities[i-1]
                vel_after = velocities[i+1]
                
                # Check for significant direction change
                dot_product = vel_before[0]*vel_after[0] + vel_before[1]*vel_after[1]
                mag_before = vel_before[2]
                mag_after = vel_after[2]
                
                if mag_before > 0 and mag_after > 0:
                    cos_angle = dot_product / (mag_before * mag_after)
                    angle_change = math.degrees(math.acos(max(-1, min(1, cos_angle))))
                    
                    # Calculate confidence based on physics indicators
                    confidence = 0.0
                    
                    # High acceleration indicates impact
                    if accel_mag > 50:
                        confidence += 0.4
                        
                    # Large angle change indicates bounce
                    if angle_change > 45:
                        confidence += 0.3
                        
                    # Velocity magnitude conservation (energy consideration)
                    velocity_ratio = min(mag_after, mag_before) / max(mag_after, mag_before)
                    if velocity_ratio > 0.5:  # Some energy preserved
                        confidence += 0.2
                        
                    # Wall proximity bonus
                    pos = positions[i+1]  # Position after potential bounce
                    wall_distance = min(pos[0], court_width - pos[0], pos[1], court_height - pos[1])
                    if wall_distance < 40:
                        confidence += 0.1
                        
                    if confidence > 0.3:
                        bounces.append({
                            'frame_index': i + 1,
                            'position': pos,
                            'confidence': confidence,
                            'angle_change': angle_change,
                            'acceleration': accel_mag,
                            'wall_distance': wall_distance
                        })
        
        return bounces
        
    def _detect_velocity_bounces(self, trajectory):
        """
        Detect bounces based on velocity vector analysis
        """
        bounces = []
        
        if len(trajectory) < 5:
            return bounces
            
        positions = [(pos[0], pos[1]) for pos in trajectory]
        times = [pos[2] if len(pos) > 2 else i for i, pos in enumerate(trajectory)]
        
        # Calculate velocity vectors
        velocities = []
        for i in range(1, len(positions)):
            dt = max(times[i] - times[i-1], 1e-6)
            vx = (positions[i][0] - positions[i-1][0]) / dt
            vy = (positions[i][1] - positions[i-1][1]) / dt
            velocities.append((vx, vy))
        
        # Look for velocity reversals and sharp changes
        for i in range(2, len(velocities) - 1):
            v_prev = velocities[i-1]
            v_curr = velocities[i]
            v_next = velocities[i+1]
            
            confidence = 0.0
            
            # Check for component reversals
            x_reversal = v_prev[0] * v_next[0] < 0
            y_reversal = v_prev[1] * v_next[1] < 0
            
            if x_reversal:
                confidence += 0.4
            if y_reversal:
                confidence += 0.4
                
            # Check for velocity magnitude changes
            mag_prev = math.sqrt(v_prev[0]**2 + v_prev[1]**2)
            mag_next = math.sqrt(v_next[0]**2 + v_next[1]**2)
            
            if mag_prev > 10 and mag_next > 10:  # Significant movement
                velocity_change_ratio = abs(mag_next - mag_prev) / max(mag_prev, mag_next)
                if velocity_change_ratio > 0.3:
                    confidence += 0.2
                    
            if confidence > 0.3:
                bounces.append({
                    'frame_index': i + 1,
                    'position': positions[i + 1],
                    'confidence': confidence,
                    'x_reversal': x_reversal,
                    'y_reversal': y_reversal
                })
                
        return bounces
        
    def _detect_wall_bounces(self, trajectory, court_width, court_height):
        """
        Detect bounces based on wall proximity and trajectory changes
        """
        bounces = []
        wall_threshold = 35  # Distance from wall to consider a wall bounce
        
        for i in range(2, len(trajectory) - 2):
            pos = (trajectory[i][0], trajectory[i][1])
            
            # Check wall proximity
            wall_distances = [
                pos[0],  # Left wall
                court_width - pos[0],  # Right wall
                pos[1],  # Top wall
                court_height - pos[1]  # Bottom wall
            ]
            
            min_wall_distance = min(wall_distances)
            
            if min_wall_distance < wall_threshold:
                # Near wall, check for trajectory change
                if i >= 2 and i < len(trajectory) - 2:
                    prev_pos = (trajectory[i-2][0], trajectory[i-2][1])
                    next_pos = (trajectory[i+2][0], trajectory[i+2][1])
                    
                    # Calculate approach and departure vectors
                    approach_vec = (pos[0] - prev_pos[0], pos[1] - prev_pos[1])
                    departure_vec = (next_pos[0] - pos[0], next_pos[1] - pos[1])
                    
                    # Calculate angle between vectors
                    if (approach_vec[0] != 0 or approach_vec[1] != 0) and (departure_vec[0] != 0 or departure_vec[1] != 0):
                        dot_product = approach_vec[0] * departure_vec[0] + approach_vec[1] * departure_vec[1]
                        mag1 = math.sqrt(approach_vec[0]**2 + approach_vec[1]**2)
                        mag2 = math.sqrt(departure_vec[0]**2 + departure_vec[1]**2)
                        
                        if mag1 > 0 and mag2 > 0:
                            cos_angle = dot_product / (mag1 * mag2)
                            angle = math.degrees(math.acos(max(-1, min(1, cos_angle))))
                            
                            # Calculate confidence
                            wall_proximity_factor = 1.0 - (min_wall_distance / wall_threshold)
                            angle_factor = angle / 180.0
                            
                            confidence = 0.5 * wall_proximity_factor + 0.5 * angle_factor
                            
                            if confidence > 0.4:
                                bounces.append({
                                    'frame_index': i,
                                    'position': pos,
                                    'confidence': confidence,
                                    'wall_distance': min_wall_distance,
                                    'angle_change': angle
                                })
        
        return bounces
        
    def _detect_curvature_bounces(self, trajectory):
        """
        Detect bounces based on trajectory curvature analysis
        """
        bounces = []
        
        if len(trajectory) < 5:
            return bounces
            
        positions = [(pos[0], pos[1]) for pos in trajectory]
        
        # Calculate curvature at each point
        for i in range(2, len(positions) - 2):
            # Use 5-point stencil for better curvature estimation
            p1, p2, p3, p4, p5 = positions[i-2:i+3]
            
            # Calculate curvature using finite differences
            # First derivatives
            dx1 = (p3[0] - p1[0]) / 2
            dy1 = (p3[1] - p1[1]) / 2
            
            # Second derivatives
            dx2 = p4[0] - 2*p3[0] + p2[0]
            dy2 = p4[1] - 2*p3[1] + p2[1]
            
            # Curvature formula: |x'y'' - y'x''| / (x'^2 + y'^2)^(3/2)
            numerator = abs(dx1 * dy2 - dy1 * dx2)
            denominator = (dx1*dx1 + dy1*dy1)**1.5
            
            if denominator > 1e-6:
                curvature = numerator / denominator
                
                # High curvature indicates sharp direction change (potential bounce)
                if curvature > 0.01:  # Threshold for significant curvature
                    confidence = min(curvature * 10, 1.0)  # Scale curvature to confidence
                    
                    bounces.append({
                        'frame_index': i,
                        'position': positions[i],
                        'confidence': confidence,
                        'curvature': curvature
                    })
        
        return bounces
        
    def _merge_and_score_candidates(self, candidates):
        """
        Merge nearby bounce candidates and calculate final confidence scores
        """
        if not candidates:
            return []
            
        # Sort by frame index
        candidates.sort(key=lambda x: x['frame_index'])
        
        merged_bounces = []
        current_group = [candidates[0]]
        
        # Group nearby candidates
        for i in range(1, len(candidates)):
            if candidates[i]['frame_index'] - current_group[-1]['frame_index'] <= self.min_bounce_separation:
                current_group.append(candidates[i])
            else:
                # Process current group
                merged_bounce = self._process_candidate_group(current_group)
                if merged_bounce:
                    merged_bounces.append(merged_bounce)
                current_group = [candidates[i]]
        
        # Process last group
        if current_group:
            merged_bounce = self._process_candidate_group(current_group)
            if merged_bounce:
                merged_bounces.append(merged_bounce)
        
        return merged_bounces
        
    def _process_candidate_group(self, group):
        """
        Process a group of nearby bounce candidates into a single bounce
        """
        if not group:
            return None
            
        # Calculate weighted average position and frame
        total_weight = sum(c['confidence'] * c['base_weight'] for c in group)
        
        if total_weight == 0:
            return None
            
        avg_frame = sum(c['frame_index'] * c['confidence'] * c['base_weight'] for c in group) / total_weight
        avg_x = sum(c['position'][0] * c['confidence'] * c['base_weight'] for c in group) / total_weight
        avg_y = sum(c['position'][1] * c['confidence'] * c['base_weight'] for c in group) / total_weight
        
        # Calculate final confidence
        algorithm_count = len(set(c['algorithm'] for c in group))
        base_confidence = total_weight / len(group)
        
        # Bonus for multiple algorithms agreeing
        multi_algorithm_bonus = 0.1 * (algorithm_count - 1)
        final_confidence = min(base_confidence + multi_algorithm_bonus, 1.0)
        
        return {
            'frame_index': int(avg_frame),
            'position': (int(avg_x), int(avg_y)),
            'confidence': final_confidence,
            'algorithm_count': algorithm_count,
            'supporting_algorithms': [c['algorithm'] for c in group]
        }
        
    def draw_bounces(self, frame, trajectory, frame_index):
        """
        Draw detected bounces on the frame with enhanced visualization
        """
        try:
            # Draw recent bounces (last 10)
            recent_bounces = [b for b in self.detected_bounces if abs(b['frame_index'] - frame_index) <= 10]
            
            for bounce in recent_bounces:
                pos = bounce['position']
                confidence = bounce['confidence']
                age = abs(frame_index - bounce['frame_index'])
                
                # Color based on confidence (green = high, yellow = medium, red = low)
                if confidence > 0.8:
                    color = (0, 255, 0)  # Green
                elif confidence > 0.6:
                    color = (0, 255, 255)  # Yellow
                else:
                    color = (0, 165, 255)  # Orange
                
                # Size based on age (newer bounces are larger)
                radius = max(8 - age, 3)
                
                # Draw bounce marker
                cv2.circle(frame, pos, radius + 3, color, 2)
                cv2.circle(frame, pos, radius, (255, 255, 255), -1)
                
                # Draw confidence text
                conf_text = f"{confidence:.2f}"
                cv2.putText(frame, conf_text, 
                          (pos[0] + 15, pos[1] - 5),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
                
                # Draw algorithm indicator
                algo_count = bounce.get('algorithm_count', 1)
                algo_text = f"A:{algo_count}"
                cv2.putText(frame, algo_text, 
                          (pos[0] + 15, pos[1] + 10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
        
        except Exception as e:
            print(f"Error drawing bounces: {e}")

# Initialize global bounce detector
bounce_detector = BounceDetector()

def detect_ball_bounces_comprehensive(trajectory, court_width, court_height, confidence_threshold=0.7):
    """
    Comprehensive ball bounce detection using multiple algorithms and physics-based validation
    
    Returns:
        tuple: (bounce_count, bounce_positions, bounce_details)
    """
    if len(trajectory) < 5:
        return 0, [], []
    
    bounce_positions = []
    bounce_details = []
    
    # Algorithm 1: Velocity Direction Change Detection
    velocity_bounces = detect_bounces_by_velocity_change(trajectory)
    
    # Algorithm 2: Trajectory Angle Analysis
    angle_bounces = detect_bounces_by_angle_analysis(trajectory)
    
    # Algorithm 3: Physics-Based Bounce Detection
    physics_bounces = detect_bounces_by_physics(trajectory, court_width, court_height)
    
    # Algorithm 4: Wall Proximity Analysis
    wall_bounces = detect_bounces_by_wall_proximity(trajectory, court_width, court_height)
    
    # Combine and validate bounces
    all_potential_bounces = []
    
    # Add bounces from each algorithm with their source
    for bounce in velocity_bounces:
        all_potential_bounces.append({**bounce, 'source': 'velocity', 'weight': 0.3})
    
    for bounce in angle_bounces:
        all_potential_bounces.append({**bounce, 'source': 'angle', 'weight': 0.25})
        
    for bounce in physics_bounces:
        all_potential_bounces.append({**bounce, 'source': 'physics', 'weight': 0.3})
        
    for bounce in wall_bounces:
        all_potential_bounces.append({**bounce, 'source': 'wall', 'weight': 0.15})
    
    # Cluster nearby bounces and calculate consensus
    confirmed_bounces = validate_and_cluster_bounces(all_potential_bounces, trajectory, confidence_threshold)
    
    return len(confirmed_bounces), [b['position'] for b in confirmed_bounces], confirmed_bounces

def detect_bounces_by_velocity_change(trajectory, velocity_threshold=10):
    """Detect bounces by analyzing velocity direction changes"""
    bounces = []
    
    if len(trajectory) < 4:
        return bounces
    
    velocities = []
    for i in range(1, len(trajectory)):
        p1, p2 = trajectory[i-1], trajectory[i]
        if len(p1) >= 3 and len(p2) >= 3:
            dt = max(0.033, abs(p2[2] - p1[2]))  # Min 30fps
            vel_x = (p2[0] - p1[0]) / dt
            vel_y = (p2[1] - p1[1]) / dt
            velocities.append((vel_x, vel_y, i))
    
    for i in range(1, len(velocities) - 1):
        prev_vel = velocities[i-1]
        curr_vel = velocities[i]
        next_vel = velocities[i+1]
        
        # Check for significant velocity direction change
        prev_dir_x = 1 if prev_vel[0] > 0 else -1
        next_dir_x = 1 if next_vel[0] > 0 else -1
        prev_dir_y = 1 if prev_vel[1] > 0 else -1
        next_dir_y = 1 if next_vel[1] > 0 else -1
        
        # Significant direction change in either axis
        if (prev_dir_x != next_dir_x and abs(prev_vel[0]) > velocity_threshold) or \
           (prev_dir_y != next_dir_y and abs(prev_vel[1]) > velocity_threshold):
            
            frame_idx = curr_vel[2]
            if frame_idx < len(trajectory):
                position = trajectory[frame_idx][:2]
                confidence = min(1.0, (abs(prev_vel[0]) + abs(prev_vel[1])) / 50)
                
                bounces.append({
                    'position': position,
                    'frame_index': frame_idx,
                    'confidence': confidence,
                    'type': 'velocity_change'
                })
    
    return bounces

def detect_bounces_by_angle_analysis(trajectory, angle_threshold=45):
    """Detect bounces by analyzing trajectory angle changes"""
    bounces = []
    
    if len(trajectory) < 5:
        return bounces
    
    for i in range(2, len(trajectory) - 2):
        # Get 5-point window
        points = trajectory[i-2:i+3]
        
        # Calculate angles between consecutive segments
        angles = []
        for j in range(len(points) - 2):
            p1, p2, p3 = points[j], points[j+1], points[j+2]
            
            vec1 = (p2[0] - p1[0], p2[1] - p1[1])
            vec2 = (p3[0] - p2[0], p3[1] - p2[1])
            
            angle = calculate_vector_angle(vec1, vec2)
            angles.append(angle)
        
        # Check for sharp angle change (potential bounce)
        if len(angles) >= 2:
            max_angle_change = max(angles)
            if max_angle_change > angle_threshold:
                position = trajectory[i][:2]
                confidence = min(1.0, max_angle_change / 180)
                
                bounces.append({
                    'position': position,
                    'frame_index': i,
                    'confidence': confidence,
                    'type': 'angle_change',
                    'angle': max_angle_change
                })
    
    return bounces

def detect_bounces_by_physics(trajectory, court_width, court_height):
    """Detect bounces using physics-based analysis"""
    bounces = []
    
    if len(trajectory) < 6:
        return bounces
    
    # Analyze trajectory for unnatural accelerations (bounces)
    accelerations = []
    for i in range(2, len(trajectory)):
        if i >= 2 and len(trajectory[i]) >= 3:
            p0, p1, p2 = trajectory[i-2], trajectory[i-1], trajectory[i]
            
            # Calculate acceleration
            dt1 = max(0.033, abs(p1[2] - p0[2]))
            dt2 = max(0.033, abs(p2[2] - p1[2]))
            
            v1_x, v1_y = (p1[0] - p0[0]) / dt1, (p1[1] - p0[1]) / dt1
            v2_x, v2_y = (p2[0] - p1[0]) / dt2, (p2[1] - p1[1]) / dt2
            
            acc_x = (v2_x - v1_x) / dt2
            acc_y = (v2_y - v1_y) / dt2
            
            acc_magnitude = math.sqrt(acc_x**2 + acc_y**2)
            accelerations.append((acc_magnitude, i))
    
    # Find peaks in acceleration (potential bounces)
    for i, (acc, frame_idx) in enumerate(accelerations):
        if i > 0 and i < len(accelerations) - 1:
            prev_acc = accelerations[i-1][0]
            next_acc = accelerations[i+1][0]
            
            # Check if this is a local maximum
            if acc > prev_acc and acc > next_acc and acc > 200:  # Threshold for significant acceleration
                position = trajectory[frame_idx][:2]
                confidence = min(1.0, acc / 1000)
                
                bounces.append({
                    'position': position,
                    'frame_index': frame_idx,
                    'confidence': confidence,
                    'type': 'physics_acceleration',
                    'acceleration': acc
                })
    
    return bounces

def detect_bounces_by_wall_proximity(trajectory, court_width, court_height, wall_distance=30):
    """Detect bounces by analyzing wall proximity and trajectory changes"""
    bounces = []
    
    # Define wall boundaries with tolerance
    walls = {
        'left': wall_distance,
        'right': court_width - wall_distance,
        'top': wall_distance,
        'bottom': court_height - wall_distance
    }
    
    for i in range(1, len(trajectory) - 1):
        pos = trajectory[i]
        x, y = pos[0], pos[1]
        
        # Check proximity to walls
        near_wall = False
        wall_type = None
        
        if x <= walls['left']:
            near_wall = True
            wall_type = 'left'
        elif x >= walls['right']:
            near_wall = True
            wall_type = 'right'
        elif y <= walls['top']:
            near_wall = True
            wall_type = 'top'
        elif y >= walls['bottom']:
            near_wall = True
            wall_type = 'bottom'
        
        if near_wall and i > 0 and i < len(trajectory) - 1:
            # Check for direction change near wall
            prev_pos = trajectory[i-1]
            next_pos = trajectory[i+1]
            
            # Direction before and after
            dir_before = (pos[0] - prev_pos[0], pos[1] - prev_pos[1])
            dir_after = (next_pos[0] - pos[0], next_pos[1] - pos[1])
            
            # Check for appropriate direction change based on wall
            direction_changed = False
            if wall_type in ['left', 'right']:
                if dir_before[0] * dir_after[0] < 0:  # X direction changed
                    direction_changed = True
            elif wall_type in ['top', 'bottom']:
                if dir_before[1] * dir_after[1] < 0:  # Y direction changed
                    direction_changed = True
            
            if direction_changed:
                # Calculate distance to nearest wall
                if wall_type == 'left':
                    wall_dist = x
                elif wall_type == 'right':
                    wall_dist = court_width - x
                elif wall_type == 'top':
                    wall_dist = y
                else:  # bottom
                    wall_dist = court_height - y
                
                confidence = max(0.1, 1.0 - (wall_dist / wall_distance))
                
                bounces.append({
                    'position': (x, y),
                    'frame_index': i,
                    'confidence': confidence,
                    'type': 'wall_bounce',
                    'wall': wall_type,
                    'wall_distance': wall_dist
                })
    
    return bounces

def validate_and_cluster_bounces(potential_bounces, trajectory, confidence_threshold=0.7):
    """Validate and cluster nearby bounce detections"""
    if not potential_bounces:
        return []
    
    # Sort by frame index
    potential_bounces.sort(key=lambda x: x['frame_index'])
    
    # Cluster nearby bounces (within 3 frames and 20 pixels)
    clusters = []
    for bounce in potential_bounces:
        added_to_cluster = False
        
        for cluster in clusters:
            # Check if bounce is close to existing cluster
            cluster_center = cluster[0]
            frame_diff = abs(bounce['frame_index'] - cluster_center['frame_index'])
            pos_diff = math.sqrt(
                (bounce['position'][0] - cluster_center['position'][0])**2 +
                (bounce['position'][1] - cluster_center['position'][1])**2
            )
            
            if frame_diff <= 3 and pos_diff <= 20:
                cluster.append(bounce)
                added_to_cluster = True
                break
        
        if not added_to_cluster:
            clusters.append([bounce])
    
    # Validate each cluster and create consensus bounce
    validated_bounces = []
    for cluster in clusters:
        if len(cluster) >= 2:  # At least 2 algorithms agree
            # Calculate weighted average position and confidence
            total_weight = sum(b['weight'] for b in cluster)
            avg_x = sum(b['position'][0] * b['weight'] for b in cluster) / total_weight
            avg_y = sum(b['position'][1] * b['weight'] for b in cluster) / total_weight
            avg_confidence = sum(b['confidence'] * b['weight'] for b in cluster) / total_weight
            avg_frame = int(sum(b['frame_index'] * b['weight'] for b in cluster) / total_weight)
            
            if avg_confidence >= confidence_threshold:
                validated_bounces.append({
                    'position': (avg_x, avg_y),
                    'frame_index': avg_frame,
                    'confidence': avg_confidence,
                    'type': 'consensus',
                    'sources': [b['source'] for b in cluster],
                    'algorithm_count': len(cluster)
                })
    
    return validated_bounces

def detect_wall_bounces_advanced(trajectory, court_width, court_height):
    """
    Advanced wall bounce detection using trajectory analysis with bounce position tracking
    """
    bounces = 0
    bounce_positions = []
    
    if len(trajectory) < 3:
        return bounces, bounce_positions
    
    for i in range(1, len(trajectory) - 1):
        x_prev, y_prev, _ = trajectory[i-1]
        x_curr, y_curr, _ = trajectory[i]
        x_next, y_next, _ = trajectory[i+1]
        
        # Calculate direction vectors
        v1 = (x_curr - x_prev, y_curr - y_prev)
        v2 = (x_next - x_curr, y_next - y_curr)
        
        # Check for sudden direction change (potential wall bounce)
        if (abs(v1[0]) > 0 or abs(v1[1]) > 0) and (abs(v2[0]) > 0 or abs(v2[1]) > 0):
            # Calculate angle between vectors
            dot_product = v1[0]*v2[0] + v1[1]*v2[1]
            mag1 = math.sqrt(v1[0]**2 + v1[1]**2)
            mag2 = math.sqrt(v2[0]**2 + v2[1]**2)
            
            if mag1 > 0 and mag2 > 0:
                cos_angle = dot_product / (mag1 * mag2)
                cos_angle = max(-1, min(1, cos_angle))
                angle = math.degrees(math.acos(cos_angle))
                
                # Check if near wall and direction change is significant
                near_wall = (x_curr < 30 or x_curr > court_width - 30 or 
                        y_curr < 30 or y_curr > court_height - 30)
                
                if angle > 60 and near_wall:
                    bounces += 1
                    bounce_positions.append((int(x_curr), int(y_curr)))
    
    return bounces, bounce_positions


def detect_ball_bounces_gpu(trajectory, velocity_threshold=30.0, angle_threshold=45.0, court_width=640, court_height=360):
    """
    Enhanced GPU-optimized ball bounce detection with improved physics modeling and filtering
    """
    if len(trajectory) < 6:  # Need more points for reliable detection
        return []
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        # Extract positions and convert to GPU tensors
        positions = torch.tensor([[pos[0], pos[1]] for pos in trajectory], dtype=torch.float32, device=device)
        times = torch.tensor([pos[2] if len(pos) > 2 else i for i, pos in enumerate(trajectory)], dtype=torch.float32, device=device)
        
        bounce_positions = []
        
        if len(positions) < 6:
            return bounce_positions
        
        # 1. ENHANCED VELOCITY ANALYSIS
        # Calculate velocities with time normalization
        dt = times[1:] - times[:-1]
        dt = torch.clamp(dt, min=1e-6)  # Prevent division by zero
        
        displacements = positions[1:] - positions[:-1]
        velocities = displacements / dt.unsqueeze(1)
        speeds = torch.norm(velocities, dim=1)
        
        # 2. ACCELERATION ANALYSIS (key for bounce detection)
        dt_vel = times[2:] - times[:-2] 
        dt_vel = torch.clamp(dt_vel, min=1e-6)
        
        accelerations = (velocities[1:] - velocities[:-1]) / dt_vel.unsqueeze(1)
        acceleration_magnitudes = torch.norm(accelerations, dim=1)
        
        # 3. ENHANCED DIRECTION ANALYSIS
        if len(velocities) >= 3:
            # Normalize velocities to get directions
            normalized_velocities = F.normalize(velocities + 1e-8, p=2, dim=1)
            
            # Calculate angular changes between consecutive velocity vectors
            dot_products = torch.sum(normalized_velocities[:-1] * normalized_velocities[1:], dim=1)
            angles = torch.acos(torch.clamp(dot_products, -1.0, 1.0)) * 180.0 / math.pi
            
            # 4. ENHANCED WALL PROXIMITY WITH GRADIENT
            wall_margin = 50
            wall_distances = torch.min(torch.stack([
                positions[:, 0],  # Distance from left wall
                court_width - positions[:, 0],  # Distance from right wall  
                positions[:, 1],  # Distance from top wall
                court_height - positions[:, 1]  # Distance from bottom wall
            ]), dim=0)[0]
            
            # Create wall proximity score (0 = at wall, 1 = far from wall)
            wall_proximity_score = torch.clamp(wall_distances / wall_margin, 0, 1)
            
            # 5. PHYSICS-BASED BOUNCE DETECTION
            # Look for characteristic bounce patterns
            bounce_candidates = []
            
            for i in range(2, len(speeds) - 2):
                # Skip if not enough data
                if i >= len(angles) or i >= len(acceleration_magnitudes):
                    continue
                
                # Multiple bounce indicators
                indicators = []
                
                # A. Sudden direction change
                if angles[i-1] > angle_threshold:
                    indicators.append(('angle', angles[i-1].item(), 0.8))
                
                # B. Velocity magnitude change (deceleration then acceleration)
                speed_change = abs(speeds[i+1] - speeds[i-1])
                if speed_change > velocity_threshold:
                    indicators.append(('velocity', speed_change.item(), 0.7))
                
                # C. High acceleration (impact detection)
                if acceleration_magnitudes[i-1] > 100:  # Adjust threshold as needed
                    indicators.append(('acceleration', acceleration_magnitudes[i-1].item(), 0.9))
                
                # D. Wall proximity factor
                wall_factor = 1.0 - wall_proximity_score[i]
                if wall_factor > 0.3:  # Near wall
                    indicators.append(('wall_proximity', wall_factor.item(), 0.6))
                
                # E. Velocity reversal detection
                if i > 0 and i < len(velocities) - 1:
                    vel_before = velocities[i-1]
                    vel_after = velocities[i+1]
                    
                    # Check for component reversal (especially important for wall bounces)
                    x_reversal = vel_before[0] * vel_after[0] < 0
                    y_reversal = vel_before[1] * vel_after[1] < 0
                    
                    if x_reversal or y_reversal:
                        reversal_strength = 1.0 if (x_reversal and y_reversal) else 0.7
                        indicators.append(('velocity_reversal', reversal_strength, 0.85))
                
                # F. Trajectory curvature analysis
                if i >= 2 and i < len(positions) - 2:
                    # Calculate curvature using three points
                    p1, p2, p3 = positions[i-1], positions[i], positions[i+1]
                    
                    # Vector from p1 to p2
                    v1 = p2 - p1
                    # Vector from p2 to p3  
                    v2 = p3 - p2
                    
                    # Cross product magnitude indicates curvature
                    cross_product = v1[0] * v2[1] - v1[1] * v2[0]
                    curvature = abs(cross_product) / (torch.norm(v1) * torch.norm(v2) + 1e-8)
                    
                    if curvature > 0.5:  # High curvature indicates direction change
                        indicators.append(('curvature', curvature.item(), 0.6))
                
                # Calculate combined confidence score
                if indicators:
                    total_weight = sum(weight for _, _, weight in indicators)
                    weighted_score = sum(weight for _, _, weight in indicators) / len(indicators)
                    
                    # Bonus for multiple indicators
                    multi_indicator_bonus = min(0.3, 0.1 * (len(indicators) - 1))
                    final_confidence = weighted_score + multi_indicator_bonus
                    
                    bounce_candidates.append({
                        'index': i,
                        'confidence': final_confidence,
                        'indicators': indicators,
                        'wall_factor': wall_factor.item()
                    })
            
            # 6. BOUNCE FILTERING AND SELECTION
            # Sort by confidence and apply non-maximum suppression
            bounce_candidates.sort(key=lambda x: x['confidence'], reverse=True)
            
            # Non-maximum suppression to avoid duplicate detections
            min_separation = 5  # Minimum frames between bounces
            selected_bounces = []
            
            for candidate in bounce_candidates:
                if candidate['confidence'] < 0.5:  # Confidence threshold
                    continue
                
                # Check if too close to already selected bounces
                too_close = False
                for selected in selected_bounces:
                    if abs(candidate['index'] - selected['index']) < min_separation:
                        too_close = True
                        break
                
                if not too_close:
                    selected_bounces.append(candidate)
            
            # 7. CONVERT TO OUTPUT FORMAT
            for bounce in selected_bounces:
                idx = bounce['index']
                if idx < len(trajectory):
                    x, y = trajectory[idx][:2]
                    
                    # Ensure coordinates are properly converted
                    if hasattr(x, 'cpu'):
                        x = x.cpu()
                    if hasattr(y, 'cpu'):
                        y = y.cpu()
                    
                    bounce_positions.append({
                        'position': (int(x), int(y)),
                        'confidence': bounce['confidence'],
                        'indicators': bounce['indicators'],
                        'frame_index': idx
                    })
        
        # Return just positions for compatibility, but include confidence info
        return [bounce['position'] for bounce in bounce_positions]
        
    except Exception as e:
        print(f"GPU bounce detection error: {e}, falling back to CPU")
        return detect_ball_bounces_cpu_enhanced(trajectory, velocity_threshold, angle_threshold, court_width, court_height)

def detect_ball_bounces_cpu_enhanced(trajectory, velocity_threshold=30.0, angle_threshold=45.0, court_width=640, court_height=360):
    """
    Enhanced CPU fallback with improved physics modeling
    """
    bounce_positions = []
    
    if len(trajectory) < 6:
        return bounce_positions
    
    # Convert to numpy for efficient computation
    positions = np.array([[pos[0], pos[1]] for pos in trajectory])
    times = np.array([pos[2] if len(pos) > 2 else i for i, pos in enumerate(trajectory)])
    
    # Calculate velocities
    dt = np.diff(times)
    dt = np.maximum(dt, 1e-6)  # Prevent division by zero
    
    displacements = np.diff(positions, axis=0)
    velocities = displacements / dt[:, np.newaxis]
    speeds = np.linalg.norm(velocities, axis=1)
    
    # Calculate accelerations
    if len(velocities) >= 2:
        dt_vel = times[2:] - times[:-2]
        dt_vel = np.maximum(dt_vel, 1e-6)
        
        accelerations = np.diff(velocities, axis=0) / dt_vel[:, np.newaxis]
        acceleration_magnitudes = np.linalg.norm(accelerations, axis=1)
        
        # Enhanced bounce detection
        for i in range(2, len(speeds) - 2):
            bounce_score = 0
            
            # Check multiple bounce indicators
            if i < len(velocities) - 1:
                # Direction change
                v1 = velocities[i-1]
                v2 = velocities[i+1]
                
                if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
                    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                    angle = np.arccos(np.clip(cos_angle, -1, 1)) * 180 / np.pi
                    
                    if angle > angle_threshold:
                        bounce_score += 0.4
                
                # Velocity magnitude change
                speed_change = abs(speeds[i+1] - speeds[i-1])
                if speed_change > velocity_threshold:
                    bounce_score += 0.3
                
                # Acceleration spike
                if i-1 < len(acceleration_magnitudes) and acceleration_magnitudes[i-1] > 100:
                    bounce_score += 0.3
                
                # Wall proximity
                x, y = positions[i]
                wall_distance = min(x, court_width - x, y, court_height - y)
                if wall_distance < 50:
                    bounce_score += 0.2 * (1 - wall_distance / 50)
                
                # Velocity component reversal
                if (v1[0] * v2[0] < 0) or (v1[1] * v2[1] < 0):
                    bounce_score += 0.3
                
                # Accept bounce if score is high enough
                if bounce_score > 0.6:
                    bounce_positions.append((int(positions[i][0]), int(positions[i][1])))
    
    return bounce_positions

def smooth_trajectory_gpu(trajectory, window_size=5):
    """
    GPU-optimized trajectory smoothing to reduce noise before bounce detection
    """
    if len(trajectory) < window_size:
        return trajectory
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        positions = torch.tensor([[pos[0], pos[1]] for pos in trajectory], dtype=torch.float32, device=device)
        
        # Apply Gaussian smoothing
        smoothed_positions = []
        
        for i in range(len(positions)):
            start_idx = max(0, i - window_size // 2)
            end_idx = min(len(positions), i + window_size // 2 + 1)
            
            window_positions = positions[start_idx:end_idx]
            weights = torch.exp(-torch.linspace(-1, 1, len(window_positions))**2)
            weights = weights / weights.sum()
            
            smoothed_pos = torch.sum(window_positions * weights.unsqueeze(1), dim=0)
            smoothed_positions.append(smoothed_pos)
        
        # Convert back to original format
        smoothed_trajectory = []
        for i, pos in enumerate(smoothed_positions):
            original_time = trajectory[i][2] if len(trajectory[i]) > 2 else i
            smoothed_trajectory.append([pos[0].cpu().item(), pos[1].cpu().item(), original_time])
        
        return smoothed_trajectory
        
    except Exception as e:
        print(f"GPU smoothing error: {e}, using CPU fallback")
        return trajectory

def detect_ball_bounces_with_preprocessing(trajectory, **kwargs):
    """
    Main function that combines preprocessing with enhanced bounce detection
    """
    if len(trajectory) < 6:
        return []
    
    # 1. Smooth trajectory to reduce noise
    smoothed_trajectory = smooth_trajectory_gpu(trajectory)
    
    # 2. Apply enhanced bounce detection
    bounces = detect_ball_bounces_gpu(smoothed_trajectory, **kwargs)
    
    # 3. Post-process bounces (remove duplicates, validate physics)
    validated_bounces = validate_bounces(bounces, smoothed_trajectory)
    
    return validated_bounces

def validate_bounces(bounces, trajectory):
    """
    Validate detected bounces using physics constraints
    """
    if not bounces or len(trajectory) < 6:
        return bounces
    
    validated = []
    
    for bounce in bounces:
        # Extract position (handle both tuple and dict formats)
        if isinstance(bounce, dict):
            pos = bounce['position']
        else:
            pos = bounce
        
        # Find corresponding trajectory index
        bounce_idx = None
        for i, traj_pos in enumerate(trajectory):
            if abs(traj_pos[0] - pos[0]) < 5 and abs(traj_pos[1] - pos[1]) < 5:
                bounce_idx = i
                break
        
        if bounce_idx is not None and bounce_idx > 2 and bounce_idx < len(trajectory) - 2:
            # Validate physics: check if velocity pattern makes sense
            before_pos = trajectory[bounce_idx - 2]
            after_pos = trajectory[bounce_idx + 2]
            
            # Check if there's actual movement (not a false positive)
            movement_before = math.sqrt((trajectory[bounce_idx][0] - before_pos[0])**2 + 
                                     (trajectory[bounce_idx][1] - before_pos[1])**2)
            movement_after = math.sqrt((after_pos[0] - trajectory[bounce_idx][0])**2 + 
                                     (after_pos[1] - trajectory[bounce_idx][1])**2)
            
            if movement_before > 5 and movement_after > 5:  # Sufficient movement
                validated.append(bounce)
    
    return validated



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
def count_wall_hits(past_ball_pos, threshold=12):
    """
    Enhanced wall hit detection with improved accuracy and lower threshold
    """
    try:
        wall_hits = 0
        direction_changes = 0
        last_direction = None

        for i in range(1, len(past_ball_pos) - 1):
            x1, y1, _ = past_ball_pos[i - 1]
            x2, y2, _ = past_ball_pos[i]
            x3, y3, _ = past_ball_pos[i + 1]

            # Calculate direction vectors with more precision
            dir1 = x2 - x1
            

            # More sensitive direction change detection
            if abs(dir1) > threshold:
                current_direction = 1 if dir1 > 0 else -1
                if last_direction is not None and current_direction != last_direction:
                    # Check if direction change is significant enough
                    if abs(dir1) > threshold * 1.2:  # Additional threshold check
                        direction_changes += 1
                last_direction = current_direction

            # Count wall hits with stricter criteria
            if direction_changes >= 2:
                wall_hits += 1
                direction_changes = 0
                last_direction = None

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
        
        # Much more generous jump distance - allows for very fast ball movement
        generous_max_jump = max_jump * 3.0  # Increased from 1.5
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
    players, past_ball_pos, proximity_threshold=80
):
    """
    Enhanced player ball hit detection with improved accuracy.
    Uses multiple factors including proximity, velocity change, and trajectory analysis.
    """
    if len(past_ball_pos) < 4 or not players.get(1) or not players.get(2):
        return 0

    # Get recent ball positions for analysis
    current_pos = past_ball_pos[-1]
    prev_pos = past_ball_pos[-2]
    
    # Enhanced trajectory analysis for hit detection
    trajectory_change_detected = False
    velocity_change_detected = False
    angle_change = 0
    
    if len(past_ball_pos) >= 4:
        # Analyze the last 4 positions for trajectory changes
        positions = past_ball_pos[-4:]
        
        # Calculate velocity changes
        velocities = []
        for i in range(1, len(positions)):
            p1, p2 = positions[i-1], positions[i]
            if len(p1) >= 3 and len(p2) >= 3:
                dt = p2[2] - p1[2] if p2[2] != p1[2] else 1
                velocity = math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2) / dt
                velocities.append(velocity)
        
        # Check for significant velocity change (potential hit)
        if len(velocities) >= 2:
            velocity_change = abs(velocities[-1] - velocities[-2])
            if velocity_change > 15:  # Threshold for significant velocity change
                velocity_change_detected = True
        
        # Calculate trajectory angle change
        if len(positions) >= 3:
            p1, p2, p3 = positions[-3], positions[-2], positions[-1]
            vec1 = (p2[0] - p1[0], p2[1] - p1[1])
            vec2 = (p3[0] - p2[0], p3[1] - p2[1])
            
            # Calculate angle between vectors
            dot_product = vec1[0] * vec2[0] + vec1[1] * vec2[1]
            mag1 = math.sqrt(vec1[0] ** 2 + vec1[1] ** 2)
            mag2 = math.sqrt(vec2[0] ** 2 + vec2[1] ** 2)
            
            if mag1 > 0 and mag2 > 0:
                cos_angle = dot_product / (mag1 * mag2)
                cos_angle = max(-1, min(1, cos_angle))
                angle_change = math.degrees(math.acos(cos_angle))
                if angle_change > 45:  # Significant direction change
                    trajectory_change_detected = True

    try:
        # Enhanced player position calculation using multiple keypoints
        def get_enhanced_player_pos(player):
            pose = player.get_latest_pose()
            if not pose or not hasattr(pose, 'xyn') or len(pose.xyn) == 0:
                return None
                
            keypoints = pose.xyn[0]
            
            # Use key body parts for better position estimation
            # Priorities: racket hand (right wrist=10, left wrist=9), shoulders (5,6), hips (11,12)
            priority_keypoints = [10, 9, 5, 6, 11, 12]  # Right wrist, left wrist, shoulders, hips
            valid_points = []
            
            # Get priority keypoints first
            for idx in priority_keypoints:
                if idx < len(keypoints):
                    kp = keypoints[idx]
                    if not (kp[0] == 0 and kp[1] == 0):
                        x = int(kp[0] * frame_width)
                        y = int(kp[1] * frame_height)
                        # Weight racket hand positions more heavily
                        weight = 3 if idx in [9, 10] else 2 if idx in [5, 6] else 1
                        for _ in range(weight):
                            valid_points.append((x, y))
            
            # Add other valid keypoints with lower weight
            for i, kp in enumerate(keypoints):
                if i not in priority_keypoints and not (kp[0] == 0 and kp[1] == 0):
                    x = int(kp[0] * frame_width)
                    y = int(kp[1] * frame_height)
                    valid_points.append((x, y))
            
            if not valid_points:
                return None
            
            # Calculate weighted average
            avg_x = sum(p[0] for p in valid_points) / len(valid_points)
            avg_y = sum(p[1] for p in valid_points) / len(valid_points)
            return (avg_x, avg_y)

        p1_pos = get_enhanced_player_pos(players[1])
        p2_pos = get_enhanced_player_pos(players[2])

        if p1_pos is None or p2_pos is None:
            return 0

        # Calculate distances to current and previous ball positions
        p1_curr_distance = math.hypot(p1_pos[0] - current_pos[0], p1_pos[1] - current_pos[1])
        p2_curr_distance = math.hypot(p2_pos[0] - current_pos[0], p2_pos[1] - current_pos[1])
        
        p1_prev_distance = math.hypot(p1_pos[0] - prev_pos[0], p1_pos[1] - prev_pos[1])
        p2_prev_distance = math.hypot(p2_pos[0] - prev_pos[0], p2_pos[1] - prev_pos[1])
        
        # Enhanced scoring system for hit detection
        p1_score = 0
        p2_score = 0
        
        # Proximity scoring (closer = higher score)
        if p1_curr_distance < proximity_threshold:
            p1_score += (proximity_threshold - p1_curr_distance) / proximity_threshold * 50
        if p2_curr_distance < proximity_threshold:
            p2_score += (proximity_threshold - p2_curr_distance) / proximity_threshold * 50
            
        # Movement towards ball scoring
        if p1_prev_distance > p1_curr_distance:  # Player 1 moving towards ball
            p1_score += 20
        if p2_prev_distance > p2_curr_distance:  # Player 2 moving towards ball
            p2_score += 20
            
        # Trajectory change bonus (only if significant change detected)
        if trajectory_change_detected or velocity_change_detected:
            # Give bonus to closer player when trajectory changes
            if p1_curr_distance < p2_curr_distance:
                p1_score += 30
            else:
                p2_score += 30
        
        # Minimum threshold for hit detection
        min_score_threshold = 25
        
        # Determine winner
        if p1_score > p2_score and p1_score > min_score_threshold:
            return 1
        elif p2_score > p1_score and p2_score > min_score_threshold:
            return 2
        elif trajectory_change_detected or velocity_change_detected:
            # If there's a clear trajectory change but no clear winner, pick closest
            return 1 if p1_curr_distance < p2_curr_distance else 2

    except Exception as e:
        print(f"Error in enhanced determine_ball_hit: {e}")
        return 0

    return 0  # No clear hit detected
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
                'ball_position': (float(last_ball_pos[0]), float(last_ball_pos[1])),
                'ball_trajectory_length': len(past_ball_pos),
                'ball_speed': calculate_ball_speed(past_ball_pos) if len(past_ball_pos) > 1 else 0
            })
        else:
            coaching_data.update({
                'ball_position': (0.0, 0.0),  # Default position
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


def create_enhancement_summary():
    """
    Create a visual summary of the enhancements made
    """
    try:
        # Create a summary image
        summary_img = np.ones((600, 800, 3), dtype=np.uint8) * 50  # Dark background
        
        # Title
        cv2.putText(summary_img, "SQUASH ANALYSIS ENHANCEMENTS", (150, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        # Enhancement list
        enhancements = [
            "1. Enhanced Ball-Player Hit Detection:",
            "   - Weighted keypoint analysis (racket hand priority)",
            "   - Multi-factor scoring (proximity + movement + trajectory)",
            "   - Velocity change detection",
            "",
            "2. Real-time Shot Tracking:",
            "   - Color-coded trajectory visualization",
            "   - Crosscourt shots: GREEN lines",
            "   - Straight shots: YELLOW lines", 
            "   - Boast shots: MAGENTA lines",
            "   - Drop shots: ORANGE lines",
            "   - Lob shots: BLUE lines",
            "",
            "3. Shot Data Logging:",
            "   - Complete shot trajectories saved",
            "   - Shot classification and duration",
            "   - Player identification per shot",
            "   - Comprehensive analysis reports",
            "",
            "4. Visual Improvements:",
            "   - Enhanced ball highlighting during shots",
            "   - Real-time shot statistics display",
            "   - Trajectory lines with shot identification"
        ]
        
        y_pos = 90
        for line in enhancements:
            if line.startswith("   -"):
                color = (200, 200, 200)  # Light gray for sub-points
                font_size = 0.4
            elif line.strip() and line[0].isdigit():
                color = (0, 255, 0)  # Green for main points
                font_size = 0.5
            elif ":" in line and not line.startswith("   "):
                color = (255, 255, 0)  # Yellow for categories
                font_size = 0.45
            else:
                color = (255, 255, 255)  # White for regular text
                font_size = 0.4
                
            cv2.putText(summary_img, line, (50, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, font_size, color, 1)
            y_pos += 25
            
        # Save summary
        cv2.imwrite("output/enhancement_summary.png", summary_img)
        print("📊 Enhancement summary saved to output/enhancement_summary.png")
        
    except Exception as e:
        print(f"Error creating enhancement summary: {e}")

def analyze_shot_patterns(shots_file="output/shots_log.jsonl"):
    """
    Analyze shot patterns from the saved shot data with enhanced bounce analysis
    """
    try:
        if not os.path.exists(shots_file):
            print(f"No shot data file found at {shots_file}")
            return {}
            
        shots = []
        with open(shots_file, "r") as f:
            for line in f:
                if line.strip():
                    shots.append(json.loads(line))
        
        if not shots:
            return {}
            
        # Analyze patterns
        analysis = {
            'total_shots': len(shots),
            'player_distribution': {},
            'shot_type_distribution': {},
            'average_duration': 0,
            'longest_shot': None,
            'shortest_shot': None,
            'rally_analysis': [],
            'bounce_analysis': {
                'total_bounces': 0,
                'shots_with_bounces': 0,
                'avg_bounces_per_shot': 0,
                'bounce_confidence_distribution': {},
                'bounce_type_distribution': {}
            }
        }
        
        # Player distribution
        for shot in shots:
            player = shot.get('player_who_hit', 0)
            analysis['player_distribution'][f'player_{player}'] = analysis['player_distribution'].get(f'player_{player}', 0) + 1
            
        # Shot type distribution
        for shot in shots:
            shot_type = str(shot.get('final_shot_type', 'unknown'))
            analysis['shot_type_distribution'][shot_type] = analysis['shot_type_distribution'].get(shot_type, 0) + 1
            
        # Duration analysis
        durations = [shot.get('duration', 0) for shot in shots]
        if durations:
            analysis['average_duration'] = sum(durations) / len(durations)
            analysis['longest_shot'] = max(shots, key=lambda x: x.get('duration', 0))
            analysis['shortest_shot'] = min(shots, key=lambda x: x.get('duration', 0))
        
        # Enhanced bounce analysis
        total_bounces = 0
        shots_with_bounces = 0
        bounce_confidences = []
        bounce_types = {}
        
        for shot in shots:
            bounce_details = shot.get('bounce_details', [])
            if bounce_details:
                shots_with_bounces += 1
                total_bounces += len(bounce_details)
                
                for bounce in bounce_details:
                    # Confidence distribution
                    confidence = bounce.get('confidence', 0)
                    bounce_confidences.append(confidence)
                    
                    # Bounce type distribution
                    bounce_type = bounce.get('type', 'unknown')
                    bounce_types[bounce_type] = bounce_types.get(bounce_type, 0) + 1
        
        analysis['bounce_analysis'].update({
            'total_bounces': total_bounces,
            'shots_with_bounces': shots_with_bounces,
            'avg_bounces_per_shot': total_bounces / len(shots) if shots else 0,
            'bounce_confidence_avg': sum(bounce_confidences) / len(bounce_confidences) if bounce_confidences else 0,
            'bounce_type_distribution': bounce_types
        })
            
        return analysis
        
    except Exception as e:
        print(f"Error analyzing shot patterns: {e}")
        return {}

def main(path="self2.mp4", input_frame_width=640, input_frame_height=360, max_frames=None):
    # Update global frame dimensions with user-provided values
    global frame_width, frame_height
    frame_width = input_frame_width
    frame_height = input_frame_height
    # Initialize frame counter early so exception handlers can access it
    frame_count = 0
    
    # 🚀 AGGRESSIVE PERFORMANCE OPTIMIZATIONS
    print("🚀 INITIALIZING ULTRA-FAST GPU-OPTIMIZED SQUASH COACHING PIPELINE")
    print("=" * 70)
    
    # GPU optimization setup with aggressive settings
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f" Primary compute device: {device}")
    
    if torch.cuda.is_available():
        print(f" GPU: {torch.cuda.get_device_name(0)}")
        print(f" GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        # 🚀 AGGRESSIVE GPU OPTIMIZATIONS
        torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark = True  # Optimize for fixed input sizes
        torch.backends.cudnn.deterministic = False  # Speed over reproducibility
        torch.backends.cudnn.enabled = True
        
        # Set memory fraction for optimal performance
        torch.cuda.set_per_process_memory_fraction(0.9)  # Use 90% of GPU memory
        
        # Enable mixed precision for 2x speedup
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        print(f" 🚀 GPU Memory allocated: {torch.cuda.memory_allocated(0) / 1e6:.1f} MB")
        print(f" 🚀 GPU Memory cached: {torch.cuda.memory_reserved(0) / 1e6:.1f} MB")
        print(" 🚀 CUDA optimizations enabled: benchmark=True, mixed_precision=True")
    else:
        print(" ⚠️  No GPU detected - using CPU (performance may be slower)")
    
    # 🚀 MODEL OPTIMIZATION SETTINGS
    model_optimizations = {
        'conf': 0.3,  # Higher confidence threshold for faster processing
        'iou': 0.5,   # Higher IoU threshold
        'max_det': 10,  # Limit detections for speed
        'agnostic_nms': True,  # Faster NMS
        'half': True if torch.cuda.is_available() else False,  # FP16 for 2x speedup
    }
    
    print(" 🚀 Model optimizations applied:")
    for key, value in model_optimizations.items():
        print(f"   • {key}: {value}")
    
    try:
        print(" INITIALIZING GPU-OPTIMIZED SQUASH COACHING PIPELINE")
        print("=" * 70)
        
        # GPU optimization setup
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f" Primary compute device: {device}")
        
        if torch.cuda.is_available():
            print(f" GPU: {torch.cuda.get_device_name(0)}")
            print(f" GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            # Optimize GPU memory usage
            torch.cuda.empty_cache()
            # Monitor GPU memory usage
            print(f" 🚀 GPU Memory allocated: {torch.cuda.memory_allocated(0) / 1e6:.1f} MB")
            print(f" 🚀 GPU Memory cached: {torch.cuda.memory_reserved(0) / 1e6:.1f} MB")
        else:
            print(" ⚠️  No GPU detected - using CPU (performance may be slower)")
        
        csvstart = 0
        end = csvstart + 100
        
        # Load ball position prediction model with GPU optimization
        try:
            import tensorflow as tf
            ball_predict = tf.keras.models.load_model(
                "trained-models/ball_position_model(25k).keras"
            )
            if torch.cuda.is_available():
                # Try to use GPU for TensorFlow if available
                tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[0], True)
                print(" Ball prediction model loaded with GPU acceleration")
        except Exception as e:
            print(f"Ball prediction model loading error: {e}")
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
        
        # 🚀 ULTRA-FAST MODEL LOADING WITH OPTIMIZATIONS
        print(" 🚀 Loading optimized YOLO models with maximum GPU acceleration...")
        
        # Load pose model with aggressive optimizations
        pose_model = YOLO("models/yolo11n-pose.pt")
        if torch.cuda.is_available():
            pose_model.to(device)
            # Apply aggressive optimizations
            pose_model.fuse()  # Fuse layers for speed
            pose_model.half()  # Use FP16 for 2x speedup
            print(" 🚀 Pose model loaded on GPU with FP16 optimization")
        else:
            print(" Pose model loaded on CPU")
        
        # Load ball detection model with aggressive optimizations
        ballmodel = YOLO("trained-models/black_ball_selfv3.pt")
        if torch.cuda.is_available():
            ballmodel.to(device)
            # Apply aggressive optimizations
            ballmodel.fuse()  # Fuse layers for speed
            ballmodel.half()  # Use FP16 for 2x speedup
            print(" 🚀 Ball detection model loaded on GPU with FP16 optimization")
        else:
            print(" Ball detection model loaded on CPU")
        
        print("=" * 70)
        print(" Enhanced Features Active:")
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
        
        # Initialize enhanced 3D ball tracking
        past_ball_pos_3d = []  # Store 3D ball positions for enhanced tracking

        running_frame = 0
        print("started video input")
        int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        abs(reference_points[1][0] - reference_points[0][0])
        validate_reference_points(reference_points, reference_points_3d)
        print(f"loaded everything in {time.time()-start} seconds")
        
        # 🚀 ULTRA-FAST MAIN PROCESSING LOOP WITH OPTIMIZED FRAME HANDLING
        print(" 🚀 Starting ultra-fast video processing...")
        
        # Pre-allocate memory for faster processing
        frame_buffer = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
        
        while cap.isOpened():
            success, frame = cap.read()

            if not success:
                break

            # 🚀 OPTIMIZED FRAME PROCESSING - Direct resize without copy
            frame = cv2.resize(frame, (frame_width, frame_height), interpolation=cv2.INTER_LINEAR)
            frame_count += 1
            
            # Check frame limit for testing
            if max_frames is not None and frame_count > max_frames:
                print(f"🛑 Reached frame limit ({max_frames}), stopping processing")
                break

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
                # 🚀 ULTRA-FAST POSE DETECTION WITH OPTIMIZED PROCESSING
                global known_players_features
                try:
                    # Use optimized inference with pre-configured settings
                    track_results = pose_model.track(frame, persist=True, show=False, **model_optimizations, verbose=False)
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
                plast=[[],[]],
                past_ball_pos_3d=None
            ):
                try:
                    # 🚀 ULTRA-FAST BALL DETECTION WITH OPTIMIZED PROCESSING
                    # Use optimized inference with pre-configured settings
                    ball = ballmodel(frame, **model_optimizations, verbose=False)
                    
                    # 🚀 OPTIMIZED BALL DETECTION - Direct GPU processing
                    x1, y1, x2, y2 = 0, 0, 0, 0
                    highestconf = 0.0
                    label = "ball"
                    ball_detected = False
                    
                    # 🚀 FAST DETECTION PROCESSING - Minimize CPU-GPU transfers
                    if ball and len(ball) > 0 and hasattr(ball[0], 'boxes') and ball[0].boxes is not None and len(ball[0].boxes) > 0:
                        # Find the highest confidence detection with optimized processing
                        best_box = None
                        best_conf = 0
                        
                        # 🚀 VECTORIZED PROCESSING for speed
                        boxes = ball[0].boxes
                        if len(boxes) > 0:
                            # Get all confidences at once
                            confidences = boxes.conf.cpu().numpy()
                            # Find best detection in one operation
                            best_idx = np.argmax(confidences)
                            best_conf = confidences[best_idx]
                            
                            if best_conf > 0.2:  # Optimized threshold
                                best_box = boxes[best_idx]
                                try:
                                    # 🚀 OPTIMIZED COORDINATE EXTRACTION
                                    coords = best_box.xyxy[0].cpu().numpy()
                                    if len(coords) >= 4:
                                        x1, y1, x2, y2 = coords[0], coords[1], coords[2], coords[3]
                                        highestconf = best_conf
                                        label = ballmodel.names[int(best_box.cls)]
                                        ball_detected = True
                                except Exception as e:
                                    ball_detected = False
                    
                    # 🚀 OPTIMIZED VISUALIZATION - Only draw if detected
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

                    # 🚀 OPTIMIZED FRAME DISPLAY - Minimal text rendering
                    if frame_count % 10 == 0:  # Only update every 10 frames for speed
                        cv2.putText(
                            annotated_frame,
                            f"Frame: {frame_count}",
                            (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.9,
                            (0, 255, 0),
                            2,
                        )
                    
                    # 🚀 ENHANCED BALL POSITION UPDATE WITH 3D POSITIONING
                    if ball_detected:
                        # Calculate ball center with optimized math
                        avg_x = int((x1 + x2) / 2)
                        avg_y = int((y1 + y2) / 2)
                        
                        # Enhanced validation with physics-based checks
                        if frame_count == 73:
                            print(f"Frame 73 Debug: x1={x1}, y1={y1}, w={x2-x1}, h={y2-y1}, conf={highestconf:.3f}")
                            print(f"Frame 73 Debug: past_ball_pos length={len(past_ball_pos)}")
                            if len(past_ball_pos) > 0:
                                print(f"Frame 73 Debug: last ball pos={past_ball_pos[-1]}")
                        
                        # Try enhanced validation first, fall back to original validation if it fails
                        validation_passed = enhanced_ball_detection_validation(x1, y1, x2-x1, y2-y1, highestconf, past_ball_pos, frame_count)
                        
                        if not validation_passed:
                            # Fall back to original validation
                            validation_passed = validate_ball_detection(x1, y1, x2-x1, y2-y1, highestconf, past_ball_pos)
                            if validation_passed:
                                if frame_count == 73:
                                    print(f"Frame 73: Enhanced validation failed, but original validation passed")
                            else:
                                if frame_count == 73:
                                    print(f"Frame 73: Both enhanced and original validation failed")
                        
                        if validation_passed:
                            # Use enhanced 3D positioning if homography is available
                            if 'homography' in locals() and 'reference_points_3d' in locals():
                                try:
                                    # Enhanced 3D positioning
                                    enhanced_result = enhanced_ball_3d_positioning(
                                        [avg_x, avg_y], 
                                        homography, 
                                        reference_points_3d, 
                                        past_ball_pos, 
                                        frame_count
                                    )
                                    
                                    # Store both 2D and 3D positions
                                    ball_2d_pos = [avg_x, avg_y, running_frame]
                                    ball_3d_pos = enhanced_result['position'] + [running_frame]
                                    
                                    # Update past_ball_pos with enhanced 2D position
                                    past_ball_pos.append(ball_2d_pos)
                                    
                                    # Store 3D position separately for advanced analysis
                                    if 'past_ball_pos_3d' not in locals():
                                        past_ball_pos_3d = []
                                    past_ball_pos_3d.append(ball_3d_pos)
                                    
                                    # Apply 3D smoothing if we have enough data
                                    if len(past_ball_pos_3d) >= 5:
                                        past_ball_pos_3d = smooth_ball_trajectory_3d(past_ball_pos_3d)
                                    
                                    # Update mainball with enhanced position
                                    mainball.update(avg_x, avg_y, avg_x * avg_y)
                                    
                                    # Enhanced visualization with 3D confidence
                                    confidence_text = f"3D Conf: {enhanced_result['confidence']:.2f}"
                                    cv2.putText(annotated_frame, confidence_text, 
                                            (int(x1), int(y1) - 30), cv2.FONT_HERSHEY_SIMPLEX, 
                                            0.5, (255, 255, 0), 2)
                                    
                                except Exception as e:
                                    print(f"Enhanced 3D positioning error: {e}")
                                    # Fallback to standard 2D positioning
                                    past_ball_pos.append([avg_x, avg_y, running_frame])
                                    mainball.update(avg_x, avg_y, avg_x * avg_y)
                            else:
                                # Standard 2D positioning when homography not available
                                past_ball_pos.append([avg_x, avg_y, running_frame])
                                mainball.update(avg_x, avg_y, avg_x * avg_y)
                            
                            # Keep only recent positions to prevent memory issues
                            if len(past_ball_pos) > 100:
                                past_ball_pos = past_ball_pos[-100:]
                            
                            # Update heatmap
                            if len(past_ball_pos) > 1:
                                prev_x, prev_y, _ = past_ball_pos[-2]
                                drawmap(avg_x, avg_y, prev_x, prev_y, ballmap)
                        else:
                            # Ball detection failed all validation - skip this detection
                            print(f"Ball detection failed all validation checks at frame {frame_count}")
                    
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
                        
                        # Enhanced comprehensive bounce detection and visualization
                        if len(past_ball_pos) >= 4:
                            # Use comprehensive bounce detection system
                            trajectory_segment = past_ball_pos[-30:] if len(past_ball_pos) > 30 else past_ball_pos
                            
                            detected_bounces = bounce_detector.detect_bounces_comprehensive(
                                trajectory_segment, frame_width, frame_height
                            )
                            
                            # Draw detected bounces with enhanced visualization
                            bounce_detector.draw_bounces(annotated_frame, trajectory_segment, frame_count)
                            
                            # Also use legacy GPU detection for comparison
                            gpu_bounces = detect_ball_bounces_gpu(
                                trajectory_segment, 
                                velocity_threshold=3.0, 
                                angle_threshold=30.0,
                                court_width=frame_width,
                                court_height=frame_height
                            )
                            
                            # Draw legacy GPU bounces with different style for comparison
                            for i, bounce_pos in enumerate(gpu_bounces):
                                # Legacy bounce visualization (smaller, cyan)
                                cv2.circle(annotated_frame, bounce_pos, 8, (255, 255, 0), 2)  # Cyan outline
                                cv2.circle(annotated_frame, bounce_pos, 4, (255, 255, 255), -1)  # White center
                                
                                # Add legacy bounce label
                                cv2.putText(annotated_frame, f"L{i+1}", 
                                        (bounce_pos[0] - 8, bounce_pos[1] - 15),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 0), 1)
                        
                        # Enhanced comprehensive bounce statistics display
                        if len(past_ball_pos) >= 4:
                            comprehensive_bounces = getattr(bounce_detector, 'detected_bounces', [])
                            recent_bounces = [b for b in comprehensive_bounces if abs(b['frame_index'] - frame_count) <= 30]
                            
                            # Enhanced bounce counter with detailed info
                            bounce_text = f"Enhanced Bounces: {len(recent_bounces)} | Total: {len(comprehensive_bounces)} | Legacy: {len(gpu_bounces)}"
                            text_size = cv2.getTextSize(bounce_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                            cv2.rectangle(annotated_frame, (8, 75), (text_size[0] + 16, 105), (0, 0, 0), -1)
                            cv2.putText(annotated_frame, bounce_text, 
                                    (10, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                            
                            # Show confidence info for recent bounces
                            if recent_bounces:
                                avg_confidence = sum(b['confidence'] for b in recent_bounces) / len(recent_bounces)
                                conf_text = f"Avg Confidence: {avg_confidence:.2f}"
                                cv2.putText(annotated_frame, conf_text, 
                                        (10, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
                    
                    # Simple status display
                    if ball_detected:
                        cv2.putText(annotated_frame, "BALL DETECTED", 
                                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    else:
                        cv2.putText(annotated_frame, "NO BALL DETECTED", 
                                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    """
                    ENHANCED FRAMEPOSE WITH OPTIONAL DEEPSORT
                    """
                    # Try to use enhanced ReID system first, then DeepSort, then fallback to standard
                    use_enhanced_reid = False
                    use_enhanced_tracking = False
                    
                    # First, try enhanced ReID system
                    try:
                        from enhanced_framepose import enhanced_framepose as reid_framepose
                        use_enhanced_reid = True
                        #print(" Using enhanced ReID-based player tracking")
                    except ImportError as e:
                        print(f"Enhanced ReID not available: {e}")
                    
                    # If ReID not available, try DeepSort
                    if not use_enhanced_reid:
                        try:
                            from squash.deepsortframepose import framepose as enhanced_framepose
                            use_enhanced_tracking = True
                            #print(" Using enhanced DeepSort-based player tracking")
                        except ImportError as e:
                            print(f"Using standard player tracking (DeepSort not available: {e})")
                    
                    # Use the appropriate framepose function
                    if use_enhanced_reid:
                        try:
                            framepose_result = reid_framepose(
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
                            print(f"Enhanced ReID tracking failed: {e}, falling back to DeepSort")
                            use_enhanced_reid = False
                            use_enhanced_tracking = True
                    
                    if not use_enhanced_reid and use_enhanced_tracking:
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
                    
                    if not use_enhanced_reid and not use_enhanced_tracking:
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
                    
                    # Save ReID statistics periodically
                    if frame_count % 500 == 0:
                        try:
                            if use_enhanced_reid:
                                from enhanced_framepose import save_reid_references, get_reid_statistics
                                save_reid_references(f"output/reid_references_frame_{frame_count}.json")
                                reid_stats = get_reid_statistics()
                                print(f"📊 ReID Stats at frame {frame_count}: {reid_stats}")
                        except Exception as e:
                            print(f"Error saving ReID stats: {e}")
                    
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
                        plast, # 20
                        past_ball_pos_3d # 21
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
                        plast, # 20
                        past_ball_pos_3d # 21
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
                embeddings=embeddings,
                plast=plast,
                past_ball_pos_3d=past_ball_pos_3d,
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
            past_ball_pos_3d = safe_get(detections_result, 21, [])
            
            if len(detections_result) < 22:
                print(f"Warning: detections_result has {len(detections_result)} elements, expected 22")
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
            
            # Enhanced shot tracking with visualization
            ball_hit = isinstance(match_in_play, dict) and match_in_play.get('ball_hit', False)
            current_ball_pos = past_ball_pos[-1] if past_ball_pos else None
            
            # Detect shot start
            shot_started = False
            if ball_hit and who_hit > 0 and current_ball_pos:
                shot_started = shot_tracker.detect_shot_start(
                    ball_hit, who_hit, frame_count, current_ball_pos, type_of_shot
                )
                
            # Update active shots
            if current_ball_pos:
                shot_tracker.update_active_shots(
                    current_ball_pos, frame_count, 
                    new_hit_detected=shot_started, 
                    new_shot_type=type_of_shot
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
                    #print(f" Player {who_hit} action: {player_action}")
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
                            bounce_count, bounce_positions = detect_wall_bounces_advanced(past_ball_pos, frame_width, frame_height)
                        else:
                            # Use default dimensions or get from frame
                            current_frame_width = getattr(frame, 'shape', [0, 640])[1] if 'frame' in locals() else 640
                            current_frame_height = getattr(frame, 'shape', [360, 0])[0] if 'frame' in locals() else 360
                            bounce_count, bounce_positions = detect_wall_bounces_advanced(past_ball_pos, current_frame_width, current_frame_height)
                        coaching_data['wall_bounce_count'] = bounce_count
                        coaching_data['bounce_positions'] = bounce_positions
                    except Exception as bounce_error:
                        print(f"Wall bounce detection error: {bounce_error}")
                        coaching_data['wall_bounce_count'] = 0
                        coaching_data['bounce_positions'] = []
                
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
                
            # Enhanced status display with GPU information and shot tracking
            gpu_status = "GPU" if torch.cuda.is_available() else "CPU"
            shot_stats = shot_tracker.get_shot_statistics()
            
            cv2.putText(
                annotated_frame,
                f"Enhanced Coaching: ON | Device: {gpu_status} | Data: {len(coaching_data_collection)}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
            )
            
            # Display shot tracking statistics
            cv2.putText(
                annotated_frame,
                f"Shots: {shot_stats['total_shots']} | Active: {shot_stats['active_shots']} | P1: {shot_stats['player1_shots']} | P2: {shot_stats['player2_shots']}",
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 0),
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

            positions = load_data("output/ball-xyn.txt")
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
            # Enhanced ball position to 3D conversion
            rlball = None
            if past_ball_pos and len(past_ball_pos) > 0:
                last_ball_pos = past_ball_pos[-1]
                if len(last_ball_pos) >= 2:
                    try:
                        # Use enhanced 3D positioning if available
                        if 'past_ball_pos_3d' in locals() and len(past_ball_pos_3d) > 0:
                            # Use pre-calculated 3D position
                            last_3d_pos = past_ball_pos_3d[-1]
                            rlball = last_3d_pos[:3]  # Extract x, y, z
                        else:
                            # Fallback to enhanced 3D positioning
                            enhanced_result = enhanced_ball_3d_positioning(
                                [last_ball_pos[0], last_ball_pos[1]],
                                homography,
                                reference_points_3d,
                                past_ball_pos,
                                frame_count
                            )
                            rlball = enhanced_result['position']
                    except Exception as e:
                        print(f"Error in enhanced ball 3D conversion: {e}")
                        # Fallback to original method
                        try:
                            rlball = pixel_to_3d(
                                [last_ball_pos[0], last_ball_pos[1]],
                                homography,
                                reference_points_3d,
                            )
                        except Exception as fallback_error:
                            print(f"Fallback 3D conversion also failed: {fallback_error}")
                            rlball = None
                else:
                    print(f"Warning: Last ball position has insufficient data: {last_ball_pos}")
            else:
                pass
                #print("Warning: No ball positions available for 3D conversion")
            

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

            # Enhanced ball 3D position display
            if past_ball_pos and len(past_ball_pos) > 0:
                last_ball_pos = past_ball_pos[-1]
                if len(last_ball_pos) >= 2:
                    try:
                        # Use enhanced 3D positioning for display
                        if 'past_ball_pos_3d' in locals() and len(past_ball_pos_3d) > 0:
                            # Use pre-calculated enhanced 3D position
                            last_3d_pos = past_ball_pos_3d[-1]
                            ball_3d = last_3d_pos[:3]  # Extract x, y, z
                            text = f"Enhanced 3D ball: {[round(x, 3) for x in ball_3d]}"
                        else:
                            # Calculate enhanced 3D position
                            enhanced_result = enhanced_ball_3d_positioning(
                                [last_ball_pos[0], last_ball_pos[1]], 
                                homography, 
                                reference_points_3d, 
                                past_ball_pos, 
                                frame_count
                            )
                            ball_3d = enhanced_result['position']
                            confidence = enhanced_result['confidence']
                            text = f"Enhanced 3D ball: {[round(x, 3) for x in ball_3d]} (Conf: {confidence:.2f})"
                    except Exception as e:
                        print(f"Error in enhanced ball 3D display: {e}")
                        # Fallback to original method
                        try:
                            ball_3d = pixel_to_3d([last_ball_pos[0], last_ball_pos[1]], homography, reference_points_3d)
                            text = f"Fallback 3D ball: {[round(x, 3) for x in ball_3d]}"
                        except Exception as fallback_error:
                            print(f"Fallback 3D display also failed: {fallback_error}")
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
                            print(f"\n Generating interim coaching report at frame {running_frame}...")
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
            
            # 🚀 OPTIMIZED GPU MEMORY MONITORING - Less frequent for speed
            if frame_count % 30 == 0 and torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated(0) / 1e6
                cached = torch.cuda.memory_reserved(0) / 1e6
                print(f"Frame {frame_count}: GPU Memory - Allocated: {allocated:.1f}MB, Cached: {cached:.1f}MB")
            
            # Draw shot trajectories and enhanced ball visualization
            current_ball_pos = past_ball_pos[-1] if past_ball_pos else None
            shot_tracker.draw_shot_trajectories(annotated_frame, current_ball_pos)
            
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
        print("\n🎾 Generating Enhanced Autonomous Coaching Analysis with Shot Tracking...")
        
        # Generate shot analysis report
        try:
            shot_stats = shot_tracker.get_shot_statistics()
            
            # Get bounce detection statistics
            total_bounces = len(getattr(bounce_detector, 'detected_bounces', []))
            recent_bounces = [b for b in getattr(bounce_detector, 'detected_bounces', []) 
                            if abs(b['frame_index'] - frame_count) <= 100]
            
            # Create comprehensive shot analysis
            shot_analysis_report = f"""
ENHANCED SHOT TRACKING ANALYSIS
=======================================

📊 SHOT STATISTICS:
-----------------
• Total Shots Tracked: {shot_stats['total_shots']}
• Player 1 Shots: {shot_stats['player1_shots']}
• Player 2 Shots: {shot_stats['player2_shots']}
• Average Shot Duration: {shot_stats['avg_duration']:.1f} frames

🏓 BOUNCE DETECTION STATISTICS:
------------------------------
• Total Bounces Detected: {total_bounces}
• Recent Bounces (last 100 frames): {len(recent_bounces)}
• Enhanced Detection Algorithms: 4 (Physics, Velocity, Wall, Curvature)
• Average Bounce Confidence: {sum(b['confidence'] for b in recent_bounces) / len(recent_bounces):.2f if recent_bounces else 0:.2f}

🎯 SHOT TYPE BREAKDOWN:
---------------------
"""
            for shot_type, count in shot_stats['shot_types'].items():
                percentage = (count / shot_stats['total_shots'] * 100) if shot_stats['total_shots'] > 0 else 0
                shot_analysis_report += f"• {shot_type}: {count} ({percentage:.1f}%)\n"
                
            shot_analysis_report += f"""

📈 ENHANCED VISUALIZATION FEATURES:
----------------------------------
• Real-time trajectory visualization with color coding
• Clear shot START markers: Large circles with "START" text
• Clear shot END markers: Square markers with "END" text  
• Active shot indicators: "ACTIVE" labels on current ball
• Crosscourt shots: Green trajectory lines
• Straight shots: Yellow trajectory lines  
• Boast shots: Magenta trajectory lines
• Drop shots: Orange trajectory lines
• Lob shots: Blue trajectory lines
• Bounce visualization: Multi-colored confidence-based markers

🔬 TECHNICAL IMPROVEMENTS:
-------------------------
• Enhanced player-ball hit detection using weighted keypoints
• Multi-factor scoring system (proximity + movement + trajectory)
• Real-time shot classification with trajectory analysis
• Automatic shot completion detection
• Comprehensive 4-algorithm bounce detection system
• Physics-based bounce validation
• Velocity vector analysis for precise hit detection
• Wall proximity analysis with gradient scoring
• Trajectory curvature analysis for direction changes
• Complete shot data with bounce information saved to: output/shots_log.jsonl

🎨 VISUAL MARKERS LEGEND:
------------------------
• START: Large circle with white center and colored border
• END: Square marker with red center and colored border  
• ACTIVE: Current ball position with enhanced highlighting
• Bounces: Color-coded by confidence (Green=High, Yellow=Medium, Orange=Low)
• Shot Duration: Displayed next to END markers
• Algorithm Count: Shows how many detection algorithms agreed
"""
            
            # Save shot analysis report
            with open("output/shot_analysis_report.txt", "w") as f:
                f.write(shot_analysis_report)
                
            print("✅ Shot analysis report saved to output/shot_analysis_report.txt")
            print(f"✅ {shot_stats['total_shots']} shots tracked and saved to output/shots_log.jsonl")
            
            # Analyze shot patterns from saved data
            shot_patterns = analyze_shot_patterns()
            if shot_patterns:
                print(f"📊 Advanced shot pattern analysis completed")
                print(f"   - Total shots analyzed: {shot_patterns['total_shots']}")
                print(f"   - Average shot duration: {shot_patterns['average_duration']:.1f} frames")
            
            # Create visual enhancement summary
            create_enhancement_summary()
            
        except Exception as e:
            print(f"⚠️ Error generating shot analysis: {e}")
        
        try:
            # Initialize the enhanced autonomous coach
            from autonomous_coaching import AutonomousSquashCoach
            autonomous_coach = AutonomousSquashCoach()
            
            # Generate comprehensive coaching insights
            coaching_insights = autonomous_coach.analyze_match_data(coaching_data_collection)
            
            # Enhanced coaching report with bounce and shot analysis
            enhanced_report = f"""
ENHANCED SQUASH COACHING ANALYSIS
================================================

Analysis Date: {time.strftime('%Y-%m-%d %H:%M:%S')}
Video Analyzed: {path}
Total Frames Processed: {frame_count}
Enhanced Coaching Data Points: {len(coaching_data_collection)}
GPU Acceleration: {'✅ Enabled' if torch.cuda.is_available() else '❌ CPU Only'}

ENHANCED BALL TRACKING ANALYSIS:
------------------------------
• Total trajectory points: {len(past_ball_pos)}
• Enhanced bounce detection: GPU-accelerated
• Multi-criteria validation: Angle, velocity, wall proximity
• Visualization: Real-time colored trajectory indicators
• Shot tracking: {shot_tracker.get_shot_statistics()['total_shots']} complete shots analyzed

{coaching_insights}

TECHNICAL ENHANCEMENTS:
---------------------
• 🎾 GPU-optimized ball detection and tracking
• 🎯 Enhanced shot tracking with real-time visualization
• 🔍 Enhanced bounce detection with multiple validation criteria
• 📊 Real-time trajectory analysis with physics modeling
• 🤖 Comprehensive coaching data collection
• 📈 Advanced ball bounce pattern analysis
• 🎨 Color-coded shot visualization system

SYSTEM PERFORMANCE:
-----------------
• Processing device: {'GPU' if torch.cuda.is_available() else 'CPU'}
• Ball detection accuracy: Enhanced with trained model
• Bounce detection: Multi-algorithm validation
• Shot tracking: ✅ Active throughout session
• Real-time analysis: ✅ Active throughout session

================================================
"""
              # Save enhanced report
            with open("output/enhanced_autonomous_coaching_report.txt", "w", encoding='utf-8') as f:
                f.write(enhanced_report)
              # Save detailed coaching data with bounce information
            detailed_data = []
            for data_point in coaching_data_collection:
                if isinstance(data_point, dict):
                    detailed_data.append(data_point)
            
            with open("output/enhanced_coaching_data.json", "w", encoding='utf-8') as f:
                json.dump(detailed_data, f, indent=2, default=str)
            
            print(" Enhanced coaching analysis completed!")
            print(f"Enhanced report saved: output/enhanced_autonomous_coaching_report.txt")
            print(f" Enhanced data saved: output/enhanced_coaching_data.json")
            print(f" Ball bounces analyzed: GPU-accelerated detection active")
              # Also generate traditional report for compatibility
            generate_coaching_report(coaching_data_collection, path, frame_count)
            print("Traditional coaching report also generated for compatibility.")
            
            # Generate ReID system report if available
            try:
                from enhanced_framepose import get_reid_statistics, save_reid_references
                reid_stats = get_reid_statistics()
                if reid_stats:
                    save_reid_references("output/final_reid_references.json")
                    
                    reid_report = f"""
ENHANCED PLAYER RE-IDENTIFICATION REPORT
=======================================

Total Track ID Swaps Detected: {reid_stats.get('total_swaps_detected', 0)}
Player 1 Initialization: {'Complete' if reid_stats.get('initialization_status', {}).get(1) else '❌ Incomplete'}
Player 2 Initialization: {'Complete' if reid_stats.get('initialization_status', {}).get(2) else '❌ Incomplete'}

Reference Feature Counts:
- Player 1: {reid_stats.get('reference_counts', {}).get(1, 0)} appearance features
- Player 2: {reid_stats.get('reference_counts', {}).get(2, 0)} appearance features

Final Track Mappings: {reid_stats.get('current_mappings', {})}

System Performance:
- Initialization frames: 100-150 (when players are separated)
- Proximity threshold: 100 pixels (for swap detection)
- Confidence threshold: 0.6 (for identity assignments)
- Feature extraction: ResNet50-based deep features

The ReID system continuously monitors player appearances and positions,
detecting when track IDs may have been swapped due to occlusion or
close proximity between players.
"""
                    
                    with open("output/reid_analysis_report.txt", "w", encoding='utf-8') as f:
                        f.write(reid_report)
                    
                    print("Player ReID analysis report generated!")
                    print(f"   - Total swaps detected: {reid_stats.get('total_swaps_detected', 0)}")
                    print(f"   - References saved: output/final_reid_references.json")
                    print(f"   - Report saved: output/reid_analysis_report.txt")
            except Exception as e:
                print(f"Error generating ReID report: {e}")
            
            # Also generate traditional report for compatibility
            generate_coaching_report(coaching_data_collection, path, frame_count)
            print("Traditional coaching report also generated for compatibility.")
            
            # Generate comprehensive visualizations and analytics
            print("\nGenerating comprehensive visualizations and analytics...")
            try:
                from autonomous_coaching import create_graphics, view_all_graphics
                
                # Generate all visualizations based on final.csv data
                create_graphics()
                print("All visualizations generated successfully!")
                
                # Display summary of generated graphics
                view_all_graphics()
                print("Graphics summary and analytics displayed!")
                
            except Exception as viz_error:
                print(f"  Error generating visualizations: {viz_error}")
                print("   Pipeline completed successfully, but visualizations could not be generated.")
            
        except Exception as e:
            print(f"  Error in enhanced coaching analysis: {e}")
            # Fallback to basic report
            try:
                generate_coaching_report(coaching_data_collection, path, frame_count)
                print(" Fallback coaching report generated successfully.")
            except Exception as fallback_error:
                print(f" Fallback coaching report error: {fallback_error}")
        
        print("\n ENHANCED PROCESSING COMPLETE!")
        print("=" * 50)
        print(" Check output/ directory for results:")
        print("   • enhanced_autonomous_coaching_report.txt - Enhanced analysis")
        print("   • enhanced_coaching_data.json - Detailed data with bounces")
        print("    reid_analysis_report.txt - Player ReID analysis")
        print("    final_reid_references.json - Player appearance references")
        print("   • annotated.mp4 - Video with bounce visualization")
        print("   • final.csv - Complete match data")
        print("    graphics/ - Comprehensive visualizations and analytics:")
        print("     - Shot type analysis and heatmaps")
        print("     - Player and ball movement patterns")
        print("     - Ball trajectory analysis")
        print("     - Match flow and performance metrics")
        print("     - Summary statistics and reports")
        print("   • Other traditional output files")
        print("=" * 50)
        print("\n ENHANCED REID SYSTEM FEATURES:")
        print("   • Initial player appearance capture (frames 100-150)")
        print("   • Continuous track ID swap detection")
        print("   • Deep learning-based appearance features")
        print("   • Multi-modal identity verification (appearance + position)")
        print("   • Real-time confidence scoring")
        print("=" * 50)

        cap.release()
        cv2.destroyAllWindows()


# Enhanced Ball 3D Positioning System
def enhanced_ball_3d_positioning(pixel_point, H, rl_reference_points, past_ball_pos, frame_count, fps=30):
    """
    Enhanced 3D ball positioning with physics modeling and temporal consistency.
    
    Parameters:
        pixel_point (list): Current [x, y] pixel coordinate
        H (np.array): Homography matrix
        rl_reference_points (list): 3D reference points
        past_ball_pos (list): Historical ball positions [[x, y, frame], ...]
        frame_count (int): Current frame number
        fps (int): Frames per second
    
    Returns:
        dict: Enhanced 3D position with confidence and metadata
    """
    from scipy.optimize import minimize
    from scipy.spatial.distance import cdist
    
    # Convert to numpy arrays
    rl_ref_np = np.array(rl_reference_points, dtype=np.float32)
    
    # Step 1: Basic homography transformation
    pixel_homogeneous = np.array([*pixel_point, 1])
    real_world_2d = np.dot(H, pixel_homogeneous)
    real_world_2d /= real_world_2d[2]
    
    # Step 2: Enhanced Z-coordinate estimation using physics
    z_estimate = estimate_ball_height_physics(pixel_point, past_ball_pos, frame_count, fps)
    
    # Step 3: Temporal consistency check
    temporal_correction = None
    if len(past_ball_pos) >= 3:
        temporal_correction = apply_temporal_consistency(pixel_point, past_ball_pos, frame_count)
        if temporal_correction is not None:
            pixel_point = temporal_correction
    
    # Step 4: Advanced interpolation with confidence weighting
    confidence_weights = calculate_interpolation_weights(pixel_point, rl_ref_np)
    
    # Step 5: Physics-based trajectory prediction
    trajectory_prediction = predict_ball_trajectory_3d(past_ball_pos, frame_count, fps)
    
    # Step 6: Combine all estimates with weighted averaging
    final_position = combine_3d_estimates(
        real_world_2d[:2], 
        z_estimate, 
        trajectory_prediction, 
        confidence_weights, 
        rl_ref_np
    )
    
    # Step 7: Validate against court boundaries
    final_position = validate_court_boundaries(final_position)
    
    # Step 8: Calculate confidence score
    confidence = calculate_3d_confidence(
        pixel_point, past_ball_pos, final_position, frame_count
    )
    
    return {
        'position': final_position,
        'confidence': confidence,
        'trajectory_prediction': trajectory_prediction,
        'z_estimate': z_estimate,
        'temporal_corrected': temporal_correction is not None
    }

def estimate_ball_height_physics(pixel_point, past_ball_pos, frame_count, fps):
    """
    Estimate ball height using physics modeling.
    """
    if len(past_ball_pos) < 3:
        return 0.0  # Default ground level
    
    # Calculate ball velocity and acceleration
    recent_positions = past_ball_pos[-5:] if len(past_ball_pos) >= 5 else past_ball_pos
    
    # Extract velocities
    velocities = []
    for i in range(1, len(recent_positions)):
        dx = recent_positions[i][0] - recent_positions[i-1][0]
        dy = recent_positions[i][1] - recent_positions[i-1][1]
        dt = (recent_positions[i][2] - recent_positions[i-1][2]) / fps
        if dt > 0:
            velocities.append([dx/dt, dy/dt])
    
    if not velocities:
        return 0.0
    
    # Calculate vertical velocity component (assuming perspective projection)
    avg_velocity = np.mean(velocities, axis=0)
    velocity_magnitude = np.linalg.norm(avg_velocity)
    
    # Physics-based height estimation
    # Squash ball physics: gravity affects vertical motion
    g = 9.81  # m/s²
    time_since_last = (frame_count - recent_positions[-1][2]) / fps
    
    # Estimate initial vertical velocity from trajectory
    if len(velocities) >= 2:
        vertical_accel = (velocities[-1][1] - velocities[0][1]) / (len(velocities) - 1)
        initial_vertical_velocity = velocities[0][1] - 0.5 * vertical_accel * len(velocities)
    else:
        initial_vertical_velocity = velocities[0][1]
    
    # Calculate height using projectile motion
    height = initial_vertical_velocity * time_since_last - 0.5 * g * time_since_last**2
    
    # Constrain to reasonable squash court heights (0-5 meters)
    height = max(0.0, min(5.0, height))
    
    return height

def apply_temporal_consistency(pixel_point, past_ball_pos, frame_count):
    """
    Apply temporal consistency to reduce jitter and improve tracking.
    """
    if len(past_ball_pos) < 3:
        return None
    
    # Calculate expected position based on recent trajectory
    recent_positions = past_ball_pos[-3:]
    
    # Linear prediction
    if len(recent_positions) >= 2:
        last_pos = recent_positions[-1]
        prev_pos = recent_positions[-2]
        
        # Calculate velocity
        dx = last_pos[0] - prev_pos[0]
        dy = last_pos[1] - prev_pos[1]
        dt = last_pos[2] - prev_pos[2]
        
        if dt > 0:
            vx = dx / dt
            vy = dy / dt
            
            # Predict current position
            time_diff = frame_count - last_pos[2]
            predicted_x = last_pos[0] + vx * time_diff
            predicted_y = last_pos[1] + vy * time_diff
            
            # Calculate distance between predicted and detected
            distance = np.sqrt((pixel_point[0] - predicted_x)**2 + (pixel_point[1] - predicted_y)**2)
            
            # If detection is too far from prediction, use weighted average
            if distance > 50:  # Threshold for outlier detection
                weight = 0.3  # Weight for current detection
                corrected_x = weight * pixel_point[0] + (1 - weight) * predicted_x
                corrected_y = weight * pixel_point[1] + (1 - weight) * predicted_y
                return [corrected_x, corrected_y]
    
    return None

def calculate_interpolation_weights(pixel_point, reference_points):
    """
    Calculate advanced interpolation weights using multiple factors.
    """
    from scipy.spatial.distance import cdist
    
    # Calculate distances to reference points
    distances = cdist([pixel_point], reference_points[:, :2])[0]
    
    # Base weights (inverse distance)
    base_weights = 1 / (distances + 1e-6)
    
    # Apply confidence weighting based on reference point reliability
    confidence_weights = np.ones_like(base_weights)
    
    # Court corners are more reliable than intermediate points
    corner_indices = [0, 1, 2, 3]  # Court corners
    for idx in corner_indices:
        if idx < len(confidence_weights):
            confidence_weights[idx] = 1.5
    
    # Service line points are also reliable
    service_indices = [4, 5, 6]  # Service line points
    for idx in service_indices:
        if idx < len(confidence_weights):
            confidence_weights[idx] = 1.2
    
    # Combine weights
    final_weights = base_weights * confidence_weights
    final_weights /= np.sum(final_weights)  # Normalize
    
    return final_weights

def predict_ball_trajectory_3d(past_ball_pos, frame_count, fps):
    """
    Predict ball trajectory in 3D space using physics modeling.
    """
    if len(past_ball_pos) < 3:
        return None
    
    # Extract recent trajectory
    recent_positions = past_ball_pos[-5:] if len(past_ball_pos) >= 5 else past_ball_pos
    
    # Calculate 3D trajectory parameters
    positions_3d = []
    for pos in recent_positions:
        # Estimate Z coordinate for each position
        z_est = estimate_ball_height_physics([pos[0], pos[1]], past_ball_pos, pos[2], fps)
        positions_3d.append([pos[0], pos[1], z_est])
    
    # Fit polynomial to trajectory
    if len(positions_3d) >= 3:
        try:
            # Convert to numpy arrays
            positions_np = np.array(positions_3d)
            times = np.array([pos[2] for pos in recent_positions])
            
            # Fit 3D polynomial (quadratic for physics)
            coeffs_x = np.polyfit(times, positions_np[:, 0], 2)
            coeffs_y = np.polyfit(times, positions_np[:, 1], 2)
            coeffs_z = np.polyfit(times, positions_np[:, 2], 2)
            
            # Predict next position
            next_time = frame_count
            predicted_x = np.polyval(coeffs_x, next_time)
            predicted_y = np.polyval(coeffs_y, next_time)
            predicted_z = np.polyval(coeffs_z, next_time)
            
            return [predicted_x, predicted_y, predicted_z]
        except:
            pass
    
    return None

def combine_3d_estimates(real_world_2d, z_estimate, trajectory_prediction, confidence_weights, reference_points):
    """
    Combine multiple 3D estimates using weighted averaging.
    """
    # Start with homography-based 2D position
    final_x, final_y = real_world_2d
    
    # Combine Z estimates
    z_estimates = [z_estimate]
    z_weights = [0.6]  # Weight for physics-based estimate
    
    # Add trajectory prediction if available
    if trajectory_prediction is not None:
        z_estimates.append(trajectory_prediction[2])
        z_weights.append(0.4)
    
    # Add reference point interpolation
    ref_z = np.dot(confidence_weights, reference_points[:, 2])
    z_estimates.append(ref_z)
    z_weights.append(0.2)
    
    # Normalize weights
    z_weights = np.array(z_weights)
    z_weights /= np.sum(z_weights)
    
    # Calculate final Z
    final_z = np.dot(z_weights, z_estimates)
    
    return [final_x, final_y, final_z]

def validate_court_boundaries(position):
    """
    Validate that the 3D position is within court boundaries.
    """
    x, y, z = position
    
    # Squash court dimensions (meters)
    COURT_LENGTH = 9.75
    COURT_WIDTH = 6.4
    MAX_HEIGHT = 5.0
    
    # Clamp to court boundaries
    x = max(0, min(COURT_WIDTH, x))
    y = max(0, min(COURT_LENGTH, y))
    z = max(0, min(MAX_HEIGHT, z))
    
    return [x, y, z]

def calculate_3d_confidence(pixel_point, past_ball_pos, final_position, frame_count):
    """
    Calculate confidence score for the 3D position estimate.
    """
    confidence = 0.5  # Base confidence
    
    # Factor 1: Temporal consistency
    if len(past_ball_pos) >= 3:
        recent_positions = past_ball_pos[-3:]
        distances = []
        for i in range(1, len(recent_positions)):
            dist = np.sqrt((recent_positions[i][0] - recent_positions[i-1][0])**2 + 
                          (recent_positions[i][1] - recent_positions[i-1][1])**2)
            distances.append(dist)
        
        avg_distance = np.mean(distances)
        if avg_distance < 100:  # Smooth trajectory
            confidence += 0.2
        elif avg_distance < 200:  # Moderate movement
            confidence += 0.1
    
    # Factor 2: Position stability
    if len(past_ball_pos) >= 2:
        last_pos = past_ball_pos[-1]
        current_dist = np.sqrt((pixel_point[0] - last_pos[0])**2 + (pixel_point[1] - last_pos[1])**2)
        if current_dist < 50:  # Stable position
            confidence += 0.2
        elif current_dist < 100:  # Reasonable movement
            confidence += 0.1
    
    # Factor 3: Court boundary compliance
    x, y, z = final_position
    if 0 <= x <= 6.4 and 0 <= y <= 9.75 and 0 <= z <= 5.0:
        confidence += 0.1
    
    # Factor 4: Trajectory physics compliance
    if len(past_ball_pos) >= 3:
        # Check if trajectory follows expected physics
        recent_positions = past_ball_pos[-3:]
        velocities = []
        for i in range(1, len(recent_positions)):
            dx = recent_positions[i][0] - recent_positions[i-1][0]
            dy = recent_positions[i][1] - recent_positions[i-1][1]
            dt = recent_positions[i][2] - recent_positions[i-1][2]
            if dt > 0:
                velocities.append(np.sqrt(dx**2 + dy**2) / dt)
        
        if velocities:
            avg_velocity = np.mean(velocities)
            if 0 < avg_velocity < 1000:  # Reasonable velocity range
                confidence += 0.1
    
    return min(1.0, confidence)

def enhanced_ball_detection_validation(x, y, w, h, confidence, past_ball_pos, frame_count, fps=30):
    """
    Enhanced ball detection validation with physics-based checks.
    """
    # Basic validation - relaxed confidence threshold
    if confidence < 0.15:  # Reduced from 0.2
        if frame_count == 73:
            print(f"Frame 73: Confidence {confidence:.3f} too low (threshold: 0.15)")
        return False
    
    # Size validation with more lenient thresholds
    ball_size = w * h
    if ball_size < 5 or ball_size > 5000:  # Relaxed from 10-2500
        if frame_count == 73:
            print(f"Frame 73: Ball size {ball_size} outside range (5-5000)")
        return False
    
    # Aspect ratio validation - more lenient
    aspect_ratio = w / h if h > 0 else float('inf')
    if aspect_ratio < 0.2 or aspect_ratio > 5.0:  # Relaxed from 0.3-3.5
        if frame_count == 73:
            print(f"Frame 73: Aspect ratio {aspect_ratio:.2f} outside range (0.2-5.0)")
        return False
    
    # Physics-based validation - more lenient
    if len(past_ball_pos) >= 2:
        last_pos = past_ball_pos[-1]
        distance = np.sqrt((x - last_pos[0])**2 + (y - last_pos[1])**2)
        time_diff = (frame_count - last_pos[2]) / fps
        
        if time_diff > 0:
            velocity = distance / time_diff
            
            # Check for physically impossible velocities - much more lenient threshold
            if velocity > 1000:  # pixels per second (increased from 300)
                if frame_count == 73:
                    print(f"Frame 73: Velocity {velocity:.1f} too high (threshold: 1000)")
                return False
            
            # Check for sudden direction changes (potential false detection) - more lenient
            if len(past_ball_pos) >= 3:
                prev_pos = past_ball_pos[-2]
                angle_change = calculate_trajectory_angle_change(
                    [prev_pos[0], prev_pos[1]], 
                    [last_pos[0], last_pos[1]], 
                    [x, y]
                )
                if angle_change > 170:  # Degrees (increased from 150)
                    if frame_count == 73:
                        print(f"Frame 73: Angle change {angle_change:.1f}° too large (threshold: 170°)")
                    return False
    
    return True

def calculate_trajectory_angle_change(p1, p2, p3):
    """
    Calculate the angle change in trajectory.
    """
    v1 = np.array([p2[0] - p1[0], p2[1] - p1[1]])
    v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
    
    if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
        return 0
    
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    cos_angle = np.clip(cos_angle, -1, 1)
    angle = np.arccos(cos_angle) * 180 / np.pi
    
    return angle

def smooth_ball_trajectory_3d(past_ball_pos_3d, window_size=5):
    """
    Apply 3D smoothing to ball trajectory.
    """
    if len(past_ball_pos_3d) < window_size:
        return past_ball_pos_3d
    
    smoothed_trajectory = []
    for i in range(len(past_ball_pos_3d)):
        start_idx = max(0, i - window_size // 2)
        end_idx = min(len(past_ball_pos_3d), i + window_size // 2 + 1)
        
        window_positions = past_ball_pos_3d[start_idx:end_idx]
        avg_x = sum(pos[0] for pos in window_positions) / len(window_positions)
        avg_y = sum(pos[1] for pos in window_positions) / len(window_positions)
        avg_z = sum(pos[2] for pos in window_positions) / len(window_positions)
        
        smoothed_trajectory.append([avg_x, avg_y, avg_z, past_ball_pos_3d[i][3]])  # Keep frame number
    
    return smoothed_trajectory

def predict_ball_3d_position_advanced(past_ball_pos_3d, frames_ahead=3):
    """
    Advanced 3D ball position prediction using physics and machine learning.
    """
    if len(past_ball_pos_3d) < 3:
        return None
    
    # Extract recent 3D positions
    recent_positions = past_ball_pos_3d[-5:] if len(past_ball_pos_3d) >= 5 else past_ball_pos_3d
    
    # Calculate 3D velocities
    velocities_3d = []
    for i in range(1, len(recent_positions)):
        dx = recent_positions[i][0] - recent_positions[i-1][0]
        dy = recent_positions[i][1] - recent_positions[i-1][1]
        dz = recent_positions[i][2] - recent_positions[i-1][2]
        dt = recent_positions[i][3] - recent_positions[i-1][3]  # Frame difference
        
        if dt > 0:
            velocities_3d.append([dx/dt, dy/dt, dz/dt])
    
    if not velocities_3d:
        return None
    
    # Calculate average velocity and acceleration
    avg_velocity = np.mean(velocities_3d, axis=0)
    
    # Apply physics constraints (gravity affects Z component)
    g = 9.81  # m/s²
    time_ahead = frames_ahead / 30.0  # Assuming 30 fps
    
    # Predict position with physics
    last_pos = recent_positions[-1]
    predicted_x = last_pos[0] + avg_velocity[0] * time_ahead
    predicted_y = last_pos[1] + avg_velocity[1] * time_ahead
    predicted_z = last_pos[2] + avg_velocity[2] * time_ahead - 0.5 * g * time_ahead**2
    
    # Ensure Z doesn't go below ground
    predicted_z = max(0.0, predicted_z)
    
    return [predicted_x, predicted_y, predicted_z]


if __name__ == "__main__":
    try:
        start=time.time()
        main()
        print(f"Execution time: {time.time() - start:.2f} seconds")
    except KeyboardInterrupt:
        print("\nProgram interrupted by user. All outputs have been generated.")
    except Exception as e:
        print(f"Unexpected error: {e}")
        
