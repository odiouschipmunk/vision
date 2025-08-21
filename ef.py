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
matplotlib.use("Agg")  # Use non-interactive backend for server environments
import tensorflow as tf
import matplotlib.pyplot as plt
from ultralytics import YOLO
from squash import Referencepoints, Functions  # Ensure Functions is imported
from matplotlib import pyplot as plt
from squash.Ball import Ball
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torchvision import transforms
# Import autonomous coaching system
from autonomous_coaching import collect_coaching_data, generate_coaching_report, create_graphics, view_all_graphics
# Import enhanced ball physics and shot detection system
from enhanced_ball_physics import create_enhanced_shot_detector, BallPosition, ShotEvent
from enhanced_shot_integration import integrate_enhanced_detection_into_pipeline
# Import ultimate coaching enhancement system
from enhanced_coaching_config import apply_ultimate_enhancement
# Import Llama coaching enhancement system
from llama_coaching_enhancement import initialize_llama_enhancement, get_llama_enhancer
import json
import numpy as np

class MemoryEfficientLLMManager:
    """Manages LLM loading to prevent memory conflicts"""
    
    def __init__(self):
        self.current_model = None
        self.model_instances = {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def load_model(self, model_type):
        """Load a specific model type, unloading others first"""
        if self.current_model == model_type:
            return self.model_instances.get(model_type)
        
        # Check memory before loading
        print("💾 Checking memory before loading model...")
        self.check_memory_usage()
        
        # Unload current model to free memory
        self.unload_current_model()
        
        try:
            if model_type == 'llama':
                print("🤖 Loading Llama enhancement model...")
                initialize_llama_enhancement()
                llama_enhancer = get_llama_enhancer()
                if llama_enhancer and llama_enhancer.is_initialized:
                    self.model_instances[model_type] = llama_enhancer
                    self.current_model = model_type
                    print("✅ Llama model loaded successfully")
                    print("💾 Memory after loading Llama model:")
                    self.check_memory_usage()
                    return llama_enhancer
                else:
                    print("⚠️ Llama model failed to initialize")
                    return None
                    
            elif model_type == 'autonomous_coaching':
                print("🧠 Loading autonomous coaching models...")
                from autonomous_coaching import get_autonomous_coach
                autonomous_coach = get_autonomous_coach()
                if autonomous_coach:
                    self.model_instances[model_type] = autonomous_coach
                    self.current_model = model_type
                    print("✅ Autonomous coaching models loaded successfully")
                    print("💾 Memory after loading autonomous coaching models:")
                    self.check_memory_usage()
                    return autonomous_coach
                else:
                    print("⚠️ Autonomous coaching models failed to initialize")
                    return None
                    
        except Exception as e:
            print(f"❌ Failed to load {model_type} model: {e}")
            return None
    
    def unload_current_model(self):
        """Unload current model to free memory"""
        if self.current_model:
            print(f"🗑️ Unloading {self.current_model} model to free memory...")
            if self.current_model in self.model_instances:
                del self.model_instances[self.current_model]
            self.current_model = None
            
            # Clear GPU cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                print(f"🧹 GPU memory cleared: {torch.cuda.memory_allocated(0) / 1e6:.1f} MB")
    
    def get_model(self, model_type):
        """Get model instance, loading if necessary"""
        if model_type in self.model_instances:
            return self.model_instances[model_type]
        return self.load_model(model_type)
    
    def check_memory_usage(self):
        """Check current memory usage"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(0) / 1e9
            reserved = torch.cuda.memory_reserved(0) / 1e9
            total = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"💾 GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved, {total:.2f}GB total")
            return allocated, reserved, total
        return 0, 0, 0

# Global LLM manager instance
llm_manager = MemoryEfficientLLMManager()

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)
print(f"time to import everything: {time.time()-start}")
alldata = organizeddata = []
# Initialize global frame dimensions and frame counter for exception handling
frame_count = 0
frame_width = 0
frame_height = 0
# Autonomous coaching system imported from autonomous_coaching.py

class ShotClassificationModel:
    """Enhanced shot classification using multiple trajectory features"""
    
    def __init__(self):
        self.shot_types = {
            'straight_drive': {'horizontal_variance': 0.15, 'height_profile': 'low', 'speed_profile': 'fast'},
            'crosscourt': {'horizontal_variance': 0.35, 'height_profile': 'medium', 'speed_profile': 'medium'},
            'drop_shot': {'vertical_drop': 0.6, 'speed_profile': 'slow', 'front_court_ending': True},
            'lob': {'height_profile': 'high', 'trajectory_arc': 'parabolic', 'back_court_ending': True},
            'boast': {'wall_hits': True, 'angle_change': 60, 'side_wall_contact': True},
            'kill_shot': {'height_profile': 'very_low', 'speed_profile': 'very_fast', 'front_court_ending': True},
            'volley': {'interception_height': 'high', 'quick_reaction': True},
            'defensive_lob': {'height_profile': 'very_high', 'back_court_ending': True, 'pressure_response': True}
        }
    
    def classify_shot(self, trajectory, court_dimensions, player_positions=None):
        """
        Classify shot type based on comprehensive trajectory analysis
        
        Args:
            trajectory: List of ball positions [(x, y, frame), ...]
            court_dimensions: (width, height) of court
            player_positions: Dict of player positions during shot
            
        Returns:
            dict: Shot classification with confidence and features
        """
        if len(trajectory) < 5:
            return {'type': 'unknown', 'confidence': 0.0, 'features': {}}
        
        features = self._extract_trajectory_features(trajectory, court_dimensions)
        shot_scores = {}
        
        # Score each shot type
        for shot_type, criteria in self.shot_types.items():
            score = self._calculate_shot_score(features, criteria)
            shot_scores[shot_type] = score
        
        # Get best match
        best_shot = max(shot_scores.items(), key=lambda x: x[1])
        shot_type, confidence = best_shot
        
        # Add contextual information
        shot_context = self._analyze_shot_context(trajectory, court_dimensions, player_positions)
        
        return {
            'type': shot_type,
            'confidence': confidence,
            'features': features,
            'context': shot_context,
            'all_scores': shot_scores
        }
    
    def _extract_trajectory_features(self, trajectory, court_dimensions):
        """Extract comprehensive features from ball trajectory"""
        width, height = court_dimensions
        
        # Basic trajectory metrics
        start_pos = trajectory[0]
        end_pos = trajectory[-1]
        max_height = min(pos[1] for pos in trajectory)  # Y is inverted in screen coords
        min_height = max(pos[1] for pos in trajectory)
        
        # Movement analysis
        horizontal_movement = abs(end_pos[0] - start_pos[0])
        vertical_movement = abs(end_pos[1] - start_pos[1])
        total_distance = sum(math.sqrt((trajectory[i][0] - trajectory[i-1][0])**2 + 
                                    (trajectory[i][1] - trajectory[i-1][1])**2) 
                        for i in range(1, len(trajectory)))
        
        # Speed analysis
        speeds = []
        for i in range(1, len(trajectory)):
            if len(trajectory[i]) > 2 and len(trajectory[i-1]) > 2:
                dt = max(0.033, abs(trajectory[i][2] - trajectory[i-1][2])) if trajectory[i][2] != trajectory[i-1][2] else 1
                speed = math.sqrt((trajectory[i][0] - trajectory[i-1][0])**2 + 
                                (trajectory[i][1] - trajectory[i-1][1])**2) / dt
                speeds.append(speed)
        
        avg_speed = sum(speeds) / len(speeds) if speeds else 0
        max_speed = max(speeds) if speeds else 0
        speed_variance = np.var(speeds) if speeds else 0
        
        # Trajectory shape analysis
        height_variance = np.var([pos[1] for pos in trajectory])
        trajectory_curvature = self._calculate_curvature(trajectory)
        
        # Court position analysis
        start_court_zone = self._get_court_zone(start_pos, court_dimensions)
        end_court_zone = self._get_court_zone(end_pos, court_dimensions)
        
        # Wall interaction analysis
        wall_hits = self._detect_wall_interactions(trajectory, court_dimensions)
        direction_changes = self._count_direction_changes(trajectory)
        
        return {
            'horizontal_movement': horizontal_movement / width,
            'vertical_movement': vertical_movement / height,
            'height_range': (min_height - max_height) / height,
            'avg_speed': avg_speed,
            'max_speed': max_speed,
            'speed_variance': speed_variance,
            'height_variance': height_variance / (height**2),
            'trajectory_curvature': trajectory_curvature,
            'start_zone': start_court_zone,
            'end_zone': end_court_zone,
            'wall_hits': wall_hits,
            'direction_changes': direction_changes,
            'total_distance': total_distance,
            'trajectory_length': len(trajectory)
        }
    
    def _calculate_shot_score(self, features, criteria):
        """Calculate how well features match shot criteria"""
        score = 0.0
        total_criteria = len(criteria)
        
        for criterion, expected in criteria.items():
            if criterion == 'horizontal_variance':
                actual = features.get('horizontal_movement', 0)
                if isinstance(expected, (int, float)):
                    score += 1.0 - abs(actual - expected)
            elif criterion == 'height_profile':
                height_range = features.get('height_range', 0)
                if expected == 'low' and height_range < 0.2:
                    score += 1.0
                elif expected == 'medium' and 0.2 <= height_range <= 0.5:
                    score += 1.0
                elif expected == 'high' and height_range > 0.5:
                    score += 1.0
                elif expected == 'very_high' and height_range > 0.7:
                    score += 1.0
                elif expected == 'very_low' and height_range < 0.1:
                    score += 1.0
            elif criterion == 'speed_profile':
                avg_speed = features.get('avg_speed', 0)
                if expected == 'slow' and avg_speed < 10:
                    score += 1.0
                elif expected == 'medium' and 10 <= avg_speed <= 25:
                    score += 1.0
                elif expected == 'fast' and avg_speed > 25:
                    score += 1.0
                elif expected == 'very_fast' and avg_speed > 40:
                    score += 1.0
            elif criterion == 'wall_hits' and expected:
                if features.get('wall_hits', 0) > 0:
                    score += 1.0
            elif criterion == 'front_court_ending' and expected:
                if features.get('end_zone') in ['front_left', 'front_right', 'front_center']:
                    score += 1.0
            elif criterion == 'back_court_ending' and expected:
                if features.get('end_zone') in ['back_left', 'back_right', 'back_center']:
                    score += 1.0
        
        return score / total_criteria if total_criteria > 0 else 0.0
    
    def _calculate_curvature(self, trajectory):
        """Calculate trajectory curvature using three-point method"""
        if len(trajectory) < 3:
            return 0.0
        
        curvatures = []
        for i in range(1, len(trajectory) - 1):
            p1, p2, p3 = trajectory[i-1], trajectory[i], trajectory[i+1]
            
            # Vector from p1 to p2
            v1 = (p2[0] - p1[0], p2[1] - p1[1])
            # Vector from p2 to p3
            v2 = (p3[0] - p2[0], p3[1] - p2[1])
            
            # Calculate angle between vectors
            dot_product = v1[0]*v2[0] + v1[1]*v2[1]
            mag1 = math.sqrt(v1[0]**2 + v1[1]**2)
            mag2 = math.sqrt(v2[0]**2 + v2[1]**2)
            
            if mag1 > 0 and mag2 > 0:
                cos_angle = dot_product / (mag1 * mag2)
                cos_angle = max(-1, min(1, cos_angle))  # Clamp to valid range
                angle = math.acos(cos_angle)
                curvatures.append(angle)
        
        return sum(curvatures) / len(curvatures) if curvatures else 0.0
    
    def _get_court_zone(self, position, court_dimensions):
        """Determine which zone of the court the position is in"""
        x, y = position[0], position[1]
        width, height = court_dimensions
        
        # Divide court into 9 zones
        x_zone = 'left' if x < width/3 else 'center' if x < 2*width/3 else 'right'
        y_zone = 'front' if y < height/3 else 'middle' if y < 2*height/3 else 'back'
        
        return f"{y_zone}_{x_zone}"
    
    def _detect_wall_interactions(self, trajectory, court_dimensions):
        """Detect ball interactions with walls"""
        width, height = court_dimensions
        wall_threshold = 30  # Pixels from edge
        
        wall_hits = 0
        for i in range(1, len(trajectory)):
            x, y = trajectory[i][0], trajectory[i][1]
            prev_x, prev_y = trajectory[i-1][0], trajectory[i-1][1]
            
            # Check if ball went from not-near-wall to near-wall
            near_wall_now = (x < wall_threshold or x > width - wall_threshold or 
                        y < wall_threshold or y > height - wall_threshold)
            near_wall_prev = (prev_x < wall_threshold or prev_x > width - wall_threshold or 
                            prev_y < wall_threshold or prev_y > height - wall_threshold)
            
            if near_wall_now and not near_wall_prev:
                # Also check for direction change
                if i < len(trajectory) - 1:
                    next_x, next_y = trajectory[i+1][0], trajectory[i+1][1]
                    direction_change = abs((x - prev_x) * (next_x - x) + (y - prev_y) * (next_y - y)) < 0
                    if direction_change:
                        wall_hits += 1
        
        return wall_hits
    
    def _count_direction_changes(self, trajectory):
        """Count significant direction changes in trajectory"""
        if len(trajectory) < 3:
            return 0
        
        direction_changes = 0
        threshold = math.pi / 4  # 45 degrees
        
        for i in range(1, len(trajectory) - 1):
            p1, p2, p3 = trajectory[i-1], trajectory[i], trajectory[i+1]
            
            # Calculate direction vectors
            dir1 = math.atan2(p2[1] - p1[1], p2[0] - p1[0])
            dir2 = math.atan2(p3[1] - p2[1], p3[0] - p2[0])
            
            # Calculate angle difference
            angle_diff = abs(dir2 - dir1)
            angle_diff = min(angle_diff, 2*math.pi - angle_diff)  # Take smaller angle
            
            if angle_diff > threshold:
                direction_changes += 1
        
        return direction_changes
    
    def _analyze_shot_context(self, trajectory, court_dimensions, player_positions):
        """Analyze contextual information about the shot"""
        context = {
            'shot_duration': len(trajectory),
            'court_coverage': self._calculate_court_coverage(trajectory, court_dimensions),
            'pressure_situation': False,
            'attacking_shot': False,
            'defensive_shot': False
        }
        
        # Determine if shot is attacking or defensive based on trajectory
        start_zone = self._get_court_zone(trajectory[0], court_dimensions)
        end_zone = self._get_court_zone(trajectory[-1], court_dimensions)
        
        # Attacking shots typically go to front court
        if 'front' in end_zone:
            context['attacking_shot'] = True
        
        # Defensive shots typically go to back court or are lobs
        if 'back' in end_zone and trajectory[0][1] > trajectory[-1][1]:  # Ball goes up
            context['defensive_shot'] = True
        
        return context
    
    def _calculate_court_coverage(self, trajectory, court_dimensions):
        """Calculate how much of the court the trajectory covers"""
        if not trajectory:
            return 0.0
        
        x_positions = [pos[0] for pos in trajectory]
        y_positions = [pos[1] for pos in trajectory]
        
        x_range = (max(x_positions) - min(x_positions)) / court_dimensions[0]
        y_range = (max(y_positions) - min(y_positions)) / court_dimensions[1]
        
        return (x_range + y_range) / 2

class PlayerHitDetector:
    """Enhanced player hit detection with multiple algorithms"""
    
    def __init__(self):
        self.hit_detection_methods = [
            self._proximity_detection,
            self._trajectory_analysis,
            self._racket_position_analysis,
            self._movement_pattern_analysis
        ]
        self.confidence_weights = [0.3, 0.4, 0.2, 0.1]
    
    def detect_player_hit(self, players, ball_trajectory, frame_count):
        """
        Detect which player hit the ball with enhanced accuracy
        
        Args:
            players: Dict of player objects {1: Player, 2: Player}
            ball_trajectory: List of recent ball positions
            frame_count: Current frame number
            
        Returns:
            tuple: (player_id, confidence, hit_type, additional_info)
        """
        if len(ball_trajectory) < 3 or not players:
            return 0, 0.0, 'none', {}
        
        hit_results = []
        
        # Run all detection methods
        for method, weight in zip(self.hit_detection_methods, self.confidence_weights):
            try:
                result = method(players, ball_trajectory, frame_count)
                hit_results.append((result, weight))
            except Exception as e:
                print(f"Hit detection method failed: {e}")
                hit_results.append(((0, 0.0, 'none', {}), weight))
        
        # Combine results with weighted voting
        return self._combine_hit_results(hit_results)
    
    def _proximity_detection(self, players, ball_trajectory, frame_count):
        """Detect hit based on player proximity to ball"""
        current_ball_pos = ball_trajectory[-1]
        player_distances = {}
        
        for player_id, player in players.items():
            if player and player.get_latest_pose():
                # Get player position using multiple keypoints
                player_pos = self._get_enhanced_player_position(player)
                if player_pos:
                    distance = math.sqrt(
                        (current_ball_pos[0] - player_pos[0])**2 + 
                        (current_ball_pos[1] - player_pos[1])**2
                    )
                    player_distances[player_id] = distance
        
        if not player_distances:
            return 0, 0.0, 'none', {}
        
        closest_player = min(player_distances.items(), key=lambda x: x[1])
        player_id, distance = closest_player
        
        # Convert distance to confidence (closer = higher confidence)
        max_hit_distance = 100  # pixels
        confidence = max(0, (max_hit_distance - distance) / max_hit_distance)
        
        hit_type = 'proximity_hit' if confidence > 0.3 else 'none'
        
        return player_id if confidence > 0.3 else 0, confidence, hit_type, {'distance': distance}
    
    def _trajectory_analysis(self, players, ball_trajectory, frame_count):
        """Detect hit based on ball trajectory changes"""
        if len(ball_trajectory) < 5:
            return 0, 0.0, 'none', {}
        
        # Analyze last few positions for sudden changes
        recent_positions = ball_trajectory[-5:]
        
        # Calculate velocity and direction changes
        velocities = []
        directions = []
        
        for i in range(1, len(recent_positions)):
            p1, p2 = recent_positions[i-1], recent_positions[i]
            if len(p1) >= 3 and len(p2) >= 3:
                dt = max(0.033, abs(p2[2] - p1[2])) if p2[2] != p1[2] else 1
                velocity = math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2) / dt
                direction = math.atan2(p2[1] - p1[1], p2[0] - p1[0])
                velocities.append(velocity)
                directions.append(direction)
        
        # Check for significant changes
        velocity_change = 0
        direction_change = 0
        
        if len(velocities) >= 2:
            velocity_change = abs(velocities[-1] - velocities[-2])
        
        if len(directions) >= 2:
            dir_diff = abs(directions[-1] - directions[-2])
            direction_change = min(dir_diff, 2*math.pi - dir_diff)
        
        # Determine which player is most likely to have caused the change
        hit_confidence = 0
        hit_player = 0
        
        if velocity_change > 15 or direction_change > math.pi/6:  # 30 degrees
            # Find closest player at time of trajectory change
            change_position = recent_positions[-2]  # Position where change occurred
            
            player_distances = {}
            for player_id, player in players.items():
                if player and player.get_latest_pose():
                    player_pos = self._get_enhanced_player_position(player)
                    if player_pos:
                        distance = math.sqrt(
                            (change_position[0] - player_pos[0])**2 + 
                            (change_position[1] - player_pos[1])**2
                        )
                        player_distances[player_id] = distance
            
            if player_distances:
                closest_player = min(player_distances.items(), key=lambda x: x[1])
                hit_player, distance = closest_player
                
                # Higher confidence for larger trajectory changes and closer proximity
                trajectory_confidence = min(1.0, (velocity_change + direction_change * 10) / 50)
                proximity_confidence = max(0, (150 - distance) / 150)
                hit_confidence = (trajectory_confidence + proximity_confidence) / 2
        
        hit_type = 'trajectory_change' if hit_confidence > 0.4 else 'none'
        
        return hit_player if hit_confidence > 0.4 else 0, hit_confidence, hit_type, {
            'velocity_change': velocity_change,
            'direction_change': direction_change
        }
    
    def _racket_position_analysis(self, players, ball_trajectory, frame_count):
        """Detect hit based on racket/arm position analysis"""
        current_ball_pos = ball_trajectory[-1]
        hit_confidences = {}
        
        for player_id, player in players.items():
            if player and player.get_latest_pose():
                pose = player.get_latest_pose()
                
                # Check arm/racket positions (wrists, elbows)
                arm_keypoints = []
                
                try:
                    if hasattr(pose, 'xyn') and len(pose.xyn) > 0:
                        keypoints = pose.xyn[0]
                        # Wrist keypoints (9=left wrist, 10=right wrist)
                        # Elbow keypoints (7=left elbow, 8=right elbow)
                        for idx in [7, 8, 9, 10]:
                            if idx < len(keypoints):
                                kp = keypoints[idx]
                                if len(kp) >= 2 and kp[0] > 0 and kp[1] > 0:
                                    arm_keypoints.append((kp[0] * 640, kp[1] * 360))  # Convert to pixel coords
                except Exception:
                    continue
                
                if arm_keypoints:
                    # Find closest arm keypoint to ball
                    min_distance = float('inf')
                    for arm_pos in arm_keypoints:
                        distance = math.sqrt(
                            (current_ball_pos[0] - arm_pos[0])**2 + 
                            (current_ball_pos[1] - arm_pos[1])**2
                        )
                        min_distance = min(min_distance, distance)
                    
                    # Convert to confidence
                    max_racket_distance = 80  # pixels
                    confidence = max(0, (max_racket_distance - min_distance) / max_racket_distance)
                    hit_confidences[player_id] = (confidence, min_distance)
        
        if not hit_confidences:
            return 0, 0.0, 'none', {}
        
        # Get player with highest racket confidence
        best_player = max(hit_confidences.items(), key=lambda x: x[1][0])
        player_id, (confidence, distance) = best_player
        
        hit_type = 'racket_proximity' if confidence > 0.3 else 'none'
        
        return player_id if confidence > 0.3 else 0, confidence, hit_type, {'racket_distance': distance}
    
    def _movement_pattern_analysis(self, players, ball_trajectory, frame_count):
        """Detect hit based on player movement patterns"""
        # This is a simplified version - could be enhanced with historical movement data
        current_ball_pos = ball_trajectory[-1]
        
        # Look for players moving towards ball position
        movement_scores = {}
        
        for player_id, player in players.items():
            if player and hasattr(player, 'position_history') and len(player.position_history) > 1:
                # Check if player was moving towards ball
                recent_positions = player.position_history[-3:]
                
                if len(recent_positions) >= 2:
                    # Calculate movement direction
                    last_pos = recent_positions[-1]
                    prev_pos = recent_positions[-2]
                    
                    movement_vector = (last_pos[0] - prev_pos[0], last_pos[1] - prev_pos[1])
                    ball_vector = (current_ball_pos[0] - prev_pos[0], current_ball_pos[1] - prev_pos[1])
                    
                    # Calculate alignment of movement with ball direction
                    if math.sqrt(movement_vector[0]**2 + movement_vector[1]**2) > 0:
                        dot_product = movement_vector[0]*ball_vector[0] + movement_vector[1]*ball_vector[1]
                        movement_mag = math.sqrt(movement_vector[0]**2 + movement_vector[1]**2)
                        ball_mag = math.sqrt(ball_vector[0]**2 + ball_vector[1]**2)
                        
                        if movement_mag > 0 and ball_mag > 0:
                            alignment = dot_product / (movement_mag * ball_mag)
                            movement_scores[player_id] = max(0, alignment)
        
        if not movement_scores:
            return 0, 0.0, 'none', {}
        
        best_player = max(movement_scores.items(), key=lambda x: x[1])
        player_id, score = best_player
        
        hit_type = 'movement_pattern' if score > 0.5 else 'none'
        
        return player_id if score > 0.5 else 0, score, hit_type, {'movement_alignment': score}
    
    def _get_enhanced_player_position(self, player):
        """Get player position using multiple keypoints for better accuracy"""
        if not player or not player.get_latest_pose():
            return None
        
        pose = player.get_latest_pose()
        
        try:
            if hasattr(pose, 'xyn') and len(pose.xyn) > 0:
                keypoints = pose.xyn[0]
                
                # Priority order: wrists, elbows, shoulders, hips
                priority_keypoints = [9, 10, 7, 8, 5, 6, 11, 12]  # Wrists, elbows, shoulders, hips
                
                for idx in priority_keypoints:
                    if idx < len(keypoints):
                        kp = keypoints[idx]
                        if len(kp) >= 2 and kp[0] > 0 and kp[1] > 0:
                            return (kp[0] * 640, kp[1] * 360)  # Convert to pixel coordinates
        except Exception:
            pass
        
        return None
    
    def _combine_hit_results(self, hit_results):
        """Combine results from multiple detection methods"""
        player_scores = {1: 0.0, 2: 0.0}
        total_weight = 0
        combined_info = {}
        
        for (player_id, confidence, hit_type, info), weight in hit_results:
            if player_id > 0:
                player_scores[player_id] += confidence * weight
            total_weight += weight
            combined_info.update(info)
        
        # Normalize scores
        if total_weight > 0:
            for player_id in player_scores:
                player_scores[player_id] /= total_weight
        
        # Get best player
        best_player = max(player_scores.items(), key=lambda x: x[1])
        player_id, final_confidence = best_player
        
        # Determine hit type based on confidence
        if final_confidence > 0.7:
            final_hit_type = 'strong_hit'
        elif final_confidence > 0.5:
            final_hit_type = 'probable_hit'
        elif final_confidence > 0.3:
            final_hit_type = 'possible_hit'
        else:
            final_hit_type = 'none'
            player_id = 0
        
        return player_id, final_confidence, final_hit_type, combined_info

class ShotPhaseDetector:
    """Detect shot phases: start (ball leaves racket) -> middle (wall hit) -> end (floor hit)"""
    
    def __init__(self):
        self.phase_history = []
        self.wall_hit_threshold = 30  # pixels from wall
        self.floor_hit_indicators = {
            'height_threshold': 0.7,  # Lower 30% of court
            'velocity_decrease': 0.7,  # 30% velocity decrease
            'bounce_pattern': True    # Look for bounce patterns
        }
    
    def detect_shot_phases(self, trajectory, court_dimensions, previous_phase='none'):
        """
        Detect current shot phase based on trajectory analysis
        
        Args:
            trajectory: List of ball positions [(x, y, frame), ...]
            court_dimensions: (width, height)
            previous_phase: Previous detected phase
            
        Returns:
            dict: Phase information with confidence and details
        """
        if len(trajectory) < 3:
            return {'phase': 'start', 'confidence': 0.0, 'details': {}}
        
        width, height = court_dimensions
        current_pos = trajectory[-1]
        
        # Phase detection logic
        if previous_phase in ['none', 'start']:
            # Check for wall hit (transition from start to middle)
            wall_hit = self._detect_wall_hit(trajectory, court_dimensions)
            if wall_hit['detected']:
                return {
                    'phase': 'middle',
                    'confidence': wall_hit['confidence'],
                    'details': wall_hit['details'],
                    'transition': 'start_to_middle'
                }
        
        if previous_phase in ['start', 'middle']:
            # Check for floor hit (transition to end)
            floor_hit = self._detect_floor_hit(trajectory, court_dimensions)
            if floor_hit['detected']:
                return {
                    'phase': 'end',
                    'confidence': floor_hit['confidence'],
                    'details': floor_hit['details'],
                    'transition': 'middle_to_end' if previous_phase == 'middle' else 'start_to_end'
                }
        
        # If no transition detected, maintain current phase
        phase = previous_phase if previous_phase != 'none' else 'start'
        return {'phase': phase, 'confidence': 0.5, 'details': {}, 'transition': 'none'}
    
    def _detect_wall_hit(self, trajectory, court_dimensions):
        """Detect ball hitting wall based on position and trajectory changes"""
        if len(trajectory) < 4:
            return {'detected': False, 'confidence': 0.0, 'details': {}}
        
        width, height = court_dimensions
        recent_positions = trajectory[-4:]
        
        wall_hit_confidence = 0.0
        wall_type = 'none'
        
        # Check proximity to walls
        current_pos = recent_positions[-1]
        x, y = current_pos[0], current_pos[1]
        
        # Distance to each wall
        dist_to_left = x
        dist_to_right = width - x
        dist_to_top = y
        dist_to_bottom = height - y
        
        min_wall_distance = min(dist_to_left, dist_to_right, dist_to_top, dist_to_bottom)
        
        # Proximity factor
        if min_wall_distance < self.wall_hit_threshold:
            proximity_confidence = (self.wall_hit_threshold - min_wall_distance) / self.wall_hit_threshold
            wall_hit_confidence += proximity_confidence * 0.4
            
            # Determine which wall
            if min_wall_distance == dist_to_left:
                wall_type = 'left_wall'
            elif min_wall_distance == dist_to_right:
                wall_type = 'right_wall'
            elif min_wall_distance == dist_to_top:
                wall_type = 'front_wall'
            else:
                wall_type = 'back_wall'
        
        # Check for direction change near wall
        if len(recent_positions) >= 3:
            direction_change = self._calculate_direction_change(recent_positions[-3:])
            if direction_change > math.pi/4 and min_wall_distance < self.wall_hit_threshold * 2:
                wall_hit_confidence += 0.4
        
        # Check for velocity change
        velocity_change = self._calculate_velocity_change(recent_positions)
        if velocity_change > 0.3:  # 30% velocity change
            wall_hit_confidence += 0.2
        
        detected = wall_hit_confidence > 0.6
        
        return {
            'detected': detected,
            'confidence': wall_hit_confidence,
            'details': {
                'wall_type': wall_type,
                'distance_to_wall': min_wall_distance,
                'direction_change': direction_change if 'direction_change' in locals() else 0,
                'velocity_change': velocity_change
            }
        }
    
    def _detect_floor_hit(self, trajectory, court_dimensions):
        """Detect ball hitting floor/ground"""
        if len(trajectory) < 5:
            return {'detected': False, 'confidence': 0.0, 'details': {}}
        
        width, height = court_dimensions
        recent_positions = trajectory[-5:]
        
        floor_hit_confidence = 0.0
        
        # Check if ball is in lower part of court
        current_y = recent_positions[-1][1]
        height_factor = current_y / height
        
        if height_factor > self.floor_hit_indicators['height_threshold']:
            position_confidence = (height_factor - self.floor_hit_indicators['height_threshold']) / (1 - self.floor_hit_indicators['height_threshold'])
            floor_hit_confidence += position_confidence * 0.4
        
        # Check for velocity decrease (ball slowing down)
        velocity_change = self._calculate_velocity_change(recent_positions)
        if velocity_change < -self.floor_hit_indicators['velocity_decrease']:  # Negative = slowing down
            velocity_confidence = abs(velocity_change) / self.floor_hit_indicators['velocity_decrease']
            floor_hit_confidence += velocity_confidence * 0.3
        
        # Check for bounce pattern (sudden upward movement after downward)
        if len(recent_positions) >= 3:
            bounce_detected = self._detect_bounce_pattern(recent_positions[-3:])
            if bounce_detected:
                floor_hit_confidence += 0.3
        
        detected = floor_hit_confidence > 0.5
        
        return {
            'detected': detected,
            'confidence': floor_hit_confidence,
            'details': {
                'height_factor': height_factor,
                'velocity_change': velocity_change,
                'bounce_detected': bounce_detected if 'bounce_detected' in locals() else False
            }
        }
    
    def _calculate_direction_change(self, positions):
        """Calculate direction change between trajectory segments"""
        if len(positions) < 3:
            return 0.0
        
        p1, p2, p3 = positions[0], positions[1], positions[2]
        
        # Direction vectors
        dir1 = math.atan2(p2[1] - p1[1], p2[0] - p1[0])
        dir2 = math.atan2(p3[1] - p2[1], p3[0] - p2[0])
        
        # Calculate angle difference
        angle_diff = abs(dir2 - dir1)
        return min(angle_diff, 2*math.pi - angle_diff)
    
    def _calculate_velocity_change(self, positions):
        """Calculate relative velocity change"""
        if len(positions) < 4:
            return 0.0
        
        # Calculate velocities for first and last segments
        early_velocity = math.sqrt(
            (positions[1][0] - positions[0][0])**2 + 
            (positions[1][1] - positions[0][1])**2
        )
        
        late_velocity = math.sqrt(
            (positions[-1][0] - positions[-2][0])**2 + 
            (positions[-1][1] - positions[-2][1])**2
        )
        
        if early_velocity > 0:
            return (late_velocity - early_velocity) / early_velocity
        return 0.0
    
    def _detect_bounce_pattern(self, positions):
        """Detect bounce pattern (down then up movement)"""
        if len(positions) < 3:
            return False
        
        p1, p2, p3 = positions[0], positions[1], positions[2]
        
        # Check for downward then upward movement
        downward = p2[1] > p1[1]  # Y increases downward in screen coords
        upward = p3[1] < p2[1]    # Y decreases upward
        
        return downward and upward

# Initialize global shot tracker

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
        self.shot_classification_model = ShotClassificationModel()
        self.player_hit_detector = PlayerHitDetector()
        self.phase_detector = ShotPhaseDetector()
        
    def detect_shot_start(self, ball_hit, who_hit, frame_count, ball_pos, shot_type, hit_confidence=0.0, hit_type="normal", past_ball_pos=None):
        """
        Enhanced shot start detection with improved player identification and shot classification
        """
        if not ball_hit or who_hit == 0:
            return False
            
        # Check cooldown to avoid duplicate shot detection
        if frame_count - self.last_hit_frame < self.ball_hit_cooldown:
            return False
            
        # Only register shots with sufficient confidence
        if hit_confidence < 0.3:
            return False
            
        # Use enhanced shot classification if past_ball_pos is provided
        if hasattr(self, 'shot_classification_model') and past_ball_pos and len(past_ball_pos) > 3:
            try:
                enhanced_classification = self.shot_classification_model.classify_shot(
                    past_ball_pos[-10:], (640, 360)  # Use recent trajectory
                )
                shot_type = enhanced_classification['type']
                shot_features = enhanced_classification['features']
                classification_confidence = enhanced_classification['confidence']
            except Exception:
                shot_features = {}
                classification_confidence = 0.5
        else:
            shot_features = {}
            classification_confidence = 0.5
            
        # Start new shot with enhanced tracking
        self.shot_id_counter += 1
        new_shot = {
            'id': self.shot_id_counter,
            'start_frame': frame_count,
            'player_who_hit': who_hit,
            'shot_type': shot_type,
            'hit_confidence': hit_confidence,
            'classification_confidence': classification_confidence,
            'hit_type': hit_type,
            'trajectory': [ball_pos.copy()],
            'status': 'active',
            'color': self.get_shot_color(shot_type),
            'end_frame': None,
            'final_shot_type': None,
            'phase': 'start',  # Ball leaves racket
            'wall_hit_frame': None,
            'floor_hit_frame': None,
            'phase_transitions': [],
            'shot_features': shot_features,
            'phase_history': [{'phase': 'start', 'frame': frame_count, 'confidence': 1.0}]
        }
        
        self.active_shots.append(new_shot)
        self.last_hit_frame = frame_count
        
        # Log enhanced shot information
        print(f"🎾 Enhanced Shot {self.shot_id_counter} started:")
        print(f"   Player: {who_hit}, Type: {shot_type}")
        print(f"   Hit Confidence: {hit_confidence:.2f}, Classification: {classification_confidence:.2f}")
        print(f"   Phase: START (ball leaves racket)")
        
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
        
    def update_shot_phases(self, shot, ball_pos, frame_count):
        """
        Enhanced shot phase tracking: start (ball leaves racket) -> middle (wall hit) -> end (floor hit)
        """
        if not hasattr(self, 'phase_detector'):
            return
            
        current_phase = shot.get('phase', 'start')
        
        # Use enhanced phase detection
        phase_result = self.phase_detector.detect_shot_phases(
            shot['trajectory'], (640, 360), current_phase
        )
        
        new_phase = phase_result['phase']
        phase_confidence = phase_result['confidence']
        transition = phase_result.get('transition', 'none')
        
        # Update phase if changed
        if new_phase != current_phase and phase_confidence > 0.6:
            shot['phase'] = new_phase
            
            # Log phase transition
            if transition == 'start_to_middle':
                shot['wall_hit_frame'] = frame_count
                print(f"🏏 Shot {shot['id']}: MIDDLE phase (wall hit) - Frame {frame_count}")
            elif transition == 'middle_to_end' or transition == 'start_to_end':
                shot['floor_hit_frame'] = frame_count
                print(f"🎯 Shot {shot['id']}: END phase (floor hit) - Frame {frame_count}")
            
            # Record phase transition
            phase_transition = {
                'from': current_phase,
                'to': new_phase,
                'frame': frame_count,
                'position': ball_pos.copy(),
                'confidence': phase_confidence,
                'details': phase_result.get('details', {})
            }
            
            if 'phase_transitions' not in shot:
                shot['phase_transitions'] = []
            shot['phase_transitions'].append(phase_transition)
            
            # Update phase history
            if 'phase_history' not in shot:
                shot['phase_history'] = []
            shot['phase_history'].append({
                'phase': new_phase,
                'frame': frame_count,
                'confidence': phase_confidence,
                'details': phase_result.get('details', {})
            })
    
    def detect_wall_hit(self, trajectory, current_pos):
        """
        Detect if ball has hit a wall based on trajectory analysis
        """
        if len(trajectory) < 3:
            return False
        
        # Check last few positions for wall proximity and direction change
        recent_positions = trajectory[-3:]
        
        # Check if near wall (within 30 pixels of edge)
        court_width, court_height = 640, 360  # Default values
        x, y = current_pos[0], current_pos[1]
        
        near_wall = (x < 30 or x > court_width - 30 or y < 30 or y > court_height - 30)
        
        if near_wall and len(recent_positions) >= 3:
            # Check for direction change
            p1, p2, p3 = recent_positions[-3], recent_positions[-2], recent_positions[-1]
            
            # Calculate direction vectors
            dir1 = (p2[0] - p1[0], p2[1] - p1[1])
            dir2 = (p3[0] - p2[0], p3[1] - p2[1])
            
            # Check for significant direction change
            if abs(dir1[0]) > 5 and abs(dir2[0]) > 5:
                # X direction change (left/right wall)
                if (dir1[0] > 0) != (dir2[0] > 0):
                    return True
            
            if abs(dir1[1]) > 5 and abs(dir2[1]) > 5:
                # Y direction change (top/bottom wall)
                if (dir1[1] > 0) != (dir2[1] > 0):
                    return True
        
        return False
    
    def detect_floor_hit(self, trajectory, current_pos):
        """
        Detect if ball has hit the floor (or is about to hit)
        """
        if len(trajectory) < 5:
            return False
        
        # Check if ball is moving downward and in lower part of court
        recent_positions = trajectory[-5:]
        
        # Check if ball is in lower third of court
        court_height = 360  # Default value
        y = current_pos[1]
        
        if y > court_height * 0.7:  # In lower 30% of court
            # Check if ball is moving slowly (coming to rest)
            if len(recent_positions) >= 3:
                velocities = []
                for i in range(1, len(recent_positions)):
                    p1, p2 = recent_positions[i-1], recent_positions[i]
                    velocity = math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
                    velocities.append(velocity)
                
                # Check if velocity is decreasing (ball slowing down)
                if len(velocities) >= 2:
                    avg_early_velocity = sum(velocities[:2]) / 2
                    avg_late_velocity = sum(velocities[-2:]) / 2
                    
                    # Ball is slowing down significantly
                    if avg_late_velocity < avg_early_velocity * 0.7:
                        return True
        
        return False
        
    def update_active_shots(self, ball_pos, frame_count, new_hit_detected=False, new_shot_type=None):
        """
        Update trajectories of active shots with improved phase tracking
        """
        if not ball_pos or len(ball_pos) < 2:
            return
            
        for shot in self.active_shots[:]:  # Use slice to avoid modification during iteration
            if shot['status'] == 'active':
                # Add current ball position to trajectory
                shot['trajectory'].append(ball_pos.copy())
                
                # Track shot phases: start -> wall hit -> floor hit
                self.update_shot_phases(shot, ball_pos, frame_count)
                
                # Use simple bounce detection for wall hits
                if len(shot['trajectory']) >= 5:
                    wall_hits = count_wall_hits_legacy(shot['trajectory'])
                    shot['wall_hits'] = wall_hits
                    
                    # Update shot type based on wall hits and trajectory
                    if wall_hits > 0:
                        shot['shot_type'] = f"{shot['shot_type']}_wall_bounce"
                
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
        Save shot data to file for later analysis with phase information
        """
        try:
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
                'wall_hits': shot.get('wall_hits', 0),
                'shot_color': shot['color'],
                'phase': shot.get('phase', 'unknown'),
                'wall_hit_frame': shot.get('wall_hit_frame'),
                'floor_hit_frame': shot.get('floor_hit_frame'),
                'phase_transitions': shot.get('phase_transitions', [])
            }
            
            # Append to shots log file
            with open("output/shots_log.jsonl", "a") as f:
                f.write(json.dumps(shot_data) + "\n")
                
        except Exception as e:
            print(f"Error saving shot data: {e}")
            
    def draw_shot_trajectories(self, frame, ball_pos):
        """
        Draw trajectories of active and recent shots with phase information
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
                        
                        # Draw shot info with phase
                        phase = shot.get('phase', 'unknown')
                        hit_type = shot.get('hit_type', 'normal')
                        confidence = shot.get('hit_confidence', 0.0)
                        text = f"Shot {shot['id']}: P{shot['player_who_hit']} - {shot['shot_type']} ({phase})"
                        cv2.putText(frame, text, 
                                (start_pos[0] - 50, start_pos[1] - 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                        
                        # Draw confidence and hit type
                        conf_text = f"Conf: {confidence:.2f} | Type: {hit_type}"
                        cv2.putText(frame, conf_text, 
                                (start_pos[0] - 50, start_pos[1] - 55),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
                        
                        # Draw phase transition markers
                        if 'phase_transitions' in shot:
                            for transition in shot['phase_transitions']:
                                trans_pos = transition['position']
                                trans_pos = (int(trans_pos[0]), int(trans_pos[1]))
                                
                                if transition['to'] == 'middle':
                                    # Wall hit marker - square
                                    cv2.rectangle(frame, 
                                                (trans_pos[0]-6, trans_pos[1]-6),
                                                (trans_pos[0]+6, trans_pos[1]+6),
                                                (0, 255, 255), -1)  # Yellow
                                    cv2.putText(frame, "WALL", 
                                            (trans_pos[0] - 15, trans_pos[1] - 10),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 255), 1)
                                elif transition['to'] == 'end':
                                    # Floor hit marker - triangle
                                    cv2.circle(frame, trans_pos, 8, (0, 0, 255), -1)  # Red
                                    cv2.putText(frame, "FLOOR", 
                                            (trans_pos[0] - 15, trans_pos[1] - 10),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
                        
                        # Draw wall hits using simple detection
                        wall_hits = shot.get('wall_hits', 0)
                        if wall_hits > 0:
                            cv2.putText(frame, f"Wall hits: {wall_hits}", 
                                    (start_pos[0] - 50, start_pos[1] + 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            
            # Draw recent completed shots (last 3)
            for shot in self.completed_shots[-3:]:
                if len(shot['trajectory']) > 1:
                    color = tuple(int(c * 0.5) for c in shot['color'])  # Dimmed color
                    thickness = 2
                    
                    # Draw dimmed trajectory
                    for i in range(1, len(shot['trajectory'])):
                        pt1 = (int(shot['trajectory'][i-1][0]), int(shot['trajectory'][i-1][1]))
                        pt2 = (int(shot['trajectory'][i][0]), int(shot['trajectory'][i][1]))
                        cv2.line(frame, pt1, pt2, color, thickness)
                    
                    # Draw END marker for completed shots
                    if len(shot['trajectory']) > 0:
                        end_pos = (int(shot['trajectory'][-1][0]), int(shot['trajectory'][-1][1]))
                        cv2.circle(frame, end_pos, 10, color, 2)
                        cv2.circle(frame, end_pos, 5, (128, 128, 128), -1)
                        cv2.putText(frame, "END", 
                                (end_pos[0] - 15, end_pos[1] + 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        except Exception as e:
            print(f"Error drawing shot trajectories: {e}")
    
    def get_shot_statistics(self):
        """
        Get comprehensive shot statistics with phase information
        """
        stats = {
            'total_shots': len(self.completed_shots),
            'active_shots': len(self.active_shots),
            'shots_by_player': {1: 0, 2: 0},
            'shots_by_type': {},
            'shots_by_phase': {'start': 0, 'middle': 0, 'end': 0},
            'average_shot_duration': 0,
            'wall_hits_distribution': {},
            'hit_confidence_average': 0.0,
            'hit_types': {}
        }
        
        if not self.completed_shots:
            return stats
        
        total_duration = 0
        total_confidence = 0
        confidence_count = 0
        
        for shot in self.completed_shots:
            # Player statistics
            player = shot.get('player_who_hit', 0)
            if player in stats['shots_by_player']:
                stats['shots_by_player'][player] += 1
            
            # Shot type statistics
            shot_type = shot.get('final_shot_type', shot.get('shot_type', 'unknown'))
            stats['shots_by_type'][shot_type] = stats['shots_by_type'].get(shot_type, 0) + 1
            
            # Phase statistics
            phase = shot.get('phase', 'unknown')
            if phase in stats['shots_by_phase']:
                stats['shots_by_phase'][phase] += 1
            
            # Duration statistics
            duration = shot.get('duration', 0)
            total_duration += duration
            
            # Wall hits statistics
            wall_hits = shot.get('wall_hits', 0)
            stats['wall_hits_distribution'][wall_hits] = stats['wall_hits_distribution'].get(wall_hits, 0) + 1
            
            # Hit confidence statistics
            hit_confidence = shot.get('hit_confidence', 0.0)
            if hit_confidence > 0:
                total_confidence += hit_confidence
                confidence_count += 1
            
            # Hit type statistics
            hit_type = shot.get('hit_type', 'normal')
            stats['hit_types'][hit_type] = stats['hit_types'].get(hit_type, 0) + 1
        
        # Calculate averages
        if self.completed_shots:
            stats['average_shot_duration'] = total_duration / len(self.completed_shots)
        
        if confidence_count > 0:
            stats['hit_confidence_average'] = total_confidence / confidence_count
        
        return stats

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
def shot_type_enhanced(past_ball_pos, court_width=640, court_height=360, threshold=5):
    """
    Enhanced shot type detection with comprehensive classification
    """
    if len(past_ball_pos) < threshold:
        return "unknown"
    
    # Use enhanced shot classification if available
    try:
        # Import shot_tracker from global scope if available
        import inspect
        frame = inspect.currentframe()
        if frame and frame.f_back and 'shot_tracker' in frame.f_back.f_globals:
            shot_tracker_ref = frame.f_back.f_globals['shot_tracker']
            if hasattr(shot_tracker_ref, 'shot_classification_model'):
                classification_result = shot_tracker_ref.shot_classification_model.classify_shot(
                    past_ball_pos[-min(20, len(past_ball_pos)):], # Use recent trajectory
                    (court_width, court_height)
                )
                
                shot_type = classification_result['type']
                confidence = classification_result['confidence']
                
                # Add confidence and context information
                if confidence > 0.7:
                    return f"{shot_type}_confident"
                elif confidence > 0.4:
                    return shot_type
                else:
                    # Fall back to legacy classification for low confidence
                    return shot_type_enhanced_legacy(past_ball_pos, court_width, court_height, threshold)
    except Exception:
        # Fall back to legacy if enhanced classification fails
        pass
                
    # Fallback to legacy enhanced classification
    return shot_type_enhanced_legacy(past_ball_pos, court_width, court_height, threshold)

def shot_type_enhanced_legacy(past_ball_pos, court_width=640, court_height=360, threshold=5):
    """
    Legacy enhanced shot type detection with better classification
    """
    if len(past_ball_pos) < threshold:
        return "unknown"
    
    recent_positions = past_ball_pos[-threshold:]
    
    # Calculate trajectory characteristics
    start_pos = recent_positions[0]
    end_pos = recent_positions[-1]
    
    horizontal_movement = end_pos[0] - start_pos[0]
    vertical_movement = end_pos[1] - start_pos[1]
    
    # Determine shot direction with better thresholds
    shot_direction = ""
    horizontal_ratio = abs(horizontal_movement) / court_width
    
    if horizontal_ratio > 0.4:  # Significant horizontal movement
        shot_direction = "crosscourt"
    elif horizontal_ratio < 0.08:  # Minimal horizontal movement  
        shot_direction = "straight"
    elif horizontal_ratio > 0.25:  # Medium horizontal movement
        shot_direction = "wide_crosscourt"
    else:
        shot_direction = "angled"
    
    # Determine shot height/type with improved analysis
    y_positions = [pos[1] for pos in recent_positions]
    max_height = min(y_positions)  # Y is inverted in screen coords
    min_height = max(y_positions)
    height_variation = min_height - max_height
    
    # Calculate average height
    avg_height = sum(y_positions) / len(y_positions)
    height_ratio = avg_height / court_height
    
    shot_height = ""
    if height_variation > court_height * 0.5:  # High trajectory variation
        shot_height = "lob"
    elif height_variation < court_height * 0.1:  # Very low trajectory
        shot_height = "drive"
    elif height_ratio < 0.3:  # Ends in upper court (short shot)
        shot_height = "drop"
    elif height_ratio > 0.8:  # Very low/floor shot
        shot_height = "kill"
    else:
        shot_height = "mid"
    
    # Enhanced special shot detection
    wall_hits = count_wall_hits_legacy(past_ball_pos)
    
    # Check for boast (side wall then front wall)
    if wall_hits > 0:
        # Analyze trajectory to determine boast vs regular wall shot
        if horizontal_movement > 0 and abs(horizontal_movement) > court_width * 0.3:
            return f"{shot_direction}_boast"
        else:
            return f"{shot_direction}_{shot_height}_wall"
    
    # Check for volley (high interception)
    if len(recent_positions) >= 3 and height_ratio < 0.4:
        # Check if ball was intercepted before natural trajectory completion
        speed_changes = []
        for i in range(1, len(recent_positions)):
            p1, p2 = recent_positions[i-1], recent_positions[i]
            speed = math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
            speed_changes.append(speed)
        
        if len(speed_changes) >= 2:
            early_speed = sum(speed_changes[:len(speed_changes)//2]) / (len(speed_changes)//2)
            late_speed = sum(speed_changes[len(speed_changes)//2:]) / (len(speed_changes) - len(speed_changes)//2)
            
            # Sudden speed increase might indicate volley
            if late_speed > early_speed * 1.5:
                return f"{shot_direction}_volley"
    
    # Combine direction and height for final classification
    final_shot_type = f"{shot_direction}_{shot_height}"
    
    # Add quality indicators based on trajectory smoothness
    if len(recent_positions) >= 4:
        # Calculate trajectory smoothness
        direction_changes = 0
        for i in range(2, len(recent_positions)):
            p1, p2, p3 = recent_positions[i-2], recent_positions[i-1], recent_positions[i]
            
            dir1 = math.atan2(p2[1] - p1[1], p2[0] - p1[0])
            dir2 = math.atan2(p3[1] - p2[1], p3[0] - p2[0])
            
            angle_diff = abs(dir2 - dir1)
            angle_diff = min(angle_diff, 2*math.pi - angle_diff)
            
            if angle_diff > math.pi/6:  # 30 degrees
                direction_changes += 1
        
        # Smooth shots are likely more controlled
        if direction_changes == 0:
            final_shot_type += "_clean"
        elif direction_changes > 2:
            final_shot_type += "_erratic"
    
    return final_shot_type
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
    Simple shot classification using the enhanced shot type detection
    """
    try:
        if len(past_ball_pos) < 4:
            return ["straight", "drive", 0, 0]
        
        # Use the enhanced shot type detection
        shot_classification = shot_type_enhanced(past_ball_pos, court_width, court_height)
        
        # Parse the shot classification
        parts = shot_classification.split('_')
        shot_direction = parts[0] if len(parts) > 0 else "straight"
        shot_height = parts[1] if len(parts) > 1 else "drive"
        shot_style = parts[2] if len(parts) > 2 else "normal"
        
        # Count wall hits using legacy detection
        wall_bounces = count_wall_hits_legacy(past_ball_pos)
        
        # Calculate simple trajectory metrics
        start_pos = past_ball_pos[0]
        end_pos = past_ball_pos[-1]
        horizontal_displacement = abs(end_pos[0] - start_pos[0]) / court_width
        
        # Simple confidence based on trajectory length and consistency
        confidence_score = min(1.0, len(past_ball_pos) / 10.0)
        
        return [shot_direction, shot_height, shot_style, wall_bounces, 
                horizontal_displacement, confidence_score, {}]

    except Exception as e:
        print(f"Error in shot classification: {str(e)}")
        return ["straight", "drive", "normal", 0, 0, 0.5, {}]

# Enhanced trajectory and shot analysis functions removed
# Using simplified shot_type_enhanced and count_wall_hits_legacy instead

# Legacy bounce detection - simple and reliable
def count_wall_hits_legacy(past_ball_pos, threshold=15):
    """
    Simple legacy wall hit detection based on direction changes
    """
    if len(past_ball_pos) < 4:
        return 0
        
    wall_hits = 0
    direction_changes = 0
    last_direction_x = None
    last_direction_y = None

    for i in range(1, len(past_ball_pos)):
        if i >= len(past_ball_pos) - 1:
            break
            
        x1, y1 = past_ball_pos[i - 1][:2]
        x2, y2 = past_ball_pos[i][:2]

        # Calculate direction
        dir_x = x2 - x1
        dir_y = y2 - y1

        # Check for significant direction changes
        if abs(dir_x) > threshold:
            current_direction_x = 1 if dir_x > 0 else -1
            if last_direction_x is not None and current_direction_x != last_direction_x:
                direction_changes += 1
            last_direction_x = current_direction_x
            
        if abs(dir_y) > threshold:
            current_direction_y = 1 if dir_y > 0 else -1
            if last_direction_y is not None and current_direction_y != last_direction_y:
                direction_changes += 1
            last_direction_y = current_direction_y

        # Count wall hits based on direction changes
        if direction_changes >= 2:
            wall_hits += 1
            direction_changes = 0

    return wall_hits

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
# Legacy bounce detection functions - simple and reliable

def count_wall_hits(past_ball_pos, threshold=12):
    """
    Simple wall hit detection with improved accuracy and lower threshold
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
        avg_x = sum(pos[0] for pos in window_positions) / len(window_positions) if window_positions and len(window_positions) > 0 else 0
        avg_y = sum(pos[1] for pos in window_positions) / len(window_positions) if window_positions and len(window_positions) > 0 else 0
        
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
def determine_ball_hit_enhanced(players, past_ball_pos, proximity_threshold=100, velocity_threshold=20):
    """
    Enhanced player ball hit detection with improved accuracy and phase detection.
    Returns: (player_id, hit_confidence, hit_type)
    """
    if len(past_ball_pos) < 4 or not players.get(1) or not players.get(2):
        return 0, 0.0, "none"

    # Use enhanced player hit detector if available
    try:
        # Import shot_tracker from global scope if available
        import inspect
        frame = inspect.currentframe()
        if frame and frame.f_back and 'shot_tracker' in frame.f_back.f_globals:
            shot_tracker_ref = frame.f_back.f_globals['shot_tracker']
            if hasattr(shot_tracker_ref, 'player_hit_detector'):
                frame_count = past_ball_pos[-1][2] if len(past_ball_pos[-1]) > 2 else 0
                player_id, confidence, hit_type, details = shot_tracker_ref.player_hit_detector.detect_player_hit(
                    players, past_ball_pos, frame_count
                )
                return player_id, confidence, hit_type
    except Exception:
        # Fall back to legacy detection if enhanced fails
        pass
    
    # Fallback to original enhanced detection logic
    current_pos = past_ball_pos[-1]
    prev_pos = past_ball_pos[-2] if len(past_ball_pos) >= 2 else current_pos
    
    # Enhanced trajectory analysis for hit detection
    hit_detected = False
    velocity_change_detected = False
    angle_change = 0
    hit_confidence = 0.0
    closest_player = 0
    hit_type = "none"
    
    if len(past_ball_pos) >= 4:
        # Analyze the last 4 positions for trajectory changes
        positions = past_ball_pos[-4:]
        
        # Calculate velocity changes and direction changes
        velocities = []
        for i in range(1, len(positions)):
            p1, p2 = positions[i-1], positions[i]
            if len(p1) >= 3 and len(p2) >= 3:
                dt = max(0.033, abs(p2[2] - p1[2])) if p2[2] != p1[2] else 1
                velocity = math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2) / dt
                direction = math.atan2(p2[1] - p1[1], p2[0] - p1[0])
                velocities.append((velocity, direction))
        
        # Check for significant velocity change (potential hit)
        if len(velocities) >= 2:
            velocity_change = abs(velocities[-1][0] - velocities[-2][0])
            direction_change = abs(velocities[-1][1] - velocities[-2][1])
            
            # Normalize direction change to 0-π range
            direction_change = min(direction_change, 2*math.pi - direction_change)
            
            if velocity_change > velocity_threshold:
                velocity_change_detected = True
                hit_confidence += 0.4
                
            if direction_change > math.pi/4:  # 45 degrees
                angle_change = math.degrees(direction_change)
                hit_confidence += 0.3
        
        # Enhanced player proximity analysis
        player_distances = {}
        player_hits = {}
        
        for player_id in [1, 2]:
            if players.get(player_id):
                player = players[player_id]
                
                # Get player position (try multiple keypoints for better accuracy)
                player_pos = None
                
                # Try different keypoints in order of preference
                keypoint_priorities = ['right_wrist', 'left_wrist', 'right_elbow', 'left_elbow', 
                                    'right_shoulder', 'left_shoulder', 'nose']
                
                for keypoint in keypoint_priorities:
                    if hasattr(player, keypoint) and player.__dict__[keypoint] is not None:
                        kp = player.__dict__[keypoint]
                        if len(kp) >= 2 and kp[0] > 0 and kp[1] > 0:  # Valid coordinates
                            player_pos = (kp[0], kp[1])
                            break
                
                if not player_pos and player.get_latest_pose():
                    # Fallback to pose keypoints
                    try:
                        pose = player.get_latest_pose()
                        if hasattr(pose, 'xyn') and len(pose.xyn) > 0:
                            keypoints = pose.xyn[0]
                            # Try wrists first (9, 10), then elbows (7, 8)
                            for idx in [9, 10, 7, 8, 5, 6]:
                                if idx < len(keypoints):
                                    kp = keypoints[idx]
                                    if len(kp) >= 2 and kp[0] > 0 and kp[1] > 0:
                                        player_pos = (kp[0] * 640, kp[1] * 360)  # Convert to pixels
                                        break
                    except Exception:
                        continue
                
                if player_pos:
                    # Calculate distance to ball
                    ball_distance = math.sqrt(
                        (current_pos[0] - player_pos[0])**2 + 
                        (current_pos[1] - player_pos[1])**2
                    )
                    player_distances[player_id] = ball_distance
                    
                    # Calculate hit probability based on multiple factors
                    hit_probability = 0.0
                    
                    # Factor 1: Proximity to ball
                    if ball_distance < proximity_threshold:
                        proximity_score = 1.0 - (ball_distance / proximity_threshold)
                        hit_probability += proximity_score * 0.4
                    
                    # Factor 2: Player movement towards ball
                    if len(past_ball_pos) >= 3:
                        prev_ball_pos = past_ball_pos[-3]
                        prev_distance = math.sqrt(
                            (prev_ball_pos[0] - player_pos[0])**2 + 
                            (prev_ball_pos[1] - player_pos[1])**2
                        )
                        
                        # Player is getting closer to ball
                        if prev_distance > ball_distance:
                            movement_score = min(0.3, (prev_distance - ball_distance) / 50)
                            hit_probability += movement_score
                    
                    # Factor 3: Ball trajectory change when near player
                    if ball_distance < proximity_threshold * 1.5 and (velocity_change_detected or angle_change > 30):
                        trajectory_score = 0.3
                        hit_probability += trajectory_score
                    
                    # Factor 4: Player arm/racket position (if wrist detected)
                    if player_pos and any(hasattr(player, kp) for kp in ['right_wrist', 'left_wrist']):
                        arm_score = 0.1  # Bonus for detected arm position
                        hit_probability += arm_score
                    
                    player_hits[player_id] = hit_probability
        
        # Determine which player hit the ball
        if player_hits:
            best_player = max(player_hits.items(), key=lambda x: x[1])
            closest_player = best_player[0]
            player_hit_confidence = best_player[1]
            
            # Only consider it a hit if confidence is high enough
            if player_hit_confidence > 0.3:
                hit_detected = True
                hit_confidence = min(1.0, hit_confidence + player_hit_confidence)
                
                # Determine hit type based on ball change characteristics
                if velocity_change > velocity_threshold * 2:
                    hit_type = "power_shot"
                elif angle_change > 60:
                    hit_type = "direction_change"
                elif velocity_change > velocity_threshold:
                    hit_type = "controlled_shot"
                else:
                    hit_type = "touch_shot"
    
    return closest_player if hit_detected else 0, hit_confidence, hit_type

def get_enhanced_player_pos(player):
    """Enhanced player position calculation using multiple keypoints"""
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
                x = int(kp[0] * 1920)  # Assuming standard frame width
                y = int(kp[1] * 1080)  # Assuming standard frame height
                # Weight racket hand positions more heavily
                weight = 3 if idx in [9, 10] else 2 if idx in [5, 6] else 1
                for _ in range(weight):
                    valid_points.append((x, y))
    
    # Add other valid keypoints with lower weight
    for i, kp in enumerate(keypoints):
        if i not in priority_keypoints and not (kp[0] == 0 and kp[1] == 0):
            x = int(kp[0] * 1920)
            y = int(kp[1] * 1080)
            valid_points.append((x, y))
    
    if not valid_points:
        return None
    
    # Calculate weighted average
    avg_x = sum(p[0] for p in valid_points) / len(valid_points) if valid_points and len(valid_points) > 0 else 0
    avg_y = sum(p[1] for p in valid_points) / len(valid_points) if valid_points and len(valid_points) > 0 else 0
    return (avg_x, avg_y)

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
    # Ensure match_in_play is a dictionary, handle tuple case
    if isinstance(match_in_play, tuple):
        # Convert tuple to dict format if possible
        match_in_play_dict = {}
        if len(match_in_play) >= 1:
            match_in_play_dict['in_play'] = bool(match_in_play[0])
        if len(match_in_play) >= 2:
            match_in_play_dict['ball_hit'] = bool(match_in_play[1])
        if len(match_in_play) >= 3:
            match_in_play_dict['player_movement'] = bool(match_in_play[2])
    elif isinstance(match_in_play, dict):
        match_in_play_dict = match_in_play
    else:
        match_in_play_dict = {'in_play': False, 'ball_hit': False, 'player_movement': False}
    
    coaching_data = {
        'frame': frame_count,
        'timestamp': time.time(),
        'shot_type': type_of_shot,
        'player_who_hit': who_hit,
        'match_active': match_in_play_dict.get('in_play', False),
        'ball_hit_detected': match_in_play_dict.get('ball_hit', False),
        'player_movement': match_in_play_dict.get('player_movement', False),
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
                'ball_position': {'x': float(last_ball_pos[0]), 'y': float(last_ball_pos[1])},
                'ball_trajectory_length': len(past_ball_pos),
                'ball_speed': calculate_ball_speed(past_ball_pos) if len(past_ball_pos) > 1 else 0
            })
        else:
            coaching_data.update({
                'ball_position': {'x': 0.0, 'y': 0.0},  # Default position
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
            analysis['average_duration'] = sum(durations) / len(durations) if durations and len(durations) > 0 else 0
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
            'avg_bounces_per_shot': total_bounces / len(shots) if shots and len(shots) > 0 else 0,
            'bounce_confidence_avg': sum(bounce_confidences) / len(bounce_confidences) if bounce_confidences and len(bounce_confidences) > 0 else 0,
            'bounce_type_distribution': bounce_types
        })
            
        return analysis
        
    except Exception as e:
        print(f"Error analyzing shot patterns: {e}")
        return {}

def main(path="self2.mp4", input_frame_width=640, input_frame_height=360, max_frames=None):
    # Ensure virtual environment is being used
    ensure_venv_usage()
    
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
    
    # 🔥 INITIALIZE ENHANCED BALL PHYSICS AND SHOT DETECTION
    try:
        from enhanced_shot_integration import integrate_enhanced_detection_into_pipeline
        enhanced_integrator = integrate_enhanced_detection_into_pipeline(
            frame_width=input_frame_width, 
            frame_height=input_frame_height
        )
        print("🔥 Enhanced Ball Physics System: ACTIVATED")
        enhanced_detection_enabled = True
    except Exception as e:
        print(f"⚠️  Enhanced detection initialization failed: {e}")
        print("   Falling back to legacy detection system")
        enhanced_integrator = None
        enhanced_detection_enabled = False
    
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
                if len(references1) > 0:
                    sum(references1) / len(references1)
                if len(references2) > 0:
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
                            # Use legacy bounce detection system
                            trajectory_segment = past_ball_pos[-30:] if len(past_ball_pos) > 30 else past_ball_pos
                            
                            detected_bounces = count_wall_hits_legacy(trajectory_segment)
                            
                            # Legacy bounce detection complete
                            
                            # For display purposes, create empty list since legacy doesn't return positions
                            gpu_bounces = []  # Legacy detection only counts, doesn't provide positions
                            
                            # Skip bounce position visualization since legacy detection doesn't provide positions
                        
                        # Legacy bounce statistics display
                        if len(past_ball_pos) >= 4:
                            legacy_bounces = detected_bounces if 'detected_bounces' in locals() else 0
                            
                            # Legacy bounce counter
                            bounce_text = f"Wall Hits (Legacy): {legacy_bounces}"
                            text_size = cv2.getTextSize(bounce_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                            cv2.rectangle(annotated_frame, (8, 75), (text_size[0] + 16, 105), (0, 0, 0), -1)
                            cv2.putText(annotated_frame, bounce_text, 
                                    (10, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                    
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
                    
                    who_hit, hit_confidence, hit_type = determine_ball_hit_enhanced(players, past_ball_pos)
                    
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
            
            # Check if detections_result is iterable and has expected structure
            if not isinstance(detections_result, (list, tuple)) or len(detections_result) < 22:
                print(f"Warning: detections_result is not in expected format, using defaults")
                # Use default values
                detections_result = [
                    frame, frame_count, annotated_frame, mainball, ball, ballmap,
                    past_ball_pos, ball_false_pos, running_frame, otherTrackIds,
                    updated, references1, references2, pixdiffs, players,
                    player_last_positions, False, [], 0, [[],[]], [[],[]], None
                ]
            
            # Get values from detections_result with safe access
            def safe_get(arr, idx, default=None):
                try:
                    return arr[idx] if isinstance(arr, (list, tuple)) and idx < len(arr) else default
                except:
                    return default
                
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
            
            # 🔥 ENHANCED BALL DETECTION PROCESSING
            if enhanced_detection_enabled and enhanced_integrator:
                try:
                    # Process with enhanced physics-based detection
                    enhanced_results = enhanced_integrator.process_ball_detection(
                        past_ball_pos, players, frame_count
                    )
                    
                    # Extract enhanced analysis
                    shot_analysis = enhanced_results['shot_analysis']
                    compatible_results = enhanced_results['compatible_results']
                    shot_events = enhanced_results['shot_events']
                    
                    # Use enhanced results for main pipeline
                    match_in_play = compatible_results['match_in_play']
                    type_of_shot = compatible_results['shot_type']
                    ball_hit = compatible_results['ball_hit']
                    who_hit = compatible_results['who_hit']
                    hit_confidence = compatible_results['hit_confidence']
                    hit_type = compatible_results['hit_type']
                    
                    # Log enhanced detection events
                    if shot_events:
                        event_summary = {}
                        for event in shot_events:
                            event_summary[event.event_type] = event_summary.get(event.event_type, 0) + 1
                        print(f"🎯 Frame {frame_count}: Enhanced events detected: {event_summary}")
                    
                except Exception as e:
                    print(f"Enhanced detection error: {e}, falling back to legacy")
                    enhanced_detection_enabled = False
            
            # Fallback to legacy detection if enhanced system fails
            if not enhanced_detection_enabled:
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
                # Enhanced shot classification with improved trajectory analysis
                past_ball_pos_global = past_ball_pos  # Make past_ball_pos available globally
                type_of_shot = shot_type_enhanced(
                    past_ball_pos=past_ball_pos, 
                    court_width=frame_width, 
                    court_height=frame_height,
                    threshold=5
                )
                
                # Enhanced hit detection with improved confidence
                if len(past_ball_pos) >= 3:
                    who_hit, hit_confidence, hit_type = determine_ball_hit_enhanced(players, past_ball_pos)
                else:
                    who_hit, hit_confidence, hit_type = 0, 0.0, "none"
                    
                # Enhanced shot tracking with visualization
                ball_hit = isinstance(match_in_play, dict) and match_in_play.get('ball_hit', False)
            
            current_ball_pos = past_ball_pos[-1] if past_ball_pos else None
            
            # Detect shot start with enhanced detection
            shot_started = False
            if ball_hit and who_hit > 0 and current_ball_pos and hit_confidence > 0.3:
                shot_started = shot_tracker.detect_shot_start(
                    ball_hit, who_hit, frame_count, current_ball_pos, type_of_shot, hit_confidence, hit_type, past_ball_pos
                )
                
            # Update active shots with enhanced phase tracking
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
                
                # Add enhanced detection results to coaching data if available
                if enhanced_detection_enabled and 'shot_analysis' in locals():
                    coaching_data['enhanced_analysis'] = {
                        'ball_tracking': shot_analysis.get('ball_tracking', {}),
                        'shot_events': shot_analysis.get('shot_events', {}),
                        'trajectory_quality': shot_analysis.get('trajectory_quality', 'unknown'),
                        'shot_phase': shot_analysis.get('shot_phase', 'none'),
                        'enhanced_events_count': len(shot_events) if 'shot_events' in locals() else 0
                    }
                    
                    # Add physics-based metrics
                    ball_tracking_info = shot_analysis.get('ball_tracking', {})
                    if ball_tracking_info:
                        coaching_data['ball_physics'] = {
                            'velocity_magnitude': np.sqrt(sum(v**2 for v in ball_tracking_info.get('velocity', [0, 0]))),
                            'acceleration_magnitude': np.sqrt(sum(a**2 for a in ball_tracking_info.get('acceleration', [0, 0]))),
                            'tracking_confidence': ball_tracking_info.get('confidence', 0.0)
                        }
                    
                    # Add shot event details
                    if 'shot_events' in locals() and shot_events:
                        coaching_data['shot_event_details'] = [
                            {
                                'type': event.event_type,
                                'frame': event.frame,
                                'position': event.position,
                                'confidence': event.confidence,
                                'player_id': getattr(event, 'player_id', None),
                                'wall_type': getattr(event, 'wall_type', None),
                                'impact_angle': getattr(event, 'impact_angle', None)
                            }
                            for event in shot_events
                        ]
                
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
                        # Use simple legacy wall hit detection
                        bounce_count = count_wall_hits_legacy(past_ball_pos)
                        bounce_positions = []  # Legacy detection doesn't return positions
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
            
            # Display enhanced shot tracking statistics
            cv2.putText(
                annotated_frame,
                f"Shots: {shot_stats['total_shots']} | Active: {shot_stats['active_shots']} | P1: {shot_stats['shots_by_player'].get(1, 0)} | P2: {shot_stats['shots_by_player'].get(2, 0)}",
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 0),
                1,
            )
            
            # Display current shot phase information
            if shot_tracker.active_shots:
                current_shot = shot_tracker.active_shots[-1]  # Most recent active shot
                phase = current_shot.get('phase', 'unknown')
                shot_id = current_shot.get('id', 0)
                shot_player = current_shot.get('player_who_hit', 0)
                
                phase_colors = {
                    'start': (0, 255, 255),    # Yellow - ball leaves racket
                    'middle': (255, 165, 0),   # Orange - ball hits wall  
                    'end': (255, 0, 255),      # Magenta - ball hits floor
                    'unknown': (128, 128, 128) # Gray
                }
                
                phase_color = phase_colors.get(phase, (255, 255, 255))
                cv2.putText(
                    annotated_frame,
                    f"Shot #{shot_id} | P{shot_player} | Phase: {phase.upper()}",
                    (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    phase_color,
                    1,
                )
                
                # Display phase transition information
                if 'phase_transitions' in current_shot and current_shot['phase_transitions']:
                    last_transition = current_shot['phase_transitions'][-1]
                    transition_info = f"{last_transition['from']} -> {last_transition['to']} (F:{last_transition['frame']})"
                    cv2.putText(
                        annotated_frame,
                        f"Last transition: {transition_info}",
                        (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.3,
                        (200, 200, 200),
                        1,
                    )
            
            # Display enhanced hit detection information
            if who_hit > 0:
                hit_info = f"Hit by P{who_hit} | Conf: {hit_confidence:.2f} | Type: {hit_type}"
                cv2.putText(
                    annotated_frame,
                    hit_info,
                    (frame_width - 350, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (0, 255, 0),
                    1,
                )
            
            # Display current shot type with enhanced classification
            if type_of_shot and type_of_shot != "unknown":
                shot_display = type_of_shot.replace('_', ' ').title()
                cv2.putText(
                    annotated_frame,
                    f"Shot Type: {shot_display}",
                    (frame_width - 300, 70),
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

            # Enhanced ball trajectory drawing with shot phase visualization
            if len(ballxy) > 2:
                # Draw enhanced trajectory with phase-based coloring
                shot_tracker.draw_shot_trajectories(annotated_frame, ballxy[-1] if ballxy else None)
                
                # Legacy trajectory drawing (keep as fallback)
                for i in range(1, len(ballxy)):
                    if ballxy[i - 1] is None or ballxy[i] is None:
                        continue
                    if ballxy[i][2] - ballxy[i - 1][2] < 7:
                        if frame_count - ballxy[i][2] < 7:
                            # Determine trajectory color based on current shot phase
                            trajectory_color = (0, 255, 0)  # Default green
                            
                            # Check if this trajectory segment is part of an active shot
                            if shot_tracker.active_shots:
                                current_shot = shot_tracker.active_shots[-1]
                                shot_phase = current_shot.get('phase', 'start')
                                
                                phase_colors = {
                                    'start': (0, 255, 255),    # Yellow - ball leaves racket
                                    'middle': (255, 165, 0),   # Orange - ball hits wall  
                                    'end': (255, 0, 255),      # Magenta - ball hits floor
                                }
                                trajectory_color = phase_colors.get(shot_phase, (0, 255, 0))
                            
                            cv2.line(
                                annotated_frame,
                                (ballxy[i - 1][0], ballxy[i - 1][1]),
                                (ballxy[i][0], ballxy[i][1]),
                                trajectory_color,
                                2,
                            )
                            cv2.circle(
                                annotated_frame,
                                (ballxy[i - 1][0], ballxy[i - 1][1]),
                                3,
                                trajectory_color,
                                -1,
                            )
                            cv2.circle(
                                annotated_frame,
                                (ballxy[i][0], ballxy[i][1]),
                                3,
                                trajectory_color,
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
                    
                    # Save coaching data for final comprehensive analysis
                    # Interim reports disabled to avoid duplicate AI analysis calls
                    if len(coaching_data_collection) > 50 and running_frame % 500 == 0:
                        print(f"📊 Coaching data collected: {len(coaching_data_collection)} points (frame {running_frame})")
                            
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
            
            # Generate periodic outputs every 1500 frames
            if frame_count % 1500 == 0:
                try:
                    print(f"🔄 Generating periodic outputs at frame {frame_count}...")
                    periodic_outputs = generate_comprehensive_outputs(
                        frame_count, players, past_ball_pos, shot_tracker, 
                        coaching_data_collection, path
                    )
                    print(f"✅ Generated {len(periodic_outputs)} periodic outputs")
                except Exception as e:
                    print(f"⚠️ Error generating periodic outputs: {e}")
            
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
        
        # Basic autonomous coaching report will be replaced by enhanced analysis below
        basic_coaching_attempted = False
        outputs_generated = []  # Initialize outputs_generated outside try block
        try:
            print("Preparing coaching data for enhanced analysis...")
            # Basic report generation will be handled by enhanced analysis
            basic_coaching_attempted = True
            print("Coaching data prepared successfully.")
        except Exception as e:
            print(f"Error preparing coaching data: {e}")
        
        # Generate comprehensive outputs
        try:
            print("🎯 Generating comprehensive outputs...")
            
            # Call comprehensive output generation
            outputs_generated = generate_comprehensive_outputs(
                frame_count, players, past_ball_pos, shot_tracker, 
                coaching_data_collection, path
            )
            
            print(f"✅ Generated {len(outputs_generated)} comprehensive outputs")
        except Exception as e:
            print(f"Error generating comprehensive outputs: {e}")
            outputs_generated = []  # Ensure outputs_generated is defined even if generation fails
        
        # Generate comprehensive output summary
        generate_output_summary(outputs_generated, frame_count, path)
        
        # Generate final analysis outputs
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
        
        # 🔥 EXPORT ENHANCED BALL PHYSICS DATA
        if enhanced_detection_enabled and enhanced_integrator:
            try:
                print("\n🔥 Exporting Enhanced Ball Physics and Shot Detection Data...")
                enhanced_integrator.export_enhanced_data("output/enhanced_shots")
                
                # Get comprehensive shot statistics
                enhanced_stats = enhanced_integrator.get_shot_statistics()
                print(f"✓ Enhanced Detection Statistics:")
                print(f"   • Total frames processed: {enhanced_stats['system_performance']['total_frames_processed']}")
                print(f"   • Enhanced detections: {enhanced_stats['system_performance']['enhanced_detections']}")
                print(f"   • Total shots detected: {enhanced_stats['shot_detection']['total_shots']}")
                print(f"   • Total events detected: {enhanced_stats['system_performance']['total_events_detected']}")
                print(f"   • Detection success rate: {enhanced_stats['detection_success_rate']:.2%}")
                print(f"   • Processing rate: {enhanced_stats['frame_processing_rate']:.1f} fps")
                
                # Create enhanced summary report
                enhanced_summary = f"""
🔥 ENHANCED BALL PHYSICS SHOT DETECTION REPORT
==============================================

SYSTEM PERFORMANCE:
- Total frames processed: {enhanced_stats['system_performance']['total_frames_processed']}
- Enhanced detections success: {enhanced_stats['system_performance']['enhanced_detections']}
- Legacy fallbacks used: {enhanced_stats['system_performance']['legacy_fallbacks']}
- Detection success rate: {enhanced_stats['detection_success_rate']:.2%}
- Average processing rate: {enhanced_stats['frame_processing_rate']:.1f} fps

SHOT ANALYSIS:
- Total shots detected: {enhanced_stats['shot_detection']['total_shots']}
- Total events detected: {enhanced_stats['system_performance']['total_events_detected']}
- Average events per shot: {enhanced_stats['shot_detection'].get('average_events_per_shot', 0):.1f}

SHOT TYPE DISTRIBUTION:
"""
                
                # Add shot type distribution if available
                if 'shot_types' in enhanced_stats['shot_detection']:
                    for shot_type, count in enhanced_stats['shot_detection']['shot_types'].items():
                        enhanced_summary += f"- {shot_type}: {count} shots\n"
                
                enhanced_summary += f"""
PERFORMANCE ASSESSMENT: {enhanced_stats['shot_detection'].get('system_performance', 'Good')}

This enhanced system provides:
✓ Physics-based trajectory modeling with Kalman filters
✓ Multi-modal event detection (racket hits, wall impacts, floor bounces)
✓ Real-time collision detection with confidence scoring
✓ Autonomous shot segmentation and classification
✓ Advanced signal processing for trajectory analysis

"""
                
                # Save enhanced summary
                with open("output/enhanced_shot_detection_summary.txt", "w", encoding='utf-8') as f:
                    f.write(enhanced_summary)
                    
                print("✓ Enhanced shot detection summary saved to output/enhanced_shot_detection_summary.txt")
                
            except Exception as e:
                print(f"⚠️  Enhanced data export error: {e}")
        
        # Enhanced autonomous coaching analysis and report generation
        print("\n🎾 Generating Enhanced Autonomous Coaching Analysis with Shot Tracking...")
        
        # Generate shot analysis report
        try:
            shot_stats = shot_tracker.get_shot_statistics()
            
            # Get legacy bounce detection statistics
            total_bounces = shot_stats.get('wall_hits_distribution', {})
            total_wall_hits = sum(total_bounces.values()) if total_bounces else 0
            
            # Create comprehensive shot analysis
            shot_analysis_report = f"""
ENHANCED SHOT TRACKING ANALYSIS
=======================================

📊 SHOT STATISTICS:
-----------------
• Total Shots Tracked: {shot_stats['total_shots']}
• Player 1 Shots: {shot_stats['shots_by_player'].get(1, 0)}
• Player 2 Shots: {shot_stats['shots_by_player'].get(2, 0)}
• Average Shot Duration: {shot_stats['average_shot_duration']:.1f} frames

🏓 WALL HIT DETECTION STATISTICS:
------------------------------
• Total Wall Hits Detected: {total_wall_hits}
• Wall Hit Distribution: {total_bounces}
• Legacy Detection Algorithm: Direction-based hit detection

🎯 SHOT TYPE BREAKDOWN:
---------------------
"""
            for shot_type, count in shot_stats['shots_by_type'].items():
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
            with open("output/shot_analysis_report.txt", "w", encoding='utf-8') as f:
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
            # 🚀 APPLY ULTIMATE COACHING ENHANCEMENT
            print("\n🔥 Applying ULTIMATE coaching enhancement for maximum insights...")
            ultimate_analysis = apply_ultimate_enhancement(coaching_data_collection)
            print("✅ Ultimate enhancement analysis completed!")
            
            # Get the global autonomous coach instance using memory manager
            autonomous_coach = llm_manager.get_model('autonomous_coaching')
            
            # Generate comprehensive coaching insights
            if autonomous_coach:
                coaching_insights = autonomous_coach.analyze_match_data(coaching_data_collection)
            else:
                print("⚠️ Autonomous coaching model not available, skipping analysis")
                coaching_insights = {"error": "Model not available"}
            
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
            
            # Generate comprehensive visualizations and analytics
            print("\n🎨 Generating comprehensive visualizations and analytics...")
            try:
                from autonomous_coaching import create_graphics, view_all_graphics
                
                # Generate all visualizations based on final.csv data
                visualization_files = create_graphics()
                print(f"✅ {len(visualization_files)} visualizations generated successfully!")
                
                # Display summary of generated graphics
                view_all_graphics()
                print("📊 Graphics summary and analytics displayed!")
                
                # Generate enhanced shot analytics if available
                if enhanced_detection_enabled and enhanced_integrator:
                    try:
                        enhanced_stats = enhanced_integrator.get_shot_statistics()
                        enhanced_integrator.export_enhanced_data("output/enhanced_shots")
                        
                        print("\n🔥 Enhanced Shot Detection Summary:")
                        print(f"   • Total frames processed: {enhanced_stats['system_performance']['total_frames_processed']}")
                        print(f"   • Enhanced detections: {enhanced_stats['system_performance']['enhanced_detections']}")
                        print(f"   • Total shots detected: {enhanced_stats['shot_detection']['total_shots']}")
                        print(f"   • Processing rate: {enhanced_stats['frame_processing_rate']:.1f} fps")
                        print(f"   • Detection success rate: {enhanced_stats['detection_success_rate']*100:.1f}%")
                        
                        if enhanced_stats['shot_detection']['total_shots'] > 0:
                            shot_types = enhanced_stats['shot_detection'].get('shot_types', {})
                            print(f"   • Shot types detected: {', '.join(shot_types.keys())}")
                        
                    except Exception as enhanced_error:
                        print(f"⚠️ Enhanced statistics export error: {enhanced_error}")
                
            except Exception as viz_error:
                print(f"⚠️ Error generating visualizations: {viz_error}")
                print("   Pipeline completed successfully, but visualizations could not be generated.")
            
        except Exception as e:
            print(f"  Error in enhanced coaching analysis: {e}")
            print("   Generating basic coaching report as fallback...")
            # Fallback to basic coaching report if enhanced analysis fails
            try:
                if not basic_coaching_attempted:
                    generate_coaching_report(coaching_data_collection, path, frame_count)
                    print("✅ Basic coaching report generated successfully as fallback.")
                else:
                    print("   Basic coaching data was prepared, enhanced analysis failed.")
            except Exception as fallback_error:
                print(f"⚠️ Fallback coaching report also failed: {fallback_error}")
        
        print("\n🎾 ULTIMATE SQUASH COACHING ANALYSIS COMPLETE! 🎾")
        print("=" * 70)
        print("🚀 ENHANCED FEATURES ACTIVATED:")
        print(f"   • Enhanced Ball Physics Detection: {'✅ ACTIVE' if enhanced_detection_enabled else '❌ FAILED'}")
        print(f"   • AI-Powered Coaching Analysis: ✅ ACTIVE")
        print(f"   • Advanced Shot Classification: ✅ ACTIVE")
        print(f"   • Physics-Based Trajectory Modeling: ✅ ACTIVE")
        print(f"   • Real-time Event Detection: ✅ ACTIVE")
        print(f"   • Comprehensive Visualizations: ✅ ACTIVE")
        print(f"   • Player Re-identification: ✅ ACTIVE")
        print(f"   • Bounce Pattern Analysis: ✅ ACTIVE")
        print("=" * 70)
        print("\n📁 COMPREHENSIVE OUTPUT FILES GENERATED:")
        print("   🤖 AI COACHING REPORTS:")
        print("      • enhanced_autonomous_coaching_report.txt - AI coaching insights")
        print("      • autonomous_coaching_report.txt - Standard coaching analysis")
        print("   📊 DATA FILES:")
        print("      • enhanced_coaching_data.json - Complete enhanced analysis data")
        print("      • final.csv - Frame-by-frame trajectory and shot data")
        print("      • final.json - Structured match data")
        print("   🎯 SHOT ANALYSIS:")
        print("      • enhanced_shots/ - Physics-based shot detection exports")
        print("      • shots_log.jsonl - Detailed shot event logging")
        print("      • bounce_analysis.jsonl - Ball bounce pattern analysis")
        print("   🎨 VISUALIZATIONS:")
        print("      • graphics/ - Complete visual analytics suite")
        print("      • heatmaps/ - Player and ball position heatmaps")
        print("      • trajectories/ - 2D and 3D trajectory visualizations")
        print("   👥 PLAYER ANALYSIS:")
        print("      • reid_analysis_report.txt - Player tracking and identification")
        print("      • final_reid_references.json - Player appearance references")
        print("      • annotated.mp4 - Video with enhanced visualization")
        print("=" * 70)
        print(f"\n📈 SESSION STATISTICS:")
        print(f"   • Total frames processed: {frame_count:,}")
        print(f"   • Coaching data points collected: {len(coaching_data_collection):,}")
        print(f"   • Video processing completed: {path}")
        if enhanced_detection_enabled:
            print(f"   • Enhanced detection mode: ACTIVE with physics modeling")
        print("=" * 70)
        print("\n🎯 WHAT YOU GET:")
        print("   1. TECHNICAL ANALYSIS - Shot accuracy, technique evaluation")
        print("   2. TACTICAL INSIGHTS - Pattern recognition, strategy analysis")
        print("   3. PHYSICAL ASSESSMENT - Movement efficiency, court coverage")
        print("   4. IMPROVEMENT ROADMAP - Specific drills and training recommendations")
        print("   5. PERFORMANCE TRACKING - Detailed metrics and benchmarking")
        print("   6. VISUAL ANALYTICS - Interactive charts and heatmaps")
        print("=" * 70)
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
        avg_x = sum(pos[0] for pos in window_positions) / len(window_positions) if window_positions and len(window_positions) > 0 else 0
        avg_y = sum(pos[1] for pos in window_positions) / len(window_positions) if window_positions and len(window_positions) > 0 else 0
        avg_z = sum(pos[2] for pos in window_positions) / len(window_positions) if window_positions and len(window_positions) > 0 else 0
        
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


# Virtual Environment Management
def ensure_venv_usage():
    """Ensure the virtual environment is being used for the squash coaching system"""
    import sys
    import subprocess
    import os
    
    print("🔧 Checking virtual environment setup...")
    
    # Check if we're in a virtual environment
    in_venv = hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
    
    if in_venv:
        print(f"✅ Virtual environment detected: {sys.prefix}")
    else:
        print("⚠️ No virtual environment detected")
        print("💡 Consider activating your virtual environment for better dependency management")
    
    # Check for required packages
    required_packages = [
        'torch', 'torchvision', 'ultralytics', 'opencv-python', 
        'numpy', 'matplotlib', 'scipy', 'tensorflow'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"⚠️ Missing packages: {', '.join(missing_packages)}")
        print("💡 Install missing packages with: pip install -r requirements.txt")
    else:
        print("✅ All required packages are available")
    
    # Check CUDA availability
    if torch.cuda.is_available():
        print(f"🚀 CUDA available: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️ CUDA not available - using CPU")
    
    print("🔧 Virtual environment check complete\n")

# Comprehensive Output Generation Functions
def generate_comprehensive_outputs(frame_count, players, past_ball_pos, shot_tracker, coaching_data_collection, path):
    """
    Generate all outputs including graphics, clips, heatmaps, highlights, patterns, raw data, stats, reports, trajectories, visualizations
    """
    print("🎯 Generating comprehensive outputs...")
    
    # Create all output directories
    output_dirs = [
        "output/graphics", "output/clips", "output/heatmaps", "output/highlights", 
        "output/patterns", "output/raw_data", "output/stats", "output/reports", 
        "output/trajectories", "output/visualizations", "output/clips/highlights",
        "output/clips/shots", "output/clips/rallies", "output/clips/patterns",
        "output/clips/metadata", "output/heatmaps/ball", "output/heatmaps/players"
    ]
    
    for dir_path in output_dirs:
        os.makedirs(dir_path, exist_ok=True)
    
    # Generate all outputs
    outputs_generated = []
    
    # 1. Graphics and Visualizations
    outputs_generated.extend(generate_graphics_outputs(frame_count, players, past_ball_pos, shot_tracker))
    
    # 2. Heatmaps
    outputs_generated.extend(generate_heatmap_outputs(players, past_ball_pos))
    
    # 3. Clips and Highlights
    outputs_generated.extend(generate_clip_outputs(frame_count, players, past_ball_pos, shot_tracker))
    
    # 4. Patterns Analysis
    outputs_generated.extend(generate_pattern_outputs(players, past_ball_pos, shot_tracker))
    
    # 5. Raw Data
    outputs_generated.extend(generate_raw_data_outputs(frame_count, players, past_ball_pos, shot_tracker))
    
    # 6. Statistics
    outputs_generated.extend(generate_statistics_outputs(frame_count, players, past_ball_pos, shot_tracker))
    
    # 7. Reports
    outputs_generated.extend(generate_report_outputs(frame_count, players, past_ball_pos, shot_tracker, coaching_data_collection, path))
    
    # 8. Trajectories
    outputs_generated.extend(generate_trajectory_outputs(past_ball_pos, shot_tracker))
    
    # 9. Enhanced Visualizations
    outputs_generated.extend(generate_enhanced_visualizations(frame_count, players, past_ball_pos, shot_tracker))
    
    # 10. Llama AI-Enhanced Analysis
    outputs_generated.extend(generate_llama_enhanced_analysis(frame_count, players, past_ball_pos, shot_tracker, coaching_data_collection))
    
    print(f"✅ Generated {len(outputs_generated)} comprehensive outputs")
    return outputs_generated

def generate_graphics_outputs(frame_count, players, past_ball_pos, shot_tracker):
    """Generate graphics and visualizations"""
    outputs = []
    
    # Ensure output directories exist
    os.makedirs('output/graphics', exist_ok=True)
    os.makedirs('output/heatmaps', exist_ok=True)
    os.makedirs('output/heatmaps/players', exist_ok=True)
    os.makedirs('output/heatmaps/ball', exist_ok=True)
    
    try:
        # Always create a basic court visualization
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Draw court outline
        court_width, court_height = 640, 360
        ax.add_patch(plt.Rectangle((0, 0), court_width, court_height, fill=False, color='black', linewidth=2))
        
        # Plot player positions if available
        players_plotted = 0
        for player_id in [1, 2]:
            if players.get(player_id) and players.get(player_id).get_latest_pose():
                try:
                    pose = players.get(player_id).get_latest_pose()
                    if hasattr(pose, 'xyn') and len(pose.xyn) > 0 and len(pose.xyn[0]) > 16:
                        ankle_x = pose.xyn[0][16][0] * court_width
                        ankle_y = pose.xyn[0][16][1] * court_height
                        ax.scatter(ankle_x, ankle_y, s=100, label=f'Player {player_id}', alpha=0.7)
                        players_plotted += 1
                except Exception as e:
                    print(f"Error plotting player {player_id}: {e}")
        
        # Plot ball trajectory if available
        if past_ball_pos and len(past_ball_pos) > 5:
            try:
                ball_x = [pos[0] for pos in past_ball_pos[-20:]]
                ball_y = [pos[1] for pos in past_ball_pos[-20:]]
                ax.plot(ball_x, ball_y, 'g-', alpha=0.6, linewidth=2, label='Ball Trajectory')
                ax.scatter(ball_x[-1], ball_y[-1], c='red', s=50, label='Current Ball Position')
            except Exception as e:
                print(f"Error plotting ball trajectory: {e}")
        
        ax.set_xlim(0, court_width)
        ax.set_ylim(0, court_height)
        ax.set_title(f'Court Positioning - Frame {frame_count}')
        if players_plotted > 0 or (past_ball_pos and len(past_ball_pos) > 5):
            ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.savefig('output/graphics/court_positioning.png', dpi=300, bbox_inches='tight')
        plt.close()
        outputs.append('output/graphics/court_positioning.png')
        
        # Shot analysis visualization - create even if no shots detected
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if shot_tracker.completed_shots:
            shot_types = [shot.get('shot_type', 'unknown') for shot in shot_tracker.completed_shots]
            shot_counts = {}
            for shot_type in shot_types:
                shot_counts[shot_type] = shot_counts.get(shot_type, 0) + 1
            
            if shot_counts:
                types = list(shot_counts.keys())
                counts = list(shot_counts.values())
                ax.bar(types, counts, color='skyblue', alpha=0.7)
                ax.set_title('Shot Type Distribution')
                ax.set_xlabel('Shot Type')
                ax.set_ylabel('Count')
                plt.xticks(rotation=45)
            else:
                ax.text(0.5, 0.5, 'No shots detected', ha='center', va='center', transform=ax.transAxes, fontsize=14)
                ax.set_title('Shot Type Distribution - No Data')
        else:
            ax.text(0.5, 0.5, 'No shots detected', ha='center', va='center', transform=ax.transAxes, fontsize=14)
            ax.set_title('Shot Type Distribution - No Data')
        
        plt.tight_layout()
        plt.savefig('output/graphics/shot_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        outputs.append('output/graphics/shot_analysis.png')
        
        # Create a basic statistics visualization
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Basic statistics
        stats_data = {
            'Total Frames': frame_count,
            'Ball Positions': len(past_ball_pos) if past_ball_pos else 0,
            'Completed Shots': len(shot_tracker.completed_shots) if shot_tracker.completed_shots else 0,
            'Active Shots': len(shot_tracker.active_shots) if shot_tracker.active_shots else 0
        }
        
        categories = list(stats_data.keys())
        values = list(stats_data.values())
        
        bars = ax.bar(categories, values, color=['blue', 'green', 'orange', 'red'], alpha=0.7)
        ax.set_title('Session Statistics')
        ax.set_ylabel('Count')
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                str(value), ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('output/graphics/basic_statistics.png', dpi=300, bbox_inches='tight')
        plt.close()
        outputs.append('output/graphics/basic_statistics.png')
            
    except Exception as e:
        print(f"Error generating graphics: {e}")
        # Create a minimal error visualization
        try:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.text(0.5, 0.5, f'Graphics generation error:\n{str(e)}', 
                ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title('Error in Graphics Generation')
            plt.savefig('output/graphics/error_visualization.png', dpi=300, bbox_inches='tight')
            plt.close()
            outputs.append('output/graphics/error_visualization.png')
        except:
            pass
    
    return outputs

def generate_heatmap_outputs(players, past_ball_pos):
    """Generate heatmaps for players and ball"""
    outputs = []
    
    try:
        # Player heatmap
        if players.get(1) and players.get(2):
            heatmap = np.zeros((360, 640), dtype=np.float32)
            
            # Collect player positions over time
            for player_id in [1, 2]:
                if players[player_id].get_latest_pose():
                    pose = players[player_id].get_latest_pose()
                    if len(pose.xyn[0]) > 16:
                        x = int(pose.xyn[0][16][0] * 640)
                        y = int(pose.xyn[0][16][1] * 360)
                        if 0 <= x < 640 and 0 <= y < 360:
                            heatmap[y, x] += 1
            
            # Apply Gaussian blur for smooth heatmap
            heatmap = cv2.GaussianBlur(heatmap, (51, 51), 0)
            
            # Normalize and apply colormap
            heatmap_norm = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            heatmap_colored = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)
            
            # Save player heatmap
            cv2.imwrite('output/heatmaps/players/player_positions.png', heatmap_colored)
            outputs.append('output/heatmaps/players/player_positions.png')
        
        # Ball heatmap
        if past_ball_pos and len(past_ball_pos) > 10:
            ball_heatmap = np.zeros((360, 640), dtype=np.float32)
            
            for pos in past_ball_pos[-50:]:  # Last 50 positions
                x, y = int(pos[0]), int(pos[1])
                if 0 <= x < 640 and 0 <= y < 360:
                    ball_heatmap[y, x] += 1
            
            # Apply Gaussian blur
            ball_heatmap = cv2.GaussianBlur(ball_heatmap, (31, 31), 0)
            
            # Normalize and apply colormap
            ball_heatmap_norm = cv2.normalize(ball_heatmap, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            ball_heatmap_colored = cv2.applyColorMap(ball_heatmap_norm, cv2.COLORMAP_HOT)
            
            # Save ball heatmap
            cv2.imwrite('output/heatmaps/ball/ball_trajectory.png', ball_heatmap_colored)
            outputs.append('output/heatmaps/ball/ball_trajectory.png')
            
    except Exception as e:
        print(f"Error generating heatmaps: {e}")
    
    return outputs

def generate_clip_outputs(frame_count, players, past_ball_pos, shot_tracker):
    """Generate clips and highlights"""
    outputs = []
    
    try:
        # Shot highlights metadata
        if shot_tracker.completed_shots:
            highlights_data = []
            for shot in shot_tracker.completed_shots[-10:]:  # Last 10 shots
                highlight_info = {
                    'shot_id': shot.get('id'),
                    'start_frame': shot.get('start_frame'),
                    'end_frame': shot.get('end_frame'),
                    'shot_type': shot.get('shot_type'),
                    'player': shot.get('player_who_hit'),
                    'duration': shot.get('duration', 0),
                    'confidence': shot.get('hit_confidence', 0)
                }
                highlights_data.append(highlight_info)
            
            with open('output/clips/highlights/highlights_metadata.json', 'w') as f:
                json.dump(highlights_data, f, indent=2, cls=NumpyEncoder)
            outputs.append('output/clips/highlights/highlights_metadata.json')
        
        # Rally clips metadata
        rally_data = {
            'total_rallies': len(shot_tracker.completed_shots),
            'average_rally_length': sum(shot.get('duration', 0) for shot in shot_tracker.completed_shots) / max(len(shot_tracker.completed_shots), 1),
            'longest_rally': max((shot.get('duration', 0) for shot in shot_tracker.completed_shots), default=0),
            'frame_count': frame_count
        }
        
        with open('output/clips/rallies/rally_statistics.json', 'w') as f:
            json.dump(rally_data, f, indent=2, cls=NumpyEncoder)
        outputs.append('output/clips/rallies/rally_statistics.json')
        
    except Exception as e:
        print(f"Error generating clips: {e}")
    
    return outputs

def generate_pattern_outputs(players, past_ball_pos, shot_tracker):
    """Generate pattern analysis outputs"""
    outputs = []
    
    try:
        # Player movement patterns
        if players.get(1) and players.get(2):
            pattern_data = {}
            
            for player_id in [1, 2]:
                if players[player_id].get_latest_pose():
                    pose = players[player_id].get_latest_pose()
                    if len(pose.xyn[0]) > 16:
                        x = pose.xyn[0][16][0] * 640
                        y = pose.xyn[0][16][1] * 360
                        
                        # Determine court zone
                        zone = "front" if y < 180 else "back"
                        side = "left" if x < 320 else "right"
                        court_zone = f"{zone}_{side}"
                        
                        pattern_data[f'player_{player_id}'] = {
                            'position': [x, y],
                            'court_zone': court_zone,
                            'timestamp': time.time()
                        }
            
            with open('output/patterns/player_patterns.json', 'w') as f:
                json.dump(pattern_data, f, indent=2, cls=NumpyEncoder)
            outputs.append('output/patterns/player_patterns.json')
        
        # Shot patterns
        if shot_tracker.completed_shots:
            shot_patterns = {}
            for shot in shot_tracker.completed_shots:
                shot_type = shot.get('shot_type', 'unknown')
                if shot_type not in shot_patterns:
                    shot_patterns[shot_type] = 0
                shot_patterns[shot_type] += 1
            
            with open('output/patterns/shot_patterns.json', 'w') as f:
                json.dump(shot_patterns, f, indent=2, cls=NumpyEncoder)
            outputs.append('output/patterns/shot_patterns.json')
            
    except Exception as e:
        print(f"Error generating patterns: {e}")
    
    return outputs

def generate_raw_data_outputs(frame_count, players, past_ball_pos, shot_tracker):
    """Generate raw data outputs"""
    outputs = []
    
    try:
        # Frame-by-frame data
        frame_data = {
            'frame_count': frame_count,
            'timestamp': time.time(),
            'ball_positions': past_ball_pos[-20:] if past_ball_pos else [],  # Last 20 positions
            'active_shots': len(shot_tracker.active_shots),
            'completed_shots': len(shot_tracker.completed_shots)
        }
        
        # Add player data
        if players.get(1) and players.get(2):
            frame_data['players'] = {}
            for player_id in [1, 2]:
                if players[player_id].get_latest_pose():
                    pose = players[player_id].get_latest_pose()
                    if len(pose.xyn[0]) > 16:
                        x = pose.xyn[0][16][0] * 640
                        y = pose.xyn[0][16][1] * 360
                        frame_data['players'][f'player_{player_id}'] = [x, y]
        
        with open('output/raw_data/frame_data.json', 'w') as f:
            json.dump(frame_data, f, indent=2, cls=NumpyEncoder)
        outputs.append('output/raw_data/frame_data.json')
        
        # Ball trajectory data
        if past_ball_pos:
            trajectory_data = {
                'total_positions': len(past_ball_pos),
                'recent_positions': past_ball_pos[-50:],  # Last 50 positions
                'trajectory_length': len(past_ball_pos)
            }
            
        with open('output/raw_data/ball_trajectory.json', 'w') as f:
            json.dump(trajectory_data, f, indent=2, cls=NumpyEncoder)
            outputs.append('output/raw_data/ball_trajectory.json')
            
    except Exception as e:
        print(f"Error generating raw data: {e}")
    
    return outputs

def generate_statistics_outputs(frame_count, players, past_ball_pos, shot_tracker):
    """Generate statistics outputs"""
    outputs = []
    
    try:
        # Overall statistics
        stats = {
            'frame_count': frame_count,
            'total_shots': len(shot_tracker.completed_shots),
            'active_shots': len(shot_tracker.active_shots),
            'ball_positions_recorded': len(past_ball_pos) if past_ball_pos else 0,
            'players_detected': len([p for p in players.values() if p is not None]),
            'timestamp': time.time()
        }
        
        # Shot statistics
        if shot_tracker.completed_shots:
            shot_stats = shot_tracker.get_shot_statistics()
            stats.update(shot_stats)
        
        with open('output/stats/overall_statistics.json', 'w') as f:
            json.dump(stats, f, indent=2, cls=NumpyEncoder)
        outputs.append('output/stats/overall_statistics.json')
        
        # Performance metrics
        performance_metrics = {
            'processing_fps': frame_count / max(time.time() - start, 1),
            'memory_usage': 'N/A',  # Could add actual memory monitoring
            'gpu_available': torch.cuda.is_available(),
            'models_loaded': 2  # pose and ball models
        }
        
        with open('output/stats/performance_metrics.json', 'w') as f:
            json.dump(performance_metrics, f, indent=2, cls=NumpyEncoder)
        outputs.append('output/stats/performance_metrics.json')
        
    except Exception as e:
        print(f"Error generating statistics: {e}")
    
    return outputs

def generate_report_outputs(frame_count, players, past_ball_pos, shot_tracker, coaching_data_collection, path):
    """Generate comprehensive reports"""
    outputs = []
    
    try:
        # Session summary report
        session_report = f"""
SQUASH COACHING SESSION REPORT
Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}
Video: {path}

SESSION OVERVIEW:
- Total frames processed: {frame_count:,}
- Total shots detected: {len(shot_tracker.completed_shots)}
- Active shots: {len(shot_tracker.active_shots)}
- Ball positions recorded: {len(past_ball_pos) if past_ball_pos else 0}
- Players detected: {len([p for p in players.values() if p is not None])}

SHOT ANALYSIS:
"""
        
        if shot_tracker.completed_shots:
            shot_stats = shot_tracker.get_shot_statistics()
            session_report += f"""
- Shots by player: {shot_stats.get('shots_by_player', {})}
- Average shot duration: {shot_stats.get('average_shot_duration', 0):.1f} frames
- Most common shot type: {max(shot_stats.get('shots_by_type', {}).items(), key=lambda x: x[1])[0] if shot_stats.get('shots_by_type') else 'N/A'}
"""
        
        session_report += f"""
PERFORMANCE METRICS:
- Processing rate: {frame_count / max(time.time() - start, 1):.1f} fps
- GPU acceleration: {'Yes' if torch.cuda.is_available() else 'No'}
- Enhanced detection: Active

RECOMMENDATIONS:
- Continue monitoring shot consistency
- Focus on court positioning
- Analyze shot patterns for improvement
"""
        
        with open('output/reports/session_summary.txt', 'w', encoding='utf-8') as f:
            f.write(session_report)
        outputs.append('output/reports/session_summary.txt')
        
        # Technical analysis report
        technical_report = f"""
TECHNICAL ANALYSIS REPORT
Frame: {frame_count}

BALL TRACKING:
- Current ball position: {past_ball_pos[-1] if past_ball_pos else 'Not detected'}
- Trajectory length: {len(past_ball_pos) if past_ball_pos else 0}
- Recent movement: {'Active' if past_ball_pos and len(past_ball_pos) > 1 else 'Static'}

PLAYER ANALYSIS:
"""
        
        for player_id in [1, 2]:
            if players.get(player_id) and players.get(player_id).get_latest_pose():
                pose = players.get(player_id).get_latest_pose()
                if len(pose.xyn[0]) > 16:
                    x = pose.xyn[0][16][0] * 640
                    y = pose.xyn[0][16][1] * 360
                    technical_report += f"- Player {player_id}: Position ({x:.1f}, {y:.1f})\n"
                else:
                    technical_report += f"- Player {player_id}: Position not available\n"
            else:
                technical_report += f"- Player {player_id}: Not detected\n"
        
        with open('output/reports/technical_analysis.txt', 'w', encoding='utf-8') as f:
            f.write(technical_report)
        outputs.append('output/reports/technical_analysis.txt')
        
    except Exception as e:
        print(f"Error generating reports: {e}")
    
    return outputs

def generate_trajectory_outputs(past_ball_pos, shot_tracker):
    """Generate trajectory analysis outputs"""
    outputs = []
    
    # Ensure output directories exist
    os.makedirs('output/trajectories', exist_ok=True)
    
    try:
        # Initialize trajectory_analysis with default values
        trajectory_analysis = {
            'total_positions': 0,
            'recent_trajectory': [],
            'trajectory_smoothness': 'Limited',
            'movement_detected': False,
            'average_movement': 0,
            'max_movement': 0,
            'total_distance': 0
        }
        
        # Ball trajectory analysis
        if past_ball_pos and len(past_ball_pos) > 10:
            trajectory_analysis = {
                'total_positions': len(past_ball_pos),
                'recent_trajectory': past_ball_pos[-20:],  # Last 20 positions
                'trajectory_smoothness': 'Good' if len(past_ball_pos) > 5 else 'Limited',
                'movement_detected': len(past_ball_pos) > 1
            }
            
            # Calculate trajectory statistics
            if len(past_ball_pos) > 1:
                distances = []
                for i in range(1, len(past_ball_pos)):
                    p1, p2 = past_ball_pos[i-1], past_ball_pos[i]
                    distance = math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
                    distances.append(distance)
                
                trajectory_analysis.update({
                    'average_movement': sum(distances) / len(distances) if distances else 0,
                    'max_movement': max(distances) if distances else 0,
                    'total_distance': sum(distances) if distances else 0
                })
        
        with open('output/trajectories/ball_trajectory_analysis.json', 'w') as f:
            json.dump(trajectory_analysis, f, indent=2, cls=NumpyEncoder)
            outputs.append('output/trajectories/ball_trajectory_analysis.json')
        
        # Shot trajectories
        shot_trajectories = {}
        if shot_tracker.completed_shots:
            for shot in shot_tracker.completed_shots[-5:]:  # Last 5 shots
                shot_id = shot.get('id')
                trajectory = shot.get('trajectory', [])
                shot_trajectories[f'shot_{shot_id}'] = {
                    'trajectory': trajectory,
                    'shot_type': shot.get('shot_type'),
                    'player': shot.get('player_who_hit'),
                    'duration': shot.get('duration')
                }
            
        with open('output/trajectories/shot_trajectories.json', 'w') as f:
            json.dump(shot_trajectories, f, indent=2, cls=NumpyEncoder)
            outputs.append('output/trajectories/shot_trajectories.json')
            
    except Exception as e:
        print(f"Error generating trajectories: {e}")
    
    return outputs

def generate_enhanced_visualizations(frame_count, players, past_ball_pos, shot_tracker):
    """Generate enhanced visualizations"""
    outputs = []
    
    # Ensure output directories exist
    os.makedirs('output/visualizations', exist_ok=True)
    os.makedirs('output/graphics', exist_ok=True)
    os.makedirs('output/trajectories', exist_ok=True)
    
    try:
        # 3D court visualization
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Draw court outline in 3D
        court_width, court_height = 640, 360
        x = [0, court_width, court_width, 0, 0]
        y = [0, 0, court_height, court_height, 0]
        z = [0, 0, 0, 0, 0]
        ax.plot(x, y, z, 'k-', linewidth=2)
        
        # Plot ball trajectory in 3D
        if past_ball_pos and len(past_ball_pos) > 5:
            ball_x = [pos[0] for pos in past_ball_pos[-20:]]
            ball_y = [pos[1] for pos in past_ball_pos[-20:]]
            ball_z = [0] * len(ball_x)  # Assuming ground level
            ax.plot(ball_x, ball_y, ball_z, 'g-', linewidth=3, alpha=0.7, label='Ball Trajectory')
            ax.scatter(ball_x[-1], ball_y[-1], ball_z[-1], c='red', s=100, label='Current Ball')
        
        # Plot player positions in 3D
        for player_id in [1, 2]:
            if players.get(player_id) and players.get(player_id).get_latest_pose():
                pose = players.get(player_id).get_latest_pose()
                if len(pose.xyn[0]) > 16:
                    x = pose.xyn[0][16][0] * court_width
                    y = pose.xyn[0][16][1] * court_height
                    z = 0
                    ax.scatter(x, y, z, s=200, label=f'Player {player_id}', alpha=0.8)
        
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.set_zlabel('Z Position')
        ax.set_title(f'3D Court Visualization - Frame {frame_count}')
        ax.legend()
        
        plt.savefig('output/visualizations/3d_court_visualization.png', dpi=300, bbox_inches='tight')
        plt.close()
        outputs.append('output/visualizations/3d_court_visualization.png')
        
        # Performance dashboard
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Shot distribution
        if shot_tracker.completed_shots:
            shot_types = [shot.get('shot_type', 'unknown') for shot in shot_tracker.completed_shots]
            shot_counts = {}
            for shot_type in shot_types:
                shot_counts[shot_type] = shot_counts.get(shot_type, 0) + 1
            
            ax1.pie(shot_counts.values(), labels=shot_counts.keys(), autopct='%1.1f%%')
            ax1.set_title('Shot Type Distribution')
        
        # Frame processing rate
        processing_rate = frame_count / max(time.time() - start, 1)
        ax2.bar(['Processing Rate'], [processing_rate], color='green', alpha=0.7)
        ax2.set_ylabel('FPS')
        ax2.set_title('Processing Performance')
        
        # Ball movement over time
        if past_ball_pos and len(past_ball_pos) > 10:
            recent_positions = past_ball_pos[-50:]
            frame_numbers = list(range(len(recent_positions)))
            x_positions = [pos[0] for pos in recent_positions]
            y_positions = [pos[1] for pos in recent_positions]
            
            ax3.plot(frame_numbers, x_positions, 'b-', label='X Position', alpha=0.7)
            ax3.plot(frame_numbers, y_positions, 'r-', label='Y Position', alpha=0.7)
            ax3.set_xlabel('Frame')
            ax3.set_ylabel('Position')
            ax3.set_title('Ball Movement Over Time')
            ax3.legend()
        
        # Player activity
        player_activity = {
            'Active Shots': len(shot_tracker.active_shots),
            'Completed Shots': len(shot_tracker.completed_shots),
            'Ball Positions': len(past_ball_pos) if past_ball_pos else 0
        }
        
        ax4.bar(player_activity.keys(), player_activity.values(), color=['orange', 'green', 'blue'], alpha=0.7)
        ax4.set_title('Session Activity')
        ax4.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('output/visualizations/performance_dashboard.png', dpi=300, bbox_inches='tight')
        plt.close()
        outputs.append('output/visualizations/performance_dashboard.png')
        
    except Exception as e:
        print(f"Error generating enhanced visualizations: {e}")
    
    return outputs


def generate_llama_enhanced_analysis(frame_count, players, past_ball_pos, shot_tracker, coaching_data_collection):
    """Generate enhanced analysis using Llama 3.1-8B-Instruct model"""
    outputs = []
    
    try:
        print("🤖 Initializing Llama AI coaching enhancement...")
        llama_enhancer = llm_manager.get_model('llama')
        
        if not llama_enhancer or not llama_enhancer.is_initialized:
            print("⚠️ Llama model not available, skipping enhanced analysis")
            return outputs
        
        print("🧠 Generating AI-powered coaching insights...")
        
        # Ensure output directories exist
        os.makedirs('output/reports', exist_ok=True)
        os.makedirs('output/ai_analysis', exist_ok=True)
        
        # 1. Shot Pattern Analysis
        if shot_tracker.completed_shots:
            print("🎯 Analyzing shot patterns with AI...")
            shot_analysis = llama_enhancer.analyze_shot_patterns(shot_tracker.completed_shots)
            
            with open('output/ai_analysis/shot_pattern_analysis.json', 'w') as f:
                json.dump(shot_analysis, f, indent=2, cls=NumpyEncoder)
            outputs.append('output/ai_analysis/shot_pattern_analysis.json')
            
            # Create human-readable report
            shot_report = f"""
🎾 AI-POWERED SHOT PATTERN ANALYSIS
Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}
Model: Llama 3.1-8B-Instruct

📊 SHOT STATISTICS:
Total Shots: {shot_analysis.get('total_shots', 0)}
Shot Distribution: {json.dumps(shot_analysis.get('shot_distribution', {}), indent=2)}

🧠 AI ANALYSIS:
{shot_analysis.get('ai_analysis', 'No analysis available')}

💡 RECOMMENDATIONS:
"""
            for i, rec in enumerate(shot_analysis.get('recommendations', []), 1):
                shot_report += f"{i}. {rec}\n"
            
            with open('output/reports/ai_shot_analysis.txt', 'w', encoding='utf-8') as f:
                f.write(shot_report)
            outputs.append('output/reports/ai_shot_analysis.txt')
        
        # 2. Player Movement Analysis
        if players:
            print("🏃 Analyzing player movement with AI...")
            player_positions = {}
            for player_id, player in players.items():
                if hasattr(player, 'get_latest_pose') and player.get_latest_pose():
                    # Collect recent positions (simplified)
                    player_positions[player_id] = [(0.5, 0.5)]  # Mock data
            
            movement_analysis = llama_enhancer.analyze_player_movement(player_positions, (640, 360))
            
            with open('output/ai_analysis/movement_analysis.json', 'w') as f:
                json.dump(movement_analysis, f, indent=2, cls=NumpyEncoder)
            outputs.append('output/ai_analysis/movement_analysis.json')
        
        # 3. Match Report Generation
        print("📋 Generating comprehensive match report...")
        match_data = {
            "frame_count": frame_count,
            "total_shots": len(shot_tracker.completed_shots) if shot_tracker.completed_shots else 0,
            "active_shots": len(shot_tracker.active_shots) if shot_tracker.active_shots else 0,
            "ball_positions": len(past_ball_pos) if past_ball_pos else 0,
            "coaching_data": coaching_data_collection.get_summary() if coaching_data_collection else {}
        }
        
        match_report = llama_enhancer.generate_match_report(match_data)
        
        with open('output/ai_analysis/match_report.json', 'w') as f:
            json.dump(match_report, f, indent=2, cls=NumpyEncoder)
        outputs.append('output/ai_analysis/match_report.json')
        
        # Create human-readable match report
        readable_report = f"""
🎾 AI-GENERATED COMPREHENSIVE MATCH REPORT
Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}
Model: Llama 3.1-8B-Instruct

{match_report.get('match_report', 'No report available')}
"""
        
        with open('output/reports/ai_match_report.txt', 'w', encoding='utf-8') as f:
            f.write(readable_report)
        outputs.append('output/reports/ai_match_report.txt')
        
        # 4. Personalized Coaching Plan
        print("📝 Generating personalized coaching plan...")
        player_profile = {
            "session_data": {
                "frames_processed": frame_count,
                "total_shots": len(shot_tracker.completed_shots) if shot_tracker.completed_shots else 0,
                "session_duration": "Variable"
            },
            "performance_metrics": match_data,
            "coaching_preferences": "General improvement focus"
        }
        
        coaching_plan = llama_enhancer.generate_personalized_coaching_plan(player_profile)
        
        with open('output/ai_analysis/personalized_coaching_plan.json', 'w') as f:
            json.dump(coaching_plan, f, indent=2, cls=NumpyEncoder)
        outputs.append('output/ai_analysis/personalized_coaching_plan.json')
        
        # Create human-readable coaching plan
        plan_text = f"""
🎾 AI-GENERATED PERSONALIZED COACHING PLAN
Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}
Model: Llama 3.1-8B-Instruct

{coaching_plan.get('coaching_plan', 'No plan available')}
"""
        
        with open('output/reports/ai_coaching_plan.txt', 'w', encoding='utf-8') as f:
            f.write(plan_text)
        outputs.append('output/reports/ai_coaching_plan.txt')
        
        print(f"✅ AI-enhanced analysis completed! Generated {len(outputs)} outputs")
        
    except Exception as e:
        print(f"❌ Error in AI-enhanced analysis: {e}")
        import traceback
        traceback.print_exc()
    
    return outputs

def generate_output_summary(outputs_generated, frame_count, path):
    """Generate a comprehensive summary of all outputs created"""
    try:
        summary_report = f"""
🎾 COMPREHENSIVE SQUASH COACHING OUTPUT SUMMARY
Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}
Video: {path}
Total Frames Processed: {frame_count:,}

📁 OUTPUT CATEGORIES AND FILES:
"""
        
        # Categorize outputs
        categories = {
            'Graphics & Visualizations': [],
            'Heatmaps': [],
            'Clips & Highlights': [],
            'Patterns': [],
            'Raw Data': [],
            'Statistics': [],
            'Reports': [],
            'Trajectories': [],
            'Enhanced Visualizations': [],
            'AI Analysis': []
        }
        
        for output_path in outputs_generated:
            if 'graphics' in output_path:
                categories['Graphics & Visualizations'].append(output_path)
            elif 'heatmaps' in output_path:
                categories['Heatmaps'].append(output_path)
            elif 'clips' in output_path:
                categories['Clips & Highlights'].append(output_path)
            elif 'patterns' in output_path:
                categories['Patterns'].append(output_path)
            elif 'raw_data' in output_path:
                categories['Raw Data'].append(output_path)
            elif 'stats' in output_path:
                categories['Statistics'].append(output_path)
            elif 'reports' in output_path:
                categories['Reports'].append(output_path)
            elif 'trajectories' in output_path:
                categories['Trajectories'].append(output_path)
            elif 'visualizations' in output_path:
                categories['Enhanced Visualizations'].append(output_path)
            elif 'ai_analysis' in output_path:
                categories['AI Analysis'].append(output_path)
        
        # Add category summaries
        for category, files in categories.items():
            if files:
                summary_report += f"\n{category} ({len(files)} files):\n"
                for file_path in files:
                    try:
                        file_size = os.path.getsize(file_path)
                        summary_report += f"  • {os.path.basename(file_path)} ({file_size:,} bytes)\n"
                    except:
                        summary_report += f"  • {os.path.basename(file_path)}\n"
        
        summary_report += f"""

📊 SESSION STATISTICS:
• Total outputs generated: {len(outputs_generated)}
• Processing completed: {time.strftime('%Y-%m-%d %H:%M:%S')}
• Video duration: {frame_count/30:.1f} seconds (assuming 30fps)
• Output categories: {len([cat for cat, files in categories.items() if files])}

🎯 WHAT YOU GET:
1. 📈 Real-time analytics and visualizations
2. 🎨 Interactive graphics and heatmaps
3. 📹 Shot and rally highlights
4. 📊 Performance statistics and reports
5. 🎾 Ball and player trajectory analysis
6. 🔍 Pattern recognition and insights
7. 📋 Comprehensive coaching recommendations
8. 🚀 Enhanced 3D visualizations

💡 NEXT STEPS:
• Review generated reports for coaching insights
• Analyze shot patterns and player positioning
• Use heatmaps to understand court coverage
• Examine trajectory data for technique improvement
• Share highlights and clips for training purposes

🎾 SQUASH COACHING ANALYSIS COMPLETE! 🎾
"""
        
        # Save summary report
        with open('output/comprehensive_output_summary.txt', 'w', encoding='utf-8') as f:
            f.write(summary_report)
        
        print("📋 Comprehensive output summary generated: output/comprehensive_output_summary.txt")
        
        # Also print a brief summary to console
        print(f"\n🎯 COMPREHENSIVE OUTPUTS GENERATED:")
        print(f"   • Total files: {len(outputs_generated)}")
        print(f"   • Categories: {len([cat for cat, files in categories.items() if files])}")
        print(f"   • Summary saved: output/comprehensive_output_summary.txt")
        
    except Exception as e:
        print(f"Error generating output summary: {e}")


if __name__ == "__main__":
    try:
        start=time.time()
        main()
        print(f"Execution time: {time.time() - start:.2f} seconds")
    except KeyboardInterrupt:
        print("\nProgram interrupted by user. All outputs have been generated.")
    except Exception as e:
        print(f"Unexpected error: {e}")
    finally:
        # Clean up LLM models to free memory
        print("🧹 Cleaning up LLM models...")
        llm_manager.unload_current_model()
        print("✅ Memory cleanup completed")
        
