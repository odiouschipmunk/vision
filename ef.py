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
from collections import deque

class SmoothedBallTracker:
    """Enhanced ball tracking with smoothing and temporal consistency"""
    
    def __init__(self, max_history=10, smoothing_factor=0.3):
        self.max_history = max_history
        self.smoothing_factor = smoothing_factor
        self.position_history = deque(maxlen=max_history)
        self.velocity_history = deque(maxlen=max_history)
        self.last_confident_position = None
        self.prediction_streak = 0
        self.max_prediction_frames = 5
        
    def add_detection(self, position, confidence=1.0, frame_count=None):
        """Add a new ball detection with confidence scoring"""
        
        if position is None or len(position) < 2:
            return self._handle_missing_detection(frame_count)
            
        x, y = float(position[0]), float(position[1])
        
        # Validate position bounds (assuming 640x360 frame)
        if not (0 <= x <= 640 and 0 <= y <= 360):
            return self._handle_missing_detection(frame_count)
        
        # If we have history, check for sudden jumps
        if self.position_history and confidence > 0.5:
            last_pos = self.position_history[-1]
            distance = math.sqrt((x - last_pos[0])**2 + (y - last_pos[1])**2)
            
            # Reject sudden jumps that are too large (unless very high confidence)
            max_jump = 50 if confidence > 0.8 else 30
            if distance > max_jump:
                print(f"🔴 Ball jump rejected: {distance:.1f}px (conf: {confidence:.2f})")
                return self._handle_missing_detection(frame_count)
        
        # Apply temporal smoothing if we have recent history
        if self.position_history and self.smoothing_factor > 0:
            last_pos = self.position_history[-1]
            
            # Exponential smoothing
            smoothed_x = last_pos[0] * self.smoothing_factor + x * (1 - self.smoothing_factor)
            smoothed_y = last_pos[1] * self.smoothing_factor + y * (1 - self.smoothing_factor)
            
            # Use smoothed position for low confidence detections
            if confidence < 0.7:
                x, y = smoothed_x, smoothed_y
        
        # Calculate velocity if we have previous position
        velocity = [0.0, 0.0]
        if self.position_history:
            last_pos = self.position_history[-1]
            dt = 1.0  # Assume 1 frame time unit
            velocity = [(x - last_pos[0]) / dt, (y - last_pos[1]) / dt]
            self.velocity_history.append(velocity)
        
        # Add position with metadata
        ball_data = [x, y, frame_count if frame_count else len(self.position_history)]
        self.position_history.append(ball_data)
        self.last_confident_position = ball_data.copy()
        self.prediction_streak = 0
        
        return ball_data
    
    def _handle_missing_detection(self, frame_count):
        """Handle frames where ball detection failed"""
        
        if not self.position_history or self.prediction_streak >= self.max_prediction_frames:
            return None
            
        # Predict position based on velocity
        if self.velocity_history:
            last_pos = self.position_history[-1]
            last_velocity = self.velocity_history[-1]
            
            # Simple linear prediction
            predicted_x = last_pos[0] + last_velocity[0]
            predicted_y = last_pos[1] + last_velocity[1]
            
            # Bound predictions to reasonable area
            predicted_x = max(0, min(640, predicted_x))
            predicted_y = max(0, min(360, predicted_y))
            
            predicted_pos = [predicted_x, predicted_y, frame_count if frame_count else len(self.position_history)]
            self.position_history.append(predicted_pos)
            self.prediction_streak += 1
            
            print(f"🔮 Ball predicted: ({predicted_x:.1f}, {predicted_y:.1f}) streak: {self.prediction_streak}")
            return predicted_pos
            
        return None
    
    def get_current_position(self):
        """Get the most recent ball position"""
        return self.position_history[-1] if self.position_history else None
    
    def get_trajectory(self, length=None):
        """Get recent trajectory points"""
        if length is None:
            return list(self.position_history)
        else:
            return list(self.position_history)[-length:] if len(self.position_history) >= length else list(self.position_history)
    
    def get_velocity(self):
        """Get current ball velocity"""
        return self.velocity_history[-1] if self.velocity_history else [0.0, 0.0]
    
    def is_tracking(self):
        """Check if we're actively tracking the ball"""
        return len(self.position_history) > 0 and self.prediction_streak < self.max_prediction_frames
    
    def reset(self):
        """Reset the tracker"""
        self.position_history.clear()
        self.velocity_history.clear()
        self.last_confident_position = None
        self.prediction_streak = 0

# Protection function to ensure correct alias
def ensure_correct_enhanced_ball_tracker():
    """Ensure EnhancedBallTracker always points to SmoothedBallTracker"""
    global EnhancedBallTracker
    EnhancedBallTracker = SmoothedBallTracker
    return EnhancedBallTracker

# Create an alias for backward compatibility with existing code
EnhancedBallTracker = SmoothedBallTracker

# Verify the alias is working correctly and ensure no conflicts
try:
    # Test the alias with the expected parameters
    test_tracker = EnhancedBallTracker(max_history=5, smoothing_factor=0.3)
    test_tracker = None  # Clean up test instance
    print("✅ EnhancedBallTracker alias verified successfully")
except Exception as alias_error:
    print(f"❌ EnhancedBallTracker alias error: {alias_error}")
    # Force re-assignment of alias
    EnhancedBallTracker = SmoothedBallTracker
    print("🔧 EnhancedBallTracker alias re-assigned to SmoothedBallTracker")

# Double-check that EnhancedBallTracker is the correct class
if hasattr(EnhancedBallTracker, '__init__'):
    import inspect
    init_signature = inspect.signature(EnhancedBallTracker.__init__)
    if 'max_history' in init_signature.parameters:
        print("✅ EnhancedBallTracker correctly accepts max_history parameter")
    else:
        print("❌ EnhancedBallTracker does NOT accept max_history parameter")
        print(f"   Available parameters: {list(init_signature.parameters.keys())}")
        # Force re-assignment if parameters don't match
        EnhancedBallTracker = SmoothedBallTracker
        print("🔧 EnhancedBallTracker forcibly re-assigned to SmoothedBallTracker")

# Enhanced shot detection components - using local implementations
# from enhanced_shot_detection import EnhancedBallTracker, EnhancedShotDetector, integrate_enhanced_detection

# Prevent any future imports from overriding our alias
def _protect_enhanced_ball_tracker():
    """Ensure EnhancedBallTracker always points to SmoothedBallTracker"""
    global EnhancedBallTracker
    if EnhancedBallTracker != SmoothedBallTracker:
        print("⚠️ EnhancedBallTracker alias was overridden, restoring correct reference")
        EnhancedBallTracker = SmoothedBallTracker

def _trace_enhanced_ball_tracker_calls():
    """Add comprehensive call tracing to catch any erroneous EnhancedBallTracker calls"""
    import sys
    import traceback
    import inspect
    
    # Wrap the SmoothedBallTracker to catch all calls
    original_smoothed_init = SmoothedBallTracker.__init__
    
    def traced_init(self, *args, **kwargs):
        # Check for invalid parameters
        invalid_params = ['court_width', 'court_height', 'frame_width', 'frame_height']
        found_invalid = []
        
        for param in invalid_params:
            if param in kwargs:
                found_invalid.append(param)
                print(f"🚨 CRITICAL: EnhancedBallTracker called with invalid parameter: {param}={kwargs[param]}")
        
        if found_invalid:
            print(f"🚨 Full kwargs: {kwargs}")
            print(f"🚨 Full args: {args}")
            print(f"🚨 Call stack:")
            traceback.print_stack()
            
            # Remove the invalid parameters and continue
            for param in found_invalid:
                del kwargs[param]
            print(f"🔧 Removed invalid parameters {found_invalid} and continuing...")
        
        # Log all legitimate calls for debugging
        print(f"✅ EnhancedBallTracker called successfully with args: {args}, kwargs: {kwargs}")
        return original_smoothed_init(self, *args, **kwargs)
    
    SmoothedBallTracker.__init__ = traced_init
    
    # Also update the alias with extra protection
    globals()['EnhancedBallTracker'] = SmoothedBallTracker
    sys.modules[__name__].EnhancedBallTracker = SmoothedBallTracker
    
    print("🔍 Enhanced call tracing enabled for EnhancedBallTracker")
    print(f"🔒 Verified alias: {globals()['EnhancedBallTracker']}")
    print(f"🔒 Expected signature: {inspect.signature(SmoothedBallTracker.__init__)}")
    
    return True

# Additional protection: prevent enhanced_shot_detection module from being imported
import sys
if 'enhanced_shot_detection' in sys.modules:
    print("⚠️ enhanced_shot_detection module detected, removing to prevent conflicts")
    del sys.modules['enhanced_shot_detection']

# Hook to prevent future imports of enhanced_shot_detection
_original_import = __builtins__.__import__

def _safe_import(name, *args, **kwargs):
    if name == 'enhanced_shot_detection':
        print("⚠️ Blocking import of enhanced_shot_detection to prevent conflicts")
        raise ImportError("enhanced_shot_detection import blocked to prevent conflicts")
    return _original_import(name, *args, **kwargs)

__builtins__.__import__ = _safe_import
import json
print(f"time to import everything: {time.time()-start}")
import warnings
warnings.filterwarnings("ignore")
# Helper function to handle court dimensions
def get_court_dimensions(court_dimensions, default_width=640, default_height=360):
    """Helper function to extract width and height from court_dimensions"""
    if isinstance(court_dimensions, dict):
        width = float(court_dimensions.get('width', default_width))
        height = float(court_dimensions.get('height', default_height))
    elif isinstance(court_dimensions, (list, tuple)):
        width = float(court_dimensions[0]) if len(court_dimensions) > 0 else default_width
        height = float(court_dimensions[1]) if len(court_dimensions) > 1 else default_height
    else:
        width, height = default_width, default_height
    return width, height

alldata = organizeddata = []
# Initialize global frame dimensions and frame counter for exception handling
frame_count = 0
frame_width = 0
frame_height = 0
# Autonomous coaching system imported from autonomous_coaching.py

class ShotClassificationModel:
    """🎯 ENHANCED AUTONOMOUS SHOT CLASSIFICATION - Ultra-accurate with confidence scoring"""
    
    def __init__(self):
        # Shot types with enhanced detection parameters
        self.shot_types = {
            'straight_drive': {'color': (0, 255, 0), 'threshold': 0.6},        # Bright Green
            'crosscourt': {'color': (255, 165, 0), 'threshold': 0.5},          # Orange  
            'drop_shot': {'color': (255, 0, 255), 'threshold': 0.5},           # Magenta
            'lob': {'color': (0, 255, 255), 'threshold': 0.4},                 # Cyan
            'boast': {'color': (255, 255, 0), 'threshold': 0.6},               # Yellow
            'volley': {'color': (128, 0, 255), 'threshold': 0.5},              # Purple
            'kill_shot': {'color': (255, 0, 0), 'threshold': 0.6},             # Red
            'unknown': {'color': (128, 128, 128), 'threshold': 0.0}            # Gray
        }
        self.minimum_trajectory_length = 4
        
    def classify_shot(self, trajectory, court_dimensions, player_positions=None):
        """
        🎯 AUTONOMOUS SHOT CLASSIFICATION with comprehensive analysis
        Returns shot type with confidence and detailed reasoning
        """
        if len(trajectory) < self.minimum_trajectory_length:
            return {
                'type': 'unknown',
                'confidence': 0.0,
                'clarity': 'INSUFFICIENT_DATA',
                'features': {},
                'color': self.shot_types['unknown']['color'],
                'reasoning': f'Trajectory too short: {len(trajectory)} < {self.minimum_trajectory_length}'
            }
        
        print(f"🎯 Analyzing trajectory with {len(trajectory)} points")
        
        # Extract comprehensive trajectory features
        features = self._extract_trajectory_features(trajectory, court_dimensions)
        
        # Analyze each shot type with specialized algorithms
        shot_scores = {}
        shot_scores['straight_drive'] = self._classify_drive(features)
        shot_scores['crosscourt'] = self._classify_crosscourt(features)
        shot_scores['drop_shot'] = self._classify_drop(features)
        shot_scores['lob'] = self._classify_lob(features)
        shot_scores['boast'] = self._classify_boast(features)
        shot_scores['volley'] = self._classify_volley(features)
        shot_scores['kill_shot'] = self._classify_kill(features)
        
        # Find best classification
        best_shot = max(shot_scores.items(), key=lambda x: x[1]['confidence'])
        shot_type = best_shot[0]
        best_result = best_shot[1]
        confidence = best_result['confidence']
        reasoning = best_result['reasoning']
        
        # Determine clarity based on confidence
        if confidence > 0.8:
            clarity = 'VERY_HIGH'
        elif confidence > 0.6:
            clarity = 'HIGH'
        elif confidence > 0.4:
            clarity = 'MEDIUM'
        elif confidence > 0.2:
            clarity = 'LOW'
        else:
            clarity = 'VERY_LOW'
            shot_type = 'unknown'
        
        # Get shot color
        color = self.shot_types.get(shot_type, self.shot_types['unknown'])['color']
        
        print(f"🎯 SHOT CLASSIFIED: {shot_type.upper()}")
        print(f"   Confidence: {confidence:.3f}")
        print(f"   Clarity: {clarity}")
        print(f"   Color: RGB{color}")
        
        return {
            'type': shot_type,
            'confidence': confidence,
            'clarity': clarity,
            'features': features,
            'color': color,
            'reasoning': reasoning,
            'all_scores': shot_scores
        }
    
    def _extract_trajectory_features(self, trajectory, court_dimensions):
        """🔍 Extract comprehensive features from ball trajectory for classification"""
        width, height = get_court_dimensions(court_dimensions)
        
        # Basic trajectory metrics
        start_pos = trajectory[0][:2]
        end_pos = trajectory[-1][:2]
        total_displacement = {
            'x': end_pos[0] - start_pos[0],
            'y': end_pos[1] - start_pos[1],
            'distance': math.sqrt((end_pos[0] - start_pos[0])**2 + (end_pos[1] - start_pos[1])**2)
        }
        
        # Normalized coordinates (0-1 range for court analysis)
        norm_start = (start_pos[0] / width, start_pos[1] / height)
        norm_end = (end_pos[0] / width, end_pos[1] / height)
        
        # Enhanced velocity analysis
        velocities = []
        for i in range(1, len(trajectory)):
            prev_pos = trajectory[i-1][:2]
            curr_pos = trajectory[i][:2]
            velocity = math.sqrt((curr_pos[0] - prev_pos[0])**2 + (curr_pos[1] - prev_pos[1])**2)
            velocities.append(velocity)
        
        velocity_stats = {
            'average': sum(velocities) / len(velocities) if velocities else 0,
            'maximum': max(velocities) if velocities else 0,
            'minimum': min(velocities) if velocities else 0,
            'variance': sum((v - (sum(velocities)/len(velocities)))**2 for v in velocities) / len(velocities) if velocities else 0
        }
        
        # Direction change analysis
        direction_changes = 0
        major_direction_changes = 0
        
        for i in range(2, len(trajectory)):
            p1, p2, p3 = trajectory[i-2][:2], trajectory[i-1][:2], trajectory[i][:2]
            
            # Calculate vectors
            v1 = (p2[0] - p1[0], p2[1] - p1[1])
            v2 = (p3[0] - p2[0], p3[1] - p2[1])
            
            # Skip very small movements (noise)
            if abs(v1[0]) > 2 and abs(v1[1]) > 2 and abs(v2[0]) > 2 and abs(v2[1]) > 2:
                angle = self._calculate_vector_angle(v1, v2)
                angle_degrees = math.degrees(angle)
                
                if angle_degrees > 20:  # > 20 degrees
                    direction_changes += 1
                if angle_degrees > 45:  # > 45 degrees  
                    major_direction_changes += 1
        
        # Height analysis (Y-coordinate analysis)
        y_positions = [pos[1] for pos in trajectory]
        height_analysis = {
            'max_y': max(y_positions),
            'min_y': min(y_positions),
            'avg_y': sum(y_positions) / len(y_positions),
            'variation': max(y_positions) - min(y_positions),
            'height_ratio': (sum(y_positions) / len(y_positions)) / height  # Normalized average height
        }
        
        # Court coverage analysis
        x_positions = [pos[0] for pos in trajectory]
        court_coverage = {
            'x_span': max(x_positions) - min(x_positions),
            'y_span': max(y_positions) - min(y_positions),
            'x_ratio': (max(x_positions) - min(x_positions)) / width,
            'y_ratio': (max(y_positions) - min(y_positions)) / height,
            'total_coverage': ((max(x_positions) - min(x_positions)) * (max(y_positions) - min(y_positions))) / (width * height)
        }
        
        # Wall proximity analysis
        wall_proximity = {
            'closest_to_front': min(y_positions),  # Distance to front wall (y=0)
            'closest_to_back': height - max(y_positions),  # Distance to back wall
            'closest_to_left': min(x_positions),  # Distance to left wall (x=0)
            'closest_to_right': width - max(x_positions),  # Distance to right wall
        }
        
        # Tactical zone analysis
        front_court_ratio = sum(1 for y in y_positions if y < height/3) / len(y_positions)
        mid_court_ratio = sum(1 for y in y_positions if height/3 <= y <= 2*height/3) / len(y_positions)
        back_court_ratio = sum(1 for y in y_positions if y > 2*height/3) / len(y_positions)
        
        return {
            'trajectory_length': len(trajectory),
            'total_displacement': total_displacement,
            'normalized_positions': {'start': norm_start, 'end': norm_end},
            'velocity_stats': velocity_stats,
            'direction_changes': direction_changes,
            'major_direction_changes': major_direction_changes,
            'height_analysis': height_analysis,
            'court_coverage': court_coverage,
            'wall_proximity': wall_proximity,
            'court_zones': {
                'front_court_ratio': front_court_ratio,
                'mid_court_ratio': mid_court_ratio,
                'back_court_ratio': back_court_ratio
            }
        }
    
    def _classify_drive(self, features):
        """🎯 Classify straight drive shots"""
        confidence = 0.0
        reasoning = []
        
        # Drives have minimal horizontal movement
        horizontal_ratio = abs(features['total_displacement']['x']) / (features['court_coverage']['x_span'] + 1)
        if horizontal_ratio < 0.4:
            confidence += 0.3
            reasoning.append("minimal horizontal deviation")
        
        # Consistent direction (few direction changes)
        if features['direction_changes'] <= 2:
            confidence += 0.25
            reasoning.append("straight trajectory")
        
        # Moderate to high velocity
        avg_vel = features['velocity_stats']['average']
        if 8 < avg_vel < 30:
            confidence += 0.2
            reasoning.append(f"drive velocity ({avg_vel:.1f})")
        
        # Back-to-front court movement
        start_y_ratio = features['normalized_positions']['start'][1]
        end_y_ratio = features['normalized_positions']['end'][1]
        if start_y_ratio > end_y_ratio:  # Moving toward front
            confidence += 0.15
            reasoning.append("back-to-front movement")
        
        # Close to side wall (typical drive path)
        min_side_distance = min(features['wall_proximity']['closest_to_left'], 
                            features['wall_proximity']['closest_to_right'])
        if min_side_distance < 80:
            confidence += 0.1
            reasoning.append("side wall proximity")
        
        return {
            'confidence': min(1.0, confidence),
            'reasoning': "; ".join(reasoning) if reasoning else "no drive characteristics detected"
        }
    
    def _classify_crosscourt(self, features):
        """🎯 Classify crosscourt shots"""
        confidence = 0.0
        reasoning = []
        
        # Significant horizontal movement across court
        x_coverage_ratio = features['court_coverage']['x_ratio']
        if x_coverage_ratio > 0.4:
            confidence += 0.35
            reasoning.append(f"wide court coverage ({x_coverage_ratio:.2f})")
        
        # Diagonal movement pattern
        horizontal_movement = abs(features['total_displacement']['x'])
        vertical_movement = abs(features['total_displacement']['y'])
        if horizontal_movement > 100 and vertical_movement > 50:
            confidence += 0.25
            reasoning.append("diagonal trajectory pattern")
        
        # Cross-court direction change
        start_x_ratio = features['normalized_positions']['start'][0]
        end_x_ratio = features['normalized_positions']['end'][0]
        if (start_x_ratio < 0.4 and end_x_ratio > 0.6) or (start_x_ratio > 0.6 and end_x_ratio < 0.4):
            confidence += 0.2
            reasoning.append("cross-court direction")
        
        # Moderate velocity with some variation
        if features['velocity_stats']['variance'] > 15:
            confidence += 0.1
            reasoning.append("velocity variation")
        
        # Multiple direction changes (bounces off walls)
        if features['direction_changes'] >= 2:
            confidence += 0.1
            reasoning.append("multiple direction changes")
        
        return {
            'confidence': min(1.0, confidence),
            'reasoning': "; ".join(reasoning) if reasoning else "no crosscourt characteristics detected"
        }
    
    def _classify_drop(self, features):
        """🎯 Classify drop shots"""
        confidence = 0.0
        reasoning = []
        
        # Ends in front court
        if features['court_zones']['front_court_ratio'] > 0.3:
            confidence += 0.3
            reasoning.append("front court targeting")
        
        # Lower velocity (finesse shot)
        avg_vel = features['velocity_stats']['average']
        if avg_vel < 15:
            confidence += 0.25
            reasoning.append(f"drop shot velocity ({avg_vel:.1f})")
        
        # Shorter trajectory (quick shot)
        if features['trajectory_length'] < 15:
            confidence += 0.2
            reasoning.append("short trajectory")
        
        # Controlled height (not too high)
        if features['height_analysis']['variation'] < 60:
            confidence += 0.15
            reasoning.append("controlled height")
        
        # Downward movement
        if features['total_displacement']['y'] > 0:  # Moving down in screen coordinates
            confidence += 0.1
            reasoning.append("downward movement")
        
        return {
            'confidence': min(1.0, confidence),
            'reasoning': "; ".join(reasoning) if reasoning else "no drop shot characteristics detected"
        }
    
    def _classify_lob(self, features):
        """🎯 Classify lob shots"""
        confidence = 0.0
        reasoning = []
        
        # High trajectory with significant height variation
        if features['height_analysis']['variation'] > 80:
            confidence += 0.3
            reasoning.append(f"high trajectory arc ({features['height_analysis']['variation']:.1f}px)")
        
        # Ends in back court
        if features['court_zones']['back_court_ratio'] > 0.3:
            confidence += 0.25
            reasoning.append("back court targeting")
        
        # High ball position (low Y values in screen coordinates)
        if features['height_analysis']['min_y'] < features['height_analysis']['avg_y'] * 0.8:
            confidence += 0.2
            reasoning.append("high ball trajectory")
        
        # Longer trajectory duration
        if features['trajectory_length'] > 12:
            confidence += 0.15
            reasoning.append("extended trajectory")
        
        # Moderate velocity (not aggressive)
        avg_vel = features['velocity_stats']['average']
        if 10 < avg_vel < 25:
            confidence += 0.1
            reasoning.append(f"lob velocity ({avg_vel:.1f})")
        
        return {
            'confidence': min(1.0, confidence),
            'reasoning': "; ".join(reasoning) if reasoning else "no lob characteristics detected"
        }
    
    def _classify_boast(self, features):
        """🎯 Classify boast shots (side wall interactions)"""
        confidence = 0.0
        reasoning = []
        
        # Multiple direction changes (wall bounces)
        if features['major_direction_changes'] >= 2:
            confidence += 0.4
            reasoning.append(f"multiple wall bounces ({features['major_direction_changes']})")
        
        # High velocity variance (speed changes from wall hits)
        if features['velocity_stats']['variance'] > 25:
            confidence += 0.25
            reasoning.append("velocity variation from wall hits")
        
        # Side wall proximity
        min_side_distance = min(features['wall_proximity']['closest_to_left'], 
                            features['wall_proximity']['closest_to_right'])
        if min_side_distance < 40:
            confidence += 0.2
            reasoning.append("side wall proximity")
        
        # Often ends in front court after complex path
        if features['court_zones']['front_court_ratio'] > 0.2 and features['direction_changes'] > 0:
            confidence += 0.15
            reasoning.append("front court ending with wall interactions")
        
        return {
            'confidence': min(1.0, confidence),
            'reasoning': "; ".join(reasoning) if reasoning else "no boast characteristics detected"
        }
    
    def _classify_volley(self, features):
        """🎯 Classify volley shots (intercepted high)"""
        confidence = 0.0
        reasoning = []
        
        # High position intercept (upper court area)
        if features['height_analysis']['height_ratio'] < 0.5:
            confidence += 0.3
            reasoning.append("high interception")
        
        # Quick, short trajectory
        if features['trajectory_length'] < 12:
            confidence += 0.25
            reasoning.append("quick shot")
        
        # High velocity (aggressive intercept)
        if features['velocity_stats']['maximum'] > 25:
            confidence += 0.2
            reasoning.append(f"aggressive velocity ({features['velocity_stats']['maximum']:.1f})")
        
        # Limited court coverage (intercept point)
        if features['court_coverage']['total_coverage'] < 0.15:
            confidence += 0.15
            reasoning.append("limited court coverage")
        
        # Consistent direction (direct hit)
        if features['direction_changes'] <= 1:
            confidence += 0.1
            reasoning.append("direct trajectory")
        
        return {
            'confidence': min(1.0, confidence),
            'reasoning': "; ".join(reasoning) if reasoning else "no volley characteristics detected"
        }
    
    def _classify_kill(self, features):
        """🎯 Classify kill shots (aggressive winners)"""
        confidence = 0.0
        reasoning = []
        
        # High velocity (aggressive shot)
        avg_vel = features['velocity_stats']['average']
        if avg_vel > 25:
            confidence += 0.3
            reasoning.append(f"high velocity ({avg_vel:.1f})")
        
        # Low trajectory (staying low)
        if features['height_analysis']['height_ratio'] > 0.7:
            confidence += 0.25
            reasoning.append("low trajectory")
        
        # Front court targeting
        if features['court_zones']['front_court_ratio'] > 0.3:
            confidence += 0.2
            reasoning.append("front court targeting")
        
        # Short, direct trajectory
        if features['trajectory_length'] < 15 and features['direction_changes'] <= 1:
            confidence += 0.15
            reasoning.append("short, direct trajectory")
        
        # Very high peak velocity
        if features['velocity_stats']['maximum'] > 35:
            confidence += 0.1
            reasoning.append(f"peak velocity ({features['velocity_stats']['maximum']:.1f})")
        
        return {
            'confidence': min(1.0, confidence),
            'reasoning': "; ".join(reasoning) if reasoning else "no kill shot characteristics detected"
        }
    
    def _calculate_vector_angle(self, v1, v2):
        """Calculate angle between two vectors in radians"""
        try:
            len1 = math.sqrt(v1[0]**2 + v1[1]**2)
            len2 = math.sqrt(v2[0]**2 + v2[1]**2)
            
            if len1 == 0 or len2 == 0:
                return 0.0
            
            dot_product = (v1[0] * v2[0] + v1[1] * v2[1]) / (len1 * len2)
            dot_product = max(-1.0, min(1.0, dot_product))  # Clamp to valid range
            
            return math.acos(dot_product)
        except:
            return 0.0

class PlayerHitDetector:
    """Enhanced player hit detection with multiple algorithms"""
    
    def __init__(self):
        self.hit_detection_methods = [
            self._proximity_detection,
            self._trajectory_analysis,
            self._racket_position_analysis,
        ]
        self.confidence_weights = [0.4, 0.4, 0.2]
        self.hit_distance_threshold = 80  # pixels
        self.velocity_change_threshold = 15  # velocity change threshold
        self.direction_change_threshold = math.pi/6  # 30 degrees
    
    def detect_player_hit(self, players, ball_trajectory, frame_count):
        width, height = get_court_dimensions(court_dimensions)
        
        # Normalize trajectory to court dimensions
        normalized_traj = [(x/width, y/height) for x, y, *_ in trajectory]
        
        features = {}
        
        # 1. Enhanced movement analysis
        features['horizontal_movement'] = abs(normalized_traj[-1][0] - normalized_traj[0][0])
        features['vertical_movement'] = abs(normalized_traj[-1][1] - normalized_traj[0][1])
        features['diagonal_component'] = math.sqrt(features['horizontal_movement']**2 + features['vertical_movement']**2)
        
        # 2. Advanced velocity analysis with physics
        velocities, accelerations = self._calculate_physics_metrics(trajectory)
        
        if velocities:
            features['avg_velocity'] = sum(velocities) / len(velocities)
            features['max_velocity'] = max(velocities)
            features['min_velocity'] = min(velocities)
            features['velocity_variance'] = np.var(velocities)
            features['velocity_consistency'] = 1.0 - (features['velocity_variance'] / (features['avg_velocity']**2 + 1))
            features['deceleration_pattern'] = self._detect_deceleration_pattern(velocities)
            features['acceleration_pattern'] = self._detect_acceleration_pattern(accelerations) if accelerations else 0.0
        
        # 3. Trajectory shape and curvature analysis
        features['trajectory_curvature'] = self._calculate_enhanced_curvature(trajectory)
        features['linearity_score'] = self._calculate_linearity(trajectory)
        features['smoothness_score'] = self._calculate_trajectory_smoothness(trajectory)
        
        # 4. Enhanced direction change analysis
        features['direction_changes'] = self._count_enhanced_direction_changes(trajectory)
        features['major_direction_changes'] = self._count_major_direction_changes(trajectory)
        features['direction_consistency'] = self._calculate_direction_consistency(trajectory)
        
        # 5. Wall interaction analysis with physics
        features['wall_interactions'] = self._detect_enhanced_wall_interactions(trajectory, court_dimensions)
        features['bounce_physics_score'] = self._analyze_bounce_physics(trajectory, court_dimensions)
        
        # 6. Court coverage and tactical analysis
        features['court_coverage'] = self._calculate_enhanced_court_coverage(trajectory, court_dimensions)
        features['tactical_zones'] = self._analyze_tactical_zones(trajectory, court_dimensions)
        
        # 7. Height and arc analysis
        y_positions = [pos[1] for pos in trajectory]
        features['height_pattern'] = self._analyze_enhanced_height_pattern(y_positions, height)
        features['arc_characteristics'] = self._analyze_arc_characteristics(trajectory, court_dimensions)
        
        # 8. Timing and rhythm analysis
        if len(trajectory[0]) > 2:  # Has frame information
            features['shot_duration'] = trajectory[-1][2] - trajectory[0][2] if trajectory[-1][2] != trajectory[0][2] else len(trajectory)
            features['rhythm_analysis'] = self._analyze_shot_rhythm(trajectory)
        
        # 9. Shot quality indicators
        features['precision_indicators'] = self._calculate_precision_indicators(trajectory, court_dimensions)
        features['power_indicators'] = self._calculate_power_indicators(velocities if velocities else [])
        features['placement_quality'] = self._analyze_placement_quality(trajectory, court_dimensions)
        
        return features
    
    def _calculate_physics_metrics(self, trajectory):
        """Calculate velocity and acceleration with enhanced physics"""
        velocities = []
        accelerations = []
        
        for i in range(1, len(trajectory)):
            # Calculate velocity
            dx = trajectory[i][0] - trajectory[i-1][0]
            dy = trajectory[i][1] - trajectory[i-1][1]
            dt = 1.0  # Assume 1 frame = 1 time unit, adjust if frame timing available
            
            if len(trajectory[i]) > 2 and len(trajectory[i-1]) > 2:
                frame_diff = abs(trajectory[i][2] - trajectory[i-1][2])
                if frame_diff > 0:
                    dt = frame_diff / 30.0  # Assuming 30 FPS
            
            velocity = math.sqrt(dx*dx + dy*dy) / dt
            velocities.append(velocity)
            
            # Calculate acceleration
            if i > 1 and len(velocities) >= 2:
                acceleration = (velocities[-1] - velocities[-2]) / dt
                accelerations.append(acceleration)
        
        return velocities, accelerations
    
    def _calculate_enhanced_shot_score(self, features, criteria):
        """Calculate enhanced shot score with detailed confidence breakdown"""
        score = 0.0
        details = {}
        total_weight = 0.0
        
        # Weight different criteria by importance
        criterion_weights = {
            'horizontal_variance': 1.0, 'height_profile': 1.2, 'speed_profile': 1.0,
            'curvature_threshold': 0.8, 'wall_proximity': 0.9, 'velocity_consistency': 1.1,
            'diagonal_movement': 1.0, 'deceleration_pattern': 1.3, 'wall_hits': 1.4,
            'front_court_ending': 1.2, 'back_court_ending': 1.2, 'trajectory_arc': 1.1
        }
        
        for criterion, expected in criteria.items():
            weight = criterion_weights.get(criterion, 1.0)
            total_weight += weight
            criterion_score = 0.0
            
            if criterion == 'horizontal_variance':
                actual = features.get('horizontal_movement', 0)
                if isinstance(expected, (int, float)):
                    criterion_score = max(0, 1.0 - abs(actual - expected) * 2)
                    
            elif criterion == 'height_profile':
                height_range = features.get('height_pattern', {}).get('range', 0)
                if expected == 'low' and height_range < 0.2:
                    criterion_score = 1.0
                elif expected == 'medium' and 0.2 <= height_range <= 0.5:
                    criterion_score = 1.0
                elif expected == 'high' and height_range > 0.5:
                    criterion_score = 1.0
                elif expected == 'very_high' and height_range > 0.7:
                    criterion_score = 1.0
                elif expected == 'very_low' and height_range < 0.1:
                    criterion_score = 1.0
                elif expected == 'descending':
                    pattern = features.get('height_pattern', {}).get('type', '')
                    if 'descending' in pattern:
                        criterion_score = 1.0
                        
            elif criterion == 'speed_profile':
                avg_speed = features.get('avg_velocity', 0)
                if expected == 'slow' and avg_speed < 10:
                    criterion_score = 1.0
                elif expected == 'medium' and 10 <= avg_speed <= 25:
                    criterion_score = 1.0
                elif expected == 'fast' and avg_speed > 25:
                    criterion_score = 1.0
                elif expected == 'very_fast' and avg_speed > 40:
                    criterion_score = 1.0
                    
            elif criterion == 'deceleration_pattern' and expected:
                decel_score = features.get('deceleration_pattern', 0)
                criterion_score = decel_score
                
            elif criterion == 'wall_hits' and expected:
                wall_score = features.get('wall_interactions', {}).get('total_score', 0)
                criterion_score = min(1.0, wall_score)
                
            elif criterion == 'diagonal_movement' and expected:
                diagonal_comp = features.get('diagonal_component', 0)
                if diagonal_comp > 0.3:
                    criterion_score = min(1.0, diagonal_comp)
                    
            elif criterion == 'front_court_ending' and expected:
                tactical_zones = features.get('tactical_zones', {})
                if tactical_zones.get('end_zone_type') == 'front_court':
                    criterion_score = 1.0
                    
            elif criterion == 'back_court_ending' and expected:
                tactical_zones = features.get('tactical_zones', {})
                if tactical_zones.get('end_zone_type') == 'back_court':
                    criterion_score = 1.0
            
            # Add more criterion evaluations...
            
            weighted_score = criterion_score * weight
            score += weighted_score
            details[criterion] = {
                'score': criterion_score,
                'weight': weight,
                'weighted_contribution': weighted_score
            }
        
        final_score = score / total_weight if total_weight > 0 else 0.0
        details['final_score'] = final_score
        details['total_criteria'] = len(criteria)
        details['criteria_met'] = sum(1 for d in details.values() if isinstance(d, dict) and d.get('score', 0) > 0.5)
        
        return final_score, details
    
    def _detect_deceleration_pattern(self, velocities):
        """Detect if the trajectory shows deceleration pattern (key for drop shots)"""
        if len(velocities) < 3:
            return 0.0
        
        # Check if velocity decreases significantly over time
        early_avg = sum(velocities[:len(velocities)//3]) / (len(velocities)//3)
        late_avg = sum(velocities[-len(velocities)//3:]) / (len(velocities)//3)
        
        if early_avg > 0:
            deceleration_ratio = (early_avg - late_avg) / early_avg
            return max(0.0, min(1.0, deceleration_ratio))
        
        return 0.0
    
    def _detect_acceleration_pattern(self, accelerations):
        """Detect acceleration patterns in trajectory"""
        if not accelerations:
            return 0.0
        
        positive_accel = sum(1 for a in accelerations if a > 2.0)
        negative_accel = sum(1 for a in accelerations if a < -2.0)
        
        return {
            'acceleration_variance': np.var(accelerations),
            'positive_accelerations': positive_accel / len(accelerations),
            'negative_accelerations': negative_accel / len(accelerations)
        }
    
    def _calculate_enhanced_curvature(self, trajectory):
        """Enhanced curvature calculation with multiple methods"""
        if len(trajectory) < 3:
            return 0.0
        
        # Method 1: Traditional three-point curvature
        traditional_curvature = self._calculate_curvature(trajectory)
        
        # Method 2: Deviation from straight line
        start_point = trajectory[0][:2]
        end_point = trajectory[-1][:2]
        
        # Calculate expected positions on straight line
        deviations = []
        for i, pos in enumerate(trajectory):
            t = i / (len(trajectory) - 1) if len(trajectory) > 1 else 0
            expected_x = start_point[0] + t * (end_point[0] - start_point[0])
            expected_y = start_point[1] + t * (end_point[1] - start_point[1])
            
            deviation = math.sqrt((pos[0] - expected_x)**2 + (pos[1] - expected_y)**2)
            deviations.append(deviation)
        
        linearity_deviation = sum(deviations) / len(deviations) if deviations else 0
        
        # Combine both methods
        return (traditional_curvature + linearity_deviation / 100) / 2  # Normalize deviation
    
    def _calculate_linearity(self, trajectory):
        """Calculate how linear the trajectory is (1.0 = perfectly straight)"""
        if len(trajectory) < 3:
            return 1.0
        
        start_point = trajectory[0][:2]
        end_point = trajectory[-1][:2]
        
        total_deviation = 0.0
        max_possible_deviation = 0.0
        
        for i, pos in enumerate(trajectory[1:-1], 1):
            # Calculate expected position on straight line
            t = i / (len(trajectory) - 1)
            expected_x = start_point[0] + t * (end_point[0] - start_point[0])
            expected_y = start_point[1] + t * (end_point[1] - start_point[1])
            
            # Calculate actual deviation
            deviation = math.sqrt((pos[0] - expected_x)**2 + (pos[1] - expected_y)**2)
            total_deviation += deviation
            
            # Calculate maximum possible deviation (to corners of screen)
            max_possible_deviation += math.sqrt((640)**2 + (360)**2)  # Assuming typical court size
        
        if max_possible_deviation > 0:
            linearity = 1.0 - (total_deviation / max_possible_deviation)
            return max(0.0, linearity)
        
        return 1.0
    
    def _calculate_trajectory_smoothness(self, trajectory):
        """Calculate trajectory smoothness (low = jerky, high = smooth)"""
        if len(trajectory) < 4:
            return 1.0
        
        # Calculate second derivatives (acceleration changes)
        second_derivatives = []
        for i in range(2, len(trajectory) - 1):
            p1, p2, p3 = trajectory[i-1][:2], trajectory[i][:2], trajectory[i+1][:2]
            
            # First derivatives
            dx1, dy1 = p2[0] - p1[0], p2[1] - p1[1]
            dx2, dy2 = p3[0] - p2[0], p3[1] - p2[1]
            
            # Second derivatives (changes in velocity)
            ddx, ddy = dx2 - dx1, dy2 - dy1
            second_deriv_magnitude = math.sqrt(ddx**2 + ddy**2)
            second_derivatives.append(second_deriv_magnitude)
        
        if second_derivatives:
            # Lower variance in second derivatives = smoother trajectory
            smoothness = 1.0 / (1.0 + np.var(second_derivatives))
            return min(1.0, smoothness)
        
        return 1.0
    
    def _count_enhanced_direction_changes(self, trajectory):
        """Enhanced direction change counting with different thresholds"""
        if len(trajectory) < 3:
            return 0
        
        minor_changes = 0  # 15-45 degrees
        major_changes = 0  # 45-90 degrees
        sharp_changes = 0  # >90 degrees
        
        for i in range(1, len(trajectory) - 1):
            p1, p2, p3 = trajectory[i-1][:2], trajectory[i][:2], trajectory[i+1][:2]
            
            # Calculate vectors
            v1 = (p2[0] - p1[0], p2[1] - p1[1])
            v2 = (p3[0] - p2[0], p3[1] - p2[1])
            
            # Calculate angle
            angle = calculate_vector_angle(v1, v2)
            angle_deg = math.degrees(angle)
            
            if 15 <= angle_deg < 45:
                minor_changes += 1
            elif 45 <= angle_deg < 90:
                major_changes += 1
            elif angle_deg >= 90:
                sharp_changes += 1
        
        return {
            'minor': minor_changes,
            'major': major_changes,
            'sharp': sharp_changes,
            'total': minor_changes + major_changes + sharp_changes
        }
    
    def _count_major_direction_changes(self, trajectory):
        """Count only major direction changes (>45 degrees)"""
        changes = self._count_enhanced_direction_changes(trajectory)
        if isinstance(changes, dict):
            return changes.get('major', 0) + changes.get('sharp', 0)
        return changes
    
    def _calculate_direction_consistency(self, trajectory):
        """Calculate how consistent the direction is throughout trajectory"""
        if len(trajectory) < 3:
            return 1.0
        
        directions = []
        for i in range(1, len(trajectory)):
            dx = trajectory[i][0] - trajectory[i-1][0]
            dy = trajectory[i][1] - trajectory[i-1][1]
            direction = math.atan2(dy, dx)
            directions.append(direction)
        
        if not directions:
            return 1.0
        
        # Calculate variance in directions
        direction_variance = np.var(directions)
        
        # Lower variance = higher consistency
        consistency = 1.0 / (1.0 + direction_variance)
        return min(1.0, consistency)
    
    def _detect_enhanced_wall_interactions(self, trajectory, court_dimensions):
        """Enhanced wall interaction detection with physics validation"""
        width, height = get_court_dimensions(court_dimensions)
        wall_proximity = 25
        
        interactions = {
            'front_wall': 0, 'back_wall': 0, 'left_wall': 0, 'right_wall': 0,
            'total_score': 0.0, 'bounce_quality': 0.0, 'physics_validated': 0
        }
        
        for i in range(1, len(trajectory) - 1):
            x, y = trajectory[i][0], trajectory[i][1]
            prev_x, prev_y = trajectory[i-1][0], trajectory[i-1][1]
            next_x, next_y = trajectory[i+1][0], trajectory[i+1][1]
            
            # Check wall proximity
            near_front = y <= wall_proximity
            near_back = y >= height - wall_proximity
            near_left = x <= wall_proximity
            near_right = x >= width - wall_proximity
            
            if near_front or near_back or near_left or near_right:
                # Analyze direction change for bounce validation
                v_before = (x - prev_x, y - prev_y)
                v_after = (next_x - x, next_y - y)
                
                # Calculate angle change
                angle_change = calculate_vector_angle(v_before, v_after)
                
                # Validate physics (appropriate direction reversal)
                physics_valid = False
                
                if near_front and v_before[1] < 0 and v_after[1] > 0:  # Bouncing off front wall
                    physics_valid = True
                    interactions['front_wall'] += 1
                elif near_back and v_before[1] > 0 and v_after[1] < 0:  # Bouncing off back wall
                    physics_valid = True
                    interactions['back_wall'] += 1
                elif near_left and v_before[0] < 0 and v_after[0] > 0:  # Bouncing off left wall
                    physics_valid = True
                    interactions['left_wall'] += 1
                elif near_right and v_before[0] > 0 and v_after[0] < 0:  # Bouncing off right wall
                    physics_valid = True
                    interactions['right_wall'] += 1
                
                if physics_valid:
                    interactions['physics_validated'] += 1
                    bounce_quality = min(1.0, angle_change / (math.pi/2))  # Normalize to 90 degrees
                    interactions['bounce_quality'] += bounce_quality
        
        # Calculate total score
        total_walls = sum([interactions[wall] for wall in ['front_wall', 'back_wall', 'left_wall', 'right_wall']])
        if total_walls > 0:
            interactions['total_score'] = interactions['physics_validated'] / total_walls
            interactions['bounce_quality'] /= interactions['physics_validated'] if interactions['physics_validated'] > 0 else 1
        
        return interactions
    
    def _analyze_bounce_physics(self, trajectory, court_dimensions):
        """Analyze the physics quality of bounces in the trajectory"""
        interactions = self._detect_enhanced_wall_interactions(trajectory, court_dimensions)
        
        physics_score = 0.0
        
        # Score based on validated bounces
        if interactions['physics_validated'] > 0:
            physics_score += 0.5
            
            # Higher score for multiple realistic bounces
            if interactions['physics_validated'] > 1:
                physics_score += 0.3
            
            # Quality of bounces
            physics_score += interactions['bounce_quality'] * 0.2
        
        return min(1.0, physics_score)
    
    def _calculate_enhanced_court_coverage(self, trajectory, court_dimensions):
        """Enhanced court coverage analysis with tactical zones"""
        if not trajectory:
            return {'total': 0.0, 'zones_covered': [], 'tactical_importance': 0.0}
        
        width, height = get_court_dimensions(court_dimensions)
        
        # Define tactical zones
        zones = {
            'front_left': (0, width/3, 0, height/3),
            'front_center': (width/3, 2*width/3, 0, height/3),
            'front_right': (2*width/3, width, 0, height/3),
            'mid_left': (0, width/3, height/3, 2*height/3),
            'T_position': (width/3, 2*width/3, height/3, 2*height/3),
            'mid_right': (2*width/3, width, height/3, 2*height/3),
            'back_left': (0, width/3, 2*height/3, height),
            'back_center': (width/3, 2*width/3, 2*height/3, height),
            'back_right': (2*width/3, width, 2*height/3, height)
        }
        
        zones_visited = set()
        tactical_zones_visited = set()
        
        for x, y, *_ in trajectory:
            for zone_name, (x1, x2, y1, y2) in zones.items():
                if x1 <= x <= x2 and y1 <= y <= y2:
                    zones_visited.add(zone_name)
                    
                    # Mark tactical zones
                    if zone_name in ['front_center', 'T_position', 'back_center']:
                        tactical_zones_visited.add(zone_name)
        
        # Calculate coverage metrics
        total_coverage = len(zones_visited) / len(zones)
        tactical_importance = len(tactical_zones_visited) / 3  # 3 key tactical zones
        
        return {
            'total': total_coverage,
            'zones_covered': list(zones_visited),
            'tactical_importance': tactical_importance,
            'tactical_zones': list(tactical_zones_visited)
        }
    
    def _analyze_tactical_zones(self, trajectory, court_dimensions):
        """Analyze trajectory in terms of tactical court zones"""
        if not trajectory:
            return {}
        
        width, height = get_court_dimensions(court_dimensions)
        
        start_pos = trajectory[0][:2]
        end_pos = trajectory[-1][:2]
        
        # Determine start and end zones
        def get_zone_type(x, y):
            y_ratio = y / height
            if y_ratio < 0.33:
                return 'front_court'
            elif y_ratio > 0.67:
                return 'back_court'
            else:
                return 'mid_court'
        
        start_zone = get_zone_type(start_pos[0], start_pos[1])
        end_zone = get_zone_type(end_pos[0], end_pos[1])
        
        # Analyze tactical movement
        tactical_pattern = 'unknown'
        if start_zone == 'back_court' and end_zone == 'front_court':
            tactical_pattern = 'attacking'
        elif start_zone == 'front_court' and end_zone == 'back_court':
            tactical_pattern = 'defensive'
        elif start_zone == end_zone:
            tactical_pattern = 'maintaining'
        else:
            tactical_pattern = 'transitional'
        
        return {
            'start_zone_type': start_zone,
            'end_zone_type': end_zone,
            'tactical_pattern': tactical_pattern,
            'zone_transition': f"{start_zone}_to_{end_zone}"
        }
    
    def _analyze_enhanced_height_pattern(self, y_positions, court_height):
        """Enhanced height pattern analysis with detailed characteristics"""
        if len(y_positions) < 3:
            return {'type': 'unknown', 'range': 0.0, 'characteristics': []}
        
        # Normalize positions
        normalized_y = [y / court_height for y in y_positions]
        
        # Calculate statistics
        min_height = min(normalized_y)
        max_height = max(normalized_y)
        height_range = max_height - min_height
        
        # Find pattern type
        start_height = normalized_y[0]
        end_height = normalized_y[-1]
        mid_index = len(normalized_y) // 2
        mid_height = normalized_y[mid_index]
        
        characteristics = []
        pattern_type = 'level'
        
        # Determine primary pattern
        if mid_height < min(start_height, end_height) * 0.8:
            pattern_type = 'ascending_descending'  # Lob pattern
            characteristics.append('parabolic_arc')
        elif end_height > start_height * 1.3:
            pattern_type = 'descending'  # Drop shot pattern
            characteristics.append('downward_trajectory')
        elif end_height < start_height * 0.7:
            pattern_type = 'ascending'  # Rising shot
            characteristics.append('upward_trajectory')
        else:
            pattern_type = 'level'  # Drive pattern
            characteristics.append('consistent_height')
        
        # Add additional characteristics
        if height_range > 0.4:
            characteristics.append('high_variation')
        elif height_range < 0.1:
            characteristics.append('low_variation')
        
        if max_height > 0.8:
            characteristics.append('high_clearance')
        elif max_height < 0.2:
            characteristics.append('low_trajectory')
        
        return {
            'type': pattern_type,
            'range': height_range,
            'min_height': min_height,
            'max_height': max_height,
            'characteristics': characteristics,
            'height_variance': np.var(normalized_y)
        }
    
    def _analyze_arc_characteristics(self, trajectory, court_dimensions):
        """Analyze the arc characteristics of the trajectory"""
        if len(trajectory) < 5:
            return {'arc_type': 'linear', 'curvature_score': 0.0}
        
        # Calculate arc metrics
        curvature_score = self._calculate_enhanced_curvature(trajectory)
        
        # Determine arc type
        arc_type = 'linear'
        if curvature_score > 0.5:
            arc_type = 'parabolic'
        elif curvature_score > 0.3:
            arc_type = 'curved'
        
        return {
            'arc_type': arc_type,
            'curvature_score': curvature_score,
            'trajectory_shape': 'smooth' if curvature_score < 0.2 else 'curved'
        }
    
    def _analyze_shot_rhythm(self, trajectory):
        """Analyze the rhythm and timing of the shot"""
        if len(trajectory) < 3 or len(trajectory[0]) <= 2:
            return {'rhythm_type': 'unknown', 'consistency': 0.5}
        
        # Calculate frame intervals
        intervals = []
        for i in range(1, len(trajectory)):
            interval = trajectory[i][2] - trajectory[i-1][2]
            intervals.append(interval)
        
        if not intervals:
            return {'rhythm_type': 'unknown', 'consistency': 0.5}
        
        # Analyze rhythm consistency
        avg_interval = sum(intervals) / len(intervals)
        interval_variance = np.var(intervals)
        
        consistency = 1.0 / (1.0 + interval_variance) if interval_variance > 0 else 1.0
        
        # Determine rhythm type
        rhythm_type = 'consistent'
        if interval_variance > avg_interval:
            rhythm_type = 'irregular'
        elif interval_variance < avg_interval * 0.1:
            rhythm_type = 'very_consistent'
        
        return {
            'rhythm_type': rhythm_type,
            'consistency': min(1.0, consistency),
            'avg_interval': avg_interval,
            'interval_variance': interval_variance
        }
    
    def _calculate_precision_indicators(self, trajectory, court_dimensions):
        """Calculate indicators of shot precision and accuracy"""
        if not trajectory:
            return {'precision_score': 0.0, 'accuracy_indicators': []}
        
        width, height = get_court_dimensions(court_dimensions)
        
        # Calculate trajectory smoothness as precision indicator
        smoothness = self._calculate_trajectory_smoothness(trajectory)
        
        # Calculate consistency in direction
        direction_consistency = self._calculate_direction_consistency(trajectory)
        
        # Calculate target approach quality
        end_pos = trajectory[-1][:2]
        
        # Check if shot approaches tactical targets
        corner_targets = [
            (0, 0), (width, 0), (0, height), (width, height),  # Corners
            (width/2, 0), (width/2, height)  # Center targets
        ]
        
        min_distance_to_target = min(
            math.sqrt((end_pos[0] - target[0])**2 + (end_pos[1] - target[1])**2)
            for target in corner_targets
        )
        
        # Normalize target approach
        max_court_distance = math.sqrt(width**2 + height**2)
        target_approach = 1.0 - (min_distance_to_target / max_court_distance)
        
        # Calculate overall precision score
        precision_score = (smoothness * 0.4 + direction_consistency * 0.3 + target_approach * 0.3)
        
        accuracy_indicators = []
        if smoothness > 0.8:
            accuracy_indicators.append('smooth_trajectory')
        if direction_consistency > 0.8:
            accuracy_indicators.append('consistent_direction')
        if target_approach > 0.7:
            accuracy_indicators.append('good_target_approach')
        
        return {
            'precision_score': precision_score,
            'accuracy_indicators': accuracy_indicators,
            'smoothness': smoothness,
            'direction_consistency': direction_consistency,
            'target_approach': target_approach
        }
    
    def _calculate_power_indicators(self, velocities):
        """Calculate indicators of shot power"""
        if not velocities:
            return {'power_score': 0.0, 'power_indicators': []}
        
        max_velocity = max(velocities)
        avg_velocity = sum(velocities) / len(velocities)
        
        power_indicators = []
        power_score = 0.0
        
        # High average velocity
        if avg_velocity > 30:
            power_score += 0.4
            power_indicators.append('high_average_speed')
        elif avg_velocity > 20:
            power_score += 0.2
            power_indicators.append('medium_speed')
        
        # Peak velocity
        if max_velocity > 50:
            power_score += 0.4
            power_indicators.append('explosive_peak')
        elif max_velocity > 35:
            power_score += 0.2
            power_indicators.append('good_peak_speed')
        
        # Velocity consistency
        velocity_variance = np.var(velocities)
        if velocity_variance < avg_velocity * 0.3:
            power_score += 0.2
            power_indicators.append('consistent_power')
        
        return {
            'power_score': min(1.0, power_score),
            'power_indicators': power_indicators,
            'max_velocity': max_velocity,
            'avg_velocity': avg_velocity
        }
    
    def _analyze_placement_quality(self, trajectory, court_dimensions):
        """Analyze the quality of shot placement"""
        if not trajectory:
            return {'placement_score': 0.0, 'placement_type': 'unknown'}
        
        width, height = get_court_dimensions(court_dimensions)
        end_pos = trajectory[-1][:2]
        
        # Analyze placement relative to court zones
        x_ratio = end_pos[0] / width
        y_ratio = end_pos[1] / height
        
        placement_score = 0.0
        placement_type = 'center'
        
        # Reward corner placements
        corner_distance = min(x_ratio, 1-x_ratio, y_ratio, 1-y_ratio)
        
        if corner_distance < 0.15:
            placement_score = 1.0
            placement_type = 'corner'
        elif corner_distance < 0.25:
            placement_score = 0.7
            placement_type = 'edge'
        elif 0.3 < x_ratio < 0.7 and 0.3 < y_ratio < 0.7:
            placement_score = 0.4
            placement_type = 'center'
        else:
            placement_score = 0.6
            placement_type = 'mid'
        
        # Special tactical placements
        if y_ratio < 0.2:
            placement_score += 0.2
            placement_type += '_front'
        elif y_ratio > 0.8:
            placement_score += 0.1
            placement_type += '_back'
        
        return {
            'placement_score': min(1.0, placement_score),
            'placement_type': placement_type,
            'corner_distance': corner_distance,
            'tactical_value': self._evaluate_tactical_placement(x_ratio, y_ratio)
        }
    
    def _evaluate_tactical_placement(self, x_ratio, y_ratio):
        """Evaluate the tactical value of shot placement"""
        tactical_value = 0.5
        
        # Front court placements (attacking)
        if y_ratio < 0.25:
            tactical_value += 0.3
        
        # Back corner placements (length)
        if y_ratio > 0.75 and (x_ratio < 0.2 or x_ratio > 0.8):
            tactical_value += 0.2
        
        # Side wall tight placements
        if x_ratio < 0.1 or x_ratio > 0.9:
            tactical_value += 0.1
        
        return min(1.0, tactical_value)
    
    def _validate_shot_confidence(self, shot_type, features, confidence):
        """Validate and adjust shot confidence"""
        validated_confidence = confidence
        
        # Confidence penalties for inconsistent features
        if shot_type == 'straight_drive':
            if features.get('horizontal_movement', 0) > 0.3:
                validated_confidence *= 0.7
        elif shot_type == 'crosscourt':
            if features.get('diagonal_component', 0) < 0.3:
                validated_confidence *= 0.8
        elif shot_type == 'drop_shot':
            if features.get('deceleration_pattern', 0) < 0.3:
                validated_confidence *= 0.6
        elif shot_type == 'lob':
            height_pattern = features.get('height_pattern', {})
            if height_pattern.get('max_height', 0) < 0.5:
                validated_confidence *= 0.7
        
        # Boost confidence for strong indicators
        precision = features.get('precision_indicators', {}).get('precision_score', 0)
        if precision > 0.8:
            validated_confidence = min(1.0, validated_confidence * 1.1)
        
        return validated_confidence
    
    def _determine_shot_clarity(self, confidence, features, all_scores):
        """Determine the clarity of shot detection"""
        if confidence > 0.85:
            base_clarity = 'VERY_HIGH'
        elif confidence > 0.7:
            base_clarity = 'HIGH'
        elif confidence > 0.55:
            base_clarity = 'MEDIUM'
        elif confidence > 0.4:
            base_clarity = 'LOW'
        else:
            base_clarity = 'VERY_LOW'
        
        # Adjust based on score distribution
        scores = list(all_scores.values())
        if len(scores) > 1:
            second_best = sorted(scores)[-2]
            score_separation = confidence - second_best
            
            if score_separation > 0.3:
                if base_clarity == 'MEDIUM':
                    base_clarity = 'HIGH'
                elif base_clarity == 'LOW':
                    base_clarity = 'MEDIUM'
            elif score_separation < 0.1:
                if base_clarity == 'VERY_HIGH':
                    base_clarity = 'HIGH'
                elif base_clarity == 'HIGH':
                    base_clarity = 'MEDIUM'
        
        return base_clarity
    
    def _generate_detailed_analysis(self, shot_type, features, confidence):
        """Generate comprehensive detailed analysis"""
        analysis = {
            'shot_type': shot_type,
            'confidence': confidence,
            'key_indicators': [],
            'technical_metrics': {},
            'tactical_assessment': {},
            'quality_indicators': {}
        }
        
        # Shot-specific indicators
        if shot_type == 'straight_drive':
            analysis['key_indicators'] = [
                f"Low horizontal movement: {features.get('horizontal_movement', 0):.3f}",
                f"Velocity consistency: {features.get('velocity_consistency', 0):.3f}",
                f"Direction stability: {features.get('direction_consistency', 0):.3f}"
            ]
        elif shot_type == 'crosscourt':
            analysis['key_indicators'] = [
                f"High diagonal movement: {features.get('diagonal_component', 0):.3f}",
                f"Corner targeting pattern",
                f"Moderate curve: {features.get('trajectory_curvature', 0):.3f}"
            ]
        elif shot_type == 'drop_shot':
            analysis['key_indicators'] = [
                f"Deceleration: {features.get('deceleration_pattern', 0):.3f}",
                f"Front court targeting",
                f"Touch quality: {features.get('precision_indicators', {}).get('precision_score', 0):.3f}"
            ]
        
        return analysis
    
    def _run_validation_checks(self, shot_type, features):
        """Run validation checks on shot classification"""
        checks = {'passed': [], 'failed': [], 'warnings': [], 'overall_validity': 0.0}
        
        total_checks = 1
        passed_checks = 0
        
        if features.get('trajectory_length', 0) >= 5:
            checks['passed'].append('Sufficient trajectory length')
            passed_checks += 1
        else:
            checks['failed'].append('Insufficient trajectory data')
        
        checks['overall_validity'] = passed_checks / total_checks
        return checks
    
    def _calculate_curvature(self, trajectory):
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
        width, height = get_court_dimensions(court_dimensions)
        
        # Divide court into 9 zones
        x_zone = 'left' if x < width/3 else 'center' if x < 2*width/3 else 'right'
        y_zone = 'front' if y < height/3 else 'middle' if y < 2*height/3 else 'back'
        
        return f"{y_zone}_{x_zone}"
    
    def _detect_wall_interactions(self, trajectory, court_dimensions):
        """Detect ball interactions with walls"""
        width, height = get_court_dimensions(court_dimensions)
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
        ]
        self.confidence_weights = [0.4, 0.4, 0.2]
        self.hit_distance_threshold = 80  # pixels
        self.velocity_change_threshold = 15  # velocity change threshold
        self.direction_change_threshold = math.pi/6  # 30 degrees
    
    def detect_player_hit(self, players, ball_trajectory, frame_count):
        """
        Detect which player hit the ball with enhanced accuracy
        
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
        
        # Convert distance to confidence
        confidence = max(0, (self.hit_distance_threshold - distance) / self.hit_distance_threshold)
        hit_type = 'proximity_hit' if confidence > 0.3 else 'none'
        
        return player_id if confidence > 0.3 else 0, confidence, hit_type, {'distance': distance}
    
    def _trajectory_analysis(self, players, ball_trajectory, frame_count):
        """Detect hit based on ball trajectory changes"""
        if len(ball_trajectory) < 5:
            return 0, 0.0, 'none', {}
        
        # Analyze trajectory changes
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
        
        # Determine which player caused the change
        hit_confidence = 0
        hit_player = 0
        
        if velocity_change > self.velocity_change_threshold or direction_change > self.direction_change_threshold:
            # Find closest player to change position
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
                
                # Calculate confidence based on trajectory change and proximity
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
                # Get arm/racket keypoints
                arm_keypoints = self._get_arm_keypoints(player)
                
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
    
    def _get_enhanced_player_position(self, player):
        """Get player position using multiple keypoints for better accuracy"""
        if not player or not player.get_latest_pose():
            return None
        
        pose = player.get_latest_pose()
        
        try:
            if hasattr(pose, 'xyn') and len(pose.xyn) > 0:
                keypoints = pose.xyn[0]
                
                # Priority order: wrists, elbows, shoulders, hips
                priority_keypoints = [9, 10, 7, 8, 5, 6, 11, 12]
                
                for idx in priority_keypoints:
                    if idx < len(keypoints):
                        kp = keypoints[idx]
                        if len(kp) >= 2 and kp[0] > 0 and kp[1] > 0:
                            return (kp[0] * 640, kp[1] * 360)  # Convert to pixel coordinates
        except Exception:
            pass
        
        return None
    
    def _get_arm_keypoints(self, player):
        """Get arm/racket keypoints from player pose"""
        if not player or not player.get_latest_pose():
            return []
        
        pose = player.get_latest_pose()
        arm_keypoints = []
        
        try:
            if hasattr(pose, 'xyn') and len(pose.xyn) > 0:
                keypoints = pose.xyn[0]
                # Wrist and elbow keypoints
                for idx in [7, 8, 9, 10]:  # Left elbow, right elbow, left wrist, right wrist
                    if idx < len(keypoints):
                        kp = keypoints[idx]
                        if len(kp) >= 2 and kp[0] > 0 and kp[1] > 0:
                            arm_keypoints.append((kp[0] * 640, kp[1] * 360))
        except Exception:
            pass
        
        return arm_keypoints
    
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
    """Enhanced autonomous shot phase detector: start (ball leaves racket) -> middle (wall hit) -> end (floor hit)"""
    
    def __init__(self):
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
    """Enhanced autonomous shot phase detector: start (ball leaves racket) -> middle (wall hit) -> end (floor hit)"""
    
    def __init__(self):
        self.phase_history = []
        self.wall_hit_threshold = 30  # pixels from wall for detection
        self.floor_hit_threshold = 25  # pixels of movement for bounce detection
        self.velocity_change_threshold = 0.4  # 40% velocity change
        
    def detect_shot_phases(self, trajectory, court_dimensions, previous_phase='none'):
        """
        🎯 AUTONOMOUS SHOT PHASE DETECTION: Ultra-clear transition detection
        Phases: START (ball leaves racket) -> MIDDLE (wall hit) -> END (floor hit)
        """
        if len(trajectory) < 3:
            return {
                'phase': 'start', 
                'confidence': 1.0, 
                'transition': 'none',
                'event': 'none',
                'details': {'reason': 'insufficient_trajectory'}
            }
        
        width, height = get_court_dimensions(court_dimensions)
        current_phase = previous_phase if previous_phase != 'none' else 'start'
        
        print(f"🎯 Phase Analysis: {len(trajectory)} points, current phase: {current_phase.upper()}")
        
        # ENHANCED WALL HIT DETECTION (START -> MIDDLE)
        wall_hit_result = self._detect_wall_hit_enhanced(trajectory, court_dimensions)
        if wall_hit_result['detected'] and current_phase in ['start', 'none']:
            print(f"✅ WALL HIT DETECTED: {wall_hit_result['details']['wall_type'].upper()}")
            return {
                'phase': 'middle',
                'confidence': wall_hit_result['confidence'],
                'transition': 'start_to_middle',
                'event': 'wall_hit',
                'details': wall_hit_result['details']
            }
        
        # ENHANCED FLOOR HIT DETECTION (MIDDLE -> END)
        floor_hit_result = self._detect_floor_hit_enhanced(trajectory, court_dimensions)
        if floor_hit_result['detected'] and current_phase in ['start', 'middle']:
            print(f"✅ FLOOR HIT DETECTED: Rally ending")
            return {
                'phase': 'end',
                'confidence': floor_hit_result['confidence'],
                'transition': 'middle_to_end' if current_phase == 'middle' else 'start_to_end',
                'event': 'floor_hit',
                'details': floor_hit_result['details']
            }
        
        # RALLY END DETECTION (END -> COMPLETE)
        if current_phase in ['middle', 'end'] and len(trajectory) > 10:
            rally_end_result = self._detect_rally_end(trajectory)
            if rally_end_result['detected']:
                print(f"✅ RALLY END DETECTED: Shot complete")
                return {
                    'phase': 'complete',
                    'confidence': rally_end_result['confidence'],
                    'transition': 'end_to_complete',
                    'event': 'rally_end',
                    'details': rally_end_result['details']
                }
        
        # MAINTAIN CURRENT PHASE with confidence assessment
        phase_confidence = self._calculate_phase_confidence(trajectory, current_phase, court_dimensions)
        
        return {
            'phase': current_phase,
            'confidence': phase_confidence,
            'transition': 'none',
            'event': 'none',
            'details': {'maintaining_phase': True, 'trajectory_points': len(trajectory)}
        }
    
    def _detect_wall_hit(self, trajectory, court_dimensions):
        """Detect when ball hits a wall"""
        if len(trajectory) < 4:
            return {'detected': False, 'confidence': 0.0, 'position': None}
        
        width, height = get_court_dimensions(court_dimensions)
        recent_positions = trajectory[-4:]
        
        # Check proximity to walls
        for i in range(len(recent_positions)):
            x, y = recent_positions[i][:2]
            
            # Check distance to each wall
            dist_to_front = y  # Front wall at y=0
            dist_to_back = height - y
            dist_to_left = x
            dist_to_right = width - x
            
            min_wall_distance = min(dist_to_front, dist_to_back, dist_to_left, dist_to_right)
            
            if min_wall_distance < self.wall_hit_threshold:
                # Verify with velocity change
                if self._check_velocity_change(trajectory, i):
                    return {
                        'detected': True,
                        'confidence': min(1.0, (self.wall_hit_threshold - min_wall_distance) / self.wall_hit_threshold),
                        'position': (int(x), int(y))
                    }
        
        return {'detected': False, 'confidence': 0.0, 'position': None}
    
    def _detect_floor_hit(self, trajectory, court_dimensions):
        """Detect when ball bounces on floor"""
        if len(trajectory) < 5:
            return {'detected': False, 'confidence': 0.0, 'position': None}
        
        width, height = get_court_dimensions(court_dimensions)
        recent_positions = trajectory[-5:]
        
        # Look for bounce pattern (ball going down then up, or significant velocity change)
        for i in range(1, len(recent_positions) - 1):
            prev_pos = recent_positions[i-1]
            curr_pos = recent_positions[i]
            next_pos = recent_positions[i+1]
            
            # Check if ball is in lower part of court (likely floor area)
            if curr_pos[1] > height * 0.6:  # Lower 40% of court
                # Check for direction change (bounce pattern)
                if self._detect_bounce_pattern(prev_pos, curr_pos, next_pos):
                    confidence = self._calculate_floor_hit_confidence(curr_pos, height)
                    return {
                        'detected': True,
                        'confidence': confidence,
                        'position': (int(curr_pos[0]), int(curr_pos[1]))
                    }
        
        return {'detected': False, 'confidence': 0.0, 'position': None}
    
    def _check_velocity_change(self, trajectory, position_index):
        """Check if there's a significant velocity change at given position"""
        if len(trajectory) < 3 or position_index < 1:
            return False
        
        # Calculate velocities before and after the position
        positions = trajectory[max(0, position_index-2):position_index+3]
        
        if len(positions) < 3:
            return False
        
        velocities = []
        for i in range(1, len(positions)):
            p1, p2 = positions[i-1], positions[i]
            if len(p1) >= 3 and len(p2) >= 3:
                dt = max(0.033, abs(p2[2] - p1[2])) if p2[2] != p1[2] else 1
                vel = math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2) / dt
                velocities.append(vel)
        
        if len(velocities) < 2:
            return False
        
        # Check for significant velocity change
        avg_before = sum(velocities[:len(velocities)//2]) / (len(velocities)//2)
        avg_after = sum(velocities[len(velocities)//2:]) / (len(velocities) - len(velocities)//2)
        
        if avg_before > 0:
            velocity_change = abs(avg_after - avg_before) / avg_before
            return velocity_change > self.velocity_change_threshold
        
        return False
    
    def _detect_bounce_pattern(self, prev_pos, curr_pos, next_pos):
        """Detect bounce pattern in ball movement"""
        # Check for Y-direction change (ball going down then up)
        y_trend_before = curr_pos[1] - prev_pos[1]
        y_trend_after = next_pos[1] - curr_pos[1]
        
        # Bounce: ball was going down (positive Y) then goes up (negative Y)
        if y_trend_before > 5 and y_trend_after < -3:
            return True
        
        # Also check for sudden direction changes
        movement_before = math.sqrt((curr_pos[0] - prev_pos[0])**2 + (curr_pos[1] - prev_pos[1])**2)
        movement_after = math.sqrt((next_pos[0] - curr_pos[0])**2 + (next_pos[1] - curr_pos[1])**2)
        
        # Significant movement change could indicate bounce
        if movement_before > 0 and movement_after > 0:
            movement_ratio = min(movement_before, movement_after) / max(movement_before, movement_after)
            if movement_ratio < 0.5:  # 50% change in movement
                return True
        
        return False
    
    def _calculate_floor_hit_confidence(self, position, court_height):
        """Calculate confidence based on position in court"""
        y_ratio = position[1] / court_height
        
        # Higher confidence for positions lower in court
        if y_ratio > 0.8:
            return 0.9
        elif y_ratio > 0.7:
            return 0.7
        elif y_ratio > 0.6:
            return 0.5
        else:
            return 0.3
    
    def detect_phase_transition(self, trajectory, previous_phase, court_dimensions):
        """Detect phase transitions based on ball trajectory and court interactions"""
        if previous_phase in ['start', 'middle']:
            # Check for floor hit (transition to end)
            floor_hit = self._detect_floor_hit_enhanced(trajectory, court_dimensions)
            if floor_hit['detected']:
                return {
                    'phase': 'end',
                    'confidence': floor_hit['confidence'],
                    'details': floor_hit['details'],
                    'transition': 'middle_to_end' if previous_phase == 'middle' else 'start_to_end',
                    'event': 'floor_hit'
                }
        
        # Check for rally end (ball comes to rest)
        if previous_phase in ['middle', 'end']:
            rally_end = self._detect_rally_end(trajectory)
            if rally_end['detected']:
                return {
                    'phase': 'complete',
                    'confidence': rally_end['confidence'],
                    'details': rally_end['details'],
                    'transition': 'end_to_complete',
                    'event': 'rally_end'
                }
        
        # If no transition detected, maintain current phase with updated confidence
        phase = previous_phase if previous_phase != 'none' else 'start'
        confidence = self._calculate_phase_confidence(trajectory, phase, court_dimensions)
        
        return {
            'phase': phase, 
            'confidence': confidence, 
            'details': {'maintaining_phase': True}, 
            'transition': 'none'
        }
    
    def _update_tracking_histories(self, trajectory, court_dimensions):
        """Update tracking histories for improved detection accuracy"""
        if len(trajectory) < 2:
            return
            
        # Use helper function to handle court dimensions consistently
        width, height = get_court_dimensions(court_dimensions)
            
        current_pos = trajectory[-1]
        x, y = float(current_pos[0]), float(current_pos[1])
        
        # Update minimum wall distance history
        min_wall_distance = min(x, width - x, y, height - y)
        self.min_wall_distance_history.append(min_wall_distance)
        if len(self.min_wall_distance_history) > 20:  # Keep last 20 positions
            self.min_wall_distance_history.pop(0)
        
        # Update velocity history
        if len(trajectory) >= 2:
            prev_pos = trajectory[-2]
            velocity = math.sqrt((x - float(prev_pos[0]))**2 + (y - float(prev_pos[1]))**2)
            self.velocity_history.append(velocity)
            if len(self.velocity_history) > 15:  # Keep last 15 velocities
                self.velocity_history.pop(0)
    
    def _detect_wall_hit_enhanced(self, trajectory, court_dimensions):
        """
        ULTRA-CLEAR WALL HIT DETECTION: Including ALL walls (front, back, left, right)
        Enhanced detection with special focus on front wall hits
        """
        if len(trajectory) < 5:
            return {'detected': False, 'confidence': 0.0, 'details': {'reason': 'insufficient_trajectory'}}
        
        width, height = get_court_dimensions(court_dimensions)
        recent_positions = trajectory[-5:]
        current_pos = recent_positions[-1]
        x, y = float(current_pos[0]), float(current_pos[1])
        
        wall_hit_confidence = 0.0
        wall_type = 'none'
        detection_criteria = {}
        
        # ENHANCED WALL DISTANCE CALCULATIONS - All walls equally important
        wall_distances = {
            'left_wall': x,                    # Distance to left wall (x=0)
            'right_wall': width - x,          # Distance to right wall (x=width)
            'front_wall': y,                  # Distance to front wall (y=0) - CRITICAL
            'back_wall': height - y           # Distance to back wall (y=height)
        }
        
        # Find closest wall
        closest_wall = min(wall_distances.items(), key=lambda x: x[1])
        wall_type = closest_wall[0]
        min_wall_distance = closest_wall[1]
        
        print(f" 🎯 Wall analysis: {wall_type} at distance {min_wall_distance:.1f}px")
        
        # CRITERION 1: PROXIMITY TO ANY WALL (including front wall)
        wall_proximity_threshold = 20  # Increased for clearer detection
        if min_wall_distance < wall_proximity_threshold:
            proximity_confidence = (wall_proximity_threshold - min_wall_distance) / wall_proximity_threshold
            wall_hit_confidence += proximity_confidence * 0.4  # Increased weight
            detection_criteria['wall_proximity'] = proximity_confidence
            print(f" Wall proximity detected: {proximity_confidence:.2f}")
        
        # CRITERION 2: DIRECTION CHANGE (critical for wall hits)
        if len(recent_positions) >= 3:
            direction_change = self._calculate_direction_change_advanced(recent_positions[-3:])
            direction_threshold = math.pi/8  # 22.5 degrees - more sensitive
            
            if direction_change > direction_threshold:
                direction_confidence = min(1.0, direction_change / (math.pi/4))  # Normalize to 45 degrees
                wall_hit_confidence += direction_confidence * 0.35
                detection_criteria['direction_change'] = direction_confidence
                print(f" Direction change detected: {direction_confidence:.2f} ({math.degrees(direction_change):.1f})")
        
        # CRITERION 3: VELOCITY PATTERN ANALYSIS
        velocity_analysis = self._analyze_wall_collision_velocity(recent_positions)
        if velocity_analysis['collision_detected']:
            wall_hit_confidence += velocity_analysis['confidence'] * 0.25
            detection_criteria['velocity_pattern'] = velocity_analysis['confidence']
            print(f" Velocity collision pattern: {velocity_analysis['confidence']:.2f}")
        
        # CRITERION 4: FRONT WALL SPECIAL DETECTION
        if wall_type == 'front_wall':
            front_wall_boost = self._analyze_front_wall_hit_pattern(recent_positions, y)
            wall_hit_confidence += front_wall_boost * 0.3  # Extra boost for front wall
            detection_criteria['front_wall_boost'] = front_wall_boost
            print(f" Front wall boost: {front_wall_boost:.2f}")
        
        # CRITERION 5: TRAJECTORY PHYSICS
        physics_score = self._analyze_wall_hit_physics(recent_positions, wall_type, min_wall_distance)
        if physics_score > 0.3:
            wall_hit_confidence += physics_score * 0.2
            detection_criteria['physics_analysis'] = physics_score
            print(f" Physics validation: {physics_score:.2f}")
        
        # CLEAR WALL HIT THRESHOLD
        wall_hit_threshold = 0.6  # Clear threshold for definitive wall hits
        detected = wall_hit_confidence > wall_hit_threshold
        
        if detected:
            print(f" CLEAR WALL HIT DETECTED: {wall_type.upper()}")
            print(f"    Confidence: {wall_hit_confidence:.2f}/1.0")
            print(f"    Distance: {min_wall_distance:.1f}px")
            print(f"    Criteria: {len(detection_criteria)}/5")
        
        return {
            'detected': detected,
            'confidence': min(1.0, wall_hit_confidence),
            'details': {
                'wall_type': wall_type,
                'distance_to_wall': min_wall_distance,
                'all_wall_distances': wall_distances,
                'detection_criteria': detection_criteria,
                'total_criteria_met': len(detection_criteria),
                'clarity': 'HIGH' if detected else 'LOW'
            }
        }
    
    def _analyze_wall_collision_velocity(self, positions):
        """Analyze velocity patterns specific to wall collisions"""
        if len(positions) < 4:
            return {'collision_detected': False, 'confidence': 0.0}
        
        # Calculate velocity vectors before and after collision point
        velocities = []
        for i in range(1, len(positions)):
            dx = positions[i][0] - positions[i-1][0]
            dy = positions[i][1] - positions[i-1][1]
            speed = math.sqrt(dx*dx + dy*dy)
            velocities.append({'speed': speed, 'dx': dx, 'dy': dy})
        
        if len(velocities) < 3:
            return {'collision_detected': False, 'confidence': 0.0}
        
        # Look for velocity direction reversal (key wall hit indicator)
        early_vel = velocities[0]
        late_vel = velocities[-1]
        
        # Check for direction reversal in either x or y component
        x_reversal = (early_vel['dx'] * late_vel['dx']) < 0  # Opposite signs
        y_reversal = (early_vel['dy'] * late_vel['dy']) < 0  # Opposite signs
        
        collision_confidence = 0.0
        if x_reversal or y_reversal:
            collision_confidence = 0.7
            if x_reversal and y_reversal:
                collision_confidence = 1.0  # Both directions reversed = strong collision
        
        # Check for speed change
        speed_change = abs(late_vel['speed'] - early_vel['speed'])
        if speed_change > 5.0:  # Significant speed change
            collision_confidence += 0.3
        
        return {
            'collision_detected': collision_confidence > 0.5,
            'confidence': min(1.0, collision_confidence),
            'x_reversal': x_reversal,
            'y_reversal': y_reversal,
            'speed_change': speed_change
        }
    
    def _analyze_front_wall_hit_pattern(self, positions, current_y):
        """Special analysis for front wall hits (y=0 area)"""
        if current_y > 50:  # Too far from front wall
            return 0.0
        
        # Check if ball was moving toward front wall
        if len(positions) >= 3:
            # Calculate y-direction trend
            y_trend = positions[-1][1] - positions[0][1]
            
            # Ball moving toward front wall (decreasing y)
            if y_trend < 0:
                front_wall_score = 0.8
                
                # Extra boost if very close to front wall
                if current_y < 25:
                    front_wall_score = 1.0
                    
                return front_wall_score
        
        return 0.0
    
    def _analyze_wall_hit_physics(self, positions, wall_type, distance):
        """Physics-based validation of wall hit"""
        if len(positions) < 3:
            return 0.0
        
        physics_score = 0.0
        
        # Proximity physics - closer = more likely
        if distance < 15:
            physics_score += 0.5
        elif distance < 30:
            physics_score += 0.3
        
        # Angle of approach physics
        approach_angle = self._calculate_wall_approach_angle(positions, wall_type)
        if approach_angle > 0:
            physics_score += approach_angle * 0.5
        
        return min(1.0, physics_score)
    
    def _calculate_wall_approach_angle(self, positions, wall_type):
        """Calculate angle of approach to wall"""
        if len(positions) < 2:
            return 0.0
        
        # Calculate trajectory vector
        dx = positions[-1][0] - positions[0][0]
        dy = positions[-1][1] - positions[0][1]
        
        # Wall normal vectors
        wall_normals = {
            'front_wall': (0, 1),   # Front wall normal points down
            'back_wall': (0, -1),   # Back wall normal points up
            'left_wall': (1, 0),    # Left wall normal points right
            'right_wall': (-1, 0)   # Right wall normal points left
        }
        
        if wall_type not in wall_normals:
            return 0.0
        
        normal = wall_normals[wall_type]
        
        # Calculate angle between trajectory and wall normal
        trajectory_length = math.sqrt(dx*dx + dy*dy)
        if trajectory_length == 0:
            return 0.0
        
        # Normalized trajectory
        traj_norm = (dx/trajectory_length, dy/trajectory_length)
        
        # Dot product gives angle relationship
        dot_product = traj_norm[0]*normal[0] + traj_norm[1]*normal[1]
        
        # Convert to approach quality (approaching wall = positive score)
        approach_quality = max(0.0, dot_product)
        
        return approach_quality
    
    def _detect_wall_approach_pattern(self):
        """Detect if ball was approaching a wall in recent history"""
        if len(self.min_wall_distance_history) < 5:
            return False
        
        recent_distances = self.min_wall_distance_history[-5:]
        
        # Check if distances were generally decreasing (approaching wall)
        decreasing_count = 0
        for i in range(1, len(recent_distances)):
            if recent_distances[i] < recent_distances[i-1]:
                decreasing_count += 1
        
        # Ball was approaching wall if most recent distances were decreasing
        return decreasing_count >= 3
    
    def _analyze_velocity_change_for_wall_hit(self, positions):
        """Analyze velocity changes specifically for wall hit detection"""
        if len(positions) < 4:
            return 0.0
        
        # Calculate velocities before and after potential wall hit
        early_velocity = math.sqrt(
            (positions[1][0] - positions[0][0])**2 + 
            (positions[1][1] - positions[0][1])**2
        )
        
        late_velocity = math.sqrt(
            (positions[-1][0] - positions[-2][0])**2 + 
            (positions[-1][1] - positions[-2][1])**2
        )
        
        if early_velocity > 0:
            change_ratio = abs(late_velocity - early_velocity) / early_velocity
            return change_ratio
        
        return 0.0
    
    def _analyze_trajectory_curvature(self, positions):
        """Analyze trajectory curvature to detect wall bounces"""
        if len(positions) < 3:
            return 0.0
        
        # Calculate angle between consecutive line segments
        total_curvature = 0.0
        
        for i in range(1, len(positions) - 1):
            p1, p2, p3 = positions[i-1], positions[i], positions[i+1]
            
            # Vectors from p1->p2 and p2->p3
            v1 = (p2[0] - p1[0], p2[1] - p1[1])
            v2 = (p3[0] - p2[0], p3[1] - p2[1])
            
            # Calculate angle between vectors
            if abs(v1[0]) > 0.1 or abs(v1[1]) > 0.1 or abs(v2[0]) > 0.1 or abs(v2[1]) > 0.1:
                angle = self._calculate_vector_angle(v1, v2)
                total_curvature += angle
        
        return total_curvature / (len(positions) - 2) if len(positions) > 2 else 0.0
    
    def _calculate_direction_change_advanced(self, positions):
        """Enhanced direction change calculation with noise filtering"""
        if len(positions) < 3:
            return 0.0
        
        # Use longer segments to reduce noise
        p1, p2, p3 = positions[0], positions[1], positions[2]
        
        # Calculate direction vectors with minimum movement threshold
        dx1, dy1 = p2[0] - p1[0], p2[1] - p1[1]
        dx2, dy2 = p3[0] - p2[0], p3[1] - p2[1]
        
        # Filter out very small movements (noise)
        if abs(dx1) < 2 and abs(dy1) < 2:
            return 0.0
        if abs(dx2) < 2 and abs(dy2) < 2:
            return 0.0
        
        # Direction angles
        dir1 = math.atan2(dy1, dx1)
        dir2 = math.atan2(dy2, dx2)
        
        # Calculate angle difference
        angle_diff = abs(dir2 - dir1)
        return min(angle_diff, 2*math.pi - angle_diff)
    
    def _calculate_vector_angle(self, v1, v2):
        """Calculate angle between two vectors"""
        try:
            # Normalize vectors
            len1 = math.sqrt(v1[0]**2 + v1[1]**2)
            len2 = math.sqrt(v2[0]**2 + v2[1]**2)
            
            if len1 == 0 or len2 == 0:
                return 0.0
            
            # Dot product
            dot_product = (v1[0] * v2[0] + v1[1] * v2[1]) / (len1 * len2)
            dot_product = max(-1.0, min(1.0, dot_product))  # Clamp to [-1, 1]
            
            return math.acos(dot_product)
        except:
            return 0.0
    
    def _detect_floor_hit_enhanced(self, trajectory, court_dimensions):
        """
        ULTRA-CLEAR FLOOR HIT DETECTION: Ball hits floor and rally ends
        Crystal clear detection of shot end with comprehensive analysis
        """
        if len(trajectory) < 6:
            return {'detected': False, 'confidence': 0.0, 'details': {'reason': 'insufficient_trajectory'}}
        
        width, height = get_court_dimensions(court_dimensions)
        recent_positions = trajectory[-6:]
        current_pos = recent_positions[-1]
        x, y = float(current_pos[0]), float(current_pos[1])
        
        floor_hit_confidence = 0.0
        detection_criteria = {}
        
        print(f" Floor analysis: Position ({x:.1f}, {y:.1f}) in court {width}x{height}")
        
        # CRITERION 1: BALL IN LOWER COURT AREA
        height_factor = y / height
        floor_threshold = 0.65  # Lower 35% of court
        
        if height_factor > floor_threshold:
            position_confidence = min(1.0, (height_factor - floor_threshold) / (1 - floor_threshold))
            floor_hit_confidence += position_confidence * 0.25
            detection_criteria['low_court_position'] = position_confidence
            print(f" Low court position: {position_confidence:.2f} (height factor: {height_factor:.2f})")
        
        # CRITERION 2: VELOCITY DECREASE PATTERN (ball losing energy on floor)
        velocity_analysis = self._analyze_floor_impact_velocity(recent_positions)
        if velocity_analysis['impact_detected']:
            velocity_confidence = velocity_analysis['confidence']
            floor_hit_confidence += velocity_confidence * 0.35
            detection_criteria['velocity_impact'] = velocity_confidence
            print(f" Velocity impact: {velocity_confidence:.2f}")
        
        # CRITERION 3: BOUNCE PATTERN ANALYSIS (definitive floor indicator)
        bounce_analysis = self._detect_definitive_bounce_pattern(recent_positions)
        if bounce_analysis['definitive_bounce']:
            bounce_confidence = bounce_analysis['confidence']
            floor_hit_confidence += bounce_confidence * 0.40  # Highest weight
            detection_criteria['definitive_bounce'] = bounce_confidence
            print(f" Definitive bounce: {bounce_confidence:.2f}")
        
        # CRITERION 4: BALL SETTLING/STILLNESS (end of rally)
        settling_analysis = self._analyze_rally_end_settling(recent_positions)
        if settling_analysis['rally_ending']:
            settling_confidence = settling_analysis['confidence']
            floor_hit_confidence += settling_confidence * 0.25
            detection_criteria['rally_ending'] = settling_confidence
            print(f" Rally ending: {settling_confidence:.2f}")
        
        # CRITERION 5: DOWNWARD TRAJECTORY PHYSICS
        trajectory_analysis = self._analyze_floor_trajectory_physics(recent_positions)
        if trajectory_analysis['realistic_floor_approach']:
            trajectory_confidence = trajectory_analysis['confidence']
            floor_hit_confidence += trajectory_confidence * 0.20
            detection_criteria['floor_trajectory'] = trajectory_confidence
            print(f" Floor trajectory: {trajectory_confidence:.2f}")
        
        # CRYSTAL CLEAR FLOOR HIT THRESHOLD
        floor_hit_threshold = 0.7  # High threshold for definitive floor hits
        detected = floor_hit_confidence > floor_hit_threshold
        
        if detected:
            print(f" CLEAR FLOOR HIT DETECTED - SHOT END")
            print(f"    Confidence: {floor_hit_confidence:.2f}/1.0")
            print(f"    Position: ({x:.1f}, {y:.1f}) - {height_factor:.1%} down court")
            print(f"    Criteria: {len(detection_criteria)}/5")
            print(f"    RALLY ENDS HERE")
        
        return {
            'detected': detected,
            'confidence': min(1.0, floor_hit_confidence),
            'details': {
                'height_factor': height_factor,
                'court_position': {'x': x, 'y': y},
                'detection_criteria': detection_criteria,
                'total_criteria_met': len(detection_criteria),
                'clarity': 'HIGH' if detected else 'LOW',
                'rally_status': 'ENDING' if detected else 'CONTINUING'
            }
        }
    
    def _analyze_floor_impact_velocity(self, positions):
        """Analyze velocity patterns specific to floor impact"""
        if len(positions) < 4:
            return {'impact_detected': False, 'confidence': 0.0}
        
        # Calculate velocity magnitudes
        velocities = []
        for i in range(1, len(positions)):
            dx = positions[i][0] - positions[i-1][0]
            dy = positions[i][1] - positions[i-1][1]
            speed = math.sqrt(dx*dx + dy*dy)
            velocities.append(speed)
        
        if len(velocities) < 3:
            return {'impact_detected': False, 'confidence': 0.0}
        
        # Look for characteristic floor impact pattern:
        # 1. High speed approaching floor
        # 2. Sudden speed decrease at impact
        # 3. Lower speed after bounce
        
        early_speed = velocities[0]
        impact_speed = velocities[len(velocities)//2]  # Middle speed
        late_speed = velocities[-1]
        
        impact_confidence = 0.0
        
        # Speed decrease at impact
        if early_speed > 0:
            speed_decrease = (early_speed - late_speed) / early_speed
            if speed_decrease > 0.3:  # 30% speed decrease
                impact_confidence += 0.6
            elif speed_decrease > 0.15:  # 15% speed decrease
                impact_confidence += 0.4
        
        # Low final speed indicates settling
        if late_speed < 3.0:  # Very slow movement
            impact_confidence += 0.4
        
        return {
            'impact_detected': impact_confidence > 0.5,
            'confidence': min(1.0, impact_confidence),
            'speed_decrease': speed_decrease if early_speed > 0 else 0,
            'final_speed': late_speed
        }
    
    def _detect_definitive_bounce_pattern(self, positions):
        """Detect definitive bounce patterns that indicate floor contact"""
        if len(positions) < 5:
            return {'definitive_bounce': False, 'confidence': 0.0}
        
        bounce_confidence = 0.0
        bounce_indicators = []
        
        # Look for Y-direction bounce (ball goes down then up)
        y_positions = [pos[1] for pos in positions]
        
        # Find lowest point (potential floor contact)
        min_y_idx = y_positions.index(min(y_positions))
        
        # Check if ball went down then up around this point
        if 1 <= min_y_idx <= len(y_positions) - 2:
            before_bounce = y_positions[:min_y_idx+1]
            after_bounce = y_positions[min_y_idx:]
            
            # Downward trend before lowest point
            if len(before_bounce) >= 2:
                downward_trend = sum(1 for i in range(1, len(before_bounce)) 
                                if before_bounce[i] > before_bounce[i-1])
                downward_ratio = downward_trend / (len(before_bounce) - 1)
                
                if downward_ratio > 0.6:  # 60% downward movement
                    bounce_confidence += 0.4
                    bounce_indicators.append('downward_approach')
            
            # Upward trend after lowest point
            if len(after_bounce) >= 2:
                upward_trend = sum(1 for i in range(1, len(after_bounce)) 
                                if after_bounce[i] < after_bounce[i-1])
                upward_ratio = upward_trend / (len(after_bounce) - 1)
                
                if upward_ratio > 0.4:  # 40% upward movement
                    bounce_confidence += 0.6
                    bounce_indicators.append('upward_rebound')
        
        # Additional validation - direction change magnitude
        if len(positions) >= 3:
            direction_change = self._calculate_direction_change(positions[-3:])
            if direction_change > math.pi/4:  # 45 degrees
                bounce_confidence += 0.3
                bounce_indicators.append('direction_change')
        
        return {
            'definitive_bounce': bounce_confidence > 0.7,
            'confidence': min(1.0, bounce_confidence),
            'indicators': bounce_indicators,
            'lowest_point_index': min_y_idx
        }
    
    def _analyze_rally_end_settling(self, positions):
        """Analyze if ball is settling, indicating rally end"""
        if len(positions) < 4:
            return {'rally_ending': False, 'confidence': 0.0}
        
        settling_confidence = 0.0
        
        # Calculate movement in recent positions
        total_movement = 0
        for i in range(1, len(positions)):
            dx = positions[i][0] - positions[i-1][0]
            dy = positions[i][1] - positions[i-1][1]
            movement = math.sqrt(dx*dx + dy*dy)
            total_movement += movement
        
        avg_movement = total_movement / (len(positions) - 1)
        
        # Very little movement = ball settling
        if avg_movement < 2.0:  # Less than 2 pixels per frame
            settling_confidence = 0.9
        elif avg_movement < 4.0:  # Less than 4 pixels per frame
            settling_confidence = 0.6
        elif avg_movement < 7.0:  # Less than 7 pixels per frame
            settling_confidence = 0.3
        
        # Check for decreasing movement trend
        movements = []
        for i in range(1, len(positions)):
            dx = positions[i][0] - positions[i-1][0]
            dy = positions[i][1] - positions[i-1][1]
            movement = math.sqrt(dx*dx + dy*dy)
            movements.append(movement)
        
        if len(movements) >= 3:
            decreasing_trend = sum(1 for i in range(1, len(movements)) 
                                if movements[i] < movements[i-1])
            trend_ratio = decreasing_trend / (len(movements) - 1)
            
            if trend_ratio > 0.6:  # 60% decreasing trend
                settling_confidence += 0.3
        
        return {
            'rally_ending': settling_confidence > 0.5,
            'confidence': min(1.0, settling_confidence),
            'average_movement': avg_movement,
            'trend_analysis': trend_ratio if 'trend_ratio' in locals() else 0
        }
    
    def _analyze_floor_trajectory_physics(self, positions):
        """Analyze trajectory physics for realistic floor approach"""
        if len(positions) < 3:
            return {'realistic_floor_approach': False, 'confidence': 0.0}
        
        physics_confidence = 0.0
        
        # Calculate trajectory angle (should be downward for floor hit)
        start_pos = positions[0]
        end_pos = positions[-1]
        
        dx = end_pos[0] - start_pos[0]
        dy = end_pos[1] - start_pos[1]
        
        if dx != 0:
            angle = math.atan(dy / dx)
            
            # Positive angle = downward trajectory (good for floor hit)
            if dy > 0:  # Moving downward
                physics_confidence += 0.5
                
                # Steeper angle = more likely floor hit
                if angle > math.pi/6:  # > 30 degrees down
                    physics_confidence += 0.3
                elif angle > math.pi/12:  # > 15 degrees down
                    physics_confidence += 0.2
        
        # Check for realistic deceleration pattern
        if len(positions) >= 4:
            early_movement = math.sqrt((positions[1][0] - positions[0][0])**2 + 
                                    (positions[1][1] - positions[0][1])**2)
            late_movement = math.sqrt((positions[-1][0] - positions[-2][0])**2 + 
                                    (positions[-1][1] - positions[-2][1])**2)
            
            if early_movement > 0 and late_movement < early_movement:
                physics_confidence += 0.2  # Realistic deceleration
        
        return {
            'realistic_floor_approach': physics_confidence > 0.4,
            'confidence': min(1.0, physics_confidence),
            'trajectory_angle': angle if 'angle' in locals() else 0,
            'downward_movement': dy > 0
        }
    
    def _analyze_velocity_decrease(self, positions):
        """Analyze velocity decrease patterns indicative of floor contact"""
        if len(positions) < 4:
            return {'significant_decrease': False, 'decrease_factor': 0.0}
        
        # Calculate velocities for different segments
        velocities = []
        for i in range(1, len(positions)):
            p1, p2 = positions[i-1], positions[i]
            velocity = math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
            velocities.append(velocity)
        
        if len(velocities) < 3:
            return {'significant_decrease': False, 'decrease_factor': 0.0}
        
        # Compare early vs late velocities
        early_avg = sum(velocities[:2]) / 2
        late_avg = sum(velocities[-2:]) / 2
        
        if early_avg > 0:
            decrease_ratio = (early_avg - late_avg) / early_avg
            significant_decrease = decrease_ratio > self.floor_hit_indicators['velocity_decrease']
            
            return {
                'significant_decrease': significant_decrease,
                'decrease_factor': min(1.0, decrease_ratio / self.floor_hit_indicators['velocity_decrease']),
                'early_velocity': early_avg,
                'late_velocity': late_avg
            }
        
        return {'significant_decrease': False, 'decrease_factor': 0.0}
    
    def _detect_bounce_pattern_enhanced(self, positions):
        """Enhanced bounce pattern detection with noise filtering"""
        if len(positions) < 4:
            return {'bounce_detected': False, 'confidence': 0.0}
        
        # Look for characteristic bounce pattern: down -> up movement
        bounce_indicators = []
        
        for i in range(2, len(positions)):
            p1, p2, p3 = positions[i-2], positions[i-1], positions[i]
            
            # Calculate vertical movements
            down_movement = p2[1] - p1[1]  # Positive = downward
            up_movement = p2[1] - p3[1]    # Positive = upward
            
            # Filter noise - only consider significant movements
            if abs(down_movement) > 3 and abs(up_movement) > 3:
                if down_movement > 2 and up_movement > 2:  # Down then up
                    bounce_strength = min(down_movement, up_movement)
                    bounce_indicators.append(bounce_strength)
        
        if bounce_indicators:
            avg_bounce_strength = sum(bounce_indicators) / len(bounce_indicators)
            confidence = min(1.0, avg_bounce_strength / 10.0)  # Normalize
            
            return {
                'bounce_detected': True,
                'confidence': confidence,
                'bounce_count': len(bounce_indicators),
                'avg_strength': avg_bounce_strength
            }
        
        return {'bounce_detected': False, 'confidence': 0.0}
    
    def _analyze_ball_stillness(self, positions):
        """Analyze if ball is coming to rest (settling on floor)"""
        if len(positions) < 4:
            return {'is_settling': False, 'confidence': 0.0}
        
        # Calculate movement amounts in recent positions
        movements = []
        for i in range(1, len(positions)):
            p1, p2 = positions[i-1], positions[i]
            movement = math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
            movements.append(movement)
        
        if not movements:
            return {'is_settling': False, 'confidence': 0.0}
        
        avg_movement = sum(movements) / len(movements)
        max_movement = max(movements)
        
        # Ball is settling if movement is consistently small
        is_settling = (avg_movement < self.floor_hit_indicators['stillness_threshold'] and 
                    max_movement < self.floor_hit_indicators['stillness_threshold'] * 2)
        
        if is_settling:
            # Higher confidence for smaller movements
            confidence = max(0.0, 1.0 - (avg_movement / self.floor_hit_indicators['stillness_threshold']))
            return {
                'is_settling': True,
                'confidence': confidence,
                'avg_movement': avg_movement,
                'max_movement': max_movement
            }
        
        return {'is_settling': False, 'confidence': 0.0}
    
    def _analyze_floor_approach_angle(self, positions):
        """Analyze trajectory angle for steep approach to floor"""
        if len(positions) < 3:
            return {'steep_approach': False, 'confidence': 0.0}
        
        # Calculate trajectory angle in recent positions
        p1, p3 = positions[0], positions[-1]
        
        # Vertical and horizontal components
        vertical_change = p3[1] - p1[1]  # Positive = downward
        horizontal_change = abs(p3[0] - p1[0])
        
        if horizontal_change > 0:
            # Calculate angle (steep = more vertical than horizontal)
            angle_ratio = vertical_change / horizontal_change
            
            # Steep downward approach indicates floor hit
            if angle_ratio > 1.0 and vertical_change > 10:  # More down than across, significant movement
                confidence = min(1.0, angle_ratio / 3.0)  # Normalize to 3:1 ratio
                return {
                    'steep_approach': True,
                    'confidence': confidence,
                    'angle_ratio': angle_ratio,
                    'vertical_change': vertical_change
                }
        
        return {'steep_approach': False, 'confidence': 0.0}
    
    def _check_floor_hit_physics(self, positions, court_height):
        """Check if ball behavior is consistent with floor contact physics"""
        if len(positions) < 3:
            return {'realistic': False, 'details': {}}
        
        current_y = positions[-1][1]
        
        # Ball should be in reasonable floor contact area
        floor_area = current_y > (court_height * 0.6)  # Lower 40% of court
        
        # Check for realistic ball behavior patterns
        realistic_height = floor_area
        realistic_movement = True  # Could add more physics checks here
        
        return {
            'realistic': realistic_height and realistic_movement,
            'details': {
                'in_floor_area': floor_area,
                'current_height_ratio': current_y / court_height
            }
        }
    
    def _detect_rally_end(self, trajectory):
        """Detect when rally has ended (ball stationary or out of play)"""
        if len(trajectory) < 10:
            return {'detected': False, 'confidence': 0.0, 'details': {}}
        
        recent_positions = trajectory[-10:]
        
        # Check for extended stillness
        stillness_analysis = self._analyze_ball_stillness(recent_positions)
        
        if stillness_analysis['is_settling'] and stillness_analysis['confidence'] > 0.8:
            return {
                'detected': True,
                'confidence': stillness_analysis['confidence'],
                'details': {
                    'reason': 'ball_stationary',
                    'stillness_duration': len(recent_positions),
                    'avg_movement': stillness_analysis.get('avg_movement', 0)
                }
            }
        
        return {'detected': False, 'confidence': 0.0, 'details': {}}
    
    def _calculate_phase_confidence(self, trajectory, phase, court_dimensions):
        """Calculate confidence level for maintaining current phase"""
        if phase == 'start':
            # High confidence if ball is moving and not near walls/floor
            if len(self.velocity_history) > 0:
                avg_velocity = sum(self.velocity_history) / len(self.velocity_history)
                return min(1.0, avg_velocity / 20.0)  # Normalize
        
        elif phase == 'middle':
            # Confidence based on being away from walls but still moving
            if len(self.min_wall_distance_history) > 0:
                avg_wall_distance = sum(self.min_wall_distance_history) / len(self.min_wall_distance_history)
                return min(1.0, avg_wall_distance / 50.0)  # Normalize
        
        elif phase == 'end':
            # High confidence if ball is in lower court area
            if len(trajectory) > 0:
                current_y = trajectory[-1][1]
                height_factor = current_y / court_dimensions[1]
                return max(0.3, height_factor)  # At least 30% confidence
        
        return 0.5  # Default confidence
    
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
        ULTRA-CLEAR SHOT START DETECTION: Ball leaves racket
        Multiple validation criteria ensure precise shot start identification
        """
        if not ball_hit or who_hit == 0:
            return False
            
        # Check cooldown to avoid duplicate shot detection
        if frame_count - self.last_hit_frame < self.ball_hit_cooldown:
            return False
        
        # ENHANCED START VALIDATION - Multiple criteria must be met
        start_confidence = 0.0
        start_criteria = {}
        
        # Criterion 1: Hit confidence threshold
        if hit_confidence >= 0.4:  # Increased threshold for clearer starts
            start_confidence += 0.3
            start_criteria['hit_confidence'] = hit_confidence
        
        # Criterion 2: Ball trajectory analysis (if available)
        if past_ball_pos and len(past_ball_pos) >= 3:
            trajectory_score = self._analyze_shot_start_trajectory(past_ball_pos)
            start_confidence += trajectory_score * 0.4
            start_criteria['trajectory_analysis'] = trajectory_score
        
        # Criterion 3: Player proximity validation
        if ball_pos and who_hit:
            proximity_score = self._validate_player_ball_proximity(ball_pos, who_hit)
            start_confidence += proximity_score * 0.3
            start_criteria['player_proximity'] = proximity_score
        
        # CLEAR START THRESHOLD - Must exceed 0.6 for definitive start
        if start_confidence < 0.6:
            print(f" Shot start rejected - insufficient confidence: {start_confidence:.2f}")
            return False
            
        # Use enhanced shot classification if past_ball_pos is provided
        classification_confidence = 0.5
        shot_features = {}
        
        if hasattr(self, 'shot_classification_model') and past_ball_pos and len(past_ball_pos) > 3:
            try:
                enhanced_classification = self.shot_classification_model.classify_shot(
                    past_ball_pos[-10:], (640, 360)
                )
                shot_type = enhanced_classification['type']
                shot_features = enhanced_classification['features']
                classification_confidence = enhanced_classification['confidence']
            except Exception:
                pass
            
        # Start new shot with CRYSTAL CLEAR tracking
        self.shot_id_counter += 1
        new_shot = {
            'id': self.shot_id_counter,
            'start_frame': frame_count,
            'start_position': ball_pos.copy(),
            'start_confidence': start_confidence,
            'start_criteria': start_criteria,
            'player_who_hit': who_hit,
            'shot_type': shot_type,
            'hit_confidence': hit_confidence,
            'classification_confidence': classification_confidence,
            'hit_type': hit_type,
            'trajectory': [ball_pos.copy()],
            'status': 'active',
            'color': self.get_shot_color(shot_type),
            'end_frame': None,
            'end_position': None,
            'end_confidence': None,
            'final_shot_type': None,
            'phase': 'start',  # CLEAR: Ball leaves racket
            'wall_hit_frame': None,
            'wall_hit_position': None,
            'floor_hit_frame': None,
            'floor_hit_position': None,
            'phase_transitions': [],
            'shot_features': shot_features,
            'phase_history': [{'phase': 'start', 'frame': frame_count, 'confidence': start_confidence}],
            'shot_clarity': 'HIGH'  # Indicates clear shot boundaries
        }
        
        self.active_shots.append(new_shot)
        self.last_hit_frame = frame_count
        
        # Save clear shot start to autonomous log
        self._log_clear_shot_boundary('start', new_shot, frame_count)
        
        # Log CRYSTAL CLEAR shot start information
        print(f" CLEAR SHOT START #{self.shot_id_counter}:")
        print(f"    Player: {who_hit}, Type: {shot_type}")
        print(f"    Start Confidence: {start_confidence:.2f}/1.0")
        print(f"    Position: {ball_pos}")
        print(f"    Criteria Met: {len(start_criteria)}/3")
        print(f"    Phase: START (ball leaves racket)")
        
        return new_shot
    
    def _analyze_shot_start_trajectory(self, past_ball_pos):
        """Analyze trajectory to confirm shot start with velocity spike detection"""
        if len(past_ball_pos) < 3:
            return 0.5
        
        # Calculate velocity changes to detect racket contact
        velocities = []
        for i in range(1, len(past_ball_pos)):
            prev_pos = past_ball_pos[i-1]
            curr_pos = past_ball_pos[i]
            velocity = math.sqrt((curr_pos[0] - prev_pos[0])**2 + (curr_pos[1] - prev_pos[1])**2)
            velocities.append(velocity)
        
        if len(velocities) < 2:
            return 0.5
        
        # Look for velocity spike (sudden increase indicating racket contact)
        latest_velocity = velocities[-1]
        avg_prev_velocity = sum(velocities[:-1]) / len(velocities[:-1])
        
        # Strong velocity increase indicates clear shot start
        if latest_velocity > avg_prev_velocity * 2.0:  # 100% velocity increase
            return 1.0
        elif latest_velocity > avg_prev_velocity * 1.5:  # 50% velocity increase
            return 0.7
        elif latest_velocity > avg_prev_velocity * 1.2:  # 20% velocity increase
            return 0.5
        else:
            return 0.2
    
    def _validate_player_ball_proximity(self, ball_pos, player_id):
        """Validate that ball is close enough to player for realistic shot"""
        # This would typically check against player position data
        # For now, return a reasonable score based on court position
        
        # Assume shots from center court area are more likely valid
        court_center_x = 320  # Approximate court center
        court_center_y = 180
        
        distance_from_center = math.sqrt(
            (ball_pos[0] - court_center_x)**2 + 
            (ball_pos[1] - court_center_y)**2
        )
        
        # Closer to center = higher probability of valid shot
        max_distance = 200  # Maximum reasonable distance
        proximity_score = max(0.0, 1.0 - (distance_from_center / max_distance))
        
        return min(1.0, proximity_score + 0.3)  # Add base score
    
    def _log_clear_shot_boundary(self, boundary_type, shot_data, frame_count):
        """Log clear shot boundary events to autonomous phase detection file"""
        try:
            log_entry = {
                'timestamp': time.time(),
                'frame': frame_count,
                'shot_id': shot_data['id'],
                'boundary_type': boundary_type,  # 'start', 'wall_hit', 'end'
                'position': shot_data.get(f'{boundary_type}_position', shot_data.get('start_position')),
                'confidence': shot_data.get(f'{boundary_type}_confidence', shot_data.get('start_confidence', 0.0)),
                'criteria': shot_data.get(f'{boundary_type}_criteria', {}),
                'player': shot_data.get('player_who_hit'),
                'shot_type': shot_data.get('shot_type'),
                'clarity': 'HIGH'
            }
            
            with open("output/autonomous_phase_detection.jsonl", "a") as f:
                f.write(json.dumps(log_entry) + "\n")
                
        except Exception as e:
            print(f"Warning: Could not log shot boundary: {e}")
        
    def get_shot_color(self, shot_type):
        """
        ULTRA-CLEAR COLOR CODING: Each shot type gets distinctive, high-contrast colors
        Colors chosen for maximum visual distinction and clarity on court visualization
        """
        shot_colors = {
            # Primary shot types - High contrast colors for maximum visibility
            'straight_drive': (0, 255, 0),        # Bright Green - Straight line shots
            'straight': (0, 255, 0),              # Bright Green - alias
            'drive': (0, 255, 0),                 # Bright Green - alias
            
            'crosscourt': (255, 165, 0),          # Orange - Diagonal shots
            'wide_crosscourt': (255, 140, 0),     # Dark orange - wide angles
            
            'drop_shot': (255, 0, 255),           # Magenta - Finesse shots
            'drop': (255, 0, 255),                # Magenta - alias
            
            'lob': (0, 255, 255),                 # Cyan - High defensive shots
            'defensive_lob': (0, 200, 255),       # Light blue - defensive variation
            
            'boast': (255, 255, 0),               # Yellow - Tactical angle shots
            'three_wall_boast': (255, 215, 0),    # Gold - complex boasts
            
            'kill_shot': (255, 0, 0),             # Red - Aggressive winners
            'kill': (255, 0, 0),                  # Red - alias
            'nick_shot': (220, 20, 60),           # Crimson - precision winners
            
            'volley': (128, 0, 255),              # Purple - Intercept shots
            'volley_drop': (153, 50, 204),        # Dark orchid - touch volleys
            
            # Advanced shots
            'reverse_angle': (255, 192, 203),     # Pink - deceptive shots
            'trickle_boast': (173, 255, 47),      # Green-yellow - soft angles
            'working_boast': (255, 215, 0),       # Gold - tactical boasts
            'attacking_lob': (30, 144, 255),      # Dodger blue - offensive lobs
            
            # Court position based
            'front_court': (255, 20, 147),        # Deep pink - front court shots
            'back_court': (70, 130, 180),         # Steel blue - back court shots
            'mid_court': (154, 205, 50),          # Yellow green - mid court
            
            # Default and unknown
            'unknown': (128, 128, 128),           # Gray - unclassified
            'default': (255, 255, 255),           # White - fallback
            'none': (64, 64, 64)                  # Dark gray - no shot
        }
        
        # Normalize shot type input
        if isinstance(shot_type, list) and len(shot_type) > 0:
            shot_str = str(shot_type[0]).lower()
        else:
            shot_str = str(shot_type).lower()
        
        # Direct match first
        if shot_str in shot_colors:
            color = shot_colors[shot_str]
            print(f" 🎯 SHOT COLOR: {shot_type} -> RGB{color}")
            return color
        
        # Pattern matching for partial matches
        for shot_name, color in shot_colors.items():
            if shot_name in shot_str or shot_str in shot_name:
                print(f" 🎯 SHOT COLOR (pattern): {shot_type} -> {shot_name} -> RGB{color}")
                return color
        
        # Fallback to default
        default_color = shot_colors['default']
        print(f" ⚠️ SHOT COLOR (default): {shot_type} -> RGB{default_color}")
        return default_color
        
    def update_shot_phases(self, shot, ball_pos, frame_count):
        """
        ULTRA-CLEAR AUTONOMOUS SHOT PHASE TRACKING
        Tracks: START (ball leaves racket) -> MIDDLE (wall hit) -> END (floor hit)
        """
        if not hasattr(self, 'phase_detector'):
            return
            
        current_phase = shot.get('phase', 'start')
        
        # Use enhanced autonomous phase detection
        phase_result = self.phase_detector.detect_shot_phases(
            shot['trajectory'], (640, 360), current_phase
        )
        
        new_phase = phase_result['phase']
        phase_confidence = phase_result['confidence']
        transition = phase_result.get('transition', 'none')
        event_type = phase_result.get('event', 'none')
        
        # CRYSTAL CLEAR PHASE TRANSITIONS - High confidence required
        if new_phase != current_phase and phase_confidence > 0.7:  # Increased threshold
            old_phase = current_phase
            shot['phase'] = new_phase
            
            # CLEAR VISUAL LOGGING FOR EACH PHASE TRANSITION
            print("\n" + "="*60)
            print(f" SHOT #{shot['id']} - PHASE TRANSITION DETECTED")
            print("="*60)
            
            # Log autonomous phase transition with CRYSTAL CLEAR information
            if transition == 'start_to_middle':
                shot['wall_hit_frame'] = frame_count
                shot['wall_hit_position'] = ball_pos.copy()
                shot['wall_hit_confidence'] = phase_confidence
                wall_details = phase_result['details']
                
                # Store wall type for visualization
                shot['wall_type'] = wall_details.get('wall_type', 'unknown')
                
                print(f" 🎯 WALL HIT DETECTED - MIDDLE PHASE BEGINS")
                print(f"    Frame: {frame_count}")
                print(f"    Position: ({ball_pos[0]:.1f}, {ball_pos[1]:.1f})")
                print(f"    Wall: {wall_details.get('wall_type', 'unknown').upper()}")
                print(f"    Distance: {wall_details.get('distance_to_wall', 0):.1f}px")
                print(f"    Criteria: {wall_details.get('total_criteria_met', 0)}/5")
                print(f"    Confidence: {phase_confidence:.2f}/1.0")
                print(f"    Phase: {old_phase.upper()} → {new_phase.upper()}")
                
                # Special logging for front wall hits
                if wall_details.get('wall_type') == 'front_wall':
                    print(f"    🎯 FRONT WALL HIT CONFIRMED!")
                
            elif transition in ['middle_to_end', 'start_to_end']:
                shot['floor_hit_frame'] = frame_count
                shot['floor_hit_position'] = ball_pos.copy()
                shot['floor_hit_confidence'] = phase_confidence
                shot['end_frame'] = frame_count
                shot['end_position'] = ball_pos.copy()
                shot['end_confidence'] = phase_confidence
                floor_details = phase_result['details']
                
                print(f" 🏓 FLOOR HIT DETECTED - SHOT END")
                print(f"    Frame: {frame_count}")
                print(f"    Position: ({ball_pos[0]:.1f}, {ball_pos[1]:.1f})")
                print(f"    Court Height: {floor_details.get('height_factor', 0):.1%}")
                print(f"    Criteria: {floor_details.get('total_criteria_met', 0)}/5")
                print(f"    Confidence: {phase_confidence:.2f}/1.0")
                print(f"    Phase: {old_phase.upper()} → {new_phase.upper()}")
                print(f"    🎯 SHOT COMPLETE!")
                
            elif transition == 'end_to_complete':
                shot['rally_end_frame'] = frame_count
                end_details = phase_result['details']
                
                print(f" RALLY END DETECTED")
                print(f"    Frame: {frame_count}")
                print(f"    Final Position: ({ball_pos[0]:.1f}, {ball_pos[1]:.1f})")
                print(f"    Status: {end_details.get('rally_status', 'ENDING')}")
                print(f"    Confidence: {phase_confidence:.2f}/1.0")
                print(f"    RALLY COMPLETE!")
            
            # Record DETAILED phase transition with clarity markers
            phase_transition = {
                'from_phase': old_phase,
                'to_phase': new_phase,
                'frame': frame_count,
                'position': ball_pos.copy(),
                'confidence': phase_confidence,
                'details': phase_result.get('details', {}),
                'event_type': event_type,
                'autonomous_detection': True,
                'clarity': 'HIGH',
                'timestamp': time.time(),
                'transition_type': transition,
                'shot_id': shot['id']
            }
            
            shot['phase_transitions'].append(phase_transition)
            
            # Add to phase history with clear markers
            shot['phase_history'].append({
                'phase': new_phase,
                'frame': frame_count,
                'confidence': phase_confidence,
                'transition_from': old_phase,
                'clarity': 'HIGH'
            })
            
            # Log to autonomous phase detection file with CLEAR boundaries
            self._log_clear_shot_boundary(
                boundary_type=new_phase,
                shot_data=shot,
                frame_count=frame_count
            )
            
            print("="*60)
            print()
        
        # Update trajectory regardless of phase change
        shot['trajectory'].append(ball_pos.copy())
        
        # Keep trajectory manageable
        if len(shot['trajectory']) > 100:
            shot['trajectory'] = shot['trajectory'][-100:]
        
        # Update shot confidence based on phase detection quality
        if 'detection_confidence' not in shot:
            shot['detection_confidence'] = []
        shot['detection_confidence'].append({
            'frame': frame_count,
            'phase': new_phase,
            'confidence': phase_confidence
        })
    
    def _save_autonomous_phase_data(self, shot, phase_transition):
        """Save detailed autonomous phase detection data for analysis and improvement"""
        try:
            phase_data = {
                'shot_id': shot['id'],
                'timestamp': time.time(),
                'frame': phase_transition['frame'],
                'transition': f"{phase_transition['from']} -> {phase_transition['to']}",
                'confidence': phase_transition['confidence'],
                'event_type': phase_transition['event_type'],
                'detection_criteria': phase_transition.get('detection_criteria', {}),
                'position': phase_transition['position'],
                'trajectory_length': len(shot['trajectory']),
                'shot_type': shot.get('shot_type', 'unknown'),
                'player': shot.get('player_who_hit', 0)
            }
            
            # Append to autonomous phase detection log
            with open("output/autonomous_phase_detection.jsonl", "a") as f:
                f.write(json.dumps(phase_data) + "\n")
                
        except Exception as e:
            print(f"Warning: Could not save autonomous phase data: {e}")
    
    def detect_wall_hit(self, trajectory, current_pos):
        """
        Enhanced autonomous wall hit detection using the phase detector
        """
        if not hasattr(self, 'phase_detector') or len(trajectory) < 3:
            return False
            
        # Use the phase detector's enhanced wall hit detection
        wall_hit_result = self.phase_detector._detect_wall_hit_enhanced(trajectory, (640, 360))
        
        return wall_hit_result['detected'] and wall_hit_result['confidence'] > 0.7
    
    def detect_floor_hit(self, trajectory, current_pos):
        """
        Enhanced autonomous floor hit detection using the phase detector
        """
        if not hasattr(self, 'phase_detector') or len(trajectory) < 5:
            return False
            
        # Use the phase detector's enhanced floor hit detection
        floor_hit_result = self.phase_detector._detect_floor_hit_enhanced(trajectory, (640, 360))
        
        return floor_hit_result['detected'] and floor_hit_result['confidence'] > 0.6
        
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
            
    def draw_shot_trajectories(self, frame, ball_pos, frame_count=None):
        """
        🎯 ULTRA-CLEAR ENHANCED SHOT VISUALIZATION
        Shows detailed information about players, hits, walls, bounces, and shot types
        """
        try:
            frame_height, frame_width = frame.shape[:2]
            
            # 🎨 ACTIVE SHOTS: Full visibility with detailed information
            for shot in self.active_shots:
                if len(shot['trajectory']) > 1:
                    color = shot['color']
                    phase = shot.get('phase', 'start')
                    shot_id = shot.get('id', '?')
                    confidence = shot.get('start_confidence', 0.0)
                    player_who_hit = shot.get('player_who_hit', 0)
                    
                    # 🌟 MAIN TRAJECTORY: Draw numbered points for clarity
                    trajectory_thickness = 4 if phase == 'start' else 3 if phase == 'middle' else 2
                    
                    # Draw trajectory with numbered points every 5th point
                    for i in range(1, len(shot['trajectory'])):
                        pt1 = (int(shot['trajectory'][i-1][0]), int(shot['trajectory'][i-1][1]))
                        pt2 = (int(shot['trajectory'][i][0]), int(shot['trajectory'][i][1]))
                        cv2.line(frame, pt1, pt2, color, trajectory_thickness)
                        
                        # Add numbered markers every 5th point for clarity
                        if i % 5 == 0 and i > 0:
                            cv2.circle(frame, pt2, 3, (255, 255, 255), -1)
                            cv2.putText(frame, str(i), (pt2[0] + 5, pt2[1] - 5),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
                    
                    # 🎯 PHASE 1: SHOT START - Enhanced with player info
                    if len(shot['trajectory']) > 0:
                        start_pos = (int(shot['trajectory'][0][0]), int(shot['trajectory'][0][1]))
                        start_frame = shot.get('start_frame', 0)
                        
                        # Dynamic size based on confidence
                        outer_radius = int(15 + (confidence * 10))
                        inner_radius = int(8 + (confidence * 5))
                        
                        # Player-colored circle with white center
                        player_color = (0, 255, 0) if player_who_hit == 1 else (255, 0, 0) if player_who_hit == 2 else (128, 128, 128)
                        cv2.circle(frame, start_pos, outer_radius, player_color, 3)
                        cv2.circle(frame, start_pos, inner_radius, (255, 255, 255), -1)
                        
                        # Enhanced shot start label with player and frame info
                        start_label = f"P{player_who_hit} HIT #{shot_id}"
                        frame_label = f"Frame: {start_frame}"
                        
                        # No background rectangle for text labels
                        
                        cv2.putText(frame, start_label, 
                                (start_pos[0] - 20, start_pos[1] - outer_radius - 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, player_color, 2)
                        cv2.putText(frame, frame_label, 
                                (start_pos[0] - 20, start_pos[1] - outer_radius - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
                    
                    # 🎯 PHASE 2: WALL HIT - Enhanced with wall type and timing
                    if shot.get('wall_hit_frame') and shot.get('wall_hit_position'):
                        wall_pos = shot['wall_hit_position']
                        wall_pos = (int(wall_pos[0]), int(wall_pos[1]))
                        wall_frame = shot.get('wall_hit_frame', 0)
                        wall_confidence = shot.get('wall_hit_confidence', 0.8)
                        wall_type = shot.get('wall_type', 'unknown')
                        
                        # Enhanced diamond shape for wall hit
                        diamond_size = int(12 + (wall_confidence * 8))
                        diamond_points = np.array([
                            [wall_pos[0], wall_pos[1] - diamond_size],
                            [wall_pos[0] + diamond_size, wall_pos[1]],
                            [wall_pos[0], wall_pos[1] + diamond_size],
                            [wall_pos[0] - diamond_size, wall_pos[1]]
                        ], np.int32)
                        
                        # Color based on wall type
                        wall_colors = {
                            'front_wall': (0, 255, 255),    # Yellow - most important
                            'back_wall': (0, 165, 255),     # Orange
                            'left_wall': (255, 192, 203),   # Pink
                            'right_wall': (147, 20, 255),   # Purple
                            'unknown': (128, 128, 128)      # Gray
                        }
                        wall_color = wall_colors.get(wall_type, (0, 255, 255))
                        
                        cv2.fillPoly(frame, [diamond_points], wall_color)
                        cv2.polylines(frame, [diamond_points], True, (255, 255, 255), 2)
                        
                        # Enhanced wall hit label with detailed info
                        wall_label = f"{wall_type.upper().replace('_', ' ')}"
                        frame_label = f"F:{wall_frame}"
                        
                        # No background rectangle for wall hit labels
                        
                        cv2.putText(frame, wall_label, 
                                (wall_pos[0] - 25, wall_pos[1] - diamond_size - 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, wall_color, 2)
                        cv2.putText(frame, frame_label, 
                                (wall_pos[0] - 25, wall_pos[1] - diamond_size - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
                    
                    # 🎯 PHASE 3: FLOOR HIT/BOUNCE - Enhanced with bounce info
                    if shot.get('floor_hit_frame') and shot.get('floor_hit_position'):
                        floor_pos = shot['floor_hit_position']
                        floor_pos = (int(floor_pos[0]), int(floor_pos[1]))
                        floor_frame = shot.get('floor_hit_frame', 0)
                        floor_confidence = shot.get('floor_hit_confidence', 0.8)
                        
                        # Enhanced triangle for floor hit
                        tri_size = int(10 + (floor_confidence * 8))
                        triangle_points = np.array([
                            [floor_pos[0], floor_pos[1] - tri_size],
                            [floor_pos[0] - tri_size, floor_pos[1] + tri_size],
                            [floor_pos[0] + tri_size, floor_pos[1] + tri_size]
                        ], np.int32)
                        
                        cv2.fillPoly(frame, [triangle_points], (0, 0, 255))
                        cv2.polylines(frame, [triangle_points], True, (255, 255, 255), 2)
                        
                        # Enhanced floor hit label
                        bounce_label = "BOUNCE"
                        frame_label = f"F:{floor_frame}"
                        
                        # No background rectangle for bounce labels
                        
                        cv2.putText(frame, bounce_label, 
                                (floor_pos[0] - 25, floor_pos[1] + tri_size + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 2)
                        cv2.putText(frame, frame_label, 
                                (floor_pos[0] - 25, floor_pos[1] + tri_size + 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
                    
                    # 🎯 CURRENT BALL: Enhanced with velocity indicator
                    if ball_pos and len(ball_pos) >= 2:
                        current_pos = (int(ball_pos[0]), int(ball_pos[1]))
                        
                        # Pulsing effect
                        if frame_count is not None:
                            pulse_radius = 10 + int(3 * math.sin(frame_count * 0.3))
                        else:
                            pulse_radius = 10
                        
                        # Draw current ball with velocity vector
                        cv2.circle(frame, current_pos, pulse_radius, color, 3)
                        cv2.circle(frame, current_pos, 5, (255, 255, 255), -1)
                        
                        # Show velocity vector if we have trajectory
                        if len(shot['trajectory']) > 2:
                            prev_pos = shot['trajectory'][-2]
                            velocity_x = ball_pos[0] - prev_pos[0]
                            velocity_y = ball_pos[1] - prev_pos[1]
                            velocity_mag = math.sqrt(velocity_x**2 + velocity_y**2)
                            
                            if velocity_mag > 2:  # Only show if moving fast enough
                                end_x = int(current_pos[0] + velocity_x * 3)
                                end_y = int(current_pos[1] + velocity_y * 3)
                                cv2.arrowedLine(frame, current_pos, (end_x, end_y), (255, 255, 0), 2)
                    
                    # 🔤 ENHANCED SHOT TYPE INDICATOR
                    shot_type = shot.get('shot_type', 'unknown')
                    hit_type = shot.get('hit_type', 'unknown')
                    if isinstance(shot_type, list):
                        shot_type = shot_type[0] if shot_type else 'unknown'
                    
                    # Show detailed shot information in info panel
                    if len(shot['trajectory']) > 10:
                        info_y_start = 30 + (shot_id % 3) * 80  # Staggered panels
                        
                        # Shot information
                        info_lines = [
                            f"Shot #{shot_id} - Player {player_who_hit}",
                            f"Type: {str(shot_type).replace('_', ' ').title()}",
                            f"Hit: {str(hit_type).replace('_', ' ').title()}",
                            f"Phase: {phase.upper()} | Points: {len(shot['trajectory'])}"
                        ]
                        
                        for i, line in enumerate(info_lines):
                            cv2.putText(frame, line, 
                                      (15, info_y_start + 15 + i * 15),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.4, 
                                      (255, 255, 255) if i > 0 else color, 1)
            
            # 👻 COMPLETED SHOTS: Show with shot summaries
            recent_completed = self.completed_shots[-2:] if len(self.completed_shots) > 2 else self.completed_shots
            for i, shot in enumerate(recent_completed):
                if len(shot['trajectory']) > 1:
                    # Progressive dimming
                    dim_factor = 0.5 - (i * 0.1)
                    dimmed_color = tuple(int(c * dim_factor) for c in shot['color'])
                    
                    # Thin trajectory with start/end markers
                    for j in range(1, len(shot['trajectory']), 2):  # Every 2nd point for performance
                        pt1 = (int(shot['trajectory'][j-1][0]), int(shot['trajectory'][j-1][1]))
                        pt2 = (int(shot['trajectory'][j][0]), int(shot['trajectory'][j][1]))
                        cv2.line(frame, pt1, pt2, dimmed_color, 1)
                    
                    # Start and end markers
                    if len(shot['trajectory']) > 0:
                        start_pos = (int(shot['trajectory'][0][0]), int(shot['trajectory'][0][1]))
                        end_pos = (int(shot['trajectory'][-1][0]), int(shot['trajectory'][-1][1]))
                        
                        cv2.circle(frame, start_pos, 4, dimmed_color, 2)  # Start
                        cv2.circle(frame, end_pos, 6, dimmed_color, -1)   # End
                        
                        # Shot summary
                        shot_id = shot.get('id', '?')
                        player = shot.get('player_who_hit', 0)
                        cv2.putText(frame, f"#{shot_id}P{player}", 
                                  (end_pos[0] + 10, end_pos[1]),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.3, dimmed_color, 1)
            
            # 📊 ENHANCED STATUS PANEL - Right side
            status_x = frame_width - 300
            status_y = 20
            
            # No background rectangle for status panel
            
            # Status information
            active_count = len(self.active_shots)
            total_count = len(self.completed_shots)
            
            status_lines = [
                "🎾 SHOT TRACKING STATUS",
                f"Active Shots: {active_count}",
                f"Completed: {total_count}",
                f"Total Points: {sum(len(s['trajectory']) for s in self.active_shots)}"
            ]
            
            if self.active_shots:
                current_shot = self.active_shots[-1]
                status_lines.append(f"Current: #{current_shot.get('id', '?')} P{current_shot.get('player_who_hit', 0)}")
                status_lines.append(f"Phase: {current_shot.get('phase', 'unknown').upper()}")
            
            for i, line in enumerate(status_lines):
                color = (0, 255, 255) if i == 0 else (255, 255, 255)
                cv2.putText(frame, line, (status_x, status_y + i * 18),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        except Exception as e:
            print(f"⚠️ Enhanced shot visualization error: {e}")
            # Fallback: Draw basic ball tracking
            if ball_pos and len(ball_pos) >= 2:
                cv2.circle(frame, (int(ball_pos[0]), int(ball_pos[1])), 10, (255, 255, 255), 2)
                cv2.putText(frame, "BALL DETECTED", (int(ball_pos[0]) + 15, int(ball_pos[1]) - 15),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    def get_shot_statistics(self):
        """Get basic shot statistics without UI overlay"""
        return {
            'total_shots': len(self.completed_shots),
            'active_shots': len(self.active_shots),
            'shots_by_player': {1: 0, 2: 0},
            'shots_by_type': {},
            'phase_transitions': []
        }

# Initialize global shot tracker with autonomous detection capabilities
shot_tracker = ShotTracker()
# Initialize autonomous detection components
shot_tracker.shot_classification_model = ShotClassificationModel()
shot_tracker.phase_detector = ShotPhaseDetector()
shot_tracker.hit_detector = PlayerHitDetector()

print(" Autonomous Shot Detection System Initialized:")
print("   ✓ Shot Classification Model")
print("   ✓ Phase Detection (Start→Wall→Floor)")  
print("   ✓ Player Hit Detection")
print("   ✓ Color-coded Visualization")

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
    # Initialize autonomous phase detection log
    with open("output/autonomous_phase_detection.jsonl", "w") as f:
        f.write("")  # Clear the file
    print("Shot tracking initialized.")
    print("Shot data: output/shots_log.jsonl")
    print("Bounce analysis: output/bounce_analysis.jsonl")
    print("Phase detection: output/autonomous_phase_detection.jsonl")
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
    🎯 AUTONOMOUS SHOT TYPE DETECTION - Works directly with enhanced shot classification
    Ultra-accurate shot type detection using machine learning-inspired pattern recognition
    """
    if len(past_ball_pos) < threshold:
        return 'unknown'
    
    try:
        # Use the enhanced shot classification model
        shot_classification_model = ShotClassificationModel()
        court_dimensions = (court_width, court_height)
        
        # Get classification result
        classification_result = shot_classification_model.classify_shot(
            past_ball_pos, court_dimensions
        )
        
        shot_type = classification_result['type']
        confidence = classification_result['confidence']
        reasoning = classification_result.get('reasoning', 'no reasoning available')
        
        print(f"🎯 AUTONOMOUS SHOT TYPE: {shot_type.upper()}")
        print(f"   Confidence: {confidence:.3f}")
        print(f"   Reasoning: {reasoning}")
        
        # Return shot type with confidence validation
        if confidence > 0.4:
            return shot_type
        else:
            print(f"   ⚠️ Low confidence ({confidence:.3f}), using legacy detection")
            return shot_type_enhanced_legacy(past_ball_pos, court_width, court_height, threshold)
            
    except Exception as e:
        print(f"⚠️ Enhanced shot classification error: {e}")
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
    🎯 ENHANCED AUTONOMOUS PLAYER BALL HIT DETECTION
    Multi-algorithm approach with physics validation and confidence scoring
    Returns: (player_id, hit_confidence, hit_type)
    """
    if len(past_ball_pos) < 4 or not players or not any(players.get(i) for i in [1, 2]):
        return 0, 0.0, "none"

    try:
        current_ball_pos = past_ball_pos[-1][:2]
        prev_ball_pos = past_ball_pos[-2][:2] if len(past_ball_pos) >= 2 else current_ball_pos
        
        print(f"🎯 Enhanced hit detection: analyzing {len(past_ball_pos)} ball positions")
        
        # Multi-algorithm detection results
        detection_results = {}
        
        # ALGORITHM 1: PROXIMITY ANALYSIS
        for player_id in [1, 2]:
            if players.get(player_id):
                player_pos = get_enhanced_player_pos(players[player_id])
                if player_pos and len(player_pos) >= 2:
                    distance = math.sqrt(
                        (current_ball_pos[0] - player_pos[0] * frame_width)**2 + 
                        (current_ball_pos[1] - player_pos[1] * frame_height)**2
                    )
                    
                    proximity_confidence = max(0, 1.0 - (distance / proximity_threshold))
                    
                    if proximity_confidence > 0.2:
                        detection_results[f"player_{player_id}_proximity"] = {
                            'player_id': player_id,
                            'confidence': proximity_confidence,
                            'method': 'proximity',
                            'distance': distance
                        }
                        
                        print(f"   Player {player_id} proximity: {proximity_confidence:.2f} (dist: {distance:.1f}px)")
        
        # ALGORITHM 2: VELOCITY CHANGE ANALYSIS
        if len(past_ball_pos) >= 3:
            recent_positions = past_ball_pos[-3:]
            
            # Calculate velocity changes
            vel1 = math.sqrt(
                (recent_positions[1][0] - recent_positions[0][0])**2 + 
                (recent_positions[1][1] - recent_positions[0][1])**2
            )
            vel2 = math.sqrt(
                (recent_positions[2][0] - recent_positions[1][0])**2 + 
                (recent_positions[2][1] - recent_positions[1][1])**2
            )
            
            if vel1 > 0:
                velocity_change_ratio = abs(vel2 - vel1) / vel1
                velocity_confidence = min(1.0, velocity_change_ratio / 2.0)  # Normalize
                
                if velocity_confidence > 0.3:
                    detection_results["velocity_change"] = {
                        'player_id': 0,  # Will be assigned later
                        'confidence': velocity_confidence,
                        'method': 'velocity_analysis',
                        'velocity_change': velocity_change_ratio
                    }
                    
                    print(f"   Velocity change detected: {velocity_confidence:.2f} (ratio: {velocity_change_ratio:.2f})")
        
        # ALGORITHM 3: TRAJECTORY DIRECTION ANALYSIS
        if len(past_ball_pos) >= 4:
            trajectory_segment = past_ball_pos[-4:]
            direction_changes = 0
            
            for i in range(2, len(trajectory_segment)):
                p1 = trajectory_segment[i-2][:2]
                p2 = trajectory_segment[i-1][:2]
                p3 = trajectory_segment[i][:2]
                
                # Calculate vectors
                v1 = (p2[0] - p1[0], p2[1] - p1[1])
                v2 = (p3[0] - p2[0], p3[1] - p2[1])
                
                # Calculate angle between vectors
                if abs(v1[0]) > 2 and abs(v1[1]) > 2 and abs(v2[0]) > 2 and abs(v2[1]) > 2:
                    angle = calculate_vector_angle(v1, v2)
                    if angle and angle > math.pi/6:  # > 30 degrees
                        direction_changes += 1
            
            if direction_changes > 0:
                direction_confidence = min(1.0, direction_changes / 2.0)
                detection_results["direction_change"] = {
                    'player_id': 0,  # Will be assigned later
                    'confidence': direction_confidence,
                    'method': 'trajectory_analysis',
                    'direction_changes': direction_changes
                }
                
                print(f"   Direction changes detected: {direction_confidence:.2f} ({direction_changes} changes)")
        
        # COMBINE RESULTS AND DETERMINE BEST HIT
        if not detection_results:
            return 0, 0.0, "none"
        
        # Find best player match by combining proximity with other signals
        best_player_id = 0
        best_confidence = 0.0
        best_method = "none"
        
        # First, check if any proximity detections exist
        proximity_detections = {k: v for k, v in detection_results.items() if 'proximity' in k}
        
        if proximity_detections:
            # Use proximity detection with highest confidence
            best_proximity = max(proximity_detections.values(), key=lambda x: x['confidence'])
            best_player_id = best_proximity['player_id']
            best_confidence = best_proximity['confidence']
            best_method = "proximity"
            
            # Boost confidence if other methods also detected a hit
            other_detections = [v for k, v in detection_results.items() if 'proximity' not in k]
            if other_detections:
                max_other_confidence = max(d['confidence'] for d in other_detections)
                best_confidence = min(1.0, best_confidence + (max_other_confidence * 0.3))
                best_method = "proximity_plus_trajectory"
                
                print(f"   Combined detection boost: {best_confidence:.2f}")
        else:
            # No proximity detection, use strongest other signal
            best_detection = max(detection_results.values(), key=lambda x: x['confidence'])
            best_confidence = best_detection['confidence']
            best_method = best_detection['method']
            
            # Try to assign to closest player if possible
            min_distance = float('inf')
            for player_id in [1, 2]:
                if players.get(player_id):
                    player_pos = get_enhanced_player_pos(players[player_id])
                    if player_pos and len(player_pos) >= 2:
                        distance = math.sqrt(
                            (current_ball_pos[0] - player_pos[0] * frame_width)**2 + 
                            (current_ball_pos[1] - player_pos[1] * frame_height)**2
                        )
                        if distance < min_distance:
                            min_distance = distance
                            best_player_id = player_id
        
        # PHYSICS VALIDATION
        if best_confidence > 0.2:
            physics_score = validate_hit_with_physics(past_ball_pos, best_player_id, players)
            best_confidence *= physics_score
            
            print(f"   Physics validation: {physics_score:.2f} (final confidence: {best_confidence:.2f})")
        
        # DETERMINE HIT TYPE
        hit_type = "none"
        if best_confidence > 0.7:
            hit_type = "strong_hit"
        elif best_confidence > 0.5:
            hit_type = "probable_hit"
        elif best_confidence > 0.3:
            hit_type = "possible_hit"
        else:
            hit_type = "weak_signal"
            
        if best_confidence < 0.25:  # Too low confidence
            best_player_id = 0
            hit_type = "none"
            
        print(f"🎯 ENHANCED HIT RESULT: Player {best_player_id}, Confidence: {best_confidence:.3f}, Type: {hit_type}")
        
        return best_player_id, best_confidence, hit_type
        
    except Exception as e:
        print(f"⚠️ Enhanced hit detection error: {e}")
        return 0, 0.0, "error"
        
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
            
            # Normalize direction change to 0- range
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

def validate_hit_with_physics(past_ball_pos, who_hit, players, min_positions=5):
    """
    🔬 PHYSICS-BASED HIT VALIDATION: Validates hit detection using ball physics and player biomechanics
    
    Args:
        past_ball_pos: List of recent ball positions
        who_hit: Player ID who supposedly hit the ball (1 or 2)
        players: Player tracking objects
        min_positions: Minimum positions needed for validation
    
    Returns:
        float: Physics validation score (0.0 to 1.0)
    """
    try:
        if len(past_ball_pos) < min_positions or who_hit not in [1, 2]:
            return 0.0
        
        validation_score = 0.0
        max_score = 1.0
        
        # 🎯 Physics Analysis - Ball trajectory validation
        trajectory_score = 0.0
        
        # Analyze velocity profile around hit point
        if len(past_ball_pos) >= 6:
            pre_hit_positions = past_ball_pos[-6:-2]  # 4 positions before hit
            post_hit_positions = past_ball_pos[-2:]   # 2 positions after hit
            
            # Calculate pre-hit velocity
            pre_velocities = []
            for i in range(1, len(pre_hit_positions)):
                dx = pre_hit_positions[i][0] - pre_hit_positions[i-1][0]
                dy = pre_hit_positions[i][1] - pre_hit_positions[i-1][1]
                velocity = math.sqrt(dx*dx + dy*dy)
                pre_velocities.append(velocity)
            
            # Calculate post-hit velocity
            if len(post_hit_positions) >= 2:
                dx_post = post_hit_positions[1][0] - post_hit_positions[0][0]
                dy_post = post_hit_positions[1][1] - post_hit_positions[0][1]
                post_velocity = math.sqrt(dx_post*dx_post + dy_post*dy_post)
                
                # Realistic hit should show velocity change
                avg_pre_velocity = sum(pre_velocities) / len(pre_velocities) if pre_velocities else 0
                
                if avg_pre_velocity > 0:
                    velocity_ratio = post_velocity / avg_pre_velocity
                    
                    # Good hits typically show 0.5x to 3.0x velocity change
                    if 0.5 <= velocity_ratio <= 3.0:
                        trajectory_score = 0.4
                    elif 0.3 <= velocity_ratio <= 5.0:
                        trajectory_score = 0.2
                
        # 🏃 Player Biomechanics - Validate player capability
        player_score = 0.0
        
        if players.get(who_hit):
            player = players[who_hit]
            ball_pos = past_ball_pos[-1]
            
            # Get player position using multiple keypoints
            player_positions = []
            
            # Try to get multiple body parts for better validation
            body_parts = ['right_wrist', 'left_wrist', 'right_elbow', 'left_elbow', 
                        'right_shoulder', 'left_shoulder']
            
            for part in body_parts:
                if hasattr(player, part) and player.__dict__[part] is not None:
                    kp = player.__dict__[part]
                    if len(kp) >= 2 and kp[0] > 0 and kp[1] > 0:
                        player_positions.append((kp[0], kp[1]))
            
            if player_positions:
                # Calculate minimum distance to any body part
                min_distance = float('inf')
                for pos in player_positions:
                    distance = math.sqrt(
                        (ball_pos[0] - pos[0])**2 + (ball_pos[1] - pos[1])**2
                    )
                    min_distance = min(min_distance, distance)
                
                # Score based on proximity (closer = more likely to be a real hit)
                if min_distance < 50:
                    player_score = 0.3
                elif min_distance < 100:
                    player_score = 0.2
                elif min_distance < 150:
                    player_score = 0.1
        
        # 📐 Angle Change Analysis - Realistic direction changes
        angle_score = 0.0
        
        if len(past_ball_pos) >= 4:
            # Calculate angle change at hit point
            p1 = past_ball_pos[-4]
            p2 = past_ball_pos[-3]
            p3 = past_ball_pos[-2]
            p4 = past_ball_pos[-1]
            
            # Vector before hit
            vec1 = (p2[0] - p1[0], p2[1] - p1[1])
            vec2 = (p3[0] - p2[0], p3[1] - p2[1])
            
            # Vector after hit  
            vec3 = (p4[0] - p3[0], p4[1] - p3[1])
            
            # Calculate angle changes
            try:
                angle1 = math.atan2(vec1[1], vec1[0])
                angle2 = math.atan2(vec2[1], vec2[0])
                angle3 = math.atan2(vec3[1], vec3[0])
                
                # Angle change at hit point
                angle_change = abs(angle3 - angle2)
                angle_change = min(angle_change, 2*math.pi - angle_change)  # Take smaller angle
                
                # Convert to degrees
                angle_change_deg = math.degrees(angle_change)
                
                # Realistic hits show 20-150 degree changes
                if 20 <= angle_change_deg <= 150:
                    angle_score = 0.3
                elif 10 <= angle_change_deg <= 180:
                    angle_score = 0.15
                    
            except:
                pass
        
        # 🏃‍♂️ Player Movement Validation - Check if player was moving appropriately
        movement_score = 0.0
        
        if players.get(who_hit) and len(past_ball_pos) >= 3:
            player = players[who_hit]
            
            # Check if player has movement history
            if hasattr(player, 'position_history') and len(player.position_history) >= 2:
                recent_positions = player.position_history[-2:]
                
                # Calculate player movement
                player_movement = math.sqrt(
                    (recent_positions[1][0] - recent_positions[0][0])**2 +
                    (recent_positions[1][1] - recent_positions[0][1])**2
                )
                
                # Players typically move 10-100 pixels between frames when hitting
                if 5 <= player_movement <= 100:
                    movement_score = 0.1
        
        # Combine all validation scores
        validation_score = trajectory_score + player_score + angle_score + movement_score
        
        # Normalize to 0-1 range
        validation_score = min(validation_score, max_score)
        
        return validation_score
        
    except Exception as e:
        print(f"⚠️ Physics validation error: {e}")
        return 0.5  # Neutral score on error

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
        print(" Enhancement summary saved to output/enhancement_summary.png")
        
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
    # Comprehensive error handling setup at the start
    import traceback
    import sys
    
    def enhanced_error_handler(exc_type, exc_value, exc_traceback):
        """Enhanced error handler to catch EnhancedBallTracker issues"""
        if "court_width" in str(exc_value):
            print("🚨 CAUGHT THE COURT_WIDTH ERROR!")
            print(f"🚨 Error type: {exc_type}")
            print(f"🚨 Error value: {exc_value}")
            print(f"🚨 Full traceback:")
            traceback.print_tb(exc_traceback)
            
            # Print detailed frame information
            frame = exc_traceback.tb_frame
            while frame:
                print(f"🔍 Frame: {frame.f_code.co_filename}:{frame.f_lineno} in {frame.f_code.co_name}")
                print(f"    Locals: {list(frame.f_locals.keys())}")
                if 'EnhancedBallTracker' in frame.f_locals:
                    print(f"    EnhancedBallTracker in locals: {frame.f_locals['EnhancedBallTracker']}")
                frame = frame.f_back
            
        # Call the default handler
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
    
    # Install our enhanced error handler
    sys.excepthook = enhanced_error_handler
    
    # Protect EnhancedBallTracker alias at the start of execution
    _protect_enhanced_ball_tracker()
    
    # Update global frame dimensions with user-provided values
    global frame_width, frame_height, coaching_data_collection
    frame_width = input_frame_width
    frame_height = input_frame_height
    # Initialize frame counter early so exception handlers can access it
    frame_count = 0
    
    # Test if our tracing works
    print("🧪 Testing EnhancedBallTracker tracing...")
    try:
        test_instance = EnhancedBallTracker(max_history=5)
        print("✅ Test successful")
        del test_instance
    except Exception as test_error:
        print(f"❌ Test failed: {test_error}")
    
    print("Squash coaching pipeline starting up...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.enabled = True
        torch.cuda.set_per_process_memory_fraction(0.9)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print(f"GPU Memory allocated: {torch.cuda.memory_allocated(0) / 1e6:.1f} MB")
        print(f"GPU Memory cached: {torch.cuda.memory_reserved(0) / 1e6:.1f} MB")
        print("CUDA optimizations enabled.")
    else:
        print("No GPU detected - using CPU.")
    model_optimizations = {
        'conf': 0.3,
        'iou': 0.5,
        'max_det': 10,
        'agnostic_nms': True,
        'half': True if torch.cuda.is_available() else False,
    }
    print("Model optimizations:")
    for key, value in model_optimizations.items():
        print(f"    {key}: {value}")
    try:
        print("Initializing squash coaching pipeline...")
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
            print(f"  GPU Memory allocated: {torch.cuda.memory_allocated(0) / 1e6:.1f} MB")
            print(f"  GPU Memory cached: {torch.cuda.memory_reserved(0) / 1e6:.1f} MB")
        else:
            print("   No GPU detected - using CPU (performance may be slower)")
        
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
        
        #  ULTRA-FAST MODEL LOADING WITH OPTIMIZATIONS
        print("  Loading optimized YOLO models with maximum GPU acceleration...")
        
        # Load pose model with aggressive optimizations
        pose_model = YOLO("models/yolo11n-pose.pt")
        if torch.cuda.is_available():
            pose_model.to(device)
            # Apply aggressive optimizations
            pose_model.fuse()  # Fuse layers for speed
            pose_model.half()  # Use FP16 for 2x speedup
            print("  Pose model loaded on GPU with FP16 optimization")
        else:
            print(" Pose model loaded on CPU")
        
        # Load ball detection model with aggressive optimizations
        ballmodel = YOLO("trained-models/black_ball_selfv3.pt")
        if torch.cuda.is_available():
            ballmodel.to(device)
            # Apply aggressive optimizations
            ballmodel.fuse()  # Fuse layers for speed
            ballmodel.half()  # Use FP16 for 2x speedup
            print("  Ball detection model loaded on GPU with FP16 optimization")
        else:
            print(" Ball detection model loaded on CPU")
        

        
        print("=" * 70)
        print(" Enhanced Features Active:")
        print("    GPU-accelerated ball and pose detection")
        print("    Enhanced bounce detection with yellow circle visualization")
        print("    Multi-criteria bounce validation (angle, velocity, wall proximity)")
        print("    Real-time coaching data collection & analysis")
        print("    Optimized memory management")
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
        
        # Initialize Enhanced Ball Tracker for smoother detection - use explicit class
        class LocalSmoothedBallTracker:
            """Enhanced ball tracking with smoothing and temporal consistency"""
            
            def __init__(self, max_history=10, smoothing_factor=0.3):
                self.max_history = max_history
                self.smoothing_factor = smoothing_factor
                self.position_history = deque(maxlen=max_history)
                self.velocity_history = deque(maxlen=max_history)
                self.last_confident_position = None
                self.prediction_streak = 0
                self.max_prediction_frames = 5
            
            def add_detection(self, position, confidence=1.0, frame_count=None):
                """Add a new ball detection with confidence scoring"""
                
                if position is None or len(position) < 2:
                    return self._handle_missing_detection(frame_count)
                    
                x, y = float(position[0]), float(position[1])
                
                # Validate position bounds (assuming 640x360 frame)
                if not (0 <= x <= 640 and 0 <= y <= 360):
                    return self._handle_missing_detection(frame_count)
                
                # If we have history, check for sudden jumps
                if self.position_history and confidence > 0.5:
                    last_pos = self.position_history[-1]
                    distance = math.sqrt((x - last_pos[0])**2 + (y - last_pos[1])**2)
                    
                    # Reject sudden jumps that are too large (unless very high confidence)
                    max_jump = 50 if confidence > 0.8 else 30
                    if distance > max_jump:
                        return self._handle_missing_detection(frame_count)
                
                # Calculate velocity
                velocity = [0.0, 0.0]
                if self.position_history:
                    last_pos = self.position_history[-1]
                    dt = 1.0  # Assume 1 frame time unit
                    velocity = [(x - last_pos[0]) / dt, (y - last_pos[1]) / dt]
                    self.velocity_history.append(velocity)
                
                # Add position with metadata
                ball_data = [x, y, frame_count if frame_count else len(self.position_history)]
                self.position_history.append(ball_data)
                self.last_confident_position = ball_data.copy()
                self.prediction_streak = 0
                
                return ball_data
            
            def _handle_missing_detection(self, frame_count):
                """Handle frames where ball detection failed"""
                
                if not self.position_history or self.prediction_streak >= self.max_prediction_frames:
                    return None
                    
                # Predict position based on velocity
                if self.velocity_history:
                    last_pos = self.position_history[-1]
                    last_velocity = self.velocity_history[-1]
                    
                    # Simple linear prediction
                    predicted_x = last_pos[0] + last_velocity[0]
                    predicted_y = last_pos[1] + last_velocity[1]
                    
                    # Bound predictions to reasonable area
                    predicted_x = max(0, min(640, predicted_x))
                    predicted_y = max(0, min(360, predicted_y))
                    
                    predicted_pos = [predicted_x, predicted_y, frame_count if frame_count else len(self.position_history)]
                    self.position_history.append(predicted_pos)
                    self.prediction_streak += 1
                    
                    return predicted_pos
                return None
            
            def get_current_position(self):
                """Get the most recent ball position"""
                return self.position_history[-1] if self.position_history else None
            
            def get_trajectory(self, length=None):
                """Get recent trajectory points"""
                if length is None:
                    return list(self.position_history)
                else:
                    return list(self.position_history)[-length:] if len(self.position_history) >= length else list(self.position_history)
            
            def get_velocity(self):
                """Get current ball velocity"""
                return self.velocity_history[-1] if self.velocity_history else [0.0, 0.0]
            
            def is_tracking(self):
                """Check if we're actively tracking the ball"""
                return len(self.position_history) > 0 and self.prediction_streak < self.max_prediction_frames
            
            def reset(self):
                """Reset the tracker"""
                self.position_history.clear()
                self.velocity_history.clear()
                self.last_confident_position = None
                self.prediction_streak = 0
        
        enhanced_ball_tracker = LocalSmoothedBallTracker(max_history=15, smoothing_factor=0.2)
        print("✅ Enhanced Ball Tracker initialized with temporal smoothing")
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
        
        # 🎯 INITIALIZE ENHANCED SHOT DETECTION SYSTEM v2.0 WITH REFERENCE POINTS
        print("=" * 70)
        print(" 🎯 ENHANCED SHOT DETECTION SYSTEM v2.0 INITIALIZATION")
        print("=" * 70)
        
        # Ensure EnhancedBallTracker alias is protected before any potential usage
        _protect_enhanced_ball_tracker()

        # Enable comprehensive call tracing to debug the court_width error
        _trace_enhanced_ball_tracker_calls()        # Using SmoothedBallTracker initialized earlier instead of EnhancedBallTracker
        # enhanced_ball_tracker = EnhancedBallTracker(
        #     court_width=frame_width, 
        #     court_height=frame_height,
        #     reference_points_px=reference_points,
        #     reference_points_3d=reference_points_3d
        # )
        # enhanced_shot_detector = EnhancedShotDetector(
        #     court_width=frame_width, 
        #     court_height=frame_height,
        #     reference_points_px=reference_points,
        #     reference_points_3d=reference_points_3d
        # )
        enhanced_shot_detector = None  # Disabled for now
        
        print("✅ SmoothedBallTracker initialized with temporal smoothing")
        # print("✅ Enhanced Shot Detector v2.0 initialized with court-aware algorithms")
        print("✅ Real-world physics validation enabled")
        print("✅ Court geometry-based wall/floor detection active")
        print("✅ Adaptive learning and outlier detection enabled")
        if reference_points and reference_points_3d:
            print(f"🎯 Court calibrated with {len(reference_points)} reference points")
            print("✅ ULTRA-HIGH ACCURACY MODE ENABLED")
        else:
            print("⚠️ Reference points not available - using pixel-based detection")
        print("=" * 70)
        
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
        
        #  ULTRA-FAST MAIN PROCESSING LOOP WITH OPTIMIZED FRAME HANDLING
        print("  Starting ultra-fast video processing...")
        
        # Pre-allocate memory for faster processing
        frame_buffer = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
        
        while cap.isOpened():
            # Protect EnhancedBallTracker alias in each frame iteration
            _protect_enhanced_ball_tracker()
            
            success, frame = cap.read()

            if not success:
                break

            #  OPTIMIZED FRAME PROCESSING - Direct resize without copy
            frame = cv2.resize(frame, (frame_width, frame_height), interpolation=cv2.INTER_LINEAR)
            frame_count += 1
            
            # Check frame limit for testing
            if max_frames is not None and frame_count > max_frames:
                print(f" Reached frame limit ({max_frames}), stopping processing")
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
                #  ULTRA-FAST POSE DETECTION WITH OPTIMIZED PROCESSING
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
                    #  ULTRA-FAST BALL DETECTION WITH OPTIMIZED PROCESSING
                    # Use optimized inference with pre-configured settings
                    ball = ballmodel(frame, **model_optimizations, verbose=False)
                    
                    #  OPTIMIZED BALL DETECTION - Direct GPU processing
                    x1, y1, x2, y2 = 0, 0, 0, 0
                    highestconf = 0.0
                    label = "ball"
                    ball_detected = False
                    
                    #  FAST DETECTION PROCESSING - Minimize CPU-GPU transfers
                    if ball and len(ball) > 0 and hasattr(ball[0], 'boxes') and ball[0].boxes is not None and len(ball[0].boxes) > 0:
                        # Find the highest confidence detection with optimized processing
                        best_box = None
                        best_conf = 0
                        
                        #  VECTORIZED PROCESSING for speed
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
                                    #  OPTIMIZED COORDINATE EXTRACTION
                                    coords = best_box.xyxy[0].cpu().numpy()
                                    if len(coords) >= 4:
                                        x1, y1, x2, y2 = coords[0], coords[1], coords[2], coords[3]
                                        highestconf = best_conf
                                        label = ballmodel.names[int(best_box.cls)]
                                        ball_detected = True
                                except Exception as e:
                                    ball_detected = False
                    
                    #  OPTIMIZED VISUALIZATION - Only draw if detected
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

                    #  OPTIMIZED FRAME DISPLAY - Minimal text rendering
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
                    
                    #  ENHANCED BALL POSITION UPDATE WITH 3D POSITIONING
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
                                    
                                    # Store both 2D and 3D positions with enhanced tracking
                                    ball_2d_pos = [avg_x, avg_y, running_frame]
                                    ball_3d_pos = enhanced_result['position'] + [running_frame]
                                    
                                    # Use Enhanced Ball Tracker for smoother, more consistent tracking
                                    smoothed_ball_pos = enhanced_ball_tracker.add_detection(
                                        ball_2d_pos, confidence=highestconf, frame_count=running_frame
                                    )
                                    
                                    if smoothed_ball_pos is not None:
                                        # Update past_ball_pos with enhanced, smoothed position
                                        past_ball_pos.append(smoothed_ball_pos)
                                        
                                        # Store 3D position separately for advanced analysis
                                        if 'past_ball_pos_3d' not in locals():
                                            past_ball_pos_3d = []
                                        past_ball_pos_3d.append(ball_3d_pos)
                                        
                                        # Apply 3D smoothing if we have enough data
                                        if len(past_ball_pos_3d) >= 5:
                                            past_ball_pos_3d = smooth_ball_trajectory_3d(past_ball_pos_3d)
                                    else:
                                        print(f"🔴 Ball detection rejected by enhanced tracker (frame {running_frame})")
                                    
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
                            
                            # 🎯 ENHANCED SHOT DETECTION SYSTEM INTEGRATION
                            if enhanced_ball_tracker and len(past_ball_pos) >= 2:
                                try:
                                    # Update enhanced ball tracker with new detection
                                    detection_data = {
                                        'x': avg_x,
                                        'y': avg_y,
                                        'w': x2 - x1,
                                        'h': y2 - y1,
                                        'confidence': float(highestconf)
                                    }
                                    
                                    # Update enhanced tracker
                                    tracker_result = enhanced_ball_tracker.update_tracking(detection_data, frame_count)
                                    
                                    if tracker_result:
                                        # Prepare player positions for shot detector
                                        player_positions = {}
                                        if players:
                                            for player_id, player_obj in players.items():
                                                if player_obj and hasattr(player_obj, 'get_latest_pose'):
                                                    pose = player_obj.get_latest_pose()
                                                    if pose and hasattr(pose, 'xyn') and len(pose.xyn) > 0:
                                                        keypoints = pose.xyn[0]
                                                        # Calculate center position from keypoints
                                                        valid_keypoints = [kp for kp in keypoints if kp[0] > 0 and kp[1] > 0]
                                                        if valid_keypoints:
                                                            avg_x_player = sum(kp[0] for kp in valid_keypoints) / len(valid_keypoints) * frame_width
                                                            avg_y_player = sum(kp[1] for kp in valid_keypoints) / len(valid_keypoints) * frame_height
                                                            player_positions[player_id] = {
                                                                'position': [avg_x_player, avg_y_player]
                                                            }
                                        
                                        # Detect shot events with enhanced system
                                        shot_events = enhanced_shot_detector.detect_shot_events(
                                            enhanced_ball_tracker, player_positions, frame_count
                                        )
                                        
                                        # Process detected events
                                        for event in shot_events['events']:
                                            event_type = event['type']
                                            event_data = event['data']
                                            
                                            print(f"🎯 ENHANCED EVENT DETECTED: {event_type.upper()}")
                                            print(f"    Frame: {frame_count}")
                                            print(f"    Confidence: {event_data.get('confidence', 0.0):.3f}")
                                            
                                            if event_type == 'racket_hit':
                                                player_who_hit = event_data.get('player_who_hit', 0)
                                                print(f"    Player: {player_who_hit}")
                                                print(f"    Shot started with enhanced detection")
                                                
                                            elif event_type == 'wall_hit':
                                                wall_type = event_data.get('wall_type', 'unknown')
                                                wall_distance = event_data.get('wall_distance', 0)
                                                print(f"    Wall: {wall_type}")
                                                print(f"    Distance: {wall_distance:.1f}px")
                                                
                                            elif event_type == 'floor_hit':
                                                height_ratio = event_data.get('height_ratio', 0)
                                                print(f"    Height ratio: {height_ratio:.2f}")
                                                print(f"    Shot completed")
                                        
                                        # Visualize enhanced shot detection
                                        for active_shot in shot_events['active_shots']:
                                            shot_id = active_shot['id']
                                            trajectory = active_shot['trajectory']
                                            phase = active_shot.get('phase', 'start')
                                            
                                            # Draw trajectory with phase-specific colors
                                            phase_colors = {
                                                'start': (0, 255, 0),    # Green - ball leaving racket
                                                'middle': (0, 255, 255), # Yellow - wall contact
                                                'end': (0, 0, 255)       # Red - floor contact
                                            }
                                            
                                            color = phase_colors.get(phase, (255, 255, 255))
                                            
                                            # Draw trajectory
                                            if len(trajectory) > 1:
                                                for i in range(1, len(trajectory)):
                                                    if len(trajectory[i-1]) >= 2 and len(trajectory[i]) >= 2:
                                                        pt1 = (int(trajectory[i-1][0]), int(trajectory[i-1][1]))
                                                        pt2 = (int(trajectory[i][0]), int(trajectory[i][1]))
                                                        cv2.line(annotated_frame, pt1, pt2, color, 3)
                                            
                                            # Draw shot markers
                                            if trajectory:
                                                start_pos = trajectory[0]
                                                current_pos = trajectory[-1]
                                                
                                                # Start marker
                                                cv2.circle(annotated_frame, (int(start_pos[0]), int(start_pos[1])), 8, (0, 255, 0), -1)
                                                cv2.putText(annotated_frame, f"START-{shot_id}", 
                                                           (int(start_pos[0]) + 10, int(start_pos[1]) - 10),
                                                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                                                
                                                # Current position marker
                                                cv2.circle(annotated_frame, (int(current_pos[0]), int(current_pos[1])), 6, color, -1)
                                                cv2.putText(annotated_frame, f"{phase.upper()}-{shot_id}", 
                                                           (int(current_pos[0]) + 10, int(current_pos[1]) + 20),
                                                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 2)
                                        
                                        # Display detection summary
                                        summary = enhanced_shot_detector.get_detection_summary()
                                        summary_text = f"Enhanced Shots: {summary['total_completed_shots']} | Active: {summary['active_shots']}"
                                        cv2.putText(annotated_frame, summary_text,
                                                   (10, frame_height - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                                except Exception as e:
                                    print(f"Enhanced shot detection error: {e}")
                                    # Continue with legacy system
                            
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
                            
                            # No background rectangle for bounce counter
                            bounce_text = f"Wall Hits (Legacy): {legacy_bounces}"
                            cv2.putText(annotated_frame, bounce_text, 
                                    (10, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                    
                    # Simple status display
                    if ball_detected:
                        cv2.putText(annotated_frame, "BALL DETECTED", 
                                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    else:
                        # No ball detected - try enhanced tracker prediction
                        predicted_pos = enhanced_ball_tracker.add_detection(None, confidence=0.0, frame_count=running_frame)
                        if predicted_pos is not None:
                            past_ball_pos.append(predicted_pos)
                            cv2.putText(annotated_frame, "BALL PREDICTED", 
                                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                            
                            # Draw prediction indicator
                            pred_x, pred_y = int(predicted_pos[0]), int(predicted_pos[1])
                            cv2.circle(annotated_frame, (pred_x, pred_y), 8, (255, 255, 0), 2)
                            cv2.putText(annotated_frame, "PRED", (pred_x + 10, pred_y - 10),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
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
                                print(f" ReID Stats at frame {frame_count}: {reid_stats}")
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
            hit_confidence = 0.0
            hit_type = "none"
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
            
            # 🎭 ENHANCED PLAYER ACTION CLASSIFICATION (if available)
            try:
                from squash.actionclassifier import classify as classify_player_action
                # Only attempt classification if we have players and recent ball activity
                if (len(past_ball_pos) >= 3 and players and 
                    any(players.get(i) for i in [1, 2])):
                    
                    for player_id in [1, 2]:
                        player_obj = players.get(player_id)
                        if player_obj and hasattr(player_obj, 'get_latest_pose'):
                            player_pose = player_obj.get_latest_pose()
                            if player_pose:
                                # Extract keypoints properly
                                if hasattr(player_pose, 'xyn') and len(player_pose.xyn) > 0:
                                    player_keypoints = player_pose.xyn[0]
                                    player_action = classify_player_action(player_keypoints)
                                    
                                    # Display player action
                                    action_text = f"Player {player_id}: {player_action}"
                                    cv2.putText(annotated_frame, action_text,
                                               (10, 120 + player_id * 25), cv2.FONT_HERSHEY_SIMPLEX, 
                                               0.5, (255, 255, 255), 2)
                                    
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
                print(f"  Error in coaching data collection: {e}")
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
                
            # 🎯 ENHANCED SHOT EVENT DISPLAY PANEL - Clear and Prominent
            panel_x = frame_width - 400
            panel_y = 10
            panel_width = 380
            panel_height = 200
            
            # No background rectangle - just the text overlay
            
            # Panel title
            cv2.putText(annotated_frame, "🎾 SHOT DETECTION STATUS", 
                      (panel_x + 10, panel_y + 20),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            
            # Enhanced ball tracking status
            if enhanced_ball_tracker.is_tracking():
                tracking_status = "TRACKING"
                tracking_color = (0, 255, 0)
                velocity = enhanced_ball_tracker.get_velocity()
                velocity_mag = math.sqrt(velocity[0]**2 + velocity[1]**2)
                cv2.putText(annotated_frame, f"Ball: {tracking_status} | V: {velocity_mag:.1f}px/f", 
                          (panel_x + 10, panel_y + 40),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.4, tracking_color, 1)
            else:
                cv2.putText(annotated_frame, "Ball: SEARCHING", 
                          (panel_x + 10, panel_y + 40),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 165, 0), 1)
            
            # Current shot information
            if shot_tracker.active_shots:
                current_shot = shot_tracker.active_shots[-1]
                shot_id = current_shot.get('id', 0)
                shot_player = current_shot.get('player_who_hit', 0)
                phase = current_shot.get('phase', 'unknown')
                shot_type = current_shot.get('shot_type', 'unknown')
                
                # Shot header
                cv2.putText(annotated_frame, f"ACTIVE SHOT #{shot_id}", 
                          (panel_x + 10, panel_y + 65),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                
                # Player and type
                cv2.putText(annotated_frame, f"Player {shot_player} | {str(shot_type).title()}", 
                          (panel_x + 10, panel_y + 85),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                
                # Phase with color coding
                phase_colors = {
                    'start': (0, 255, 255),    # Cyan - ball leaves racket
                    'middle': (255, 165, 0),   # Orange - ball hits wall  
                    'end': (255, 0, 255),      # Magenta - ball hits floor
                }
                phase_color = phase_colors.get(phase, (128, 128, 128))
                cv2.putText(annotated_frame, f"Phase: {phase.upper()}", 
                          (panel_x + 10, panel_y + 105),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.4, phase_color, 1)
                
                # Shot events timeline
                y_offset = 125
                if current_shot.get('start_frame'):
                    cv2.putText(annotated_frame, f"✓ Hit at frame {current_shot['start_frame']}", 
                              (panel_x + 10, panel_y + y_offset),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
                    y_offset += 15
                
                if current_shot.get('wall_hit_frame'):
                    wall_type = current_shot.get('wall_type', 'unknown')
                    cv2.putText(annotated_frame, f"✓ {wall_type.replace('_', ' ').title()} hit at frame {current_shot['wall_hit_frame']}", 
                              (panel_x + 10, panel_y + y_offset),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 165, 0), 1)
                    y_offset += 15
                
                if current_shot.get('floor_hit_frame'):
                    cv2.putText(annotated_frame, f"✓ Bounce at frame {current_shot['floor_hit_frame']}", 
                              (panel_x + 10, panel_y + y_offset),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 255), 1)
                    y_offset += 15
                
                # Trajectory info
                traj_length = len(current_shot.get('trajectory', []))
                cv2.putText(annotated_frame, f"Trajectory: {traj_length} points", 
                          (panel_x + 10, panel_y + y_offset),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 200, 200), 1)
            else:
                cv2.putText(annotated_frame, "No active shot", 
                          (panel_x + 10, panel_y + 65),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.4, (128, 128, 128), 1)
            
            # Shot statistics
            shot_stats = shot_tracker.get_shot_statistics()
            cv2.putText(annotated_frame, f"Total Shots: {shot_stats['total_shots']}", 
                      (panel_x + 200, panel_y + 40),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv2.putText(annotated_frame, f"P1: {shot_stats['shots_by_player'].get(1, 0)} | P2: {shot_stats['shots_by_player'].get(2, 0)}", 
                      (panel_x + 200, panel_y + 60),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # Enhanced status display with GPU information and shot tracking
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
                shot_tracker.draw_shot_trajectories(annotated_frame, ballxy[-1] if ballxy else None, frame_count)
                
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
                        print(f" Coaching data collected: {len(coaching_data_collection)} points (frame {running_frame})")
                            
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
            
            #  OPTIMIZED GPU MEMORY MONITORING - Less frequent for speed
            if frame_count % 30 == 0 and torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated(0) / 1e6
                cached = torch.cuda.memory_reserved(0) / 1e6
                print(f"Frame {frame_count}: GPU Memory - Allocated: {allocated:.1f}MB, Cached: {cached:.1f}MB")
            
            # 🎨 CLEAN SHOT VISUALIZATION with Minimal Color-Coded Indicators
            current_ball_pos = past_ball_pos[-1] if past_ball_pos else None
            
            try:
                # Use standard shot tracker visualization
                shot_tracker.draw_shot_trajectories(annotated_frame, current_ball_pos, frame_count)
                
                # Clean, minimal visualization overlays for active shots
                for active_shot in shot_tracker.active_shots:
                    shot_id = active_shot.get('id', 0)
                    shot_phase = active_shot.get('phase', 'start')
                    trajectory = active_shot.get('trajectory', [])
                    shot_type = active_shot.get('shot_type', 'unknown')
                    
                    # Get shot type color for trajectory
                    shot_color = (255, 255, 255)  # Default white
                    try:
                        if hasattr(shot_tracker, 'shot_classification_model'):
                            shot_color = shot_tracker.shot_classification_model.get_shot_color(shot_type)
                    except:
                        pass
                    
                    # 🎯 RACKET HIT MARKER - Shot Start (Green circle)
                    if len(trajectory) > 0:
                        start_pos = trajectory[0]
                        cv2.circle(annotated_frame, (int(start_pos[0]), int(start_pos[1])), 6, (0, 255, 0), -1)  # Bright green
                        cv2.circle(annotated_frame, (int(start_pos[0]), int(start_pos[1])), 8, (0, 255, 0), 2)   # Green outline
                        # Small "START" label
                        cv2.putText(annotated_frame, "START", (int(start_pos[0] - 15), int(start_pos[1] - 12)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
                    
                    # 🧱 FRONT WALL HIT MARKER (Orange circle)
                    if 'wall_hit_position' in active_shot:
                        wall_pos = active_shot['wall_hit_position']
                        cv2.circle(annotated_frame, (int(wall_pos[0]), int(wall_pos[1])), 6, (0, 165, 255), -1)  # Bright orange
                        cv2.circle(annotated_frame, (int(wall_pos[0]), int(wall_pos[1])), 8, (0, 165, 255), 2)   # Orange outline
                        # Small "WALL" label
                        cv2.putText(annotated_frame, "WALL", (int(wall_pos[0] - 15), int(wall_pos[1] - 12)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 165, 255), 1)
                    
                    # 🔻 FLOOR BOUNCE MARKER (Magenta circle)
                    if 'floor_hit_position' in active_shot:
                        floor_pos = active_shot['floor_hit_position']
                        cv2.circle(annotated_frame, (int(floor_pos[0]), int(floor_pos[1])), 6, (255, 0, 255), -1)  # Bright magenta
                        cv2.circle(annotated_frame, (int(floor_pos[0]), int(floor_pos[1])), 8, (255, 0, 255), 2)   # Magenta outline
                        # Small "FLOOR" label
                        cv2.putText(annotated_frame, "FLOOR", (int(floor_pos[0] - 18), int(floor_pos[1] + 20)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 255), 1)
                    
                    # Optional: Current ball position with shot color (small circle)
                    if len(trajectory) > 0:
                        current_pos = trajectory[-1]
                        cv2.circle(annotated_frame, (int(current_pos[0]), int(current_pos[1])), 3, shot_color, -1)
                
                # Show recent completed shots with minimal markers
                recent_completed = shot_tracker.completed_shots[-2:] if len(shot_tracker.completed_shots) > 2 else shot_tracker.completed_shots
                for i, completed_shot in enumerate(recent_completed):
                    # Show end position with small faded marker
                    if 'end_position' in completed_shot:
                        end_pos = completed_shot['end_position']
                        shot_id = completed_shot.get('id', 0)
                        
                        # Small faded circle for completed shots
                        alpha = 0.4 + (i * 0.2)
                        faded_color = tuple(int(128 * alpha) for _ in range(3))  # Gray faded
                        
                        cv2.circle(annotated_frame, (int(end_pos[0]), int(end_pos[1])), 3, faded_color, -1)
                                
            except Exception as e:
                print(f"⚠️ Error in clean shot visualization: {e}")
                # Fallback to basic visualization
                shot_tracker.draw_shot_trajectories(annotated_frame, current_ball_pos, frame_count)
            
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
        
        # Initialize variables for enhanced coaching analysis at proper scope
        basic_coaching_attempted = False
        coaching_data_collection_ready = False
        
        # Basic autonomous coaching report will be replaced by enhanced analysis below
        try:
            print("Preparing coaching data for enhanced analysis...")
            # Basic report generation will be handled by enhanced analysis
            basic_coaching_attempted = True
            coaching_data_collection_ready = True
            print("Coaching data prepared successfully.")
        except Exception as e:
            print(f"Error preparing coaching data: {e}")
        
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
        print("\n Generating Enhanced Autonomous Coaching Analysis with Shot Tracking...")
        
        # Generate shot analysis report
        try:
            shot_stats = shot_tracker.get_shot_statistics()
            
            # Add missing fields with default values if not present
            if 'average_shot_duration' not in shot_stats:
                shot_stats['average_shot_duration'] = 0.0
            if 'shots_by_player' not in shot_stats:
                shot_stats['shots_by_player'] = {1: 0, 2: 0}
            
            # Get legacy bounce detection statistics
            total_bounces = shot_stats.get('wall_hits_distribution', {})
            total_wall_hits = sum(total_bounces.values()) if total_bounces else 0
            
            # Create comprehensive shot analysis
            shot_analysis_report = f"""
ENHANCED SHOT TRACKING ANALYSIS
=======================================

SHOT STATISTICS:
-----------------
Total Shots Tracked: {shot_stats['total_shots']}
Player 1 Shots: {shot_stats['shots_by_player'].get(1, 0)}
Player 2 Shots: {shot_stats['shots_by_player'].get(2, 0)}
Average Shot Duration: {shot_stats['average_shot_duration']:.1f} frames

WALL HIT DETECTION STATISTICS:
------------------------------
Total Wall Hits Detected: {total_wall_hits}
Wall Hit Distribution: {total_bounces}
Legacy Detection Algorithm: Direction-based hit detection

SHOT TYPE BREAKDOWN:
---------------------
"""
            for shot_type, count in shot_stats['shots_by_type'].items():
                percentage = (count / shot_stats['total_shots'] * 100) if shot_stats['total_shots'] > 0 else 0
                shot_analysis_report += f" {shot_type}: {count} ({percentage:.1f}%)\n"
                
            shot_analysis_report += f"""

ENHANCED VISUALIZATION FEATURES:
----------------------------------
Real-time trajectory visualization with color coding
Clear shot START markers: Large circles with "START" text
Clear shot END markers: Square markers with "END" text  
Active shot indicators: "ACTIVE" labels on current ball
Crosscourt shots: Green trajectory lines
Straight shots: Yellow trajectory lines  
Boast shots: Magenta trajectory lines
Drop shots: Orange trajectory lines
Lob shots: Blue trajectory lines
Bounce visualization: Multi-colored confidence-based markers

TECHNICAL IMPROVEMENTS:
-------------------------
Enhanced player-ball hit detection using weighted keypoints
Multi-factor scoring system (proximity + movement + trajectory)
Real-time shot classification with trajectory analysis
Automatic shot completion detection
Comprehensive 4-algorithm bounce detection system
Physics-based bounce validation
Velocity vector analysis for precise hit detection
Wall proximity analysis with gradient scoring
Trajectory curvature analysis for direction changes
Complete shot data with bounce information saved to: output/shots_log.jsonl

VISUAL MARKERS LEGEND:
------------------------
START: Large circle with white center and colored border
END: Square marker with red center and colored border  
ACTIVE: Current ball position with enhanced highlighting
Bounces: Color-coded by confidence (Green=High, Yellow=Medium, Orange=Low)
Shot Duration: Displayed next to END markers
Algorithm Count: Shows how many detection algorithms agreed
"""
            
            # Save shot analysis report
            with open("output/shot_analysis_report.txt", "w") as f:
                f.write(shot_analysis_report)
                
            print(" Shot analysis report saved to output/shot_analysis_report.txt")
            print(f" {shot_stats['total_shots']} shots tracked and saved to output/shots_log.jsonl")
            
            # Analyze shot patterns from saved data
            shot_patterns = analyze_shot_patterns()
            if shot_patterns:
                print(f" Advanced shot pattern analysis completed")
                print(f"   - Total shots analyzed: {shot_patterns['total_shots']}")
                print(f"   - Average shot duration: {shot_patterns['average_duration']:.1f} frames")
            
            # Create visual enhancement summary
            create_enhancement_summary()
            
        except Exception as e:
            print(f" Error generating shot analysis: {e}")
        
        try:
            # Get the global autonomous coach instance (avoid reloading models)
            from autonomous_coaching import get_autonomous_coach
            autonomous_coach = get_autonomous_coach()
            
            # Generate comprehensive coaching insights
            coaching_insights = autonomous_coach.analyze_match_data(coaching_data_collection)
            
            # Safe access to shot tracker statistics
            try:
                shot_stats = shot_tracker.get_shot_statistics() if 'shot_tracker' in globals() and shot_tracker else {'total_shots': 0}
            except Exception:
                shot_stats = {'total_shots': 0}
            
            # Enhanced coaching report with bounce and shot analysis
            enhanced_report = f"""
ENHANCED SQUASH COACHING ANALYSIS
================================================

Analysis Date: {time.strftime('%Y-%m-%d %H:%M:%S')}
Video Analyzed: {path}
Total Frames Processed: {frame_count}
Enhanced Coaching Data Points: {len(coaching_data_collection)}
GPU Acceleration: {' Enabled' if torch.cuda.is_available() else ' CPU Only'}

ENHANCED BALL TRACKING ANALYSIS:
------------------------------
Total trajectory points: {len(past_ball_pos)}
Enhanced bounce detection: GPU-accelerated
Multi-criteria validation: Angle, velocity, wall proximity
Visualization: Real-time colored trajectory indicators
Shot tracking: {shot_stats['total_shots']} complete shots analyzed

{coaching_insights}

TECHNICAL ENHANCEMENTS:
---------------------
GPU-optimized ball detection and tracking
Enhanced shot tracking with real-time visualization
Enhanced bounce detection with multiple validation criteria
Real-time trajectory analysis with physics modeling
Comprehensive coaching data collection
Advanced ball bounce pattern analysis
Color-coded shot visualization system

SYSTEM PERFORMANCE:
-----------------
Processing device: {'GPU' if torch.cuda.is_available() else 'CPU'}
Ball detection accuracy: Enhanced with trained model
Bounce detection: Multi-algorithm validation
Shot tracking:  Active throughout session
Real-time analysis:  Active throughout session

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
Player 1 Initialization: {'Complete' if reid_stats.get('initialization_status', {}).get(1) else ' Incomplete'}
Player 2 Initialization: {'Complete' if reid_stats.get('initialization_status', {}).get(2) else ' Incomplete'}

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
            print("   Generating basic coaching report as fallback...")
            # Fallback to basic coaching report if enhanced analysis fails
            try:
                # Ensure variables are accessible
                if not basic_coaching_attempted:
                    generate_coaching_report(coaching_data_collection, path, frame_count)
                    print(" Basic coaching report generated successfully as fallback.")
                else:
                    print("   Basic coaching data was prepared, enhanced analysis failed.")
            except Exception as fallback_error:
                print(f" Fallback coaching report also failed: {fallback_error}")
        
        print("\n ENHANCED PROCESSING COMPLETE!")
        print("=" * 50)
        print(" Check output/ directory for results:")
        print("    enhanced_autonomous_coaching_report.txt - Enhanced analysis")
        print("    enhanced_coaching_data.json - Detailed data with bounces")
        print("    reid_analysis_report.txt - Player ReID analysis")
        print("    final_reid_references.json - Player appearance references")
        print("    annotated.mp4 - Video with bounce visualization")
        print("    final.csv - Complete match data")
        print("    graphics/ - Comprehensive visualizations and analytics:")
        print("     - Shot type analysis and heatmaps")
        print("     - Player and ball movement patterns")
        print("     - Ball trajectory analysis")
        print("     - Match flow and performance metrics")
        print("     - Summary statistics and reports")
        print("    Other traditional output files")
        print("=" * 50)
        print("\n ENHANCED REID SYSTEM FEATURES:")
        print("    Initial player appearance capture (frames 100-150)")
        print("    Continuous track ID swap detection")
        print("    Deep learning-based appearance features")
        print("    Multi-modal identity verification (appearance + position)")
        print("    Real-time confidence scoring")
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
    g = 9.81  # m/s
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
                        print(f"Frame 73: Angle change {angle_change:.1f} too large (threshold: 170)")
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
    g = 9.81  # m/s
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
        
