"""
Enhanced Ball Physics and Shot Detection System for Squash Coaching Pipeline

This module provides advanced ball trajectory analysis with precise detection of:
1. Racket hits (with velocity and direction changes)
2. Wall impacts (front wall, side walls)
3. Floor bounces (with realistic physics modeling)

Key improvements:
- Physics-based trajectory modeling using Kalman filters
- Multi-modal event detection using signal processing
- Advanced trajectory segmentation for shot analysis
- Real-time collision detection with confidence scoring
"""

import numpy as np
import cv2
import math
import scipy.signal
from scipy.signal import find_peaks, savgol_filter
from scipy.spatial.distance import euclidean
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
import torch
import torch.nn as nn
from collections import deque, namedtuple
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any
import json
import time

# Data structures for enhanced shot detection
@dataclass
class BallPosition:
    """Enhanced ball position with physics properties"""
    x: float
    y: float
    frame: int
    velocity_x: float = 0.0
    velocity_y: float = 0.0
    acceleration_x: float = 0.0
    acceleration_y: float = 0.0
    confidence: float = 1.0
    timestamp: float = 0.0

@dataclass 
class ShotEvent:
    """Shot event with detailed physics data"""
    event_type: str  # 'racket_hit', 'wall_hit', 'floor_bounce'
    frame: int
    position: Tuple[float, float]
    velocity_before: Tuple[float, float]
    velocity_after: Tuple[float, float]
    confidence: float
    player_id: Optional[int] = None
    wall_type: Optional[str] = None  # 'front', 'side_left', 'side_right', 'back'
    impact_angle: Optional[float] = None
    spin_detected: bool = False
    
@dataclass
class Shot:
    """Complete shot with trajectory and events"""
    shot_id: int
    start_frame: int
    end_frame: int
    trajectory: List[BallPosition]
    events: List[ShotEvent]
    player_id: int
    shot_type: str
    confidence: float
    duration: float
    
class AdvancedKalmanFilter:
    """
    Enhanced Kalman Filter for ball tracking with physics-based motion model
    State vector: [x, y, vx, vy, ax, ay]
    """
    
    def __init__(self, dt=1/30.0, process_noise=1e-2, measurement_noise=1e-1):
        self.dt = dt
        self.kf = KalmanFilter(dim_x=6, dim_z=2)
        
        # State transition matrix (constant acceleration model)
        self.kf.F = np.array([
            [1, 0, dt, 0, 0.5*dt**2, 0],
            [0, 1, 0, dt, 0, 0.5*dt**2],
            [0, 0, 1, 0, dt, 0],
            [0, 0, 0, 1, 0, dt],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ])
        
        # Measurement matrix (we observe position only)
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0]
        ])
        
        # Process noise
        self.kf.Q = Q_discrete_white_noise(dim=2, dt=dt, var=process_noise, block_size=3)
        
        # Measurement noise
        self.kf.R *= measurement_noise
        
        # Initial state covariance
        self.kf.P *= 100
        
        self.initialized = False
        
    def initialize(self, position):
        """Initialize filter with first position"""
        self.kf.x = np.array([position[0], position[1], 0, 0, 0, 0])
        self.initialized = True
        
    def predict(self):
        """Predict next state"""
        if self.initialized:
            self.kf.predict()
            return self.kf.x[:2]  # Return predicted position
        return None
        
    def update(self, measurement):
        """Update with new measurement"""
        if not self.initialized:
            self.initialize(measurement)
        else:
            self.kf.update(measurement)
            
    def get_state(self):
        """Get current state [x, y, vx, vy, ax, ay]"""
        return self.kf.x.copy()
        
    def get_velocity(self):
        """Get current velocity [vx, vy]"""
        return self.kf.x[2:4]
        
    def get_acceleration(self):
        """Get current acceleration [ax, ay]"""
        return self.kf.x[4:6]

class PhysicsBasedEventDetector:
    """
    Advanced event detection using physics-based analysis and signal processing
    """
    
    def __init__(self, court_dimensions=(640, 360)):
        self.court_width, self.court_height = court_dimensions
        self.min_trajectory_length = 10
        
        # Physics constants for squash ball
        self.gravity = 9.81  # m/s^2 (adjusted for court scale)
        self.court_scale = 0.01  # pixels to meters conversion
        self.ball_mass = 0.024  # kg
        self.coefficient_of_restitution = 0.45  # squash ball bouncing
        
        # Detection thresholds
        self.velocity_change_threshold = 0.3  # Significant velocity change
        self.direction_change_threshold = 30  # degrees
        self.acceleration_threshold = 2.0  # m/s^2
        
        # Wall boundaries (adjustable based on court detection)
        self.wall_boundaries = {
            'front': 50,  # pixels from front wall
            'back': self.court_width - 50,
            'left': 50,
            'right': self.court_height - 50
        }
        
    def detect_racket_hits(self, trajectory: List[BallPosition], players_data: Dict) -> List[ShotEvent]:
        """
        Detect racket hits using advanced trajectory analysis
        """
        events = []
        
        if len(trajectory) < self.min_trajectory_length:
            return events
            
        # Convert to numpy arrays for efficient processing
        positions = np.array([[p.x, p.y] for p in trajectory])
        velocities = np.array([[p.velocity_x, p.velocity_y] for p in trajectory])
        frames = np.array([p.frame for p in trajectory])
        
        # Apply Savitzky-Golay filter to smooth the trajectory
        if len(positions) > 5:
            smoothed_x = savgol_filter(positions[:, 0], window_length=5, polyorder=2)
            smoothed_y = savgol_filter(positions[:, 1], window_length=5, polyorder=2)
            smoothed_positions = np.column_stack([smoothed_x, smoothed_y])
        else:
            smoothed_positions = positions
            
        # Calculate velocity magnitude changes
        velocity_magnitudes = np.linalg.norm(velocities, axis=1)
        
        # Find peaks in velocity changes (indicating potential hits)
        velocity_changes = np.diff(velocity_magnitudes)
        velocity_peaks, _ = find_peaks(np.abs(velocity_changes), 
                                     height=self.velocity_change_threshold,
                                     distance=10)  # Minimum 10 frames between hits
        
        for peak_idx in velocity_peaks:
            frame_idx = peak_idx + 1  # Adjust for diff offset
            if frame_idx >= len(trajectory):
                continue
                
            event_frame = frames[frame_idx]
            event_position = (trajectory[frame_idx].x, trajectory[frame_idx].y)
            
            # Calculate velocity before and after
            velocity_before = (0, 0)
            velocity_after = (0, 0)
            
            if frame_idx > 0:
                velocity_before = (velocities[frame_idx-1, 0], velocities[frame_idx-1, 1])
            if frame_idx < len(velocities):
                velocity_after = (velocities[frame_idx, 0], velocities[frame_idx, 1])
            
            # Calculate direction change
            direction_change = self._calculate_direction_change(velocity_before, velocity_after)
            
            # Determine player proximity and likely hitter
            player_id = self._determine_hitting_player(event_position, players_data, event_frame)
            
            # Calculate confidence based on multiple factors
            confidence = self._calculate_racket_hit_confidence(
                velocity_changes[peak_idx], direction_change, player_id, event_position
            )
            
            if confidence > 0.4:  # Threshold for valid racket hit
                event = ShotEvent(
                    event_type='racket_hit',
                    frame=event_frame,
                    position=event_position,
                    velocity_before=velocity_before,
                    velocity_after=velocity_after,
                    confidence=confidence,
                    player_id=player_id,
                    impact_angle=direction_change
                )
                events.append(event)
                
        return events
    
    def detect_wall_hits(self, trajectory: List[BallPosition]) -> List[ShotEvent]:
        """
        Detect wall impacts using trajectory analysis and court boundaries
        """
        events = []
        
        if len(trajectory) < 5:
            return events
            
        for i in range(2, len(trajectory) - 2):
            current_pos = trajectory[i]
            prev_pos = trajectory[i-1]
            next_pos = trajectory[i+1]
            
            # Check proximity to walls
            wall_type = self._detect_wall_proximity(current_pos.x, current_pos.y)
            
            if wall_type:
                # Analyze trajectory for bounce pattern
                velocity_before = (current_pos.x - prev_pos.x, current_pos.y - prev_pos.y)
                velocity_after = (next_pos.x - current_pos.x, next_pos.y - current_pos.y)
                
                # Check for direction change characteristic of wall bounce
                direction_change = self._calculate_direction_change(velocity_before, velocity_after)
                
                # Check for velocity magnitude conservation (elastic collision)
                velocity_mag_before = math.sqrt(velocity_before[0]**2 + velocity_before[1]**2)
                velocity_mag_after = math.sqrt(velocity_after[0]**2 + velocity_after[1]**2)
                
                velocity_ratio = velocity_mag_after / max(velocity_mag_before, 0.001)
                
                # Wall hit confidence based on physics
                confidence = self._calculate_wall_hit_confidence(
                    direction_change, velocity_ratio, wall_type, current_pos
                )
                
                if confidence > 0.5:
                    event = ShotEvent(
                        event_type='wall_hit',
                        frame=current_pos.frame,
                        position=(current_pos.x, current_pos.y),
                        velocity_before=velocity_before,
                        velocity_after=velocity_after,
                        confidence=confidence,
                        wall_type=wall_type,
                        impact_angle=direction_change
                    )
                    events.append(event)
                    
        return events
    
    def detect_floor_bounces(self, trajectory: List[BallPosition]) -> List[ShotEvent]:
        """
        Detect floor bounces using physics-based trajectory analysis
        """
        events = []
        
        if len(trajectory) < 6:
            return events
            
        # Extract y-coordinates and apply smoothing
        y_positions = np.array([p.y for p in trajectory])
        frames = np.array([p.frame for p in trajectory])
        
        if len(y_positions) > 5:
            smoothed_y = savgol_filter(y_positions, window_length=5, polyorder=2)
        else:
            smoothed_y = y_positions
            
        # Find local minima (potential floor bounces)
        minima_indices, _ = find_peaks(-smoothed_y, height=-self.court_height*0.8, distance=8)
        
        for min_idx in minima_indices:
            if min_idx < 2 or min_idx >= len(trajectory) - 2:
                continue
                
            bounce_pos = trajectory[min_idx]
            
            # Analyze trajectory before and after bounce
            pre_bounce_trajectory = trajectory[max(0, min_idx-3):min_idx]
            post_bounce_trajectory = trajectory[min_idx:min(len(trajectory), min_idx+4)]
            
            # Check for characteristic bounce pattern
            bounce_confidence = self._analyze_bounce_pattern(
                pre_bounce_trajectory, bounce_pos, post_bounce_trajectory
            )
            
            if bounce_confidence > 0.6:
                # Calculate velocities
                velocity_before = (0, 0)
                velocity_after = (0, 0)
                
                if min_idx > 0:
                    prev_pos = trajectory[min_idx-1]
                    velocity_before = (bounce_pos.x - prev_pos.x, bounce_pos.y - prev_pos.y)
                    
                if min_idx < len(trajectory) - 1:
                    next_pos = trajectory[min_idx+1]
                    velocity_after = (next_pos.x - bounce_pos.x, next_pos.y - bounce_pos.y)
                
                event = ShotEvent(
                    event_type='floor_bounce',
                    frame=bounce_pos.frame,
                    position=(bounce_pos.x, bounce_pos.y),
                    velocity_before=velocity_before,
                    velocity_after=velocity_after,
                    confidence=bounce_confidence
                )
                events.append(event)
                
        return events
    
    def _calculate_direction_change(self, velocity_before: Tuple[float, float], 
                                  velocity_after: Tuple[float, float]) -> float:
        """Calculate angle change in trajectory"""
        if not velocity_before or not velocity_after:
            return 0
            
        v1_mag = math.sqrt(velocity_before[0]**2 + velocity_before[1]**2)
        v2_mag = math.sqrt(velocity_after[0]**2 + velocity_after[1]**2)
        
        if v1_mag < 0.001 or v2_mag < 0.001:
            return 0
            
        # Normalize vectors
        v1_norm = (velocity_before[0]/v1_mag, velocity_before[1]/v1_mag)
        v2_norm = (velocity_after[0]/v2_mag, velocity_after[1]/v2_mag)
        
        # Calculate dot product
        dot_product = v1_norm[0]*v2_norm[0] + v1_norm[1]*v2_norm[1]
        dot_product = max(-1, min(1, dot_product))  # Clamp to valid range
        
        angle_rad = math.acos(dot_product)
        angle_deg = math.degrees(angle_rad)
        
        return angle_deg
    
    def _detect_wall_proximity(self, x: float, y: float) -> Optional[str]:
        """Detect which wall the ball is near"""
        wall_threshold = 40  # pixels
        
        if x < wall_threshold:
            return 'left'
        elif x > self.court_width - wall_threshold:
            return 'right'
        elif y < wall_threshold:
            return 'front'
        elif y > self.court_height - wall_threshold:
            return 'back'
        return None
    
    def _determine_hitting_player(self, ball_position: Tuple[float, float], 
                                players_data: Dict, frame: int) -> Optional[int]:
        """Determine which player likely hit the ball"""
        if not players_data:
            return None
            
        min_distance = float('inf')
        hitting_player = None
        
        for player_id, player_data in players_data.items():
            if player_data and len(player_data) > 0:
                # Get player position at the frame
                player_pos = self._get_player_position_at_frame(player_data, frame)
                if player_pos:
                    distance = euclidean(ball_position, player_pos)
                    if distance < min_distance and distance < 100:  # Max racket reach
                        min_distance = distance
                        hitting_player = player_id
                        
        return hitting_player
    
    def _get_player_position_at_frame(self, player_data: List, frame: int) -> Optional[Tuple[float, float]]:
        """Get player position at specific frame"""
        # This would need to be implemented based on your player data structure
        # For now, return a placeholder
        return None
    
    def _calculate_racket_hit_confidence(self, velocity_change: float, 
                                       direction_change: float, player_id: Optional[int],
                                       position: Tuple[float, float]) -> float:
        """Calculate confidence score for racket hit detection"""
        confidence = 0.0
        
        # Velocity change factor
        velocity_factor = min(1.0, abs(velocity_change) / 5.0)
        confidence += velocity_factor * 0.4
        
        # Direction change factor
        direction_factor = min(1.0, direction_change / 90.0)
        confidence += direction_factor * 0.3
        
        # Player proximity factor
        if player_id is not None:
            confidence += 0.3
            
        return min(1.0, confidence)
    
    def _calculate_wall_hit_confidence(self, direction_change: float, velocity_ratio: float,
                                     wall_type: str, position: BallPosition) -> float:
        """Calculate confidence score for wall hit detection"""
        confidence = 0.0
        
        # Direction change should be significant for wall hits
        if direction_change > 20:
            confidence += 0.4
            
        # Velocity should be partially conserved (some energy lost)
        if 0.6 < velocity_ratio < 1.2:
            confidence += 0.3
            
        # Position should be very close to wall
        wall_distance = self._calculate_wall_distance(position.x, position.y, wall_type)
        if wall_distance < 30:
            confidence += 0.3
            
        return min(1.0, confidence)
    
    def _calculate_wall_distance(self, x: float, y: float, wall_type: str) -> float:
        """Calculate distance to specified wall"""
        if wall_type == 'left':
            return x
        elif wall_type == 'right':
            return self.court_width - x
        elif wall_type == 'front':
            return y
        elif wall_type == 'back':
            return self.court_height - y
        return float('inf')
    
    def _analyze_bounce_pattern(self, pre_trajectory: List[BallPosition], 
                              bounce_pos: BallPosition, 
                              post_trajectory: List[BallPosition]) -> float:
        """Analyze trajectory pattern for floor bounce validation"""
        if len(pre_trajectory) < 2 or len(post_trajectory) < 2:
            return 0.0
            
        confidence = 0.0
        
        # Check if ball was moving downward before bounce
        if len(pre_trajectory) >= 2:
            y_velocity_before = pre_trajectory[-1].y - pre_trajectory[-2].y
            if y_velocity_before > 0:  # Moving down (y increases downward)
                confidence += 0.3
                
        # Check if ball moves upward after bounce
        if len(post_trajectory) >= 2:
            y_velocity_after = post_trajectory[1].y - post_trajectory[0].y
            if y_velocity_after < 0:  # Moving up (y decreases upward)
                confidence += 0.3
                
        # Check if bounce is in lower part of court
        if bounce_pos.y > self.court_height * 0.6:
            confidence += 0.4
            
        return min(1.0, confidence)

class EnhancedShotDetector:
    """
    Main enhanced shot detection system
    """
    
    def __init__(self, court_dimensions=(640, 360)):
        self.court_dimensions = court_dimensions
        self.kalman_filter = AdvancedKalmanFilter()
        self.event_detector = PhysicsBasedEventDetector(court_dimensions)
        
        # Shot tracking
        self.active_shots = []
        self.completed_shots = []
        self.shot_id_counter = 0
        
        # Trajectory buffer
        self.trajectory_buffer = deque(maxlen=300)  # 10 seconds at 30fps
        self.min_shot_length = 15  # Minimum frames for a valid shot
        
        # Performance metrics
        self.processing_times = []
        
    def process_frame(self, ball_detections: List, players_data: Dict, frame_number: int) -> Dict:
        """
        Process single frame and update shot detection
        
        Args:
            ball_detections: List of ball detections [x, y, confidence]
            players_data: Dictionary of player data
            frame_number: Current frame number
            
        Returns:
            Dictionary with shot analysis results
        """
        start_time = time.time()
        
        # Update ball tracking with Kalman filter
        ball_position = None
        if ball_detections:
            # Select best detection (highest confidence)
            best_detection = max(ball_detections, key=lambda x: x[2])
            ball_position = BallPosition(
                x=best_detection[0],
                y=best_detection[1], 
                frame=frame_number,
                confidence=best_detection[2],
                timestamp=time.time()
            )
            
            # Update Kalman filter
            self.kalman_filter.update([ball_position.x, ball_position.y])
            
            # Get velocity and acceleration from Kalman filter
            state = self.kalman_filter.get_state()
            ball_position.velocity_x = state[2]
            ball_position.velocity_y = state[3]
            ball_position.acceleration_x = state[4]
            ball_position.acceleration_y = state[5]
            
        else:
            # Try to predict ball position
            predicted_pos = self.kalman_filter.predict()
            if predicted_pos is not None:
                ball_position = BallPosition(
                    x=predicted_pos[0],
                    y=predicted_pos[1],
                    frame=frame_number,
                    confidence=0.3,  # Lower confidence for predicted
                    timestamp=time.time()
                )
                
        # Add to trajectory buffer
        if ball_position:
            self.trajectory_buffer.append(ball_position)
            
        # Analyze trajectory for shot events
        shot_events = []
        if len(self.trajectory_buffer) >= self.min_shot_length:
            trajectory_list = list(self.trajectory_buffer)
            
            # Detect different types of events
            racket_hits = self.event_detector.detect_racket_hits(trajectory_list, players_data)
            wall_hits = self.event_detector.detect_wall_hits(trajectory_list)
            floor_bounces = self.event_detector.detect_floor_bounces(trajectory_list)
            
            shot_events = racket_hits + wall_hits + floor_bounces
            
        # Update active shots and create new shots
        self._update_shot_tracking(shot_events, frame_number)
        
        # Performance tracking
        processing_time = time.time() - start_time
        self.processing_times.append(processing_time)
        
        # Prepare results
        results = {
            'ball_position': ball_position,
            'events': shot_events,
            'active_shots': len(self.active_shots),
            'completed_shots': len(self.completed_shots),
            'processing_time': processing_time,
            'trajectory_length': len(self.trajectory_buffer)
        }
        
        return results
    
    def _update_shot_tracking(self, events: List[ShotEvent], frame_number: int):
        """Update shot tracking based on detected events"""
        
        # Check for new racket hits (start of new shots)
        racket_hits = [e for e in events if e.event_type == 'racket_hit']
        
        for hit in racket_hits:
            # Check if this starts a new shot
            if not self.active_shots or (frame_number - self.active_shots[-1].start_frame) > 30:
                new_shot = Shot(
                    shot_id=self.shot_id_counter,
                    start_frame=hit.frame,
                    end_frame=hit.frame,
                    trajectory=list(self.trajectory_buffer)[-50:],  # Recent trajectory
                    events=[hit],
                    player_id=hit.player_id or 0,
                    shot_type='unknown',
                    confidence=hit.confidence,
                    duration=0
                )
                self.active_shots.append(new_shot)
                self.shot_id_counter += 1
                
        # Update active shots with new events
        for shot in self.active_shots:
            shot.end_frame = frame_number
            shot.duration = (frame_number - shot.start_frame) / 30.0  # Assuming 30fps
            
            # Add relevant events to shot
            for event in events:
                if (event.frame >= shot.start_frame and 
                    event.frame <= shot.end_frame and 
                    event not in shot.events):
                    shot.events.append(event)
                    
            # Update trajectory
            recent_trajectory = [p for p in self.trajectory_buffer 
                               if p.frame >= shot.start_frame]
            shot.trajectory = recent_trajectory
            
        # Check for completed shots (no new events for a while)
        completed_indices = []
        for i, shot in enumerate(self.active_shots):
            if frame_number - shot.end_frame > 45:  # 1.5 seconds of inactivity
                self.completed_shots.append(shot)
                completed_indices.append(i)
                
        # Remove completed shots from active list
        for i in reversed(completed_indices):
            del self.active_shots[i]
    
    def get_shot_analysis(self) -> Dict:
        """Get comprehensive shot analysis"""
        total_shots = len(self.completed_shots)
        
        if total_shots == 0:
            return {'total_shots': 0, 'analysis': 'No shots detected yet'}
            
        # Analyze shot types
        shot_types = {}
        total_events = 0
        
        for shot in self.completed_shots:
            # Classify shot based on events
            shot_type = self._classify_shot(shot)
            shot_types[shot_type] = shot_types.get(shot_type, 0) + 1
            total_events += len(shot.events)
            
        # Performance metrics
        avg_processing_time = np.mean(self.processing_times) if self.processing_times else 0
        
        analysis = {
            'total_shots': total_shots,
            'shot_types': shot_types,
            'total_events': total_events,
            'average_events_per_shot': total_events / total_shots,
            'average_processing_time': avg_processing_time,
            'system_performance': 'Good' if avg_processing_time < 0.01 else 'Needs optimization'
        }
        
        return analysis
    
    def _classify_shot(self, shot: Shot) -> str:
        """Classify shot type based on events and trajectory"""
        events = shot.events
        
        # Count event types
        racket_hits = len([e for e in events if e.event_type == 'racket_hit'])
        wall_hits = len([e for e in events if e.event_type == 'wall_hit'])
        floor_bounces = len([e for e in events if e.event_type == 'floor_bounce'])
        
        # Basic classification
        if wall_hits == 0:
            return 'direct_shot'
        elif wall_hits == 1:
            return 'single_wall_shot'
        elif wall_hits >= 2:
            return 'multi_wall_shot'
        else:
            return 'unknown'
    
    def export_shot_data(self, filename: str):
        """Export shot data to JSON file"""
        export_data = {
            'completed_shots': [],
            'analysis': self.get_shot_analysis(),
            'export_timestamp': time.time()
        }
        
        for shot in self.completed_shots:
            shot_data = {
                'shot_id': shot.shot_id,
                'start_frame': shot.start_frame,
                'end_frame': shot.end_frame,
                'player_id': shot.player_id,
                'shot_type': shot.shot_type,
                'confidence': shot.confidence,
                'duration': shot.duration,
                'trajectory': [
                    {
                        'x': p.x, 'y': p.y, 'frame': p.frame,
                        'velocity_x': p.velocity_x, 'velocity_y': p.velocity_y,
                        'confidence': p.confidence
                    } for p in shot.trajectory
                ],
                'events': [
                    {
                        'type': e.event_type,
                        'frame': e.frame,
                        'position': e.position,
                        'velocity_before': e.velocity_before,
                        'velocity_after': e.velocity_after,
                        'confidence': e.confidence,
                        'player_id': e.player_id,
                        'wall_type': e.wall_type,
                        'impact_angle': e.impact_angle
                    } for e in shot.events
                ]
            }
            export_data['completed_shots'].append(shot_data)
            
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2)
            
        print(f"Shot data exported to {filename}")

# Integration function for existing pipeline
def create_enhanced_shot_detector(court_dimensions=(640, 360)) -> EnhancedShotDetector:
    """
    Factory function to create enhanced shot detector
    """
    print("🚀 Initializing Enhanced Ball Physics and Shot Detection System")
    print("=" * 60)
    print("✓ Physics-based trajectory modeling")
    print("✓ Advanced Kalman filtering for ball tracking")
    print("✓ Multi-modal event detection (racket, wall, floor)")
    print("✓ Real-time collision detection with confidence scoring")
    print("✓ Autonomous shot segmentation and classification")
    print("=" * 60)
    
    return EnhancedShotDetector(court_dimensions)

if __name__ == "__main__":
    # Test the enhanced shot detector
    detector = create_enhanced_shot_detector()
    
    # Simulate some ball detections
    test_detections = [
        [100, 200, 0.9],  # x, y, confidence
        [105, 205, 0.8],
        [110, 210, 0.85]
    ]
    
    for i, detection in enumerate(test_detections):
        result = detector.process_frame([detection], {}, i)
        print(f"Frame {i}: {result}")
    
    analysis = detector.get_shot_analysis()
    print(f"Analysis: {analysis}")
