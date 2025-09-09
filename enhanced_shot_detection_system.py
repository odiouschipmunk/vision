"""
Enhanced Shot Detection System for Squash Analysis
Improved detection of:
1. Ball hit from racket (racket contact detection)
2. Ball hit front wall (wall impact detection)  
3. Ball hit by opponent's racket (new shot detection)
4. Ball bounced to ground (floor bounce detection)

This system provides autonomous, accurate shot detection with clear event identification.
"""

import numpy as np
import cv2
import math
import time
from typing import List, Tuple, Dict, Optional, Any
from dataclasses import dataclass
from collections import deque
import json

@dataclass
class ShotEvent:
    """Represents a detected shot event with confidence scoring"""
    event_type: str  # 'racket_hit', 'wall_hit', 'floor_bounce', 'shot_start'
    frame_number: int
    ball_position: Tuple[float, float]
    confidence: float
    player_id: Optional[int] = None
    wall_type: Optional[str] = None  # 'front', 'side_left', 'side_right', 'back'
    velocity_before: Tuple[float, float] = (0, 0)
    velocity_after: Tuple[float, float] = (0, 0)
    trajectory_change: float = 0.0
    physics_score: float = 0.0
    details: Dict[str, Any] = None

    def __post_init__(self):
        if self.details is None:
            self.details = {}

@dataclass
class ShotSequence:
    """Complete shot sequence from start to end"""
    shot_id: int
    start_frame: int
    end_frame: Optional[int]
    player_id: int
    events: List[ShotEvent]
    trajectory: List[Tuple[float, float]]
    shot_type: str = "unknown"
    confidence: float = 0.0
    is_complete: bool = False

class PhysicsBasedDetector:
    """Physics-based detection engine for shot events"""
    
    def __init__(self, court_width=640, court_height=360):
        self.court_width = court_width
        self.court_height = court_height
        
        # Physics constants
        self.gravity = 9.81 * 30  # Adjust for pixel space
        self.ball_mass = 0.024  # kg (squash ball)
        self.restitution_wall = 0.85  # Energy retention after wall hit
        self.restitution_floor = 0.75  # Energy retention after floor bounce
        
        # Detection thresholds
        self.min_velocity_change = 15  # pixels/frame
        self.min_direction_change = math.pi / 6  # 30 degrees
        self.wall_proximity_threshold = 25  # pixels from wall
        self.floor_proximity_threshold = 0.8  # 80% down the court
        
        # Confidence scoring weights
        self.weights = {
            'velocity_change': 0.3,
            'direction_change': 0.25,
            'position_validity': 0.2,
            'physics_consistency': 0.15,
            'temporal_consistency': 0.1
        }

    def detect_racket_hit(self, trajectory: List[Tuple[float, float]], 
                         players_data: Dict, frame_number: int) -> ShotEvent:
        """
        Detect when ball is hit by racket using trajectory analysis and player proximity
        """
        if len(trajectory) < 5:
            return ShotEvent("racket_hit", frame_number, trajectory[-1], 0.0)
        
        # Analyze trajectory changes
        recent_positions = trajectory[-5:]
        velocity_change = self._calculate_velocity_change(recent_positions)
        direction_change = self._calculate_direction_change(recent_positions)
        
        # Player proximity analysis
        current_pos = trajectory[-1]
        player_proximity = self._analyze_player_proximity(current_pos, players_data)
        
        # Physics validation
        physics_score = self._validate_racket_hit_physics(recent_positions)
        
        # Calculate confidence
        confidence = self._calculate_racket_hit_confidence(
            velocity_change, direction_change, player_proximity, physics_score
        )
        
        # Determine hitting player
        hitting_player = self._identify_hitting_player(current_pos, players_data, confidence)
        
        return ShotEvent(
            event_type="racket_hit",
            frame_number=frame_number,
            ball_position=current_pos,
            confidence=confidence,
            player_id=hitting_player,
            velocity_before=self._get_velocity(recent_positions, -2),
            velocity_after=self._get_velocity(recent_positions, -1),
            trajectory_change=direction_change,
            physics_score=physics_score,
            details={
                'velocity_change': velocity_change,
                'direction_change': direction_change,
                'player_proximity': player_proximity
            }
        )

    def detect_wall_hit(self, trajectory: List[Tuple[float, float]], 
                       frame_number: int) -> ShotEvent:
        """
        Detect when ball hits front wall or side walls with improved accuracy
        """
        if len(trajectory) < 4:
            return ShotEvent("wall_hit", frame_number, trajectory[-1], 0.0)
        
        recent_positions = trajectory[-4:]
        current_pos = trajectory[-1]
        
        # Wall proximity analysis
        wall_distances = self._calculate_wall_distances(current_pos)
        closest_wall, min_distance = min(wall_distances.items(), key=lambda x: x[1])
        
        # Trajectory analysis for wall impact
        velocity_change = self._calculate_velocity_change(recent_positions)
        direction_change = self._calculate_direction_change(recent_positions)
        
        # Physics validation for wall bounce
        physics_score = self._validate_wall_hit_physics(recent_positions, closest_wall)
        
        # Confidence calculation
        confidence = self._calculate_wall_hit_confidence(
            min_distance, velocity_change, direction_change, physics_score
        )
        
        return ShotEvent(
            event_type="wall_hit",
            frame_number=frame_number,
            ball_position=current_pos,
            confidence=confidence,
            wall_type=closest_wall,
            velocity_before=self._get_velocity(recent_positions, -2),
            velocity_after=self._get_velocity(recent_positions, -1),
            trajectory_change=direction_change,
            physics_score=physics_score,
            details={
                'wall_distances': wall_distances,
                'velocity_change': velocity_change,
                'impact_angle': self._calculate_impact_angle(recent_positions, closest_wall)
            }
        )

    def detect_floor_bounce(self, trajectory: List[Tuple[float, float]], 
                           frame_number: int) -> ShotEvent:
        """
        Detect when ball bounces on the floor with physics validation
        """
        if len(trajectory) < 4:
            return ShotEvent("floor_bounce", frame_number, trajectory[-1], 0.0)
        
        recent_positions = trajectory[-4:]
        current_pos = trajectory[-1]
        
        # Floor proximity (bottom area of court)
        height_ratio = current_pos[1] / self.court_height
        floor_proximity = max(0, height_ratio - self.floor_proximity_threshold) / (1 - self.floor_proximity_threshold)
        
        # Bounce pattern detection (downward then upward motion)
        bounce_pattern = self._detect_bounce_pattern(recent_positions)
        
        # Velocity analysis for floor bounce
        velocity_change = self._calculate_velocity_change(recent_positions)
        
        # Physics validation
        physics_score = self._validate_floor_bounce_physics(recent_positions)
        
        # Confidence calculation
        confidence = self._calculate_floor_bounce_confidence(
            floor_proximity, bounce_pattern, velocity_change, physics_score
        )
        
        return ShotEvent(
            event_type="floor_bounce",
            frame_number=frame_number,
            ball_position=current_pos,
            confidence=confidence,
            velocity_before=self._get_velocity(recent_positions, -2),
            velocity_after=self._get_velocity(recent_positions, -1),
            physics_score=physics_score,
            details={
                'floor_proximity': floor_proximity,
                'bounce_pattern_score': bounce_pattern,
                'velocity_change': velocity_change,
                'height_ratio': height_ratio
            }
        )

    def detect_shot_transition(self, current_shot: Optional[ShotSequence], 
                              new_event: ShotEvent, 
                              frame_number: int) -> Optional[ShotSequence]:
        """
        Detect when a new shot starts (opponent hits ball)
        """
        if new_event.event_type != "racket_hit" or new_event.confidence < 0.6:
            return None
        
        # Check if this is a different player than current shot
        if current_shot and current_shot.player_id != new_event.player_id:
            # New shot detected - opponent hit the ball
            new_shot = ShotSequence(
                shot_id=int(time.time() * 1000),  # Unique ID
                start_frame=frame_number,
                end_frame=None,
                player_id=new_event.player_id,
                events=[new_event],
                trajectory=[new_event.ball_position]
            )
            
            # Mark previous shot as complete
            if current_shot:
                current_shot.end_frame = frame_number - 1
                current_shot.is_complete = True
            
            return new_shot
        
        return None

    # Helper methods for calculations
    
    def _calculate_velocity_change(self, positions: List[Tuple[float, float]]) -> float:
        """Calculate magnitude of velocity change"""
        if len(positions) < 3:
            return 0.0
        
        # Calculate velocities
        v1 = self._get_velocity(positions, -3, -2)
        v2 = self._get_velocity(positions, -2, -1)
        
        # Magnitude change
        mag1 = math.sqrt(v1[0]**2 + v1[1]**2)
        mag2 = math.sqrt(v2[0]**2 + v2[1]**2)
        
        return abs(mag2 - mag1)

    def _calculate_direction_change(self, positions: List[Tuple[float, float]]) -> float:
        """Calculate direction change in radians"""
        if len(positions) < 3:
            return 0.0
        
        v1 = self._get_velocity(positions, -3, -2)
        v2 = self._get_velocity(positions, -2, -1)
        
        # Calculate angles
        angle1 = math.atan2(v1[1], v1[0])
        angle2 = math.atan2(v2[1], v2[0])
        
        # Angle difference
        diff = abs(angle2 - angle1)
        return min(diff, 2*math.pi - diff)

    def _get_velocity(self, positions: List[Tuple[float, float]], 
                     start_idx: int, end_idx: Optional[int] = None) -> Tuple[float, float]:
        """Get velocity vector between two positions"""
        if end_idx is None:
            end_idx = start_idx + 1
        
        if len(positions) <= max(abs(start_idx), abs(end_idx)):
            return (0.0, 0.0)
        
        p1 = positions[start_idx]
        p2 = positions[end_idx]
        
        return (p2[0] - p1[0], p2[1] - p1[1])

    def _calculate_wall_distances(self, position: Tuple[float, float]) -> Dict[str, float]:
        """Calculate distances to all walls"""
        x, y = position
        return {
            'front': y,  # Top wall (front wall in squash)
            'back': self.court_height - y,
            'left': x,
            'right': self.court_width - x
        }

    def _analyze_player_proximity(self, ball_pos: Tuple[float, float], 
                                 players_data: Dict) -> Dict[str, float]:
        """Analyze proximity to each player"""
        proximities = {}
        
        for player_id, player in players_data.items():
            if player and hasattr(player, 'get_latest_pose'):
                pose = player.get_latest_pose()
                if pose and hasattr(pose, 'xyn') and len(pose.xyn) > 0:
                    keypoints = pose.xyn[0]
                    
                    # Use multiple keypoints for better accuracy
                    relevant_keypoints = [9, 10, 7, 8]  # wrists and elbows
                    min_distance = float('inf')
                    
                    for kp_idx in relevant_keypoints:
                        if kp_idx < len(keypoints):
                            kp = keypoints[kp_idx]
                            if len(kp) >= 2 and kp[0] > 0 and kp[1] > 0:
                                kp_pos = (kp[0] * self.court_width, kp[1] * self.court_height)
                                distance = math.sqrt(
                                    (ball_pos[0] - kp_pos[0])**2 + 
                                    (ball_pos[1] - kp_pos[1])**2
                                )
                                min_distance = min(min_distance, distance)
                    
                    proximities[f'player_{player_id}'] = min_distance if min_distance != float('inf') else 999
        
        return proximities

    def _validate_racket_hit_physics(self, positions: List[Tuple[float, float]]) -> float:
        """Validate racket hit using physics principles"""
        if len(positions) < 4:
            return 0.0
        
        # Check for sudden energy increase (typical of racket hit)
        velocities = []
        for i in range(len(positions) - 1):
            v = self._get_velocity(positions, i, i + 1)
            speed = math.sqrt(v[0]**2 + v[1]**2)
            velocities.append(speed)
        
        if len(velocities) < 3:
            return 0.0
        
        # Look for acceleration pattern typical of racket hit
        before_speed = velocities[-3]
        after_speed = velocities[-1]
        
        # Racket hits typically increase ball speed
        acceleration = (after_speed - before_speed) / max(before_speed, 1)
        
        # Score based on reasonable acceleration range
        if acceleration > 0.5:  # Significant speed increase
            return min(1.0, acceleration / 2.0)
        
        return max(0.0, acceleration + 0.5)

    def _validate_wall_hit_physics(self, positions: List[Tuple[float, float]], 
                                  wall_type: str) -> float:
        """Validate wall hit using physics principles"""
        if len(positions) < 3:
            return 0.0
        
        # Check velocity direction relative to wall
        v_before = self._get_velocity(positions, -3, -2)
        v_after = self._get_velocity(positions, -2, -1)
        
        # Wall hit should reverse appropriate velocity component
        if wall_type in ['front', 'back']:
            # Vertical component should reverse
            if v_before[1] * v_after[1] < 0:  # Opposite signs
                return min(1.0, abs(v_after[1] / max(abs(v_before[1]), 1)) * self.restitution_wall)
        elif wall_type in ['left', 'right']:
            # Horizontal component should reverse
            if v_before[0] * v_after[0] < 0:  # Opposite signs
                return min(1.0, abs(v_after[0] / max(abs(v_before[0]), 1)) * self.restitution_wall)
        
        return 0.0

    def _validate_floor_bounce_physics(self, positions: List[Tuple[float, float]]) -> float:
        """Validate floor bounce using physics principles"""
        if len(positions) < 3:
            return 0.0
        
        # Check for downward then upward motion
        v1 = self._get_velocity(positions, -3, -2)
        v2 = self._get_velocity(positions, -2, -1)
        
        # Floor bounce: downward motion followed by upward motion
        if v1[1] > 0 and v2[1] < 0:  # Y increases downward in screen coords
            # Check energy conservation (with loss)
            speed_before = math.sqrt(v1[0]**2 + v1[1]**2)
            speed_after = math.sqrt(v2[0]**2 + v2[1]**2)
            
            energy_ratio = speed_after / max(speed_before, 1)
            
            # Should be less than 1 due to energy loss, but not too small
            if 0.3 <= energy_ratio <= self.restitution_floor:
                return energy_ratio / self.restitution_floor
        
        return 0.0

    def _detect_bounce_pattern(self, positions: List[Tuple[float, float]]) -> float:
        """Detect bounce pattern in trajectory"""
        if len(positions) < 3:
            return 0.0
        
        # Look for V-shaped pattern in Y coordinates
        y_coords = [pos[1] for pos in positions]
        
        # Check if middle position is lowest (highest Y value in screen coords)
        if len(y_coords) >= 3:
            if y_coords[-2] > y_coords[-3] and y_coords[-1] < y_coords[-2]:
                # Calculate "sharpness" of bounce
                descent = y_coords[-2] - y_coords[-3]
                ascent = y_coords[-2] - y_coords[-1]
                
                if descent > 0 and ascent > 0:
                    return min(1.0, (descent + ascent) / 20.0)  # Normalize
        
        return 0.0

    def _calculate_impact_angle(self, positions: List[Tuple[float, float]], 
                               wall_type: str) -> float:
        """Calculate impact angle with wall"""
        if len(positions) < 2:
            return 0.0
        
        velocity = self._get_velocity(positions, -2, -1)
        
        if wall_type in ['front', 'back']:
            # Angle with horizontal (wall is vertical)
            return math.atan2(abs(velocity[1]), abs(velocity[0]))
        else:
            # Angle with vertical (wall is horizontal)
            return math.atan2(abs(velocity[0]), abs(velocity[1]))

    def _identify_hitting_player(self, ball_pos: Tuple[float, float], 
                                players_data: Dict, min_confidence: float) -> Optional[int]:
        """Identify which player hit the ball"""
        if min_confidence < 0.5:
            return None
        
        proximities = self._analyze_player_proximity(ball_pos, players_data)
        
        if not proximities:
            return None
        
        # Find closest player
        closest_player = min(proximities.items(), key=lambda x: x[1])
        player_key, distance = closest_player
        
        # Extract player ID
        if distance < 80:  # Reasonable hitting distance
            try:
                player_id = int(player_key.split('_')[1])
                return player_id
            except:
                pass
        
        return None

    # Confidence calculation methods
    
    def _calculate_racket_hit_confidence(self, velocity_change: float, 
                                        direction_change: float,
                                        player_proximity: Dict[str, float], 
                                        physics_score: float) -> float:
        """Calculate confidence for racket hit detection"""
        
        # Velocity change component
        vel_score = min(1.0, velocity_change / 30.0)  # Normalize to 30 pixels/frame
        
        # Direction change component
        dir_score = min(1.0, direction_change / (math.pi / 2))  # Normalize to 90 degrees
        
        # Proximity component
        prox_score = 0.0
        if player_proximity:
            min_distance = min(player_proximity.values())
            prox_score = max(0.0, (100 - min_distance) / 100.0)  # Normalize to 100 pixels
        
        # Combine scores
        confidence = (
            self.weights['velocity_change'] * vel_score +
            self.weights['direction_change'] * dir_score +
            self.weights['position_validity'] * prox_score +
            self.weights['physics_consistency'] * physics_score
        )
        
        return min(1.0, confidence)

    def _calculate_wall_hit_confidence(self, distance: float, velocity_change: float,
                                      direction_change: float, physics_score: float) -> float:
        """Calculate confidence for wall hit detection"""
        
        # Distance component (closer to wall = higher confidence)
        dist_score = max(0.0, (self.wall_proximity_threshold - distance) / self.wall_proximity_threshold)
        
        # Velocity change component
        vel_score = min(1.0, velocity_change / 25.0)
        
        # Direction change component
        dir_score = min(1.0, direction_change / (math.pi / 3))  # 60 degrees
        
        # Combine scores
        confidence = (
            self.weights['position_validity'] * dist_score +
            self.weights['velocity_change'] * vel_score +
            self.weights['direction_change'] * dir_score +
            self.weights['physics_consistency'] * physics_score
        )
        
        return min(1.0, confidence)

    def _calculate_floor_bounce_confidence(self, floor_proximity: float, 
                                          bounce_pattern: float,
                                          velocity_change: float, 
                                          physics_score: float) -> float:
        """Calculate confidence for floor bounce detection"""
        
        # Combine all components
        confidence = (
            self.weights['position_validity'] * floor_proximity +
            0.3 * bounce_pattern +  # High weight for bounce pattern
            self.weights['velocity_change'] * min(1.0, velocity_change / 20.0) +
            self.weights['physics_consistency'] * physics_score
        )
        
        return min(1.0, confidence)


class EnhancedShotDetectionSystem:
    """
    Main enhanced shot detection system that integrates all detection methods
    """
    
    def __init__(self, court_width=640, court_height=360):
        self.physics_detector = PhysicsBasedDetector(court_width, court_height)
        self.active_shots = []
        self.completed_shots = []
        self.shot_id_counter = 0
        self.frame_history = deque(maxlen=30)  # Keep 30 frames of history
        
        # Detection parameters
        self.min_trajectory_length = 5
        self.shot_timeout_frames = 90  # 3 seconds at 30fps
        
    def process_frame(self, ball_position: Tuple[float, float], 
                     players_data: Dict, frame_number: int) -> Dict[str, Any]:
        """
        Process a single frame and detect shot events
        
        Returns:
            Dictionary containing detected events and current shot status
        """
        
        # Add to frame history
        self.frame_history.append({
            'frame': frame_number,
            'ball_pos': ball_position,
            'players': players_data
        })
        
        # Get trajectory from recent frames
        trajectory = [frame['ball_pos'] for frame in self.frame_history 
                     if frame['ball_pos'] is not None]
        
        if len(trajectory) < self.min_trajectory_length:
            return self._create_empty_result(frame_number)
        
        # Detect various events
        events = []
        
        # 1. Detect racket hits
        racket_hit = self.physics_detector.detect_racket_hit(
            trajectory, players_data, frame_number
        )
        if racket_hit.confidence > 0.5:
            events.append(racket_hit)
        
        # 2. Detect wall hits
        wall_hit = self.physics_detector.detect_wall_hit(trajectory, frame_number)
        if wall_hit.confidence > 0.6:
            events.append(wall_hit)
        
        # 3. Detect floor bounces
        floor_bounce = self.physics_detector.detect_floor_bounce(trajectory, frame_number)
        if floor_bounce.confidence > 0.5:
            events.append(floor_bounce)
        
        # 4. Update shot sequences
        shot_updates = self._update_shot_sequences(events, frame_number, trajectory)
        
        # 5. Clean up old shots
        self._cleanup_old_shots(frame_number)
        
        return {
            'frame_number': frame_number,
            'events': events,
            'active_shots': len(self.active_shots),
            'completed_shots': len(self.completed_shots),
            'shot_updates': shot_updates,
            'ball_position': ball_position,
            'trajectory_length': len(trajectory)
        }
    
    def _update_shot_sequences(self, events: List[ShotEvent], 
                              frame_number: int, 
                              trajectory: List[Tuple[float, float]]) -> Dict[str, Any]:
        """Update shot sequences based on detected events"""
        
        updates = {
            'new_shots': [],
            'updated_shots': [],
            'completed_shots': []
        }
        
        # Check for new shots (racket hits)
        for event in events:
            if event.event_type == "racket_hit" and event.confidence > 0.7:
                
                # Check if this starts a new shot
                current_shot = self.active_shots[-1] if self.active_shots else None
                
                new_shot = self.physics_detector.detect_shot_transition(
                    current_shot, event, frame_number
                )
                
                if new_shot:
                    self.active_shots.append(new_shot)
                    updates['new_shots'].append(new_shot)
                    
                    # Complete previous shot if it exists
                    if current_shot:
                        current_shot.is_complete = True
                        self.completed_shots.append(current_shot)
                        self.active_shots.remove(current_shot)
                        updates['completed_shots'].append(current_shot)
        
        # Update active shots with new events
        for shot in self.active_shots:
            shot.trajectory.extend(trajectory[-3:])  # Add recent positions
            
            for event in events:
                if event.confidence > 0.5:
                    shot.events.append(event)
                    updates['updated_shots'].append(shot.shot_id)
        
        return updates
    
    def _cleanup_old_shots(self, frame_number: int):
        """Clean up shots that have timed out"""
        
        shots_to_complete = []
        
        for shot in self.active_shots:
            if frame_number - shot.start_frame > self.shot_timeout_frames:
                shot.is_complete = True
                shot.end_frame = frame_number
                shots_to_complete.append(shot)
        
        for shot in shots_to_complete:
            self.active_shots.remove(shot)
            self.completed_shots.append(shot)
    
    def _create_empty_result(self, frame_number: int) -> Dict[str, Any]:
        """Create empty result for insufficient data"""
        return {
            'frame_number': frame_number,
            'events': [],
            'active_shots': len(self.active_shots),
            'completed_shots': len(self.completed_shots),
            'shot_updates': {'new_shots': [], 'updated_shots': [], 'completed_shots': []},
            'ball_position': None,
            'trajectory_length': 0
        }
    
    def get_current_status(self) -> Dict[str, Any]:
        """Get current system status"""
        
        return {
            'total_completed_shots': len(self.completed_shots),
            'active_shots': len(self.active_shots),
            'active_shot_details': [
                {
                    'shot_id': shot.shot_id,
                    'player_id': shot.player_id,
                    'start_frame': shot.start_frame,
                    'events_count': len(shot.events),
                    'trajectory_length': len(shot.trajectory)
                }
                for shot in self.active_shots
            ],
            'recent_events': [
                {
                    'type': event.event_type,
                    'frame': event.frame_number,
                    'confidence': event.confidence,
                    'player_id': event.player_id
                }
                for shot in self.active_shots[-3:] 
                for event in shot.events[-5:]  # Last 5 events from last 3 shots
            ]
        }
    
    def export_shot_data(self, filepath: str = "/tmp/enhanced_shot_data.json"):
        """Export detected shot data for analysis"""
        
        data = {
            'metadata': {
                'total_shots': len(self.completed_shots),
                'active_shots': len(self.active_shots),
                'export_timestamp': time.time()
            },
            'completed_shots': [
                {
                    'shot_id': shot.shot_id,
                    'player_id': shot.player_id,
                    'start_frame': shot.start_frame,
                    'end_frame': shot.end_frame,
                    'duration_frames': shot.end_frame - shot.start_frame if shot.end_frame else None,
                    'events': [
                        {
                            'type': event.event_type,
                            'frame': event.frame_number,
                            'position': event.ball_position,
                            'confidence': event.confidence,
                            'player_id': event.player_id,
                            'wall_type': event.wall_type,
                            'details': event.details
                        }
                        for event in shot.events
                    ],
                    'trajectory_length': len(shot.trajectory)
                }
                for shot in self.completed_shots
            ]
        }
        
        try:
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"✅ Enhanced shot data exported to {filepath}")
        except Exception as e:
            print(f"❌ Failed to export shot data: {e}")
            
        return data


def create_enhanced_shot_detector(court_width=640, court_height=360) -> EnhancedShotDetectionSystem:
    """
    Factory function to create enhanced shot detection system
    """
    return EnhancedShotDetectionSystem(court_width, court_height)


# Example usage and testing functions
def demo_enhanced_detection():
    """Demonstrate the enhanced shot detection system"""
    
    print("🎾 Enhanced Shot Detection System Demo")
    print("=" * 50)
    
    # Create detector
    detector = create_enhanced_shot_detector()
    
    # Simulate ball trajectory with various events
    # This would normally come from actual ball tracking
    simulated_trajectory = [
        # Player 1 serves
        (100, 50), (110, 52), (120, 55), (130, 58), (140, 62),
        # Ball hits front wall
        (20, 80), (25, 85), (30, 90), (35, 95), (40, 100),
        # Ball travels to Player 2
        (200, 120), (250, 140), (300, 160), (350, 180), (400, 200),
        # Player 2 hits back
        (420, 190), (430, 185), (440, 180), (450, 175), (460, 170),
        # Ball hits side wall
        (620, 150), (615, 155), (610, 160), (605, 165), (600, 170),
        # Ball bounces on floor
        (500, 300), (490, 310), (480, 305), (470, 300), (460, 295)
    ]
    
    # Mock players data
    mock_players = {
        1: type('Player', (), {
            'get_latest_pose': lambda: type('Pose', (), {
                'xyn': [[(0.15, 0.3, 0.9), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), 
                        (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0),
                        (0.0, 0.0, 0.0), (0.12, 0.28, 0.8), (0.18, 0.32, 0.8),
                        (0.10, 0.26, 0.85), (0.20, 0.34, 0.85)]]  # Mock keypoints
            })()
        })(),
        2: type('Player', (), {
            'get_latest_pose': lambda: type('Pose', (), {
                'xyn': [[(0.65, 0.4, 0.9), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0),
                        (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0),
                        (0.0, 0.0, 0.0), (0.62, 0.38, 0.8), (0.68, 0.42, 0.8),
                        (0.60, 0.36, 0.85), (0.70, 0.44, 0.85)]]  # Mock keypoints
            })()
        })()
    }
    
    # Process each frame
    results = []
    for frame_num, ball_pos in enumerate(simulated_trajectory):
        result = detector.process_frame(ball_pos, mock_players, frame_num)
        results.append(result)
        
        # Print significant events
        if result['events']:
            for event in result['events']:
                if event.confidence > 0.5:
                    print(f"Frame {frame_num}: {event.event_type.upper()} "
                          f"(confidence: {event.confidence:.2f}) "
                          f"at position {event.ball_position}")
                    if event.player_id:
                        print(f"  → Player {event.player_id}")
                    if event.wall_type:
                        print(f"  → Wall: {event.wall_type}")
    
    # Print final status
    print("\n📊 Final Detection Summary:")
    status = detector.get_current_status()
    print(f"  • Total completed shots: {status['total_completed_shots']}")
    print(f"  • Active shots: {status['active_shots']}")
    print(f"  • Total events detected: {sum(len(r['events']) for r in results)}")
    
    # Export data
    detector.export_shot_data()
    
    return detector, results


if __name__ == "__main__":
    # Run demo
    demo_enhanced_detection()