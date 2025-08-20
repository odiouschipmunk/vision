#!/usr/bin/env python3
"""
🎯 ENHANCED SHOT DETECTION UI DEMO
Demonstrates the improved shot detection system with clear UI feedback
"""

import cv2
import numpy as np
import math
import time
from collections import deque
import json

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

class EnhancedShotDetector:
    """Enhanced shot detection with clear event tracking"""
    
    def __init__(self):
        self.active_shots = []
        self.completed_shots = []
        self.shot_id_counter = 0
        self.last_hit_frame = 0
        
    def detect_player_hit(self, ball_pos, players, frame_count):
        """Detect when a player hits the ball"""
        # Simplified hit detection for demo
        if not ball_pos or frame_count - self.last_hit_frame < 30:
            return None, 0.0, 'none'
            
        # Mock player hit detection
        player_id = 1 if frame_count % 100 < 50 else 2
        confidence = 0.8
        hit_type = 'racket_hit'
        
        return player_id, confidence, hit_type
    
    def detect_wall_hit(self, trajectory):
        """Detect wall hits"""
        if len(trajectory) < 3:
            return False, 'none', 0.0
            
        current_pos = trajectory[-1]
        x, y = current_pos[0], current_pos[1]
        
        # Check proximity to walls
        wall_threshold = 30
        wall_type = 'none'
        confidence = 0.0
        
        if y < wall_threshold:
            wall_type = 'front_wall'
            confidence = 0.9
        elif y > 330:
            wall_type = 'back_wall' 
            confidence = 0.8
        elif x < wall_threshold:
            wall_type = 'left_wall'
            confidence = 0.7
        elif x > 610:
            wall_type = 'right_wall'
            confidence = 0.7
            
        return confidence > 0.6, wall_type, confidence
    
    def detect_floor_bounce(self, trajectory):
        """Detect floor bounces"""
        if len(trajectory) < 5:
            return False, 0.0
            
        # Simple bounce detection based on Y movement
        recent_y = [pos[1] for pos in trajectory[-5:]]
        
        # Look for down-then-up pattern
        for i in range(2, len(recent_y)):
            if recent_y[i-2] < recent_y[i-1] and recent_y[i-1] > recent_y[i]:
                # Potential bounce detected
                return True, 0.8
                
        return False, 0.0

def draw_enhanced_ui(frame, ball_tracker, shot_detector, frame_count):
    """Draw enhanced UI with clear shot information"""
    
    frame_height, frame_width = frame.shape[:2]
    
    # 🎯 ENHANCED SHOT EVENT DISPLAY PANEL
    panel_x = frame_width - 400
    panel_y = 10
    panel_width = 380
    panel_height = 200
    
    # No background rectangle - clean text overlay
    
    # Panel title
    cv2.putText(frame, "🎾 SHOT DETECTION STATUS", 
              (panel_x + 10, panel_y + 20),
              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    
    # Ball tracking status
    if ball_tracker.is_tracking():
        tracking_status = "TRACKING"
        tracking_color = (0, 255, 0)
        velocity = ball_tracker.get_velocity()
        velocity_mag = math.sqrt(velocity[0]**2 + velocity[1]**2)
        cv2.putText(frame, f"Ball: {tracking_status} | V: {velocity_mag:.1f}px/f", 
                  (panel_x + 10, panel_y + 40),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.4, tracking_color, 1)
    else:
        cv2.putText(frame, "Ball: SEARCHING", 
                  (panel_x + 10, panel_y + 40),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 165, 0), 1)
    
    # Shot information
    if shot_detector.active_shots:
        current_shot = shot_detector.active_shots[-1]
        shot_id = current_shot.get('id', 0)
        shot_player = current_shot.get('player_who_hit', 0)
        
        # Shot header
        cv2.putText(frame, f"ACTIVE SHOT #{shot_id}", 
                  (panel_x + 10, panel_y + 65),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        
        # Player info
        cv2.putText(frame, f"Player {shot_player}", 
                  (panel_x + 10, panel_y + 85),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Shot events timeline
        y_offset = 105
        if current_shot.get('start_frame'):
            cv2.putText(frame, f"✓ Hit at frame {current_shot['start_frame']}", 
                      (panel_x + 10, panel_y + y_offset),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
            y_offset += 15
        
        if current_shot.get('wall_hit_frame'):
            wall_type = current_shot.get('wall_type', 'unknown')
            cv2.putText(frame, f"✓ {wall_type.replace('_', ' ').title()} hit at frame {current_shot['wall_hit_frame']}", 
                      (panel_x + 10, panel_y + y_offset),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 165, 0), 1)
            y_offset += 15
        
        if current_shot.get('floor_hit_frame'):
            cv2.putText(frame, f"✓ Bounce at frame {current_shot['floor_hit_frame']}", 
                      (panel_x + 10, panel_y + y_offset),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 255), 1)
    else:
        cv2.putText(frame, "No active shot", 
                  (panel_x + 10, panel_y + 65),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.4, (128, 128, 128), 1)
    
    # Draw ball trajectory
    trajectory = ball_tracker.get_trajectory()
    if len(trajectory) > 1:
        # Draw trajectory with numbered points
        for i in range(1, len(trajectory)):
            pt1 = (int(trajectory[i-1][0]), int(trajectory[i-1][1]))
            pt2 = (int(trajectory[i][0]), int(trajectory[i][1]))
            cv2.line(frame, pt1, pt2, (0, 255, 255), 2)
            
            # Add numbered markers every 5th point
            if i % 5 == 0:
                cv2.circle(frame, pt2, 3, (255, 255, 255), -1)
                cv2.putText(frame, str(i), (pt2[0] + 5, pt2[1] - 5),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        
        # Current ball position
        current_pos = trajectory[-1]
        cv2.circle(frame, (int(current_pos[0]), int(current_pos[1])), 8, (0, 255, 0), 3)

def demo_shot_detection():
    """Run demo of enhanced shot detection"""
    
    # Create demo frame
    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    
    # Initialize trackers
    ball_tracker = SmoothedBallTracker()
    shot_detector = EnhancedShotDetector()
    
    # Demo trajectory - simulate ball movement
    demo_trajectory = [
        [100, 200], [110, 190], [120, 180], [130, 170], [140, 160],  # Moving up-right
        [150, 50], [160, 55], [170, 60], [180, 65], [190, 70],       # Hit front wall
        [200, 80], [210, 90], [220, 100], [230, 110], [240, 120],    # Bouncing back
        [250, 250], [260, 260], [270, 270], [280, 280], [290, 290]   # Moving down
    ]
    
    print("🎯 Enhanced Shot Detection Demo")
    print("=" * 50)
    
    for frame_count, ball_pos in enumerate(demo_trajectory):
        # Clear frame
        frame.fill(0)
        
        # Add ball detection to tracker
        confidence = 0.9 if frame_count < 15 else 0.7  # Lower confidence later
        smoothed_pos = ball_tracker.add_detection(ball_pos, confidence, frame_count)
        
        if smoothed_pos:
            trajectory = ball_tracker.get_trajectory()
            
            # Check for events
            player_hit, hit_conf, hit_type = shot_detector.detect_player_hit(smoothed_pos, None, frame_count)
            if player_hit and not shot_detector.active_shots:
                # Start new shot
                shot_detector.shot_id_counter += 1
                new_shot = {
                    'id': shot_detector.shot_id_counter,
                    'player_who_hit': player_hit,
                    'start_frame': frame_count,
                    'trajectory': trajectory.copy()
                }
                shot_detector.active_shots.append(new_shot)
                shot_detector.last_hit_frame = frame_count
                print(f"🎯 SHOT #{shot_detector.shot_id_counter} started by Player {player_hit} at frame {frame_count}")
            
            # Check for wall hits
            if shot_detector.active_shots:
                current_shot = shot_detector.active_shots[-1]
                wall_hit, wall_type, wall_conf = shot_detector.detect_wall_hit(trajectory)
                if wall_hit and not current_shot.get('wall_hit_frame'):
                    current_shot['wall_hit_frame'] = frame_count
                    current_shot['wall_type'] = wall_type
                    print(f"🎯 {wall_type.upper()} HIT detected at frame {frame_count}")
                
                # Check for floor bounces
                floor_hit, floor_conf = shot_detector.detect_floor_bounce(trajectory)
                if floor_hit and not current_shot.get('floor_hit_frame'):
                    current_shot['floor_hit_frame'] = frame_count
                    print(f"🎯 FLOOR BOUNCE detected at frame {frame_count}")
                    
                    # Complete the shot
                    shot_detector.completed_shots.append(shot_detector.active_shots.pop())
                    print(f"🎯 SHOT #{current_shot['id']} COMPLETED")
        
        # Draw enhanced UI
        draw_enhanced_ui(frame, ball_tracker, shot_detector, frame_count)
        
        # Add frame info
        cv2.putText(frame, f"Frame: {frame_count}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Display frame
        cv2.imshow("Enhanced Shot Detection Demo", frame)
        
        # Wait for key press or auto-advance
        key = cv2.waitKey(500) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            cv2.waitKey(0)  # Pause on spacebar
    
    cv2.destroyAllWindows()
    
    print("\n🎯 Demo completed!")
    print(f"Total shots detected: {len(shot_detector.completed_shots)}")
    for shot in shot_detector.completed_shots:
        print(f"  Shot #{shot['id']}: Player {shot['player_who_hit']} -> {shot.get('wall_type', 'no wall')} -> Floor")

if __name__ == "__main__":
    demo_shot_detection()
