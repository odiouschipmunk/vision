import math
import numpy as np
import cv2

class EnhancedBallTracker:
    """
    Enhanced ball tracker with state management, prediction, and validation
    """
    def __init__(self, max_history=100, confidence_threshold=0.3):
        self.trajectory = []
        self.max_history = max_history
        self.confidence_threshold = confidence_threshold
        self.lost_frames = 0
        self.max_lost_frames = 10
        self.last_valid_detection = None
        self.confidence_history = []
        self.adaptive_threshold = confidence_threshold
    
    def update_adaptive_threshold(self):
        """Update adaptive confidence threshold based on recent detection quality"""
        if len(self.confidence_history) > 10:
            # Keep only recent history
            self.confidence_history = self.confidence_history[-20:]
            
            # Calculate average confidence
            avg_confidence = sum(self.confidence_history) / len(self.confidence_history)
            
            # Adjust threshold - lower if we're getting good detections consistently
            if avg_confidence > 0.7:
                self.adaptive_threshold = max(0.2, self.confidence_threshold - 0.1)
            elif avg_confidence > 0.5:
                self.adaptive_threshold = self.confidence_threshold
            else:
                self.adaptive_threshold = min(0.6, self.confidence_threshold + 0.1)
    
    def validate_detection(self, x, y, w, h, confidence):
        """Validate if a detection is likely a real ball"""
        # Use adaptive threshold based on recent confidence history
        self.update_adaptive_threshold()
        
        # Confidence check with adaptive threshold
        if confidence < self.adaptive_threshold:
            return False
        
        # Size validation
        ball_size = w * h
        if ball_size < 15 or ball_size > 1000:
            return False
        
        # Aspect ratio check
        aspect_ratio = w / h if h > 0 else float('inf')
        if aspect_ratio < 0.4 or aspect_ratio > 2.5:
            return False
        
        # Temporal consistency
        if len(self.trajectory) > 0:
            last_x, last_y, _ = self.trajectory[-1]
            distance = math.sqrt((x - last_x)**2 + (y - last_y)**2)
            
            # Adaptive distance threshold based on recent movement
            if len(self.trajectory) > 3:
                recent_distances = []
                for i in range(1, min(6, len(self.trajectory))):
                    px, py, _ = self.trajectory[-i-1]
                    curr_x, curr_y, _ = self.trajectory[-i]
                    recent_distances.append(math.sqrt((curr_x - px)**2 + (curr_y - py)**2))
                
                avg_distance = sum(recent_distances) / len(recent_distances)
                max_distance = max(80, avg_distance * 3)  # Adaptive threshold
            else:
                max_distance = 80
            
            if distance > max_distance:
                return False
        
        return True
    
    def update(self, detections, frame_number):
        """Update tracker with new detections"""
        valid_detections = []
        
        # Filter valid detections
        for detection in detections:
            x, y, w, h, conf = detection
            if self.validate_detection(x, y, w, h, conf):
                valid_detections.append(detection)
        if valid_detections:
            # Choose best detection
            best_detection = self.select_best_detection(valid_detections)
            x, y, w, h, conf = best_detection
            
            # Record confidence for adaptive threshold
            self.confidence_history.append(conf)
            
            # Add to trajectory
            self.trajectory.append([x, y, frame_number])
            self.last_valid_detection = best_detection
            self.lost_frames = 0
            
            # Maintain trajectory size
            if len(self.trajectory) > self.max_history:
                self.trajectory = self.trajectory[-self.max_history:]
            
            return True, (x, y)
        else:
            # No valid detection
            self.lost_frames += 1
            
            # Try to predict position
            if self.lost_frames <= self.max_lost_frames and len(self.trajectory) > 2:
                predicted_pos = self.predict_position(frame_number)
                return False, predicted_pos
            
            return False, None
    
    def select_best_detection(self, detections):
        """Select best detection from multiple candidates"""
        if len(detections) == 1:
            return detections[0]
        
        # Score detections based on confidence and temporal consistency
        scored_detections = []
        
        for detection in detections:
            x, y, w, h, conf = detection
            score = conf
            
            # Temporal consistency bonus
            if len(self.trajectory) > 0:
                last_x, last_y, _ = self.trajectory[-1]
                distance = math.sqrt((x - last_x)**2 + (y - last_y)**2)
                distance_penalty = distance / 100.0
                score = conf - distance_penalty
            
            scored_detections.append((score, detection))
        
        # Return detection with highest score
        return max(scored_detections, key=lambda x: x[0])[1]
    
    def predict_position(self, frame_number):
        """Predict ball position using recent trajectory"""
        if len(self.trajectory) < 3:
            return None
        
        # Simple linear prediction using last few points
        recent_points = self.trajectory[-3:]
        
        # Calculate velocity from recent points
        x1, y1, t1 = recent_points[-2]
        x2, y2, t2 = recent_points[-1]
        
        if t2 == t1:
            return (x2, y2)
        
        vx = (x2 - x1) / (t2 - t1)
        vy = (y2 - y1) / (t2 - t1)
        
        # Predict position
        dt = frame_number - t2
        pred_x = x2 + vx * dt
        pred_y = y2 + vy * dt
        
        return (int(pred_x), int(pred_y))
    
    def get_smoothed_trajectory(self, window_size=5):
        """Get smoothed trajectory for visualization"""
        if len(self.trajectory) < window_size:
            return self.trajectory
        
        smoothed = []
        for i in range(len(self.trajectory)):
            start_idx = max(0, i - window_size // 2)
            end_idx = min(len(self.trajectory), i + window_size // 2 + 1)
            
            window_points = self.trajectory[start_idx:end_idx]
            avg_x = sum(p[0] for p in window_points) / len(window_points)
            avg_y = sum(p[1] for p in window_points) / len(window_points)
            
            smoothed.append([int(avg_x), int(avg_y), self.trajectory[i][2]])
        
        return smoothed
    
    def is_tracking_lost(self):
        """Check if tracking is lost"""
        return self.lost_frames > self.max_lost_frames
    
    def get_trajectory(self):
        """Get current trajectory"""
        return self.trajectory.copy()
def apply_morphological_operations(frame, ball_detections):
    """
    Apply morphological operations to reduce noise in ball detection
    """
    if not ball_detections:
        return []
    
    # Create a mask for potential ball regions
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    
    for detection in ball_detections:
        x, y, w, h, conf = detection
        # Create small region around detection
        x1, y1 = max(0, int(x - w/2)), max(0, int(y - h/2))
        x2, y2 = min(frame.shape[1], int(x + w/2)), min(frame.shape[0], int(y + h/2))
        mask[y1:y2, x1:x2] = 255
    
    # Apply morphological operations
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # Find contours in the cleaned mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours by area and aspect ratio
    filtered_detections = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if 20 < area < 800:  # Reasonable ball size range
            x, y, w, h = cv2.boundingRect(contour)
            center_x, center_y = x + w//2, y + h//2
            
            # Find corresponding original detection for visualization
            best_conf = 0.0
            for orig_detection in ball_detections:
                orig_x, orig_y, _, _, conf = orig_detection
                if abs(orig_x - center_x) < 20 and abs(orig_y - center_y) < 20:
                    best_conf = max(best_conf, conf)
            if best_conf > 0.2:  # Only keep if we have reasonable confidence
                filtered_detections.append((center_x, center_y, w, h, best_conf))
    
    return filtered_detections

def temporal_consistency_filter(detections, past_positions, max_velocity=50):
    """
    Filter detections based on temporal consistency with past positions
    """
    if not past_positions or not detections:
        return detections
    
    last_x, last_y, last_frame = past_positions[-1]
    filtered = []
    
    for detection in detections:
        x, y, w, h, conf = detection
        distance = math.sqrt((x - last_x)**2 + (y - last_y)**2)
        
        # Allow higher movement if confidence is very high
        velocity_threshold = max_velocity * (2.0 if conf > 0.8 else 1.0)
        
        if distance <= velocity_threshold:
            filtered.append(detection)
    
    return filtered if filtered else detections  # Return original if all filtered out



