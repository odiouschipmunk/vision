import math
import numpy as np
import cv2
import torch  # Add PyTorch for GPU acceleration

class EnhancedBallTracker:
    """
    Improved ball tracker - optimized for your trained model with GPU acceleration
    """
    def __init__(self, max_history=150, confidence_threshold=0.25):
        self.trajectory = []
        self.max_history = max_history
        self.base_confidence_threshold = confidence_threshold
        self.confidence_threshold = confidence_threshold
        self.lost_frames = 0
        self.max_lost_frames = 45  # Increased for better prediction in coaching pipeline
        self.last_valid_detection = None
        self.confidence_history = []
        self.adaptive_threshold = confidence_threshold
        self.velocity_history = []
        self.tracking_state = "searching"  # searching, tracking, predicting, lost
        self.tracking_confidence = 0.0
        self.consecutive_good_detections = 0
        self.speed_history = []
        
        # Enhanced parameters for coaching pipeline stability
        self.min_detections_for_tracking = 2  # Reduced for faster startup
        self.confidence_recovery_rate = 0.08  # Slower recovery for stability
        self.velocity_smoothing_factor = 0.6  # Smoother velocity changes
        
        # Enhanced GPU optimization setup
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_gpu = torch.cuda.is_available()
        
        # GPU memory optimization
        if self.use_gpu:
            torch.cuda.empty_cache()
            # Pre-allocate GPU tensors for common operations
            self.gpu_trajectory_buffer = torch.zeros((max_history, 3), device=self.device)
            self.gpu_velocity_buffer = torch.zeros((max_history-1, 2), device=self.device)
            
        print(f"🚀 Enhanced BallTracker using {'GPU' if self.use_gpu else 'CPU'} acceleration")
        if self.use_gpu:
            print(f"   💾 GPU Memory allocated for trajectory buffering")
        
    def update_adaptive_threshold(self):
        """Simplified adaptive threshold adjustment"""
        if len(self.confidence_history) > 10:
            recent_avg = sum(self.confidence_history[-5:]) / 5
            
            # Simple adaptive logic
            if self.tracking_state == "tracking" and recent_avg > 0.5:
                self.adaptive_threshold = max(0.15, self.base_confidence_threshold - 0.1)
            elif recent_avg < 0.2:
                self.adaptive_threshold = min(0.4, self.base_confidence_threshold + 0.1)
            else:
                # Gradually return to baseline
                self.adaptive_threshold = (self.adaptive_threshold + self.base_confidence_threshold) / 2

    def validate_detection(self, x, y, w, h, confidence):
        """Simplified but effective validation for ball detections"""
        # Primary confidence check
        self.update_adaptive_threshold()
        
        if confidence < self.adaptive_threshold:
            return False, "Low confidence"
        
        # Basic size validation - more lenient
        ball_size = w * h
        if ball_size < 10 or ball_size > 3000:  # Very generous bounds
            return False, "Invalid size"
        
        # Basic aspect ratio validation - more lenient for coaching pipeline
        aspect_ratio = w / h if h > 0 else float('inf')
        if aspect_ratio < 0.15 or aspect_ratio > 6.0:  # Even more generous for varied ball shapes
            return False, "Invalid aspect ratio"
        
        # Enhanced temporal consistency - more forgiving for coaching scenarios
        if len(self.trajectory) > 0:
            last_x, last_y, _ = self.trajectory[-1]
            distance = math.sqrt((x - last_x)**2 + (y - last_y)**2)
            
            # Adaptive distance check based on tracking state and confidence
            if self.tracking_state == "tracking":
                max_distance = 140  # More lenient for established tracking
            elif self.tracking_state == "searching":
                max_distance = 200  # Very lenient when searching
            else:
                max_distance = 160  # Balanced for other states
            
            # Allow larger jumps for high-confidence detections
            if confidence > 0.7:
                max_distance *= 1.5
                
            if distance > max_distance:
                # Don't immediately reject - just note the large jump
                if distance > max_distance * 1.5:
                    return False, f"Extremely far from last position ({distance:.1f})"
                else:
                    # Accept but with lower confidence
                    return True, f"Large jump detected ({distance:.1f})"
        
        return True, "Valid"

    def update(self, detections, frame_number):
        """Simplified but robust update method"""
        try:
            # Basic validation for detections
            valid_detections = []
            for detection in detections:
                x, y, w, h, conf = detection
                is_valid, reason = self.validate_detection(x, y, w, h, conf)
                if is_valid:
                    valid_detections.append(detection)
            
            if valid_detections:
                # Select best detection (highest confidence with position preference)
                best_detection = self.select_best_detection(valid_detections, frame_number)
                x, y, w, h, conf = best_detection
                
                # Update trajectory
                self.trajectory.append([x, y, frame_number])
                self.confidence_history.append(conf)
                self.last_valid_detection = best_detection
                self.lost_frames = 0
                self.consecutive_good_detections += 1
                
                # Update velocity for prediction
                if len(self.trajectory) >= 2:
                    self.update_velocity_history()
                
                # Update tracking state with improved logic for coaching pipeline
                if self.tracking_state in ["searching", "lost"]:
                    if self.consecutive_good_detections >= self.min_detections_for_tracking:
                        self.tracking_state = "tracking"
                        print(f"🟢 Ball tracking established (confidence: {conf:.2f})")
                elif self.tracking_state == "predicting":
                    self.tracking_state = "tracking"
                    print(f"🔄 Ball tracking recovered from prediction")
                    
                # More gradual confidence adjustment
                confidence_boost = self.confidence_recovery_rate if conf > 0.6 else self.confidence_recovery_rate * 0.5
                self.tracking_confidence = min(1.0, self.tracking_confidence + confidence_boost)
                
                # Maintain trajectory size
                if len(self.trajectory) > self.max_history:
                    self.trajectory = self.trajectory[-self.max_history:]
                    self.confidence_history = self.confidence_history[-self.max_history:]
                
                return True, [x, y]
            
            else:
                # No valid detections - try prediction with improved logic
                self.lost_frames += 1
                self.consecutive_good_detections = 0
                
                # Enhanced prediction logic for coaching pipeline
                if (self.lost_frames <= self.max_lost_frames and 
                    len(self.trajectory) > 2):
                    
                    # More forgiving prediction conditions
                    if self.tracking_state in ["tracking", "predicting"]:
                        predicted_pos = self.predict_position(frame_number)
                        if predicted_pos:
                            self.tracking_state = "predicting"
                            # Slower confidence decay for stability
                            decay_rate = 0.05 if self.lost_frames < 15 else 0.1
                            self.tracking_confidence = max(0.1, self.tracking_confidence - decay_rate)
                            
                            # Add predicted position to trajectory with lower confidence
                            self.trajectory.append([predicted_pos[0], predicted_pos[1], frame_number])
                            self.confidence_history.append(self.tracking_confidence)
                            
                            if self.lost_frames == 1:
                                print(f"🟡 Switching to ball prediction mode")
                            
                            return True, predicted_pos
                
                # Lost tracking with improved recovery
                if self.lost_frames > self.max_lost_frames:
                    if self.tracking_state != "lost":
                        self.tracking_state = "lost"
                        print(f"🔴 Ball tracking lost after {self.lost_frames} frames")
                    self.tracking_confidence = max(0.0, self.tracking_confidence - 0.02)
                    
                    # Gradual reset instead of immediate reset
                    if self.lost_frames > self.max_lost_frames * 3:
                        print("🔄 Resetting ball tracker for fresh start")
                        self.adaptive_threshold = max(0.1, self.adaptive_threshold - 0.05)  # Lower threshold
                        self.lost_frames = 0  # Reset counter but keep trajectory
                
                return False, None
                
        except Exception as e:
            print(f"Error in ball tracker update: {e}")
            return False, None

    def select_best_detection(self, detections, frame_number):
        """Simplified detection selection with focus on confidence and position"""
        if len(detections) == 1:
            return detections[0]
        
        best_score = -1
        best_detection = detections[0]
        
        for detection in detections:
            x, y, w, h, conf = detection
            score = conf * 0.6  # Base confidence score
            
            # Prefer detections close to predicted position
            if len(self.trajectory) > 0:
                last_x, last_y, _ = self.trajectory[-1]
                distance = math.sqrt((x - last_x)**2 + (y - last_y)**2)
                
                # Distance penalty (closer is better)
                distance_score = max(0, 1 - distance / 100)
                score += distance_score * 0.4
            
            if score > best_score:
                best_score = score
                best_detection = detection
        
        return best_detection

    def update_velocity_history(self):
        """Simple velocity calculation for prediction"""
        if len(self.trajectory) < 2:
            return
        
        p1, p2 = self.trajectory[-2], self.trajectory[-1]
        dt = p2[2] - p1[2] if p2[2] != p1[2] else 1
        velocity = [(p2[0] - p1[0]) / dt, (p2[1] - p1[1]) / dt]
        
        self.velocity_history.append(velocity)
        if len(self.velocity_history) > 10:  # Keep recent history
            self.velocity_history = self.velocity_history[-10:]
        
        # Calculate speed for monitoring
        speed = math.sqrt(velocity[0]**2 + velocity[1]**2)
        self.speed_history.append(speed)
        if len(self.speed_history) > 10:
            self.speed_history = self.speed_history[-10:]

    def predict_position(self, frame_number):
        """Enhanced position prediction with GPU acceleration for coaching pipeline"""
        if len(self.trajectory) < 2:
            return None
        
        if self.use_gpu and len(self.trajectory) > 3:
            return self.predict_position_gpu(frame_number)
        else:
            return self.predict_position_cpu(frame_number)
    
    def predict_position_gpu(self, frame_number):
        """Enhanced GPU-accelerated position prediction using pre-allocated buffers"""
        try:
            # Convert recent trajectory to GPU tensors using pre-allocated buffers
            recent_traj = self.trajectory[-10:] if len(self.trajectory) > 10 else self.trajectory
            traj_len = len(recent_traj)
            
            if traj_len < 3:
                return self.predict_position_cpu(frame_number)
            
            # Use pre-allocated buffer efficiently
            positions = self.gpu_trajectory_buffer[:traj_len].clone()
            for i, pos in enumerate(recent_traj):
                positions[i, 0] = pos[0]
                positions[i, 1] = pos[1] 
                positions[i, 2] = pos[2]
            
            # GPU-accelerated velocity calculation with optimized operations
            times = positions[:traj_len, 2]
            coords = positions[:traj_len, :2]
            
            # Calculate velocities using vectorized operations
            dt = times[1:] - times[:-1]
            dt = torch.clamp(dt, min=1e-6)  # Avoid division by zero
            velocities = (coords[1:] - coords[:-1]) / dt.unsqueeze(1)
            
            # Enhanced weighted average with exponential decay
            if len(velocities) > 1:
                # Exponential weights - more recent velocities have higher weight
                weights = torch.exp(torch.linspace(-2, 0, len(velocities), device=self.device))
                weights = weights / weights.sum()
                avg_velocity = torch.sum(velocities * weights.unsqueeze(1), dim=0)
            else:
                avg_velocity = velocities[-1]
            
            # Enhanced prediction with physics-based damping
            last_pos = positions[traj_len-1]
            dt_future = frame_number - last_pos[2]
            
            # Apply advanced damping model for ball physics
            if dt_future <= 3:
                damping = torch.tensor(1.0, device=self.device)
            elif dt_future <= 8:
                damping = torch.tensor(0.9 - (dt_future - 3) * 0.1, device=self.device)
            else:
                # Gravity and air resistance simulation
                damping = torch.tensor(max(0.2, 0.8 - (dt_future - 8) * 0.05), device=self.device)
            
            # Predict with enhanced physics model
            pred_pos = last_pos[:2] + avg_velocity * dt_future * damping
            
            # Enhanced bounds checking with court awareness
            pred_pos[0] = torch.clamp(pred_pos[0], 5, 635)  # Leave small margin from court edges
            pred_pos[1] = torch.clamp(pred_pos[1], 5, 355)
            
            return [int(pred_pos[0].cpu()), int(pred_pos[1].cpu())]
            
        except Exception as e:
            print(f"Enhanced GPU prediction failed: {e}, falling back to CPU")
            return self.predict_position_cpu(frame_number)
    
    def predict_position_cpu(self, frame_number):
        """CPU fallback for position prediction"""
        if len(self.trajectory) < 2:
            return None
        
        # Use enhanced prediction with velocity smoothing
        if len(self.velocity_history) > 0:
            # Weighted average of recent velocities - more recent = higher weight
            recent_velocities = self.velocity_history[-5:] if len(self.velocity_history) >= 5 else self.velocity_history
            weights = [0.4, 0.3, 0.2, 0.1] if len(recent_velocities) >= 4 else [1.0/len(recent_velocities)] * len(recent_velocities)
            
            avg_vx = sum(v[0] * w for v, w in zip(recent_velocities, weights[:len(recent_velocities)]))
            avg_vy = sum(v[1] * w for v, w in zip(recent_velocities, weights[:len(recent_velocities)]))
            
            # Apply velocity smoothing factor
            if hasattr(self, 'velocity_smoothing_factor'):
                avg_vx *= self.velocity_smoothing_factor
                avg_vy *= self.velocity_smoothing_factor
        else:
            # Enhanced fallback prediction using multiple points
            if len(self.trajectory) >= 3:
                p1, p2, p3 = self.trajectory[-3], self.trajectory[-2], self.trajectory[-1]
                dt1 = p2[2] - p1[2] if p2[2] != p1[2] else 1
                dt2 = p3[2] - p2[2] if p3[2] != p2[2] else 1
                
                vx1 = (p2[0] - p1[0]) / dt1
                vy1 = (p2[1] - p1[1]) / dt1
                vx2 = (p3[0] - p2[0]) / dt2
                vy2 = (p3[1] - p2[1]) / dt2
                
                # Average the velocities
                avg_vx = (vx1 + vx2) / 2
                avg_vy = (vy1 + vy2) / 2
            else:
                # Simple linear prediction
                p1, p2 = self.trajectory[-2], self.trajectory[-1]
                dt = p2[2] - p1[2] if p2[2] != p1[2] else 1
                avg_vx = (p2[0] - p1[0]) / dt
                avg_vy = (p2[1] - p1[1]) / dt
        
        # Predict position with damping for longer predictions
        last_pos = self.trajectory[-1]
        dt = frame_number - last_pos[2]
        
        # Apply damping factor for predictions far into the future
        damping = 1.0 if dt <= 5 else max(0.3, 1.0 - (dt - 5) * 0.1)
        
        pred_x = last_pos[0] + avg_vx * dt * damping
        pred_y = last_pos[1] + avg_vy * dt * damping
        pred_y = last_pos[1] + avg_vy * dt
        
        # Basic bounds checking
        pred_x = max(0, min(640, pred_x))  # Assuming 640 width
        pred_y = max(0, min(360, pred_y))  # Assuming 360 height
        
        return [int(pred_x), int(pred_y)]

    def predict_position_simple(self, frame_number):
        """Fallback simple prediction"""
        if len(self.trajectory) < 2:
            return None
        
        p1, p2 = self.trajectory[-2], self.trajectory[-1]
        dt_past = p2[2] - p1[2] if p2[2] != p1[2] else 1
        dt_future = frame_number - p2[2]
        
        vx = (p2[0] - p1[0]) / dt_past
        vy = (p2[1] - p1[1]) / dt_past
        
        pred_x = p2[0] + vx * dt_future
        pred_y = p2[1] + vy * dt_future
        
        return [int(pred_x), int(pred_y)]
    
    def get_smoothed_trajectory(self, window_size=5):
        """Get smoothed trajectory for visualization"""
        if len(self.trajectory) < window_size:
            return [[pos[0], pos[1]] for pos in self.trajectory]
        
        smoothed = []
        for i in range(len(self.trajectory)):
            start_idx = max(0, i - window_size // 2)
            end_idx = min(len(self.trajectory), i + window_size // 2 + 1)
            
            window_positions = self.trajectory[start_idx:end_idx]
            avg_x = sum(pos[0] for pos in window_positions) / len(window_positions)
            avg_y = sum(pos[1] for pos in window_positions) / len(window_positions)
            
            smoothed.append([int(avg_x), int(avg_y)])
        
        return smoothed
    
    def is_tracking_lost(self):
        """Check if tracking is lost"""
        return self.tracking_state == "lost"
    
    def get_trajectory(self):
        """Get current trajectory"""
        return self.trajectory.copy()
    
    def get_tracking_confidence(self):
        """Get current tracking confidence"""
        return self.tracking_confidence
    
    def get_tracking_state(self):
        """Get current tracking state"""
        return self.tracking_state
    
    def reset_tracking(self):
        """Reset tracker to initial state"""
        self.trajectory = []
        self.confidence_history = []
        self.velocity_history = []
        self.speed_history = []
        self.lost_frames = 0
        self.consecutive_good_detections = 0
        self.tracking_state = "searching"
        self.tracking_confidence = 0.0
        self.last_valid_detection = None
        self.adaptive_threshold = self.base_confidence_threshold


# Simplified helper functions optimized for your use case
def apply_morphological_operations(frame, ball_detections):
    """
    Simplified morphological operations to reduce noise in ball detection
    """
    if not ball_detections:
        return ball_detections
    
    # Create a mask for potential ball regions
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    
    for detection in ball_detections:
        x, y, w, h, conf = detection
        # Draw filled circle instead of rectangle for ball
        radius = max(int(w/2), int(h/2), 5)
        cv2.circle(mask, (int(x), int(y)), radius, 255, -1)
    
    # Simple morphological operations
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # Find contours and filter detections
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    filtered_detections = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if 20 < area < 2000:  # Reasonable ball size range
            # Find the detection closest to this contour
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                
                # Find closest original detection
                min_dist = float('inf')
                closest_detection = None
                
                for detection in ball_detections:
                    det_x, det_y, det_w, det_h, det_conf = detection
                    dist = math.sqrt((cx - det_x)**2 + (cy - det_y)**2)
                    if dist < min_dist:
                        min_dist = dist
                        closest_detection = detection
                
                if closest_detection and min_dist < 30:  # Within reasonable distance
                    filtered_detections.append(closest_detection)
    
    return filtered_detections if filtered_detections else ball_detections


def temporal_consistency_filter(detections, past_positions, max_velocity=70, prediction_tolerance=40):
    """
    Simplified temporal consistency filter
    """
    if not past_positions or not detections:
        return detections
    
    if len(past_positions) == 0:
        return detections
    
    last_x, last_y, last_frame = past_positions[-1]
    
    # Calculate expected velocity if we have enough history
    expected_velocity = None
    if len(past_positions) >= 2:
        prev_x, prev_y, prev_frame = past_positions[-2]
        dt = last_frame - prev_frame if last_frame != prev_frame else 1
        expected_velocity = [(last_x - prev_x) / dt, (last_y - prev_y) / dt]
    
    filtered_detections = []
    
    for detection in detections:
        x, y, w, h, conf = detection
        
        # Distance check
        distance = math.sqrt((x - last_x)**2 + (y - last_y)**2)
        if distance > max_velocity:
            continue
        
        # Velocity consistency check
        if expected_velocity:
            predicted_x = last_x + expected_velocity[0]
            predicted_y = last_y + expected_velocity[1]
            pred_distance = math.sqrt((x - predicted_x)**2 + (y - predicted_y)**2)
            
            if pred_distance > prediction_tolerance:
                continue
        
        filtered_detections.append(detection)
    
    # If no detections pass strict filter, return best detection
    if not filtered_detections and detections:
        # Return detection with highest confidence
        best_detection = max(detections, key=lambda d: d[4])
        return [best_detection]
    
    return filtered_detections if filtered_detections else detections
