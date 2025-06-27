# given player keypoints in xyn, classify it as a backhand, forehand
# player keypoints as 0: Nose 1: Left Eye 2: Right Eye 3: Left Ear 4: Right Ear 5: Left Shoulder 6: Right Shoulder 7: Left Elbow 8: Right Elbow 9: Left Wrist 10: Right Wrist 11: Left Hip 12: Right Hip 13: Left Knee 14: Right Knee 15: Left Ankle 16: Right Ankle

import numpy as np


def classify(keypoints):
    # Check if keypoints are valid
    if keypoints is None or len(keypoints) < 17:
        return "unknown"
    
    try:
        # Convert to numpy array if needed and ensure proper format
        if not isinstance(keypoints, np.ndarray):
            keypoints = np.array(keypoints)
        
        # Handle different keypoint formats
        if keypoints.ndim == 1:
            # Flatten array, reshape to (n, 3) format
            if len(keypoints) >= 51:  # 17 keypoints * 3 values each
                keypoints = keypoints.reshape(-1, 3)
            else:
                return "insufficient_data"
        
        # Extract keypoint coordinates safely with proper type conversion
        def get_keypoint(idx):
            if idx < len(keypoints) and len(keypoints[idx]) >= 3:
                return [float(keypoints[idx][0]), float(keypoints[idx][1]), float(keypoints[idx][2])]
            return [0.0, 0.0, 0.0]  # Default if keypoint not available
        
        left_wrist = get_keypoint(9)    # Left Wrist
        right_wrist = get_keypoint(10)  # Right Wrist
        left_shoulder = get_keypoint(5)  # Left Shoulder
        right_shoulder = get_keypoint(6) # Right Shoulder
        
        # Check if keypoints are detected (confidence > 0)
        left_wrist_conf = float(left_wrist[2]) if len(left_wrist) > 2 else 0
        right_wrist_conf = float(right_wrist[2]) if len(right_wrist) > 2 else 0
        left_shoulder_conf = float(left_shoulder[2]) if len(left_shoulder) > 2 else 0
        right_shoulder_conf = float(right_shoulder[2]) if len(right_shoulder) > 2 else 0
        
        # Use proper boolean operations to avoid array ambiguity with explicit type conversion
        left_wrist_low = bool(left_wrist_conf < 0.3)
        right_wrist_low = bool(right_wrist_conf < 0.3)
        left_shoulder_low = bool(left_shoulder_conf < 0.3)
        right_shoulder_low = bool(right_shoulder_conf < 0.3)
        
        if (left_wrist_low and right_wrist_low) or (left_shoulder_low and right_shoulder_low):
            return "no_shot_detected"
        
        # Determine which arm is more active (higher confidence and movement) with explicit boolean conversion
        left_arm_active = bool(left_wrist_conf > 0.5) and bool(left_shoulder_conf > 0.5)
        right_arm_active = bool(right_wrist_conf > 0.5) and bool(right_shoulder_conf > 0.5)
        
        # Check if wrist is above shoulder (shot indication) with explicit type conversion
        shot_detected = False
        
        if left_arm_active and (float(left_wrist[1]) < float(left_shoulder[1])):  # y decreases upward
            shot_detected = True
            # Backhand: left wrist crosses to right side of body
            if float(left_wrist[0]) > float(left_shoulder[0]):
                return "backhand"
            else:
                return "forehand"
                
        if right_arm_active and (float(right_wrist[1]) < float(right_shoulder[1])):
            shot_detected = True
            # Forehand: right wrist on right side, backhand: crosses to left
            if float(right_wrist[0]) < float(right_shoulder[0]):
                return "backhand"
            else:
                return "forehand"
        
        return "no_shot_detected" if not shot_detected else "unknown"
        
    except (IndexError, TypeError, ValueError) as e:
        print(f"Keypoint classification error: {e}")
        return "error"
