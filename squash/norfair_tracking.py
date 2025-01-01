import numpy as np
from norfair import Detection, Tracker, draw_tracked_objects
from norfair.distances import create_keypoints_voting_distance

# Configure Norfair tracker with optimized parameters for squash
DISTANCE_THRESHOLD = 0.3  # Threshold for keypoint distance matching
PAST_DETECTIONS_LENGTH = 10  # Number of past detections to consider
HIT_COUNTER_MAX = 10  # Maximum number of frames to keep track
MIN_DETECTED_FRAMES = 3  # Minimum frames before considering it a valid track

<<<<<<< HEAD

=======
>>>>>>> 443d2dcfee5557e2e69e926b7d2deb8c4f1f62d4
def setup_keypoint_tracking(pose_model=None):
    """
    Initialize Norfair tracker with custom distance function
    """
    distance_function = create_keypoints_voting_distance(
        keypoint_distance_threshold=DISTANCE_THRESHOLD,
<<<<<<< HEAD
        detection_threshold=0.5,  # Minimum confidence score to consider a keypoint
    )

=======
        detection_threshold=0.5  # Minimum confidence score to consider a keypoint
    )
    
>>>>>>> 443d2dcfee5557e2e69e926b7d2deb8c4f1f62d4
    tracker = Tracker(
        distance_function=distance_function,
        distance_threshold=DISTANCE_THRESHOLD,
        past_detections_length=PAST_DETECTIONS_LENGTH,
        hit_counter_max=HIT_COUNTER_MAX,
        initialization_delay=MIN_DETECTED_FRAMES,
    )
<<<<<<< HEAD

    def track_poses(frame):
        # Get pose detections from YOLO model
        results = pose_model(frame)

        if not results or not hasattr(results[0], "keypoints"):
            return [], frame

        keypoints = results[0].keypoints.cpu().numpy()
        boxes = results[0].boxes.xywh.cpu()

=======
    
    def track_poses(frame):
        # Get pose detections from YOLO model
        results = pose_model(frame)
        
        if not results or not hasattr(results[0], "keypoints"):
            return [], frame
            
        keypoints = results[0].keypoints.cpu().numpy()
        boxes = results[0].boxes.xywh.cpu()
        
>>>>>>> 443d2dcfee5557e2e69e926b7d2deb8c4f1f62d4
        # Convert YOLO detections to Norfair format
        detections = []
        for kps, box in zip(keypoints, boxes):
            if kps.shape[0] == 0:
                continue
<<<<<<< HEAD

=======
                
>>>>>>> 443d2dcfee5557e2e69e926b7d2deb8c4f1f62d4
            # Extract keypoint coordinates and confidences
            # YOLO keypoints are in format [N, 17, 3] where N is number of detections
            # Each keypoint has (x, y, confidence)
            kp_data = kps[0].data  # Get first detection's keypoints
            points = kp_data[:, :2]  # Get x,y coordinates (17, 2)
<<<<<<< HEAD
            scores = kp_data[:, 2]  # Get confidence scores (17,)

            # Create Detection object with correct shape
            detections.append(Detection(points=points, scores=scores))

        # Update tracker
        tracked_objects = tracker.update(detections=detections)

        # Draw tracked poses
        annotated_frame = frame.copy()
        draw_tracked_objects(
            annotated_frame,
            tracked_objects,
            id_size=2,
            id_thickness=2,
            color_by_id=True,
        )

        return tracked_objects, annotated_frame

    return track_poses


=======
            scores = kp_data[:, 2]   # Get confidence scores (17,)
            
            # Create Detection object with correct shape
            detections.append(Detection(points=points, scores=scores))
        
        # Update tracker
        tracked_objects = tracker.update(detections=detections)
        
        # Draw tracked poses
        annotated_frame = frame.copy()
        draw_tracked_objects(
            annotated_frame, 
            tracked_objects,
            id_size=2,
            id_thickness=2,
            color_by_id=True
        )
        
        return tracked_objects, annotated_frame
        
    return track_poses

>>>>>>> 443d2dcfee5557e2e69e926b7d2deb8c4f1f62d4
def convert_norfair_to_yolo(tracked_objects, frame_width, frame_height):
    """
    Convert Norfair tracked objects to YOLO format for compatibility
    """
    converted_results = []
    for obj in tracked_objects:
        if not obj.last_detection:
            continue
<<<<<<< HEAD

        # Convert points back to YOLO format
        points = obj.last_detection.points  # Shape is (17, 2)
        confidences = obj.last_detection.scores  # Shape is (17,)

=======
            
        # Convert points back to YOLO format
        points = obj.last_detection.points  # Shape is (17, 2)
        confidences = obj.last_detection.scores  # Shape is (17,)
        
>>>>>>> 443d2dcfee5557e2e69e926b7d2deb8c4f1f62d4
        # Calculate bounding box from keypoints
        x_min, y_min = np.min(points, axis=0)
        x_max, y_max = np.max(points, axis=0)
        width = x_max - x_min
        height = y_max - y_min
<<<<<<< HEAD

        # Create result in YOLO format
        result = {
            "track_id": obj.id,
            "keypoints": points,
            "confidences": confidences,
            "bbox": [x_min, y_min, width, height],
            "frame_width": frame_width,
            "frame_height": frame_height,
        }
        converted_results.append(result)

    return converted_results
=======
        
        # Create result in YOLO format
        result = {
            'track_id': obj.id,
            'keypoints': points,
            'confidences': confidences,
            'bbox': [x_min, y_min, width, height],
            'frame_width': frame_width,
            'frame_height': frame_height
        }
        converted_results.append(result)
        
    return converted_results 
>>>>>>> 443d2dcfee5557e2e69e926b7d2deb8c4f1f62d4
