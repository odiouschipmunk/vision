"""
Simplified ByteTracker Integration for YOLO Player Detections
This module provides a simple integration without complex dependencies.
"""

import numpy as np
import torch
from dataclasses import dataclass
from typing import List, Tuple, Optional, Any
import cv2
from collections import OrderedDict
import math

@dataclass
class TrackingArgs:
    """Arguments for simplified tracker configuration"""
    track_thresh: float = 0.5      # Detection confidence threshold
    track_buffer: int = 30         # Number of frames to keep lost tracks
    match_thresh: float = 0.8      # IOU threshold for track matching
    frame_rate: int = 30           # Video frame rate

class SimpleTrack:
    """Simple track object for player tracking"""
    def __init__(self, bbox, score, track_id):
        self.bbox = bbox  # [x1, y1, x2, y2]
        self.score = score
        self.track_id = track_id
        self.hits = 1
        self.time_since_update = 0
        self.age = 1
        self.state = 'active'
        
    def update(self, bbox, score):
        """Update track with new detection"""
        self.bbox = bbox
        self.score = score
        self.hits += 1
        self.time_since_update = 0
        self.age += 1
        
    def predict(self):
        """Simple prediction - just keep current position"""
        self.time_since_update += 1
        self.age += 1
        if self.time_since_update > 3:
            self.state = 'lost'

class YOLODetection:
    """Container for YOLO detection results"""
    def __init__(self, bbox: List[float], score: float, class_id: int = 0, keypoints: Optional[np.ndarray] = None):
        """
        Args:
            bbox: Bounding box in [x1, y1, x2, y2] format (top-left, bottom-right)
            score: Detection confidence score
            class_id: Class ID (0 for person)
            keypoints: Optional keypoints array
        """
        self.bbox = bbox
        self.score = score
        self.class_id = class_id
        self.keypoints = keypoints

class SimpleBYTETracker:
    """Simplified ByteTracker implementation"""
    
    def __init__(self, args: TrackingArgs):
        self.args = args
        self.tracks = []
        self.frame_id = 0
        self.next_track_id = 1
        
    def calculate_iou(self, bbox1, bbox2):
        """Calculate IoU between two bounding boxes"""
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
            
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def associate_detections_to_tracks(self, detections, tracks):
        """Associate detections to existing tracks using IoU"""
        if len(tracks) == 0:
            return [], list(range(len(detections))), []
            
        iou_matrix = np.zeros((len(detections), len(tracks)))
        
        for d, det in enumerate(detections):
            for t, track in enumerate(tracks):
                iou_matrix[d, t] = self.calculate_iou(det.bbox, track.bbox)
        
        # Simple greedy matching
        matches = []
        unmatched_dets = list(range(len(detections)))
        unmatched_trks = list(range(len(tracks)))
        
        # Find matches above threshold
        while len(unmatched_dets) > 0 and len(unmatched_trks) > 0:
            # Find maximum IoU
            max_iou = 0
            max_det = -1
            max_trk = -1
            
            for d in unmatched_dets:
                for t in unmatched_trks:
                    if iou_matrix[d, t] > max_iou and iou_matrix[d, t] > self.args.match_thresh:
                        max_iou = iou_matrix[d, t]
                        max_det = d
                        max_trk = t
            
            if max_det >= 0 and max_trk >= 0:
                matches.append([max_det, max_trk])
                unmatched_dets.remove(max_det)
                unmatched_trks.remove(max_trk)
            else:
                break
                
        return matches, unmatched_dets, unmatched_trks
    
    def update(self, detections_array, img_info, img_size):
        """Update tracker with new detections"""
        self.frame_id += 1
        
        # Convert numpy array to detection objects
        detections = []
        if len(detections_array) > 0:
            for det in detections_array:
                if len(det) >= 5:
                    bbox = det[:4]  # [x1, y1, x2, y2]
                    score = det[4]
                    if score > self.args.track_thresh:
                        detections.append(YOLODetection(bbox.tolist(), float(score)))
        
        # Predict existing tracks
        active_tracks = [t for t in self.tracks if t.state == 'active']
        for track in active_tracks:
            track.predict()
        
        # Associate detections to tracks
        matches, unmatched_dets, unmatched_trks = self.associate_detections_to_tracks(
            detections, active_tracks
        )
        
        # Update matched tracks
        for match in matches:
            det_idx, trk_idx = match
            active_tracks[trk_idx].update(detections[det_idx].bbox, detections[det_idx].score)
        
        # Create new tracks for unmatched detections
        for det_idx in unmatched_dets:
            new_track = SimpleTrack(
                detections[det_idx].bbox, 
                detections[det_idx].score, 
                self.next_track_id
            )
            self.next_track_id += 1
            self.tracks.append(new_track)
        
        # Remove lost tracks
        self.tracks = [t for t in self.tracks if t.time_since_update < self.args.track_buffer]
        
        # Return active tracks with required attributes
        output_tracks = []
        for track in self.tracks:
            if track.state == 'active' and track.hits >= 3:  # Require at least 3 hits
                # Create mock track object with required attributes
                class MockTrack:
                    def __init__(self, track):
                        self.track_id = track.track_id
                        self.tlbr = track.bbox  # [x1, y1, x2, y2]
                        self.score = track.score
                
                output_tracks.append(MockTrack(track))
        
        return output_tracks

class SimpleBytePtrackIntegration:
    """
    Simplified Integration class that combines YOLO player detection with simple tracking
    """
    
    def __init__(self, args: Optional[TrackingArgs] = None):
        """Initialize simplified tracker integration"""
        self.args = args or TrackingArgs()
        self.tracker = SimpleBYTETracker(self.args)
        self.frame_count = 0
    
    def convert_yolo_to_detections(self, boxes: torch.Tensor, track_ids: List[int], 
                                 keypoints: np.ndarray, confidences: Optional[torch.Tensor] = None) -> List[YOLODetection]:
        """Convert YOLO pose detection results to detection format"""
        detections = []
        
        for i, (box, track_id, kp) in enumerate(zip(boxes, track_ids, keypoints)):
            # Convert from YOLO xywh (center) to xyxy (corners) format
            x_center, y_center, width, height = box
            x1 = float(x_center - width / 2)
            y1 = float(y_center - height / 2)
            x2 = float(x_center + width / 2)
            y2 = float(y_center + height / 2)
            
            # Use confidence score if provided, otherwise use default
            score = float(confidences[i]) if confidences is not None else 0.9
            
            detection = YOLODetection(
                bbox=[x1, y1, x2, y2],
                score=score,
                class_id=0,  # Person class
                keypoints=kp
            )
            detections.append(detection)
            
        return detections
    
    def detections_to_numpy(self, detections: List[YOLODetection]) -> np.ndarray:
        """Convert detections to numpy array format expected by tracker"""
        if not detections:
            return np.empty((0, 5))
            
        det_array = np.array([
            det.bbox + [det.score] for det in detections
        ])
        
        return det_array
    
    def update_tracks(self, detections: List[YOLODetection], img_info: Tuple[int, int], 
                     img_size: Tuple[int, int]) -> List[Any]:
        """Update tracker with new detections"""
        self.frame_count += 1
        
        # Convert detections to numpy format
        dets = self.detections_to_numpy(detections)
        
        # Update tracker
        online_targets = self.tracker.update(dets, img_info, img_size)
        
        return online_targets
    
    def integrate_with_framepose(self, pose_model, frame: np.ndarray, 
                                other_track_ids: List, updated: List, 
                                references1: List, references2: List, 
                                pixdiffs: List, players: dict, 
                                frame_count: int, player_last_positions: dict,
                                frame_width: int, frame_height: int, 
                                annotated_frame: np.ndarray, max_players: int = 2) -> List:
        """Enhanced framepose function that integrates simple tracker with YOLO pose detection"""
        
        # Run YOLO pose detection as before
        track_results = pose_model.track(frame, persist=True, show=False)
        
        if (track_results and hasattr(track_results[0], "keypoints") 
            and track_results[0].keypoints is not None):
            
            # Extract results from YOLO
            boxes = track_results[0].boxes.xywh.cpu()
            yolo_track_ids = track_results[0].boxes.id.int().cpu().tolist()
            keypoints = track_results[0].keypoints.cpu().numpy()
            confidences = track_results[0].boxes.conf.cpu() if hasattr(track_results[0].boxes, 'conf') else None
            
            # Convert to detection format
            detections = self.convert_yolo_to_detections(boxes, yolo_track_ids, keypoints, confidences)
            
            # Update simple tracker
            img_info = (frame_height, frame_width)
            img_size = (frame_height, frame_width)
            byte_tracks = self.update_tracks(detections, img_info, img_size)
            
            # Map tracker results back to your player system
            self._update_players_with_tracking(
                byte_tracks, detections, other_track_ids, updated, 
                references1, references2, pixdiffs, players, 
                frame_count, player_last_positions, frame_width, 
                frame_height, annotated_frame, max_players, frame
            )
            
            # Draw tracking results
            self._draw_tracking_results(annotated_frame, byte_tracks, detections)
        
        return [
            pose_model, frame, other_track_ids, updated, references1, 
            references2, pixdiffs, players, frame_count, player_last_positions,
            frame_width, frame_height, annotated_frame
        ]
    
    def _update_players_with_tracking(self, byte_tracks, detections, other_track_ids, 
                                     updated, references1, references2, pixdiffs, 
                                     players, frame_count, player_last_positions,
                                     frame_width, frame_height, annotated_frame, max_players, frame):
        """Update your player tracking system with simple tracker results"""
        # Import here to avoid circular imports
        from squash import Functions
        from squash.Player import Player
        
        # Define the find_match_2d_array function locally since it's in ef.py
        def find_match_2d_array(array, x):
            for i in range(len(array)):
                if array[i][0] == x:
                    return True
            return False
        
        # Define the sum_pixels_in_bbox function locally since it's in ef.py
        def sum_pixels_in_bbox(frame, bbox):
            x, y, w, h = bbox
            roi = frame[int(y) : int(y + h), int(x) : int(x + w)]
            return np.sum(roi, dtype=np.int64)
        
        # Create mapping from detection to tracker track
        detection_to_track = {}
        for track in byte_tracks:
            # Find closest detection to this track
            track_bbox = track.tlbr  # [x1, y1, x2, y2]
            best_det_idx = None
            best_iou = 0
            
            for i, det in enumerate(detections):
                iou = self._calculate_iou(track_bbox, det.bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_det_idx = i
            
            if best_det_idx is not None and best_iou > 0.3:
                detection_to_track[best_det_idx] = track
        
        # Process each detection with its corresponding tracker track
        for det_idx, detection in enumerate(detections):
            if det_idx not in detection_to_track:
                continue
                
            track = detection_to_track[det_idx]
            byte_track_id = track.track_id
            keypoints = detection.keypoints
            
            # Convert bbox back to xywh center format for compatibility
            x1, y1, x2, y2 = detection.bbox
            x = (x1 + x2) / 2
            y = (y1 + y2) / 2
            w = x2 - x1
            h = y2 - y1
            
            # Determine player ID using your existing logic
            if not find_match_2d_array(other_track_ids, byte_track_id):
                # New track - assign to player using your existing logic
                if updated[0][1] > updated[1][1]:
                    other_track_ids.append([byte_track_id, 2])
                    playerid = 2
                    print(f"SimpleTracker: added track id {byte_track_id} to player 2")
                else:
                    other_track_ids.append([byte_track_id, 1])
                    playerid = 1
                    print(f"SimpleTracker: added track id {byte_track_id} to player 1")
            else:
                # Existing track - find player ID
                for track_mapping in other_track_ids:
                    if track_mapping[0] == byte_track_id:
                        playerid = track_mapping[1]
                        break
                else:
                    continue  # Skip if mapping not found
            
            # Update player references using your existing logic
            if playerid == 1:
                references1.append(sum_pixels_in_bbox(frame, [x, y, w, h]))
                if len(references1) > 1 and len(references2) > 1 and len(pixdiffs) < 5:
                    pixdiffs.append(abs(references1[-1] - references2[-1]))
            elif playerid == 2:
                references2.append(sum_pixels_in_bbox(frame, [x, y, w, h]))
                if len(references1) > 1 and len(references2) > 1 and len(pixdiffs) < 5:
                    pixdiffs.append(abs(references1[-1] - references2[-1]))
            
            # Update or create player
            if playerid in players:
                players[playerid].add_pose(keypoints)
                player_last_positions[playerid] = (x, y)
                if playerid == 1:
                    updated[0][0] = True
                    updated[0][1] = frame_count
                elif playerid == 2:
                    updated[1][0] = True
                    updated[1][1] = frame_count
            elif len(players) < max_players:
                players[playerid] = Player(player_id=playerid)
                players[playerid].add_pose(keypoints)
                player_last_positions[playerid] = (x, y)
                if playerid == 1:
                    updated[0][0] = True
                    updated[0][1] = frame_count
                elif playerid == 2:
                    updated[1][0] = True
                    updated[1][1] = frame_count
                print(f"SimpleTracker: Player {playerid} added.")
            
            # Draw keypoints on the frame
            self._draw_keypoints(annotated_frame, keypoints, playerid, frame_width, frame_height)
    
    def _draw_tracking_results(self, annotated_frame: np.ndarray, byte_tracks, detections):
        """Draw simple tracker tracking results on the frame"""
        for track in byte_tracks:
            track_bbox = track.tlbr  # [x1, y1, x2, y2]
            track_id = track.track_id
            
            # Draw bounding box in cyan for simple tracker
            cv2.rectangle(annotated_frame, 
                         (int(track_bbox[0]), int(track_bbox[1])), 
                         (int(track_bbox[2]), int(track_bbox[3])), 
                         (255, 255, 0), 2)  # Cyan for SimpleTracker
            
            # Draw track ID
            cv2.putText(annotated_frame, f'ST:{track_id}', 
                       (int(track_bbox[0]), int(track_bbox[1]) - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    
    def _draw_keypoints(self, annotated_frame: np.ndarray, keypoints: np.ndarray, playerid: int, frame_width: int, frame_height: int):
        """Draw keypoints on the frame for a specific player"""
        if keypoints is None or len(keypoints) == 0:
            return
            
        # Handle keypoints format - could be different structures
        try:
            # If keypoints has .xyn attribute (YOLO format)
            if hasattr(keypoints, 'xyn'):
                kp_data = keypoints.xyn[0]
            # If keypoints is numpy array directly
            elif isinstance(keypoints, np.ndarray):
                if len(keypoints.shape) == 3:  # Shape like (1, 17, 3)
                    kp_data = keypoints[0]
                elif len(keypoints.shape) == 2:  # Shape like (17, 3) or (17, 2)
                    kp_data = keypoints
                else:
                    return
            else:
                return
                
            # Draw each keypoint
            for i, keypoint in enumerate(kp_data):
                # Handle both 2D (x, y) and 3D (x, y, conf) keypoints
                if len(keypoint) == 3:
                    x, y, conf = keypoint
                elif len(keypoint) == 2:
                    x, y = keypoint
                    conf = 1.0  # Default confidence
                else:
                    continue
                    
                # Skip invalid keypoints
                if x == 0 and y == 0:
                    continue
                    
                # Convert normalized coordinates to pixel coordinates
                pixel_x = int(x * frame_width)
                pixel_y = int(y * frame_height)
                
                # Skip out-of-bounds keypoints
                if pixel_x < 0 or pixel_x >= frame_width or pixel_y < 0 or pixel_y >= frame_height:
                    continue
                
                # Choose color based on player ID
                if playerid == 1:
                    color = (0, 0, 255)  # Red for player 1
                else:
                    color = (255, 0, 0)  # Blue for player 2
                
                # Draw keypoint circle
                cv2.circle(annotated_frame, (pixel_x, pixel_y), 3, color, -1)
                
                # Draw player ID on ankle keypoint (index 16)
                if i == 16:  # Right ankle
                    cv2.putText(
                        annotated_frame,
                        f"P{playerid}",
                        (pixel_x, pixel_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (255, 255, 255),
                        2,
                    )
                    
        except Exception as e:
            print(f"Error drawing keypoints for player {playerid}: {e}")
            print(f"Keypoints shape: {keypoints.shape if hasattr(keypoints, 'shape') else 'no shape'}")
            print(f"Keypoints type: {type(keypoints)}")
            if hasattr(keypoints, 'xyn'):
                print(f"xyn shape: {keypoints.xyn[0].shape if len(keypoints.xyn) > 0 else 'empty'}")
    
    def _calculate_iou(self, bbox1: List[float], bbox2: List[float]) -> float:
        """Calculate Intersection over Union (IoU) between two bounding boxes"""
        # Calculate intersection area
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        
        # Calculate union area
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0

# Convenience function for easy integration
def create_simple_bytetrack_integration(track_thresh: float = 0.5, 
                                       track_buffer: int = 30,
                                       match_thresh: float = 0.8) -> SimpleBytePtrackIntegration:
    """
    Create a simplified ByteTracker integration instance
    
    Args:
        track_thresh: Confidence threshold for detections
        track_buffer: Number of frames to keep lost tracks
        match_thresh: IoU threshold for matching detections to tracks
        
    Returns:
        SimpleBytePtrackIntegration instance
    """
    args = TrackingArgs(
        track_thresh=track_thresh,
        track_buffer=track_buffer,
        match_thresh=match_thresh
    )
    return SimpleBytePtrackIntegration(args)
