"""
Enhanced Framepose with Advanced Player Re-Identification
========================================================

This module provides an enhanced framepose function that integrates the 
advanced player re-identification system for better tracking accuracy.
"""

import cv2
import numpy as np
from PIL import Image
from squash.Player import Player
from enhanced_player_reid import EnhancedPlayerReID
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global ReID system instance
reid_system = None

def sum_pixels_in_bbox(frame, bbox):
    """Sum pixels in bounding box region"""
    x, y, w, h = bbox
    roi = frame[int(y) : int(y + h), int(x) : int(x + w)]
    return np.sum(roi, dtype=np.int64)

def find_match_2d_array(array, x):
    """Find match in 2D array"""
    for i in range(len(array)):
        if array[i][0] == x:
            return True
    return False

def initialize_reid_system():
    """Initialize the ReID system once"""
    global reid_system
    if reid_system is None:
        reid_system = EnhancedPlayerReID()
        logger.info("Enhanced ReID system initialized")
    return reid_system

def enhanced_framepose(
    pose_model,
    frame,
    otherTrackIds,
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
    embeddings=[[], []],
    plast=[[], []]
):
    """
    Enhanced framepose function with advanced player re-identification
    
    Args:
        pose_model: YOLO pose model
        frame: Current frame
        otherTrackIds: Track ID to player ID mapping
        updated: Player update status
        references1: Player 1 reference data
        references2: Player 2 reference data
        pixdiffs: Pixel differences
        players: Player objects dictionary
        frame_count: Current frame number
        player_last_positions: Last known positions
        frame_width: Frame width
        frame_height: Frame height
        annotated_frame: Frame for annotations
        max_players: Maximum number of players (default 2)
        occluded: Occlusion status
        importantdata: Important data list
        embeddings: Player embeddings
        plast: Player last positions
        
    Returns:
        Updated frame processing results
    """
    global reid_system
    
    try:
        # Initialize ReID system if not already done
        if reid_system is None:
            reid_system = initialize_reid_system()
        
        # Run pose detection
        track_results = pose_model.track(frame, persist=True, show=False)
        
        if (track_results and 
            hasattr(track_results[0], "keypoints") and 
            track_results[0].keypoints is not None):
            
            # Extract pose results
            boxes = track_results[0].boxes.xywh.cpu()
            track_ids = track_results[0].boxes.id.int().cpu().tolist()
            keypoints = track_results[0].keypoints.cpu().numpy()
            
            # Prepare detections for ReID system
            reid_detections = []
            
            for box, track_id, kp in zip(boxes, track_ids, keypoints):
                x, y, w, h = box
                
                # Extract player crop
                crop_x1 = max(0, int(x - w/2))
                crop_y1 = max(0, int(y - h/2))
                crop_x2 = min(frame_width, int(x + w/2))
                crop_y2 = min(frame_height, int(y + h/2))
                
                player_crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
                
                if player_crop.size == 0:
                    continue
                
                # Create detection for ReID system
                detection = {
                    'track_id': track_id,
                    'crop': player_crop,
                    'position': (int(x), int(y)),
                    'bbox': (crop_x1, crop_y1, crop_x2, crop_y2),
                    'keypoints': kp,
                    'confidence': 1.0  # Default confidence
                }
                
                reid_detections.append(detection)
            
            # Process detections through ReID system
            reid_result = reid_system.process_frame(reid_detections, frame_count)
            
            # Update otherTrackIds based on ReID results
            for detection in reid_detections:
                track_id = detection['track_id']
                
                if track_id in reid_result['player_assignments']:
                    player_id = reid_result['player_assignments'][track_id]
                    
                    # Update otherTrackIds if not already present
                    if not find_match_2d_array(otherTrackIds, track_id):
                        otherTrackIds.append([track_id, player_id])
                        logger.info(f"Added track id {track_id} to player {player_id} (ReID)")
                    else:
                        # Check if ReID suggests a different player ID
                        current_mapping = None
                        for mapping in otherTrackIds:
                            if mapping[0] == track_id:
                                current_mapping = mapping[1]
                                break
                        
                        if current_mapping != player_id:
                            # Update mapping based on ReID
                            for mapping in otherTrackIds:
                                if mapping[0] == track_id:
                                    mapping[1] = player_id
                                    logger.info(f"Updated track id {track_id} mapping to player {player_id} (ReID correction)")
                                    break
            
            # Display ReID information on frame
            if reid_result['swap_detection'] and reid_result['swap_detection']['swap_detected']:
                swap_info = reid_result['swap_detection']
                cv2.putText(
                    annotated_frame,
                    f"SWAP DETECTED: {swap_info['corrected_mapping']}",
                    (10, 130),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2
                )
            
            # Display initialization status
            init_status = reid_result['initialization_status']
            status_text = f"Init: P1={init_status[1]}, P2={init_status[2]}"
            cv2.putText(
                annotated_frame,
                status_text,
                (10, 110),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 0),
                1
            )
            
            # Process each detection for standard tracking
            for detection in reid_detections:
                track_id = detection['track_id']
                kp = detection['keypoints']
                x, y = detection['position']
                w, h = detection['bbox'][2] - detection['bbox'][0], detection['bbox'][3] - detection['bbox'][1]
                
                # Find player ID from updated mapping
                playerid = None
                for mapping in otherTrackIds:
                    if mapping[0] == track_id:
                        playerid = mapping[1]
                        break
                
                if playerid is None:
                    continue
                
                # Update player references (existing logic)
                if track_id == 1:
                    references1.append(sum_pixels_in_bbox(frame, [x, y, w, h]))
                    if len(references1) > 1 and len(references2) > 1 and len(pixdiffs) < 5:
                        pixdiffs.append(abs(references1[-1] - references2[-1]))
                
                elif track_id == 2:
                    references2.append(sum_pixels_in_bbox(frame, [x, y, w, h]))
                    if len(references1) > 1 and len(references2) > 1 and len(pixdiffs) < 5:
                        pixdiffs.append(abs(references1[-1] - references2[-1]))
                
                # Update player objects
                if playerid in players:
                    players[playerid].add_pose(kp)
                    player_last_positions[playerid] = (x, y)
                    
                    # Update timing
                    if playerid == 1:
                        updated[0][0] = True
                        updated[0][1] = frame_count
                    elif playerid == 2:
                        updated[1][0] = True
                        updated[1][1] = frame_count
                
                elif len(players) < max_players:
                    players[playerid] = Player(player_id=playerid)
                    players[playerid].add_pose(kp)
                    player_last_positions[playerid] = (x, y)
                    
                    if playerid == 1:
                        updated[0][0] = True
                        updated[0][1] = frame_count
                    elif playerid == 2:
                        updated[1][0] = True
                        updated[1][1] = frame_count
                    
                    logger.info(f"Enhanced: Player {playerid} added")
                
                # Update embeddings if available
                if len(embeddings) >= 2:
                    if playerid == 1 and len(embeddings[0]) < 100:  # Limit embedding storage
                        try:
                            player_crop_pil = Image.fromarray(cv2.cvtColor(detection['crop'], cv2.COLOR_BGR2RGB))
                            # You can add embedding extraction here if needed
                            # embeddings[0].append(extract_embeddings(player_crop_pil))
                        except Exception as e:
                            logger.error(f"Error processing player 1 embedding: {e}")
                    
                    elif playerid == 2 and len(embeddings[1]) < 100:  # Limit embedding storage
                        try:
                            player_crop_pil = Image.fromarray(cv2.cvtColor(detection['crop'], cv2.COLOR_BGR2RGB))
                            # You can add embedding extraction here if needed
                            # embeddings[1].append(extract_embeddings(player_crop_pil))
                        except Exception as e:
                            logger.error(f"Error processing player 2 embedding: {e}")
                
                # Update plast for position tracking
                if len(plast) >= 2:
                    if playerid == 1:
                        plast[0].append([x, y, frame_count])
                        if len(plast[0]) > 50:  # Keep only recent positions
                            plast[0].pop(0)
                    elif playerid == 2:
                        plast[1].append([x, y, frame_count])
                        if len(plast[1]) > 50:  # Keep only recent positions
                            plast[1].pop(0)
                
                # Draw keypoints with player ID
                color = (0, 0, 255) if playerid == 1 else (255, 0, 0)
                
                try:
                    if hasattr(kp, '__len__') and len(kp) > 0:
                        for keypoint in kp:
                            if hasattr(keypoint, 'xyn') and len(keypoint.xyn) > 0:
                                for i, k in enumerate(keypoint.xyn[0]):
                                    if len(k) >= 2:
                                        kx, ky = k[0], k[1]
                                        if kx > 0 and ky > 0:  # Valid keypoint
                                            pixel_x = int(kx * frame_width)
                                            pixel_y = int(ky * frame_height)
                                            cv2.circle(annotated_frame, (pixel_x, pixel_y), 3, color, -1)
                                            
                                            # Draw player ID on ankle keypoint
                                            if i == 16:  # Right ankle
                                                cv2.putText(
                                                    annotated_frame,
                                                    f"P{playerid}",
                                                    (pixel_x, pixel_y - 10),
                                                    cv2.FONT_HERSHEY_SIMPLEX,
                                                    0.8,
                                                    color,
                                                    2
                                                )
                except Exception as e:
                    logger.error(f"Error drawing keypoints for player {playerid}: {e}")
                
                # Draw bounding box
                bbox = detection['bbox']
                cv2.rectangle(
                    annotated_frame,
                    (bbox[0], bbox[1]),
                    (bbox[2], bbox[3]),
                    color,
                    2
                )
                
                # Draw track ID and player ID
                cv2.putText(
                    annotated_frame,
                    f"T{track_id}->P{playerid}",
                    (bbox[0], bbox[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2
                )
        
        # Display ReID statistics
        if frame_count % 100 == 0:  # Every 100 frames
            stats = reid_system.get_statistics()
            logger.info(f"ReID Stats at frame {frame_count}: {stats}")
        
        return [
            pose_model,
            frame,
            otherTrackIds,
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
        logger.error(f"Error in enhanced_framepose: {e}")
        logger.error(f"Line: {e.__traceback__.tb_lineno}")
        
        # Return safe defaults
        return [
            pose_model,
            frame,
            otherTrackIds,
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

def get_reid_statistics():
    """Get ReID system statistics"""
    global reid_system
    if reid_system is not None:
        return reid_system.get_statistics()
    return {}

def save_reid_references(filepath):
    """Save ReID references to file"""
    global reid_system
    if reid_system is not None:
        reid_system.save_references(filepath)

def load_reid_references(filepath):
    """Load ReID references from file"""
    global reid_system
    if reid_system is not None:
        reid_system.load_references(filepath)
