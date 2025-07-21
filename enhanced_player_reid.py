"""
Enhanced Player Re-Identification (ReID) System for Squash Analysis
================================================================

This module provides an advanced player re-identification system that:
1. Uses a proper person ReID model (OSNet) for robust feature extraction
2. Captures initial player appearances when they are separated (frames 100-150)
3. Continuously monitors for track ID swapping when players come close
4. Uses multiple features: appearance, position, and temporal consistency
5. Ensures unique player assignments (no duplicate player IDs)
6. Provides confidence scores for identity assignments
"""

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
import math
from typing import Dict, List, Tuple, Optional
import json
from collections import deque
import logging

# Import proper person ReID model
try:
    import torchreid
    from torchreid.reid.utils import FeatureExtractor
    TORCHREID_AVAILABLE = True
except ImportError as e:
    print(f"Warning: torchreid not available, falling back to ResNet50, error as {e}")
    from torchvision.models import resnet50, ResNet50_Weights
    TORCHREID_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedPlayerReID:
    """Enhanced Player Re-Identification System"""
    
    def __init__(self, device=None):
        """
        Initialize the enhanced ReID system
        
        Args:
            device: PyTorch device (cuda/cpu)
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"ReID system initialized on {self.device}")
        
        # Initialize feature extractor
        self._init_feature_extractor()
        
        # Player reference storage
        self.player_references = {
            1: {
                'appearance_features': [],
                'position_history': deque(maxlen=50),
                'last_seen_frame': 0,
                'confidence_scores': [],
                'initialization_complete': False
            },
            2: {
                'appearance_features': [],
                'position_history': deque(maxlen=50),
                'last_seen_frame': 0,
                'confidence_scores': [],
                'initialization_complete': False
            }
        }
        
        # System state
        self.initialization_frames = (100, 150)  # Frames to capture initial references
        self.proximity_threshold = 100  # Distance threshold for "close" players
        self.swap_detection_window = 10  # Frames to analyze for swapping
        self.min_separation_distance = 150  # Minimum distance for reliable initialization
        
        # Track ID mapping and history
        self.track_to_player_mapping = {}
        self.mapping_history = deque(maxlen=100)
        self.swap_candidates = []
        
        # Performance metrics
        self.swap_detections = 0
        self.confidence_threshold = 0.6
        
    def _init_feature_extractor(self):
        """Initialize the deep learning feature extractor"""
        try:
            if TORCHREID_AVAILABLE:
                # Use proper person ReID model (OSNet)
                logger.info("Initializing OSNet person ReID model...")
                self.feature_extractor = FeatureExtractor(
                    model_name='osnet_x1_0',  # State-of-the-art person ReID model
                    model_path=None,  # Will download pretrained weights
                    device=str(self.device).replace('cuda:', 'cuda').replace('cpu', 'cpu')
                )
                logger.info("OSNet person ReID model initialized successfully")
                self.use_osnet = True
                
            else:
                # Fallback to ResNet50
                logger.warning("Using ResNet50 fallback (install torchreid for better results)")
                self.feature_extractor = resnet50(weights=ResNet50_Weights.DEFAULT)
                self.feature_extractor.fc = torch.nn.Identity()  # Remove final classification layer
                self.feature_extractor.to(self.device)
                self.feature_extractor.eval()
                self.use_osnet = False
                
                # Preprocessing pipeline for ResNet50
                self.preprocess = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
            
        except Exception as e:
            logger.error(f"Error initializing feature extractor: {e}")
            raise
    
    def extract_appearance_features(self, player_crop: np.ndarray) -> np.ndarray:
        """
        Extract deep appearance features from player crop using proper person ReID model
        
        Args:
            player_crop: RGB/BGR image crop of player
            
        Returns:
            Feature vector as numpy array
        """
        try:
            if player_crop.size == 0:
                return np.zeros(2048)
            
            if self.use_osnet:
                # Use OSNet person ReID model
                # OSNet expects RGB images
                if len(player_crop.shape) == 3 and player_crop.shape[2] == 3:
                    # Convert BGR to RGB if needed
                    if player_crop.dtype == np.uint8 and player_crop.max() > 1.0:
                        player_crop_rgb = cv2.cvtColor(player_crop, cv2.COLOR_BGR2RGB)
                    else:
                        player_crop_rgb = player_crop
                else:
                    return np.zeros(2048)
                
                # OSNet expects list of images
                features = self.feature_extractor([player_crop_rgb])
                # Ensure we return numpy array
                if hasattr(features[0], 'cpu'):  # It's a torch tensor
                    return features[0].cpu().numpy() if features[0].is_cuda else features[0].numpy()
                return features[0]  # Already numpy array
                
            else:
                # Use ResNet50 fallback
                # Ensure RGB format
                if len(player_crop.shape) == 3 and player_crop.shape[2] == 3:
                    # Convert BGR to RGB if needed
                    if player_crop.max() > 1.0:  # Assume 0-255 range
                        player_crop = cv2.cvtColor(player_crop, cv2.COLOR_BGR2RGB)
                
                # Preprocess and extract features
                input_tensor = self.preprocess(player_crop).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    features = self.feature_extractor(input_tensor)
                
                return features.cpu().numpy().flatten()
            
        except Exception as e:
            logger.error(f"Error extracting appearance features: {e}")
            return np.zeros(2048)
    
    def calculate_appearance_similarity(self, features1: np.ndarray, features2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two feature vectors
        
        Args:
            features1: First feature vector
            features2: Second feature vector
            
        Returns:
            Similarity score (0-1)
        """
        try:
            # Normalize features
            norm1 = np.linalg.norm(features1)
            norm2 = np.linalg.norm(features2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            # Calculate cosine similarity
            similarity = np.dot(features1, features2) / (norm1 * norm2)
            return max(0.0, similarity)  # Ensure non-negative
            
        except Exception as e:
            logger.error(f"Error calculating appearance similarity: {e}")
            return 0.0
    
    def is_initialization_frame(self, frame_count: int) -> bool:
        """Check if current frame is in initialization range"""
        return self.initialization_frames[0] <= frame_count <= self.initialization_frames[1]
    
    def are_players_separated(self, positions: List[Tuple[int, int]]) -> bool:
        """
        Check if players are sufficiently separated for reliable initialization
        
        Args:
            positions: List of (x, y) positions
            
        Returns:
            True if players are well separated
        """
        if len(positions) < 2:
            return False
        
        distance = math.sqrt((positions[0][0] - positions[1][0])**2 + 
                           (positions[0][1] - positions[1][1])**2)
        return distance > self.min_separation_distance
    
    def initialize_player_references(self, detections: List[Dict], frame_count: int) -> bool:
        """
        Initialize player references during early frames when players are separated
        
        Args:
            detections: List of player detections with crops and positions
            frame_count: Current frame number
            
        Returns:
            True if initialization successful
        """
        if not self.is_initialization_frame(frame_count):
            return False
        
        if len(detections) != 2:
            return False
        
        # Extract positions
        positions = [(det['position'][0], det['position'][1]) for det in detections]
        
        # Check if players are sufficiently separated
        if not self.are_players_separated(positions):
            return False
        
        # Sort detections by x-coordinate (left player = 1, right player = 2)
        sorted_detections = sorted(detections, key=lambda x: x['position'][0])
        
        # Initialize references for both players
        for i, (player_id, detection) in enumerate(zip([1, 2], sorted_detections)):
            track_id = detection['track_id']
            
            if not self.player_references[player_id]['initialization_complete']:
                # Extract appearance features
                features = self.extract_appearance_features(detection['crop'])
                
                # Store reference data
                self.player_references[player_id]['appearance_features'].append(features)
                self.player_references[player_id]['position_history'].append(detection['position'])
                self.player_references[player_id]['last_seen_frame'] = frame_count
                
                # Create track to player mapping
                self.track_to_player_mapping[track_id] = player_id
                
                # Mark as initialized if we have enough references
                if len(self.player_references[player_id]['appearance_features']) >= 3:
                    self.player_references[player_id]['initialization_complete'] = True
                    logger.info(f"Player {player_id} reference initialized at frame {frame_count} with track ID {track_id}")
        
        return True
    
    def detect_players_in_proximity(self, positions: List[Tuple[int, int]]) -> bool:
        """
        Detect if players are close enough that track ID swapping might occur
        
        Args:
            positions: List of player positions
            
        Returns:
            True if players are in proximity
        """
        if len(positions) < 2:
            return False
        
        distance = math.sqrt((positions[0][0] - positions[1][0])**2 + 
                           (positions[0][1] - positions[1][1])**2)
        return distance < self.proximity_threshold
    
    def predict_player_identity(self, detection: Dict, frame_count: int) -> Tuple[int, float]:
        """
        Predict player identity based on appearance and position
        
        Args:
            detection: Detection dictionary with crop and position
            frame_count: Current frame number
            
        Returns:
            Tuple of (predicted_player_id, confidence_score)
        """
        # Extract features from current detection
        current_features = self.extract_appearance_features(detection['crop'])
        current_position = detection['position']
        
        best_match = 1
        best_confidence = 0.0
        
        # Compare with both player references
        for player_id in [1, 2]:
            if not self.player_references[player_id]['initialization_complete']:
                continue
            
            # Calculate appearance similarity
            appearance_scores = []
            for ref_features in self.player_references[player_id]['appearance_features']:
                similarity = self.calculate_appearance_similarity(current_features, ref_features)
                appearance_scores.append(similarity)
            
            avg_appearance_score = np.mean(appearance_scores) if appearance_scores else 0.0
            
            # Calculate position consistency
            position_score = 0.0
            if self.player_references[player_id]['position_history']:
                recent_positions = list(self.player_references[player_id]['position_history'])[-5:]
                avg_position = np.mean(recent_positions, axis=0)
                position_distance = math.sqrt((current_position[0] - avg_position[0])**2 + 
                                            (current_position[1] - avg_position[1])**2)
                position_score = max(0.0, 1.0 - position_distance / 200.0)  # Normalize by max expected distance
            
            # Temporal consistency score
            temporal_score = 0.0
            frames_since_last = frame_count - self.player_references[player_id]['last_seen_frame']
            if frames_since_last < 10:  # Recent sighting
                temporal_score = 1.0 - (frames_since_last / 10.0)
            
            # Combined confidence score
            combined_confidence = (avg_appearance_score * 0.6 + 
                                 position_score * 0.3 + 
                                 temporal_score * 0.1)
            
            if combined_confidence > best_confidence:
                best_confidence = combined_confidence
                best_match = player_id
        
        return best_match, best_confidence
    
    def detect_track_id_swap(self, track_detections: List[Dict], frame_count: int) -> Dict:
        """
        Detect if track IDs have been swapped between players
        
        Args:
            track_detections: List of detections with track IDs
            frame_count: Current frame number
            
        Returns:
            Dictionary with swap detection results
        """
        swap_result = {
            'swap_detected': False,
            'swapped_tracks': [],
            'confidence': 0.0,
            'corrected_mapping': {}
        }
        
        if len(track_detections) != 2:
            return swap_result
        
        # Check if players are in proximity (swap more likely)
        positions = [(det['position'][0], det['position'][1]) for det in track_detections]
        if not self.detect_players_in_proximity(positions):
            return swap_result
        
        # Get current mappings
        current_mapping = {}
        for detection in track_detections:
            track_id = detection['track_id']
            if track_id in self.track_to_player_mapping:
                current_mapping[track_id] = self.track_to_player_mapping[track_id]
        
        # If we don't have both tracks mapped, can't detect swaps
        if len(current_mapping) != 2:
            return swap_result
        
        # Check if both tracks are currently assigned to the same player (major issue)
        assigned_players = list(current_mapping.values())
        if len(set(assigned_players)) == 1:
            logger.warning(f"Both tracks assigned to same player {assigned_players[0]} - forcing correction")
            # Force position-based reassignment
            sorted_detections = sorted(track_detections, key=lambda x: x['position'][0])
            for i, detection in enumerate(sorted_detections):
                new_player_id = i + 1
                old_player_id = current_mapping[detection['track_id']]
                if new_player_id != old_player_id:
                    swap_result['swap_detected'] = True
                    swap_result['swapped_tracks'].append(detection['track_id'])
                    swap_result['corrected_mapping'][detection['track_id']] = new_player_id
                    swap_result['confidence'] = 1.0  # High confidence for this type of correction
                    
            if swap_result['swap_detected']:
                self.swap_detections += 1
                logger.info(f"Forced swap correction at frame {frame_count}: {swap_result['corrected_mapping']}")
                
                # Update track mapping
                for track_id, player_id in swap_result['corrected_mapping'].items():
                    self.track_to_player_mapping[track_id] = player_id
            
            return swap_result
        
        # Predict identity for each detection and check for mismatches
        identity_predictions = []
        for detection in track_detections:
            predicted_id, confidence = self.predict_player_identity(detection, frame_count)
            current_id = current_mapping[detection['track_id']]
            
            identity_predictions.append({
                'track_id': detection['track_id'],
                'predicted_player_id': predicted_id,
                'current_player_id': current_id,
                'confidence': confidence,
                'mismatch': predicted_id != current_id
            })
        
        # Check for swaps - both predictions should be swapped for high confidence
        mismatched_predictions = [p for p in identity_predictions if p['mismatch'] and p['confidence'] > self.confidence_threshold]
        
        if len(mismatched_predictions) == 2:
            # Check if it's a true swap (predictions are swapped)
            pred1, pred2 = mismatched_predictions
            if (pred1['predicted_player_id'] == pred2['current_player_id'] and 
                pred2['predicted_player_id'] == pred1['current_player_id']):
                
                # This is a true swap
                swap_result['swap_detected'] = True
                swap_result['confidence'] = (pred1['confidence'] + pred2['confidence']) / 2
                
                for pred in mismatched_predictions:
                    swap_result['swapped_tracks'].append(pred['track_id'])
                    swap_result['corrected_mapping'][pred['track_id']] = pred['predicted_player_id']
                
                self.swap_detections += 1
                logger.info(f"Track ID swap detected at frame {frame_count}: {swap_result['corrected_mapping']}")
                
                # Update track mapping
                for track_id, player_id in swap_result['corrected_mapping'].items():
                    self.track_to_player_mapping[track_id] = player_id
        
        return swap_result
    
    def update_player_references(self, detections: List[Dict], frame_count: int):
        """
        Update player references with new detections
        
        Args:
            detections: List of player detections
            frame_count: Current frame number
        """
        for detection in detections:
            track_id = detection['track_id']
            
            # Get player ID from current mapping
            if track_id in self.track_to_player_mapping:
                player_id = self.track_to_player_mapping[track_id]
            else:
                # Skip if no mapping exists
                continue
            
            # Update references only for valid player IDs
            if player_id in self.player_references:
                features = self.extract_appearance_features(detection['crop'])
                
                # Only add valid features
                if features is not None and len(features) > 0:
                    # Ensure features is a numpy array for the all() check
                    if hasattr(features, 'cpu'):  # It's a torch tensor
                        features_np = features.cpu().numpy() if features.is_cuda else features.numpy()
                    else:
                        features_np = features
                    
                    if not np.all(features_np == 0):
                        self.player_references[player_id]['appearance_features'].append(features)
                    self.player_references[player_id]['position_history'].append(detection['position'])
                    self.player_references[player_id]['last_seen_frame'] = frame_count
                    
                    # Keep only recent features (sliding window)
                    if len(self.player_references[player_id]['appearance_features']) > 20:
                        self.player_references[player_id]['appearance_features'].pop(0)
    
    def _handle_two_player_assignment(self, detections: List[Dict], frame_count: int):
        """
        Handle assignment when we have exactly 2 detections to ensure one goes to P1 and one to P2
        This is the critical function that prevents both players getting the same ID
        """
        if len(detections) != 2:
            return
        
        # Get current track IDs
        track_ids = [det['track_id'] for det in detections]
        
        # Check current mappings
        current_mappings = {}
        unmapped_tracks = []
        
        for track_id in track_ids:
            if track_id in self.track_to_player_mapping:
                current_mappings[track_id] = self.track_to_player_mapping[track_id]
            else:
                unmapped_tracks.append(track_id)
        
        # Case 1: Both tracks unmapped - assign based on position/appearance
        if len(unmapped_tracks) == 2:
            logger.info("Both tracks unmapped, performing initial assignment")
            self._assign_unmapped_tracks(detections, frame_count)
            return
        
        # Case 2: One track mapped, one unmapped
        if len(unmapped_tracks) == 1:
            mapped_players = set(current_mappings.values())
            available_players = {1, 2} - mapped_players
            
            if available_players:
                unmapped_track = unmapped_tracks[0]
                unmapped_detection = next(det for det in detections if det['track_id'] == unmapped_track)
                
                # Try to predict based on appearance
                predicted_id, confidence = self.predict_player_identity(unmapped_detection, frame_count)
                
                if predicted_id in available_players and confidence > self.confidence_threshold:
                    player_id = predicted_id
                else:
                    # Assign to the only available player
                    player_id = list(available_players)[0]
                
                self.track_to_player_mapping[unmapped_track] = player_id
                logger.info(f"Assigned unmapped track ID {unmapped_track} to player {player_id}")
            return
        
        # Case 3: Both tracks mapped - check for conflicts
        mapped_players = list(current_mappings.values())
        
        # If both tracks mapped to the same player - THIS IS THE CRITICAL FIX
        if len(set(mapped_players)) == 1:
            logger.warning(f"Both tracks mapped to same player {mapped_players[0]} - fixing assignment")
            self._fix_duplicate_assignment(detections, frame_count)
            return
        
        # Case 4: Both tracks mapped to different players - check for swaps
        if len(set(mapped_players)) == 2:
            # Check if we need to swap based on appearance/position
            self._check_and_fix_swaps(detections, frame_count)
            return
    
    def _assign_unmapped_tracks(self, detections: List[Dict], frame_count: int):
        """Assign two unmapped tracks to players 1 and 2"""
        # Sort by x-coordinate (left = player 1, right = player 2)
        sorted_detections = sorted(detections, key=lambda x: x['position'][0])
        
        # Try appearance-based assignment first if references exist
        assignments_made = []
        
        for target_player in [1, 2]:
            if self.player_references[target_player]['initialization_complete']:
                best_detection = None
                best_confidence = 0.0
                
                for detection in sorted_detections:
                    if detection['track_id'] not in [a[0] for a in assignments_made]:
                        predicted_id, confidence = self.predict_player_identity(detection, frame_count)
                        if predicted_id == target_player and confidence > best_confidence:
                            best_confidence = confidence
                            best_detection = detection
                
                if best_detection and best_confidence > self.confidence_threshold:
                    self.track_to_player_mapping[best_detection['track_id']] = target_player
                    assignments_made.append((best_detection['track_id'], target_player))
                    logger.info(f"Appearance-based assignment: Track {best_detection['track_id']} -> Player {target_player} (conf: {best_confidence:.3f})")
        
        # Assign remaining tracks by position
        assigned_tracks = [a[0] for a in assignments_made]
        remaining_detections = [det for det in sorted_detections if det['track_id'] not in assigned_tracks]
        assigned_players = [a[1] for a in assignments_made]
        available_players = [p for p in [1, 2] if p not in assigned_players]
        
        for detection, player_id in zip(remaining_detections, available_players):
            self.track_to_player_mapping[detection['track_id']] = player_id
            logger.info(f"Position-based assignment: Track {detection['track_id']} -> Player {player_id}")
    
    def _fix_duplicate_assignment(self, detections: List[Dict], frame_count: int):
        """Fix the case where both tracks are assigned to the same player"""
        # Sort detections by x-coordinate
        sorted_detections = sorted(detections, key=lambda x: x['position'][0])
        
        # Force assignment: left player = 1, right player = 2
        for i, (target_player, detection) in enumerate(zip([1, 2], sorted_detections)):
            old_assignment = self.track_to_player_mapping.get(detection['track_id'], None)
            self.track_to_player_mapping[detection['track_id']] = target_player
            
            if old_assignment != target_player:
                logger.info(f"Fixed duplicate assignment: Track {detection['track_id']} changed from Player {old_assignment} to Player {target_player}")
    
    def _check_and_fix_swaps(self, detections: List[Dict], frame_count: int):
        """Check if tracks need to be swapped based on appearance"""
        if not (self.player_references[1]['initialization_complete'] and 
                self.player_references[2]['initialization_complete']):
            return
        
        # Check if current assignments make sense based on appearance
        swaps_needed = []
        
        for detection in detections:
            track_id = detection['track_id']
            current_player = self.track_to_player_mapping[track_id]
            predicted_player, confidence = self.predict_player_identity(detection, frame_count)
            
            if (predicted_player != current_player and 
                confidence > self.confidence_threshold and
                confidence > 0.8):  # High confidence threshold for swapping
                swaps_needed.append((track_id, current_player, predicted_player, confidence))
        
        # Apply swaps if confident
        if swaps_needed:
            for track_id, old_player, new_player, confidence in swaps_needed:
                # Check if the target player assignment is available or would create a valid swap
                conflicting_track = None
                for other_track, other_player in self.track_to_player_mapping.items():
                    if other_player == new_player and other_track != track_id:
                        conflicting_track = other_track
                        break
                
                if conflicting_track:
                    # Swap the assignments
                    self.track_to_player_mapping[track_id] = new_player
                    self.track_to_player_mapping[conflicting_track] = old_player
                    logger.info(f"Swapped assignments: Track {track_id} -> Player {new_player}, Track {conflicting_track} -> Player {old_player}")
                else:
                    # Simple reassignment
                    self.track_to_player_mapping[track_id] = new_player
                    logger.info(f"Reassigned: Track {track_id} -> Player {new_player} (conf: {confidence:.3f})")
        
    def ensure_unique_assignments(self):
        """Ensure each player ID is assigned to at most one track"""
        # Get current assignments
        player_to_tracks = {1: [], 2: []}
        
        for track_id, player_id in self.track_to_player_mapping.items():
            if player_id in player_to_tracks:
                player_to_tracks[player_id].append(track_id)
        
        # Fix duplicates
        for player_id, track_list in player_to_tracks.items():
            if len(track_list) > 1:
                logger.warning(f"Player {player_id} assigned to multiple tracks: {track_list}")
                # Keep only the most recent assignment
                most_recent_track = max(track_list)
                for track_id in track_list:
                    if track_id != most_recent_track:
                        del self.track_to_player_mapping[track_id]
                        logger.info(f"Removed duplicate assignment: Track {track_id}")
        
        # Ensure both players are represented if we have 2 tracks
        assigned_players = set(self.track_to_player_mapping.values())
        all_tracks = list(self.track_to_player_mapping.keys())
        
        if len(all_tracks) == 2 and len(assigned_players) == 1:
            # Both tracks assigned to same player - fix it
            sorted_tracks = sorted(all_tracks)
            self.track_to_player_mapping[sorted_tracks[0]] = 1
            self.track_to_player_mapping[sorted_tracks[1]] = 2
            logger.info(f"Fixed same-player assignment: Track {sorted_tracks[0]} -> Player 1, Track {sorted_tracks[1]} -> Player 2")
    
    def process_frame(self, detections: List[Dict], frame_count: int) -> Dict:
        """
        Process a single frame for player re-identification
        
        Args:
            detections: List of player detections with crops, positions, and track IDs
            frame_count: Current frame number
            
        Returns:
            Dictionary with processing results
        """
        result = {
            'frame_count': frame_count,
            'player_assignments': {},
            'swap_detection': None,
            'initialization_status': {
                1: self.player_references[1]['initialization_complete'],
                2: self.player_references[2]['initialization_complete']
            }
        }
        
        if not detections:
            return result
        
        # Initialize references if in initialization window
        if self.is_initialization_frame(frame_count):
            self.initialize_player_references(detections, frame_count)
        
        # CRITICAL: Handle assignment logic to prevent duplicate player IDs
        if len(detections) == 2:
            # This is the most common case - ensure unique assignments
            self._handle_two_player_assignment(detections, frame_count)
        elif len(detections) == 1:
            # Single detection - assign to best matching player
            detection = detections[0]
            track_id = detection['track_id']
            
            if track_id not in self.track_to_player_mapping:
                predicted_id, confidence = self.predict_player_identity(detection, frame_count)
                if confidence > self.confidence_threshold:
                    # Check if this player ID is already taken
                    occupied_players = set(self.track_to_player_mapping.values())
                    if predicted_id not in occupied_players:
                        self.track_to_player_mapping[track_id] = predicted_id
                    else:
                        # Assign to the first available player ID
                        available_players = {1, 2} - occupied_players
                        if available_players:
                            self.track_to_player_mapping[track_id] = list(available_players)[0]
        
        # Always ensure unique assignments after any changes
        self.ensure_unique_assignments()
        
        # Check for track ID swaps (after ensuring unique assignments)
        if (len(detections) >= 2 and 
            self.player_references[1]['initialization_complete'] and 
            self.player_references[2]['initialization_complete']):
            result['swap_detection'] = self.detect_track_id_swap(detections, frame_count)
        
        # Update references
        self.update_player_references(detections, frame_count)
        
        # Create final player assignments
        for detection in detections:
            track_id = detection['track_id']
            if track_id in self.track_to_player_mapping:
                player_id = self.track_to_player_mapping[track_id]
                result['player_assignments'][track_id] = player_id
        
        # Validate assignments (ensure no duplicates)
        assigned_players = list(result['player_assignments'].values())
        if len(assigned_players) != len(set(assigned_players)):
            logger.error(f"Duplicate player assignments detected: {result['player_assignments']}")
            # Emergency fix: reassign by position
            if len(detections) == 2:
                sorted_detections = sorted(detections, key=lambda x: x['position'][0])
                result['player_assignments'] = {}
                for i, detection in enumerate(sorted_detections):
                    player_id = i + 1
                    result['player_assignments'][detection['track_id']] = player_id
                    self.track_to_player_mapping[detection['track_id']] = player_id
                logger.info("Emergency reassignment by position applied")
        
        return result
    
    def get_statistics(self) -> Dict:
        """Get system statistics and performance metrics"""
        return {
            'total_swaps_detected': self.swap_detections,
            'initialization_status': {
                1: self.player_references[1]['initialization_complete'],
                2: self.player_references[2]['initialization_complete']
            },
            'reference_counts': {
                1: len(self.player_references[1]['appearance_features']),
                2: len(self.player_references[2]['appearance_features'])
            },
            'current_mappings': dict(self.track_to_player_mapping)
        }
    
    def save_references(self, filepath: str):
        """Save player references to file"""
        try:
            # Convert numpy arrays to lists for JSON serialization
            references_data = {}
            for player_id, data in self.player_references.items():
                references_data[str(player_id)] = {
                    'appearance_features': [feat.tolist() for feat in data['appearance_features']],
                    'position_history': list(data['position_history']),
                    'last_seen_frame': data['last_seen_frame'],
                    'initialization_complete': data['initialization_complete']
                }
            
            with open(filepath, 'w') as f:
                json.dump(references_data, f, indent=2)
            
            logger.info(f"Player references saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving references: {e}")
    
    def load_references(self, filepath: str):
        """Load player references from file"""
        try:
            with open(filepath, 'r') as f:
                references_data = json.load(f)
            
            # Convert back to numpy arrays
            for player_id_str, data in references_data.items():
                player_id = int(player_id_str)
                self.player_references[player_id]['appearance_features'] = [
                    np.array(feat) for feat in data['appearance_features']
                ]
                self.player_references[player_id]['position_history'] = deque(
                    data['position_history'], maxlen=50
                )
                self.player_references[player_id]['last_seen_frame'] = data['last_seen_frame']
                self.player_references[player_id]['initialization_complete'] = data['initialization_complete']
            
            logger.info(f"Player references loaded from {filepath}")
            
        except Exception as e:
            logger.error(f"Error loading references: {e}")
