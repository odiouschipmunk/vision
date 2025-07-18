"""
Enhanced Player Re-Identification (ReID) System for Squash Analysis
================================================================

This module provides an advanced player re-identification system that:
1. Captures initial player appearances when they are separated (frames 100-150)
2. Continuously monitors for track ID swapping when players come close
3. Uses multiple features: appearance, position, and temporal consistency
4. Provides confidence scores for identity assignments
"""

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
import math
from typing import Dict, List, Tuple, Optional
import json
from collections import deque
import logging

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
            # Use ResNet50 for robust feature extraction
            self.feature_extractor = resnet50(weights=ResNet50_Weights.DEFAULT)
            self.feature_extractor.fc = torch.nn.Identity()  # Remove final classification layer
            self.feature_extractor.to(self.device)
            self.feature_extractor.eval()
            
            # Preprocessing pipeline
            self.preprocess = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            logger.info("Feature extractor initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing feature extractor: {e}")
            raise
    
    def extract_appearance_features(self, player_crop: np.ndarray) -> np.ndarray:
        """
        Extract deep appearance features from player crop
        
        Args:
            player_crop: RGB image crop of player
            
        Returns:
            Feature vector as numpy array
        """
        try:
            if player_crop.size == 0:
                return np.zeros(2048)
            
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
        
        # Predict identity for each detection
        identity_predictions = []
        for detection in track_detections:
            predicted_id, confidence = self.predict_player_identity(detection, frame_count)
            identity_predictions.append({
                'track_id': detection['track_id'],
                'predicted_player_id': predicted_id,
                'confidence': confidence,
                'current_mapping': self.track_to_player_mapping.get(detection['track_id'], predicted_id)
            })
        
        # Check for inconsistencies in mapping
        for pred in identity_predictions:
            if (pred['track_id'] in self.track_to_player_mapping and 
                pred['predicted_player_id'] != self.track_to_player_mapping[pred['track_id']] and
                pred['confidence'] > self.confidence_threshold):
                
                swap_result['swap_detected'] = True
                swap_result['swapped_tracks'].append(pred['track_id'])
                swap_result['corrected_mapping'][pred['track_id']] = pred['predicted_player_id']
                swap_result['confidence'] = max(swap_result['confidence'], pred['confidence'])
        
        # If swap detected, update mapping
        if swap_result['swap_detected']:
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
        # Handle case where we have 2 detections but need to assign them to players 1 and 2
        if len(detections) == 2:
            self._handle_two_player_assignment(detections, frame_count)
        
        for detection in detections:
            track_id = detection['track_id']
            
            # Determine player ID
            if track_id in self.track_to_player_mapping:
                player_id = self.track_to_player_mapping[track_id]
            else:
                # Predict player ID
                player_id, confidence = self.predict_player_identity(detection, frame_count)
                if confidence > self.confidence_threshold:
                    self.track_to_player_mapping[track_id] = player_id
                else:
                    continue  # Skip if confidence too low
            
            # Update references
            if player_id in self.player_references:
                features = self.extract_appearance_features(detection['crop'])
                
                self.player_references[player_id]['appearance_features'].append(features)
                self.player_references[player_id]['position_history'].append(detection['position'])
                self.player_references[player_id]['last_seen_frame'] = frame_count
                
                # Keep only recent features (sliding window)
                if len(self.player_references[player_id]['appearance_features']) > 20:
                    self.player_references[player_id]['appearance_features'].pop(0)
    
    def _handle_two_player_assignment(self, detections: List[Dict], frame_count: int):
        """
        Handle assignment when we have exactly 2 detections to ensure one goes to P1 and one to P2
        """
        if len(detections) != 2:
            return
        
        # Get current mappings
        mapped_players = set()
        unmapped_detections = []
        
        for detection in detections:
            track_id = detection['track_id']
            if track_id in self.track_to_player_mapping:
                mapped_players.add(self.track_to_player_mapping[track_id])
            else:
                unmapped_detections.append(detection)
        
        # If both detections map to the same player, we need to reassign one
        assigned_players = [self.track_to_player_mapping.get(det['track_id']) for det in detections]
        if len(set(filter(None, assigned_players))) == 1 and None not in assigned_players:
            # Both track IDs are mapped to the same player - reassign based on position/appearance
            self._reassign_duplicate_mapping(detections, frame_count)
        
        # Assign unmapped detections to available player IDs
        available_players = {1, 2} - mapped_players
        for detection in unmapped_detections:
            if available_players:
                # Try to predict based on appearance if possible
                predicted_id, confidence = self.predict_player_identity(detection, frame_count)
                
                if predicted_id in available_players and confidence > self.confidence_threshold:
                    player_id = predicted_id
                else:
                    # Assign to any available player
                    player_id = available_players.pop()
                
                self.track_to_player_mapping[detection['track_id']] = player_id
                available_players.discard(player_id)
                logger.info(f"Assigned track ID {detection['track_id']} to player {player_id}")
    
    def _reassign_duplicate_mapping(self, detections: List[Dict], frame_count: int):
        """
        Reassign when both detections map to the same player
        """
        if len(detections) != 2:
            return
        
        # Sort by x-coordinate (left = player 1, right = player 2)
        sorted_detections = sorted(detections, key=lambda x: x['position'][0])
        
        # Try to use appearance-based prediction first
        reassigned = False
        for i, (target_player, detection) in enumerate(zip([1, 2], sorted_detections)):
            predicted_id, confidence = self.predict_player_identity(detection, frame_count)
            
            if predicted_id == target_player and confidence > self.confidence_threshold:
                self.track_to_player_mapping[detection['track_id']] = target_player
                reassigned = True
        
        # If appearance-based reassignment didn't work, use position-based
        if not reassigned:
            for i, (target_player, detection) in enumerate(zip([1, 2], sorted_detections)):
                self.track_to_player_mapping[detection['track_id']] = target_player
                logger.info(f"Position-based reassignment: Track ID {detection['track_id']} -> Player {target_player}")
    
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
        
        # Initialize references if in initialization window
        if self.is_initialization_frame(frame_count):
            self.initialize_player_references(detections, frame_count)
        
        # Check for track ID swaps
        if (self.player_references[1]['initialization_complete'] and 
            self.player_references[2]['initialization_complete']):
            result['swap_detection'] = self.detect_track_id_swap(detections, frame_count)
        
        # Update references
        self.update_player_references(detections, frame_count)
        
        # Create player assignments
        for detection in detections:
            track_id = detection['track_id']
            if track_id in self.track_to_player_mapping:
                player_id = self.track_to_player_mapping[track_id]
                result['player_assignments'][track_id] = player_id
        
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
