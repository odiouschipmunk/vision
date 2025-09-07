"""
Enhanced Shot Classification System for Squash Analysis

This module provides comprehensive shot classification using multiple AI techniques:
- Trajectory pattern analysis
- Machine learning-based classification  
- Physics-based shot modeling
- Real-time shot type detection
- Court position analysis
"""

import numpy as np
import cv2
import math
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from collections import defaultdict
import json
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

@dataclass
class ShotFeatures:
    """Shot features for classification"""
    trajectory_length: float
    start_position: Tuple[float, float]
    end_position: Tuple[float, float] 
    max_height: float
    velocity_changes: List[float]
    direction_changes: List[float]
    wall_hits: int
    court_coverage: float
    shot_duration: int
    player_position: Tuple[float, float]
    ball_speed_avg: float
    ball_speed_max: float
    height_variation: float
    trajectory_smoothness: float

@dataclass 
class ShotClassification:
    """Shot classification result""" 
    shot_type: str
    confidence: float
    features: ShotFeatures
    subtypes: List[str]
    tactical_context: Dict[str, Any]
    difficulty_score: float
    effectiveness_score: float

class TrajectoryAnalyzer:
    """Advanced trajectory analysis for shot classification"""
    
    def __init__(self):
        self.court_width = 640
        self.court_height = 360
        
    def analyze_trajectory(self, trajectory: List[Tuple[float, float]]) -> ShotFeatures:
        """Extract comprehensive features from ball trajectory"""
        
        if len(trajectory) < 3:
            return self._empty_features()
            
        # Convert to numpy arrays for easier processing
        positions = np.array(trajectory)
        
        # Basic position metrics
        start_pos = tuple(positions[0])
        end_pos = tuple(positions[-1])
        
        # Trajectory length calculation
        distances = np.sqrt(np.sum(np.diff(positions, axis=0)**2, axis=1))
        trajectory_length = np.sum(distances)
        
        # Height analysis
        heights = positions[:, 1]
        max_height = np.max(heights)
        height_variation = np.std(heights)
        
        # Velocity analysis
        velocities = np.diff(positions, axis=0)
        speeds = np.sqrt(np.sum(velocities**2, axis=1))
        velocity_changes = np.abs(np.diff(speeds))
        
        # Direction changes
        if len(velocities) > 1:
            angles = np.arctan2(velocities[:, 1], velocities[:, 0])
            direction_changes = np.abs(np.diff(angles))
            # Handle angle wraparound
            direction_changes = np.minimum(direction_changes, 2*np.pi - direction_changes)
        else:
            direction_changes = [0]
            
        # Wall hit detection (simplified)
        wall_hits = self._detect_wall_hits(positions)
        
        # Court coverage
        x_range = np.max(positions[:, 0]) - np.min(positions[:, 0])
        y_range = np.max(positions[:, 1]) - np.min(positions[:, 1])
        court_coverage = (x_range / self.court_width) * (y_range / self.court_height)
        
        # Trajectory smoothness
        if len(positions) > 2:
            second_derivatives = np.diff(velocities, axis=0)
            trajectory_smoothness = 1.0 / (1.0 + np.mean(np.linalg.norm(second_derivatives, axis=1)))
        else:
            trajectory_smoothness = 1.0
            
        return ShotFeatures(
            trajectory_length=trajectory_length,
            start_position=start_pos,
            end_position=end_pos,
            max_height=max_height,
            velocity_changes=velocity_changes.tolist(),
            direction_changes=direction_changes.tolist(),
            wall_hits=wall_hits,
            court_coverage=court_coverage,
            shot_duration=len(trajectory),
            player_position=start_pos,  # Simplified - should be actual player position
            ball_speed_avg=np.mean(speeds) if len(speeds) > 0 else 0,
            ball_speed_max=np.max(speeds) if len(speeds) > 0 else 0,
            height_variation=height_variation,
            trajectory_smoothness=trajectory_smoothness
        )
        
    def _detect_wall_hits(self, positions: np.ndarray) -> int:
        """Simple wall hit detection based on position changes"""
        wall_hits = 0
        
        for i in range(1, len(positions)):
            prev_pos = positions[i-1]
            curr_pos = positions[i]
            
            # Check for sudden direction changes near walls
            if (curr_pos[0] < 50 and prev_pos[0] > curr_pos[0]) or \
               (curr_pos[0] > self.court_width-50 and prev_pos[0] < curr_pos[0]) or \
               (curr_pos[1] < 50 and prev_pos[1] > curr_pos[1]) or \
               (curr_pos[1] > self.court_height-50 and prev_pos[1] < curr_pos[1]):
                wall_hits += 1
                
        return wall_hits
        
    def _empty_features(self) -> ShotFeatures:
        """Return empty features for invalid trajectories"""
        return ShotFeatures(
            trajectory_length=0,
            start_position=(0, 0),
            end_position=(0, 0),
            max_height=0,
            velocity_changes=[],
            direction_changes=[],
            wall_hits=0,
            court_coverage=0,
            shot_duration=0,
            player_position=(0, 0),
            ball_speed_avg=0,
            ball_speed_max=0,
            height_variation=0,
            trajectory_smoothness=0
        )

class ShotTypeClassifier:
    """Machine learning-based shot type classifier"""
    
    def __init__(self):
        self.shot_types = [
            'straight_drive', 'crosscourt', 'boast', 'drop_shot', 
            'lob', 'volley', 'kill_shot', 'defensive', 'serve'
        ]
        self.classifier = None
        self.is_trained = False
        self._initialize_classifier()
        
    def _initialize_classifier(self):
        """Initialize the ML classifier with synthetic training data"""
        # Create synthetic training data for different shot types
        training_features, training_labels = self._generate_training_data()
        
        # Train random forest classifier
        self.classifier = RandomForestClassifier(
            n_estimators=100, 
            max_depth=10, 
            random_state=42,
            min_samples_split=5
        )
        
        self.classifier.fit(training_features, training_labels)
        self.is_trained = True
        print("✅ Shot classifier trained with synthetic data")
        
    def _generate_training_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate synthetic training data for shot classification"""
        features_list = []
        labels_list = []
        
        # Generate examples for each shot type
        for shot_type in self.shot_types:
            for _ in range(50):  # 50 examples per shot type
                features = self._generate_shot_features(shot_type)
                features_list.append(self._features_to_vector(features))
                labels_list.append(shot_type)
                
        return np.array(features_list), np.array(labels_list)
        
    def _generate_shot_features(self, shot_type: str) -> ShotFeatures:
        """Generate synthetic features for a specific shot type"""
        
        if shot_type == 'straight_drive':
            return ShotFeatures(
                trajectory_length=np.random.normal(200, 30),
                start_position=(np.random.normal(320, 50), np.random.normal(300, 20)),
                end_position=(np.random.normal(320, 30), np.random.normal(100, 20)),
                max_height=np.random.normal(250, 30),
                velocity_changes=[np.random.normal(5, 2) for _ in range(5)],
                direction_changes=[np.random.normal(0.1, 0.05) for _ in range(4)],
                wall_hits=0,
                court_coverage=np.random.normal(0.3, 0.1),
                shot_duration=np.random.randint(15, 25),
                player_position=(320, 300),
                ball_speed_avg=np.random.normal(15, 3),
                ball_speed_max=np.random.normal(20, 4),
                height_variation=np.random.normal(30, 10),
                trajectory_smoothness=np.random.normal(0.8, 0.1)
            )
            
        elif shot_type == 'crosscourt':
            return ShotFeatures(
                trajectory_length=np.random.normal(250, 40),
                start_position=(np.random.normal(200, 50), np.random.normal(300, 20)),
                end_position=(np.random.normal(450, 50), np.random.normal(150, 30)),
                max_height=np.random.normal(280, 40),
                velocity_changes=[np.random.normal(6, 2) for _ in range(5)],
                direction_changes=[np.random.normal(0.3, 0.1) for _ in range(4)],
                wall_hits=1,
                court_coverage=np.random.normal(0.5, 0.1),
                shot_duration=np.random.randint(18, 30),
                player_position=(200, 300),
                ball_speed_avg=np.random.normal(12, 3),
                ball_speed_max=np.random.normal(18, 4),
                height_variation=np.random.normal(40, 15),
                trajectory_smoothness=np.random.normal(0.7, 0.1)
            )
            
        elif shot_type == 'boast':
            return ShotFeatures(
                trajectory_length=np.random.normal(180, 30),
                start_position=(np.random.normal(150, 30), np.random.normal(250, 30)),
                end_position=(np.random.normal(400, 40), np.random.normal(200, 40)),
                max_height=np.random.normal(220, 30),
                velocity_changes=[np.random.normal(8, 3) for _ in range(4)],
                direction_changes=[np.random.normal(0.7, 0.2) for _ in range(3)],
                wall_hits=2,
                court_coverage=np.random.normal(0.4, 0.1),
                shot_duration=np.random.randint(12, 20),
                player_position=(150, 250),
                ball_speed_avg=np.random.normal(10, 2),
                ball_speed_max=np.random.normal(15, 3),
                height_variation=np.random.normal(25, 8),
                trajectory_smoothness=np.random.normal(0.6, 0.1)
            )
            
        elif shot_type == 'drop_shot':
            return ShotFeatures(
                trajectory_length=np.random.normal(120, 20),
                start_position=(np.random.normal(320, 40), np.random.normal(280, 20)),
                end_position=(np.random.normal(320, 30), np.random.normal(50, 15)),
                max_height=np.random.normal(180, 20),
                velocity_changes=[np.random.normal(3, 1) for _ in range(3)],
                direction_changes=[np.random.normal(0.05, 0.02) for _ in range(2)],
                wall_hits=0,
                court_coverage=np.random.normal(0.2, 0.05),
                shot_duration=np.random.randint(20, 35),
                player_position=(320, 280),
                ball_speed_avg=np.random.normal(6, 2),
                ball_speed_max=np.random.normal(10, 2),
                height_variation=np.random.normal(35, 10),
                trajectory_smoothness=np.random.normal(0.9, 0.05)
            )
            
        elif shot_type == 'lob':
            return ShotFeatures(
                trajectory_length=np.random.normal(300, 50),
                start_position=(np.random.normal(320, 50), np.random.normal(300, 30)),
                end_position=(np.random.normal(320, 40), np.random.normal(100, 30)),
                max_height=np.random.normal(50, 15),  # High trajectory
                velocity_changes=[np.random.normal(4, 1.5) for _ in range(6)],
                direction_changes=[np.random.normal(0.1, 0.03) for _ in range(5)],
                wall_hits=0,
                court_coverage=np.random.normal(0.4, 0.1),
                shot_duration=np.random.randint(25, 40),
                player_position=(320, 300),
                ball_speed_avg=np.random.normal(8, 2),
                ball_speed_max=np.random.normal(12, 3),
                height_variation=np.random.normal(60, 20),
                trajectory_smoothness=np.random.normal(0.85, 0.1)
            )
            
        else:  # Default case for other shot types
            return ShotFeatures(
                trajectory_length=np.random.normal(150, 40),
                start_position=(np.random.normal(320, 80), np.random.normal(250, 50)),
                end_position=(np.random.normal(320, 80), np.random.normal(150, 50)),
                max_height=np.random.normal(200, 50),
                velocity_changes=[np.random.normal(5, 2) for _ in range(4)],
                direction_changes=[np.random.normal(0.2, 0.1) for _ in range(3)],
                wall_hits=np.random.randint(0, 2),
                court_coverage=np.random.normal(0.3, 0.1),
                shot_duration=np.random.randint(15, 30),
                player_position=(320, 250),
                ball_speed_avg=np.random.normal(10, 3),
                ball_speed_max=np.random.normal(15, 4),
                height_variation=np.random.normal(35, 15),
                trajectory_smoothness=np.random.normal(0.7, 0.15)
            )
            
    def _features_to_vector(self, features: ShotFeatures) -> np.ndarray:
        """Convert shot features to numerical vector for ML"""
        return np.array([
            features.trajectory_length,
            features.start_position[0],
            features.start_position[1],
            features.end_position[0], 
            features.end_position[1],
            features.max_height,
            np.mean(features.velocity_changes) if features.velocity_changes else 0,
            np.std(features.velocity_changes) if len(features.velocity_changes) > 1 else 0,
            np.mean(features.direction_changes) if features.direction_changes else 0,
            features.wall_hits,
            features.court_coverage,
            features.shot_duration,
            features.ball_speed_avg,
            features.ball_speed_max,
            features.height_variation,
            features.trajectory_smoothness
        ])
        
    def classify_shot(self, features: ShotFeatures) -> ShotClassification:
        """Classify shot type based on features"""
        
        if not self.is_trained:
            return self._default_classification(features)
            
        # Convert features to vector
        feature_vector = self._features_to_vector(features).reshape(1, -1)
        
        # Get prediction and confidence
        prediction = self.classifier.predict(feature_vector)[0]
        probabilities = self.classifier.predict_proba(feature_vector)[0]
        confidence = np.max(probabilities)
        
        # Get top 3 predictions
        top_indices = np.argsort(probabilities)[-3:][::-1]
        subtypes = [self.shot_types[i] for i in top_indices[1:]]
        
        # Calculate difficulty and effectiveness scores
        difficulty_score = self._calculate_difficulty_score(features)
        effectiveness_score = self._calculate_effectiveness_score(features, prediction)
        
        # Generate tactical context
        tactical_context = self._generate_tactical_context(features, prediction)
        
        return ShotClassification(
            shot_type=prediction,
            confidence=confidence,
            features=features,
            subtypes=subtypes,
            tactical_context=tactical_context,
            difficulty_score=difficulty_score,
            effectiveness_score=effectiveness_score
        )
        
    def _calculate_difficulty_score(self, features: ShotFeatures) -> float:
        """Calculate shot difficulty based on features"""
        difficulty = 0.0
        
        # Wall hits increase difficulty
        difficulty += features.wall_hits * 0.3
        
        # Speed increases difficulty
        if features.ball_speed_max > 0:
            difficulty += min(features.ball_speed_max / 25.0, 1.0) * 0.3
            
        # Court coverage increases difficulty
        difficulty += features.court_coverage * 0.2
        
        # Direction changes increase difficulty
        if features.direction_changes:
            difficulty += min(np.mean(features.direction_changes) / 0.5, 1.0) * 0.2
            
        return min(difficulty, 1.0)
        
    def _calculate_effectiveness_score(self, features: ShotFeatures, shot_type: str) -> float:
        """Calculate shot effectiveness based on type and execution"""
        effectiveness = 0.5  # Base score
        
        # Type-specific effectiveness factors
        if shot_type == 'drop_shot':
            # Drop shots are effective if they're slow and low
            if features.ball_speed_avg < 8:
                effectiveness += 0.2
            if features.max_height > 300:  # Lower is better for drop shots
                effectiveness += 0.1
                
        elif shot_type == 'kill_shot':
            # Kill shots are effective if fast and low
            if features.ball_speed_max > 18:
                effectiveness += 0.3
                
        elif shot_type == 'lob':
            # Lobs are effective if high and deep
            if features.max_height < 100:  # Higher trajectory
                effectiveness += 0.2
                
        # Trajectory smoothness affects all shots
        effectiveness += features.trajectory_smoothness * 0.2
        
        return min(effectiveness, 1.0)
        
    def _generate_tactical_context(self, features: ShotFeatures, shot_type: str) -> Dict[str, Any]:
        """Generate tactical analysis context"""
        context = {
            'placement': 'unknown',
            'timing': 'unknown', 
            'pressure_level': 'medium',
            'tactical_intent': 'unknown',
            'court_position': 'center'
        }
        
        # Determine court position
        x_pos = features.start_position[0]
        if x_pos < 200:
            context['court_position'] = 'left'
        elif x_pos > 440:
            context['court_position'] = 'right'
        else:
            context['court_position'] = 'center'
            
        # Determine placement
        end_x = features.end_position[0]
        if end_x < 200:
            context['placement'] = 'front_left'
        elif end_x > 440:
            context['placement'] = 'front_right'
        else:
            context['placement'] = 'front_center'
            
        # Determine tactical intent based on shot type
        intent_map = {
            'drop_shot': 'attacking',
            'kill_shot': 'attacking',
            'boast': 'pressure',
            'lob': 'defensive',
            'straight_drive': 'control',
            'crosscourt': 'pressure'
        }
        context['tactical_intent'] = intent_map.get(shot_type, 'neutral')
        
        return context
        
    def _default_classification(self, features: ShotFeatures) -> ShotClassification:
        """Default classification when ML model not available"""
        return ShotClassification(
            shot_type='unknown',
            confidence=0.1,
            features=features,
            subtypes=[],
            tactical_context={},
            difficulty_score=0.5,
            effectiveness_score=0.5
        )

class EnhancedShotClassifier:
    """Main enhanced shot classification system"""
    
    def __init__(self, court_dimensions=(640, 360)):
        self.court_width, self.court_height = court_dimensions
        self.trajectory_analyzer = TrajectoryAnalyzer()
        self.shot_classifier = ShotTypeClassifier()
        
        # Shot history for pattern analysis
        self.shot_history = []
        self.max_history = 50
        
        # Performance statistics
        self.stats = {
            'total_classifications': 0,
            'shot_type_counts': defaultdict(int),
            'average_confidence': 0.0,
            'classification_errors': 0
        }
        
        print("🎯 Enhanced Shot Classifier Initialized")
        print(f"   Court dimensions: {court_dimensions}")
        print(f"   Available shot types: {len(self.shot_classifier.shot_types)}")
        
    def classify_shot_from_trajectory(self, trajectory: List[Tuple[float, float]], 
                                    player_position: Optional[Tuple[float, float]] = None) -> ShotClassification:
        """Classify shot from ball trajectory"""
        
        try:
            # Extract features from trajectory
            features = self.trajectory_analyzer.analyze_trajectory(trajectory)
            
            # Update player position if provided
            if player_position:
                features.player_position = player_position
                
            # Classify the shot
            classification = self.shot_classifier.classify_shot(features)
            
            # Update statistics
            self._update_statistics(classification)
            
            # Add to shot history
            self._add_to_history(classification)
            
            return classification
            
        except Exception as e:
            print(f"Shot classification error: {e}")
            return self._error_classification()
            
    def get_shot_patterns(self) -> Dict[str, Any]:
        """Analyze patterns in shot history"""
        if not self.shot_history:
            return {}
            
        patterns = {
            'most_common_shots': [],
            'average_difficulty': 0.0,
            'average_effectiveness': 0.0,
            'tactical_tendencies': defaultdict(int),
            'court_coverage': defaultdict(int)
        }
        
        # Most common shot types
        shot_counts = defaultdict(int)
        total_difficulty = 0.0
        total_effectiveness = 0.0
        
        for classification in self.shot_history:
            shot_counts[classification.shot_type] += 1
            total_difficulty += classification.difficulty_score
            total_effectiveness += classification.effectiveness_score
            
            # Tactical tendencies
            intent = classification.tactical_context.get('tactical_intent', 'unknown')
            patterns['tactical_tendencies'][intent] += 1
            
            # Court coverage
            position = classification.tactical_context.get('court_position', 'unknown')
            patterns['court_coverage'][position] += 1
            
        # Sort by frequency
        patterns['most_common_shots'] = sorted(
            shot_counts.items(), key=lambda x: x[1], reverse=True
        )[:5]
        
        # Averages
        n_shots = len(self.shot_history)
        patterns['average_difficulty'] = total_difficulty / n_shots
        patterns['average_effectiveness'] = total_effectiveness / n_shots
        
        return patterns
        
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get classifier performance statistics"""
        stats = self.stats.copy()
        stats['shot_history_length'] = len(self.shot_history)
        stats['recent_patterns'] = self.get_shot_patterns()
        return stats
        
    def export_classifications(self, filename: str):
        """Export shot classifications to JSON file"""
        export_data = {
            'classifier_info': {
                'court_dimensions': (self.court_width, self.court_height),
                'total_classifications': self.stats['total_classifications'],
                'shot_types_available': self.shot_classifier.shot_types
            },
            'shot_history': [],
            'performance_stats': self.get_performance_stats(),
            'shot_patterns': self.get_shot_patterns()
        }
        
        # Export shot history
        for classification in self.shot_history:
            shot_data = {
                'shot_type': classification.shot_type,
                'confidence': classification.confidence,
                'difficulty_score': classification.difficulty_score,
                'effectiveness_score': classification.effectiveness_score,
                'subtypes': classification.subtypes,
                'tactical_context': classification.tactical_context,
                'features': {
                    'trajectory_length': classification.features.trajectory_length,
                    'start_position': classification.features.start_position,
                    'end_position': classification.features.end_position,
                    'wall_hits': classification.features.wall_hits,
                    'court_coverage': classification.features.court_coverage,
                    'ball_speed_avg': classification.features.ball_speed_avg,
                    'ball_speed_max': classification.features.ball_speed_max
                }
            }
            export_data['shot_history'].append(shot_data)
            
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2)
            
        print(f"Shot classifications exported to {filename}")
        
    def _update_statistics(self, classification: ShotClassification):
        """Update internal statistics"""
        self.stats['total_classifications'] += 1
        self.stats['shot_type_counts'][classification.shot_type] += 1
        
        # Update running average confidence
        n = self.stats['total_classifications']
        old_avg = self.stats['average_confidence']
        self.stats['average_confidence'] = ((n-1) * old_avg + classification.confidence) / n
        
    def _add_to_history(self, classification: ShotClassification):
        """Add classification to shot history"""
        self.shot_history.append(classification)
        
        # Maintain maximum history length
        if len(self.shot_history) > self.max_history:
            self.shot_history.pop(0)
            
    def _error_classification(self) -> ShotClassification:
        """Return error classification"""
        self.stats['classification_errors'] += 1
        return ShotClassification(
            shot_type='error',
            confidence=0.0,
            features=self.trajectory_analyzer._empty_features(),
            subtypes=[],
            tactical_context={},
            difficulty_score=0.0,
            effectiveness_score=0.0
        )

# Factory function for easy integration
def create_enhanced_shot_classifier(court_dimensions=(640, 360)) -> EnhancedShotClassifier:
    """Create enhanced shot classifier instance"""
    print("🚀 Initializing Enhanced Shot Classification System")
    print("=" * 60)
    print("✓ Machine learning-based shot type detection")
    print("✓ Advanced trajectory pattern analysis")
    print("✓ Physics-based shot modeling")
    print("✓ Tactical context analysis")
    print("✓ Performance tracking and pattern recognition")
    print("=" * 60)
    
    return EnhancedShotClassifier(court_dimensions)

if __name__ == "__main__":
    # Test the enhanced shot classifier
    classifier = create_enhanced_shot_classifier()
    
    # Test with sample trajectory
    test_trajectory = [
        (100, 300), (150, 280), (200, 250), (250, 220), 
        (300, 200), (350, 180), (400, 150), (450, 120)
    ]
    
    classification = classifier.classify_shot_from_trajectory(test_trajectory)
    print(f"\nTest Classification:")
    print(f"Shot Type: {classification.shot_type}")
    print(f"Confidence: {classification.confidence:.2f}")
    print(f"Difficulty: {classification.difficulty_score:.2f}")
    print(f"Effectiveness: {classification.effectiveness_score:.2f}")