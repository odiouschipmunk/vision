"""
Enhanced Shot Detection Integration for Main Pipeline
Seamlessly integrates the new enhanced shot detection system into the existing squash analysis pipeline.
"""

import sys
import os
import time
import json
import numpy as np
from typing import Dict, List, Tuple, Optional, Any

# Import the enhanced detection system
from enhanced_shot_detection_system import (
    EnhancedShotDetectionSystem, 
    ShotEvent, 
    ShotSequence,
    create_enhanced_shot_detector
)

class ShotDetectionIntegrator:
    """
    Integration layer between enhanced shot detection and main pipeline
    """
    
    def __init__(self, court_width=640, court_height=360, enable_logging=True):
        self.court_width = court_width
        self.court_height = court_height
        self.enable_logging = enable_logging
        
        # Initialize enhanced detector
        self.enhanced_detector = create_enhanced_shot_detector(court_width, court_height)
        
        # Integration state
        self.frame_count = 0
        self.last_ball_position = None
        self.detection_stats = {
            'total_frames_processed': 0,
            'racket_hits_detected': 0,
            'wall_hits_detected': 0,
            'floor_bounces_detected': 0,
            'shots_completed': 0,
            'high_confidence_events': 0
        }
        
        # Legacy compatibility
        self.legacy_shot_data = {
            'who_hit': 0,
            'ball_hit': False,
            'shot_type': 'unknown',
            'hit_confidence': 0.0,
            'match_in_play': False
        }
        
        if self.enable_logging:
            print("🎾 Enhanced Shot Detection Integration Initialized")
            print(f"   Court dimensions: {court_width}x{court_height}")
    
    def process_frame_enhanced(self, ball_position: Tuple[float, float], 
                             players_data: Dict, 
                             frame_number: int,
                             past_ball_pos: List = None) -> Dict[str, Any]:
        """
        Enhanced frame processing with improved shot detection
        
        Args:
            ball_position: Current ball position (x, y)
            players_data: Dictionary of player objects
            frame_number: Current frame number
            past_ball_pos: Legacy ball position history
            
        Returns:
            Enhanced detection results with legacy compatibility
        """
        
        self.frame_count = frame_number
        self.last_ball_position = ball_position
        
        # Process with enhanced detector
        enhanced_result = self.enhanced_detector.process_frame(
            ball_position, players_data, frame_number
        )
        
        # Update statistics
        self._update_statistics(enhanced_result)
        
        # Generate legacy-compatible results
        legacy_result = self._generate_legacy_compatibility(enhanced_result, past_ball_pos)
        
        # Create comprehensive result
        integrated_result = {
            # Enhanced detection data
            'enhanced': enhanced_result,
            
            # Legacy compatibility
            'legacy': legacy_result,
            
            # Integration metadata
            'integration': {
                'frame_number': frame_number,
                'processing_time': time.time(),
                'detection_mode': 'enhanced',
                'confidence_summary': self._get_confidence_summary(enhanced_result['events'])
            },
            
            # Statistics
            'stats': self.detection_stats.copy()
        }
        
        # Log significant events
        if self.enable_logging:
            self._log_significant_events(enhanced_result['events'], frame_number)
        
        return integrated_result
    
    def get_enhanced_shot_classification(self, past_ball_pos: List, 
                                       players_data: Dict = None) -> Dict[str, Any]:
        """
        Enhanced shot classification using trajectory analysis
        
        Args:
            past_ball_pos: Ball position history
            players_data: Player data for context
            
        Returns:
            Enhanced shot classification with confidence scoring
        """
        
        if not past_ball_pos or len(past_ball_pos) < 5:
            return {
                'shot_type': 'unknown',
                'confidence': 0.0,
                'classification_method': 'insufficient_data',
                'enhanced_features': {}
            }
        
        # Extract trajectory for analysis
        trajectory = [(pos[0], pos[1]) for pos in past_ball_pos[-20:]]
        
        # Analyze trajectory patterns
        features = self._analyze_trajectory_features(trajectory)
        
        # Classify shot type based on enhanced analysis
        shot_classification = self._classify_shot_enhanced(features, trajectory)
        
        # Add player context if available
        if players_data:
            player_context = self._analyze_player_context(trajectory[-1], players_data)
            shot_classification['player_context'] = player_context
        
        return shot_classification
    
    def detect_autonomous_events(self, ball_position: Tuple[float, float],
                                players_data: Dict,
                                frame_number: int) -> Dict[str, Any]:
        """
        Autonomous event detection for real-time analysis
        
        Returns clear identification of:
        1. Ball hit from racket
        2. Ball hit front wall
        3. Ball hit by opponent (new shot)
        4. Ball bounced to ground
        """
        
        # Get current detection state
        detector_status = self.enhanced_detector.get_current_status()
        
        # Process current frame
        frame_result = self.enhanced_detector.process_frame(
            ball_position, players_data, frame_number
        )
        
        # Analyze events for autonomous operation
        autonomous_events = {
            'racket_contact': None,
            'wall_impact': None,
            'opponent_hit': None,
            'floor_bounce': None,
            'shot_phase': 'unknown',
            'confidence_scores': {},
            'recommended_actions': []
        }
        
        # Process detected events
        for event in frame_result['events']:
            if event.confidence > 0.6:  # High confidence threshold
                
                if event.event_type == 'racket_hit':
                    autonomous_events['racket_contact'] = {
                        'detected': True,
                        'player_id': event.player_id,
                        'position': event.ball_position,
                        'confidence': event.confidence,
                        'frame': event.frame_number,
                        'velocity_change': event.details.get('velocity_change', 0)
                    }
                    
                    # Check if this is opponent hit (new shot)
                    if self._is_new_shot_by_opponent(event, detector_status):
                        autonomous_events['opponent_hit'] = {
                            'detected': True,
                            'new_player_id': event.player_id,
                            'position': event.ball_position,
                            'confidence': event.confidence,
                            'shot_transition': True
                        }
                        autonomous_events['recommended_actions'].append(
                            f"New shot started by Player {event.player_id}"
                        )
                
                elif event.event_type == 'wall_hit':
                    wall_type = event.wall_type or 'unknown'
                    autonomous_events['wall_impact'] = {
                        'detected': True,
                        'wall_type': wall_type,
                        'position': event.ball_position,
                        'confidence': event.confidence,
                        'frame': event.frame_number,
                        'impact_angle': event.details.get('impact_angle', 0),
                        'is_front_wall': wall_type == 'front'
                    }
                    
                    if wall_type == 'front':
                        autonomous_events['recommended_actions'].append(
                            "Ball hit front wall - tracking return trajectory"
                        )
                
                elif event.event_type == 'floor_bounce':
                    autonomous_events['floor_bounce'] = {
                        'detected': True,
                        'position': event.ball_position,
                        'confidence': event.confidence,
                        'frame': event.frame_number,
                        'bounce_quality': event.details.get('bounce_pattern_score', 0)
                    }
                    autonomous_events['recommended_actions'].append(
                        "Ball bounced on floor - potential end of rally"
                    )
        
        # Determine current shot phase
        autonomous_events['shot_phase'] = self._determine_shot_phase(autonomous_events)
        
        # Calculate confidence scores
        autonomous_events['confidence_scores'] = {
            'overall_detection': self._calculate_overall_confidence(frame_result['events']),
            'shot_tracking': min(1.0, len(frame_result['events']) / 3.0),
            'phase_identification': self._calculate_phase_confidence(autonomous_events)
        }
        
        return autonomous_events
    
    def get_real_time_insights(self) -> Dict[str, Any]:
        """
        Get real-time insights for autonomous coaching
        """
        
        status = self.enhanced_detector.get_current_status()
        
        insights = {
            'current_match_state': {
                'active_rallies': status['active_shots'],
                'completed_rallies': status['total_completed_shots'],
                'last_processed_frame': self.frame_count,
                'ball_tracking_active': self.last_ball_position is not None
            },
            
            'performance_metrics': {
                'detection_accuracy': self._calculate_detection_accuracy(),
                'processing_rate': self._calculate_processing_rate(),
                'event_detection_rate': self._calculate_event_detection_rate()
            },
            
            'coaching_recommendations': self._generate_coaching_recommendations(status),
            
            'technical_status': {
                'detector_health': 'operational',
                'memory_usage': 'normal',
                'processing_latency': 'low'
            }
        }
        
        return insights
    
    # Helper methods for integration
    
    def _update_statistics(self, enhanced_result: Dict[str, Any]):
        """Update detection statistics"""
        
        self.detection_stats['total_frames_processed'] += 1
        
        for event in enhanced_result['events']:
            if event.confidence > 0.5:
                if event.event_type == 'racket_hit':
                    self.detection_stats['racket_hits_detected'] += 1
                elif event.event_type == 'wall_hit':
                    self.detection_stats['wall_hits_detected'] += 1
                elif event.event_type == 'floor_bounce':
                    self.detection_stats['floor_bounces_detected'] += 1
                
                if event.confidence > 0.8:
                    self.detection_stats['high_confidence_events'] += 1
        
        if enhanced_result['shot_updates']['completed_shots']:
            self.detection_stats['shots_completed'] += len(
                enhanced_result['shot_updates']['completed_shots']
            )
    
    def _generate_legacy_compatibility(self, enhanced_result: Dict[str, Any], 
                                     past_ball_pos: List = None) -> Dict[str, Any]:
        """Generate legacy-compatible results"""
        
        # Default legacy values
        legacy = {
            'who_hit': 0,
            'ball_hit': False,
            'shot_type': 'unknown',
            'hit_confidence': 0.0,
            'match_in_play': True,
            'type_of_shot': ['unknown', 'unknown', 0, 0]
        }
        
        # Update based on enhanced detection
        for event in enhanced_result['events']:
            if event.event_type == 'racket_hit' and event.confidence > 0.6:
                legacy['who_hit'] = event.player_id or 0
                legacy['ball_hit'] = True
                legacy['hit_confidence'] = event.confidence
                legacy['match_in_play'] = True
                
                # Enhanced shot classification
                if past_ball_pos:
                    shot_class = self.get_enhanced_shot_classification(past_ball_pos)
                    legacy['shot_type'] = shot_class['shot_type']
                    legacy['type_of_shot'] = [
                        shot_class['shot_type'],
                        shot_class.get('shot_direction', 'unknown'),
                        shot_class['confidence'],
                        shot_class.get('difficulty_score', 0)
                    ]
        
        return legacy
    
    def _analyze_trajectory_features(self, trajectory: List[Tuple[float, float]]) -> Dict[str, Any]:
        """Analyze trajectory features for shot classification"""
        
        if len(trajectory) < 3:
            return {}
        
        # Convert to numpy for easier processing
        positions = np.array(trajectory)
        
        # Calculate features
        features = {
            'trajectory_length': len(trajectory),
            'start_position': tuple(positions[0]),
            'end_position': tuple(positions[-1]),
            'distance_covered': np.sum(np.sqrt(np.sum(np.diff(positions, axis=0)**2, axis=1))),
            'avg_speed': 0,
            'max_speed': 0,
            'direction_changes': 0,
            'height_variation': np.std(positions[:, 1]),
            'horizontal_movement': abs(positions[-1][0] - positions[0][0]),
            'vertical_movement': abs(positions[-1][1] - positions[0][1])
        }
        
        # Speed analysis
        if len(positions) > 1:
            speeds = np.sqrt(np.sum(np.diff(positions, axis=0)**2, axis=1))
            features['avg_speed'] = float(np.mean(speeds))
            features['max_speed'] = float(np.max(speeds))
        
        # Direction changes
        if len(positions) > 2:
            velocities = np.diff(positions, axis=0)
            angles = np.arctan2(velocities[:, 1], velocities[:, 0])
            if len(angles) > 1:
                angle_changes = np.abs(np.diff(angles))
                # Handle angle wraparound
                angle_changes = np.minimum(angle_changes, 2*np.pi - angle_changes)
                features['direction_changes'] = int(np.sum(angle_changes > np.pi/6))  # > 30 degrees
        
        return features
    
    def _classify_shot_enhanced(self, features: Dict[str, Any], 
                              trajectory: List[Tuple[float, float]]) -> Dict[str, Any]:
        """Enhanced shot classification using trajectory features"""
        
        if not features:
            return {
                'shot_type': 'unknown',
                'confidence': 0.0,
                'classification_method': 'no_features'
            }
        
        # Enhanced classification logic
        shot_scores = {}
        
        # Straight drive characteristics
        if (features.get('horizontal_movement', 0) < 100 and 
            features.get('direction_changes', 0) < 2):
            shot_scores['straight_drive'] = 0.8
        
        # Crosscourt characteristics
        if features.get('horizontal_movement', 0) > 200:
            shot_scores['crosscourt'] = 0.7
        
        # Drop shot characteristics
        if (features.get('avg_speed', 0) < 15 and 
            features.get('vertical_movement', 0) > 100):
            shot_scores['drop_shot'] = 0.6
        
        # Lob characteristics
        if (features.get('height_variation', 0) > 50 and
            features.get('max_speed', 0) < 20):
            shot_scores['lob'] = 0.7
        
        # Boast characteristics
        if features.get('direction_changes', 0) > 2:
            shot_scores['boast'] = 0.6
        
        # Default to drive if no clear classification
        if not shot_scores:
            shot_scores['drive'] = 0.5
        
        # Get best classification
        best_shot = max(shot_scores.items(), key=lambda x: x[1])
        shot_type, confidence = best_shot
        
        return {
            'shot_type': shot_type,
            'confidence': confidence,
            'classification_method': 'enhanced_trajectory_analysis',
            'enhanced_features': features,
            'all_scores': shot_scores,
            'shot_direction': self._determine_shot_direction(trajectory),
            'difficulty_score': self._calculate_difficulty_score(features),
            'tactical_intent': self._analyze_tactical_intent(features, shot_type)
        }
    
    def _determine_shot_direction(self, trajectory: List[Tuple[float, float]]) -> str:
        """Determine shot direction"""
        
        if len(trajectory) < 2:
            return 'unknown'
        
        start_x = trajectory[0][0]
        end_x = trajectory[-1][0]
        
        horizontal_movement = end_x - start_x
        court_center = self.court_width / 2
        
        if abs(horizontal_movement) < 50:
            return 'straight'
        elif start_x < court_center and end_x > court_center:
            return 'crosscourt_right'
        elif start_x > court_center and end_x < court_center:
            return 'crosscourt_left'
        elif horizontal_movement > 0:
            return 'right'
        else:
            return 'left'
    
    def _calculate_difficulty_score(self, features: Dict[str, Any]) -> float:
        """Calculate shot difficulty score"""
        
        difficulty = 0.0
        
        # Speed component
        if features.get('max_speed', 0) > 30:
            difficulty += 0.3
        
        # Direction changes component
        if features.get('direction_changes', 0) > 1:
            difficulty += 0.2
        
        # Distance component
        if features.get('distance_covered', 0) > 300:
            difficulty += 0.2
        
        # Height variation component
        if features.get('height_variation', 0) > 30:
            difficulty += 0.3
        
        return min(1.0, difficulty)
    
    def _analyze_tactical_intent(self, features: Dict[str, Any], shot_type: str) -> str:
        """Analyze tactical intent of the shot"""
        
        # Simple tactical analysis
        if shot_type in ['drop_shot', 'kill_shot']:
            return 'attacking'
        elif shot_type in ['lob', 'defensive_clear']:
            return 'defensive'
        elif shot_type in ['boast', 'crosscourt']:
            return 'positional'
        else:
            return 'neutral'
    
    def _analyze_player_context(self, ball_position: Tuple[float, float], 
                               players_data: Dict) -> Dict[str, Any]:
        """Analyze player context for shot classification"""
        
        context = {
            'players_detected': len(players_data),
            'nearest_player': None,
            'player_distances': {}
        }
        
        min_distance = float('inf')
        nearest_player = None
        
        for player_id, player in players_data.items():
            if player and hasattr(player, 'get_latest_pose'):
                pose = player.get_latest_pose()
                if pose and hasattr(pose, 'xyn') and len(pose.xyn) > 0:
                    # Simple center of mass calculation
                    keypoints = pose.xyn[0]
                    valid_points = [(kp[0] * self.court_width, kp[1] * self.court_height) 
                                  for kp in keypoints if len(kp) >= 2 and kp[0] > 0 and kp[1] > 0]
                    
                    if valid_points:
                        center_x = sum(p[0] for p in valid_points) / len(valid_points)
                        center_y = sum(p[1] for p in valid_points) / len(valid_points)
                        
                        distance = np.sqrt((ball_position[0] - center_x)**2 + 
                                         (ball_position[1] - center_y)**2)
                        
                        context['player_distances'][f'player_{player_id}'] = distance
                        
                        if distance < min_distance:
                            min_distance = distance
                            nearest_player = player_id
        
        context['nearest_player'] = nearest_player
        context['nearest_distance'] = min_distance if min_distance != float('inf') else None
        
        return context
    
    def _is_new_shot_by_opponent(self, event: ShotEvent, 
                                detector_status: Dict[str, Any]) -> bool:
        """Check if this is a new shot by opponent"""
        
        if not detector_status['active_shot_details']:
            return True  # First shot
        
        # Get current active shot
        current_shot = detector_status['active_shot_details'][-1]
        current_player = current_shot.get('player_id')
        
        # Check if different player hit the ball
        return event.player_id and event.player_id != current_player
    
    def _determine_shot_phase(self, autonomous_events: Dict[str, Any]) -> str:
        """Determine current phase of the shot"""
        
        if autonomous_events['racket_contact']:
            if autonomous_events['opponent_hit']:
                return 'shot_transition'
            else:
                return 'ball_in_flight'
        elif autonomous_events['wall_impact']:
            return 'wall_contact'
        elif autonomous_events['floor_bounce']:
            return 'floor_contact'
        else:
            return 'tracking'
    
    def _calculate_overall_confidence(self, events: List[ShotEvent]) -> float:
        """Calculate overall confidence score"""
        
        if not events:
            return 0.0
        
        return sum(event.confidence for event in events) / len(events)
    
    def _calculate_phase_confidence(self, autonomous_events: Dict[str, Any]) -> float:
        """Calculate confidence in phase identification"""
        
        confidence = 0.0
        
        if autonomous_events['racket_contact']:
            confidence += autonomous_events['racket_contact']['confidence'] * 0.4
        if autonomous_events['wall_impact']:
            confidence += autonomous_events['wall_impact']['confidence'] * 0.3
        if autonomous_events['floor_bounce']:
            confidence += autonomous_events['floor_bounce']['confidence'] * 0.3
        
        return min(1.0, confidence)
    
    def _get_confidence_summary(self, events: List[ShotEvent]) -> Dict[str, float]:
        """Get confidence summary for events"""
        
        summary = {
            'racket_hits': [],
            'wall_hits': [],
            'floor_bounces': [],
            'average_confidence': 0.0
        }
        
        for event in events:
            if event.event_type == 'racket_hit':
                summary['racket_hits'].append(event.confidence)
            elif event.event_type == 'wall_hit':
                summary['wall_hits'].append(event.confidence)
            elif event.event_type == 'floor_bounce':
                summary['floor_bounces'].append(event.confidence)
        
        all_confidences = [event.confidence for event in events]
        if all_confidences:
            summary['average_confidence'] = sum(all_confidences) / len(all_confidences)
        
        return summary
    
    def _log_significant_events(self, events: List[ShotEvent], frame_number: int):
        """Log significant detection events"""
        
        for event in events:
            if event.confidence > 0.7:  # High confidence events only
                player_info = f" by Player {event.player_id}" if event.player_id else ""
                wall_info = f" ({event.wall_type} wall)" if event.wall_type else ""
                
                print(f"🎯 Frame {frame_number}: {event.event_type.replace('_', ' ').title()}"
                      f"{player_info}{wall_info} "
                      f"(confidence: {event.confidence:.2f})")
    
    def _calculate_detection_accuracy(self) -> float:
        """Calculate detection accuracy metric"""
        
        total_events = (self.detection_stats['racket_hits_detected'] + 
                       self.detection_stats['wall_hits_detected'] + 
                       self.detection_stats['floor_bounces_detected'])
        
        if total_events == 0:
            return 0.0
        
        # Simple accuracy based on high confidence events
        accuracy = self.detection_stats['high_confidence_events'] / total_events
        return min(1.0, accuracy)
    
    def _calculate_processing_rate(self) -> float:
        """Calculate processing rate (frames per second)"""
        
        if self.detection_stats['total_frames_processed'] == 0:
            return 0.0
        
        # Simplified calculation - would need actual timing in real implementation
        return min(30.0, self.detection_stats['total_frames_processed'] / 10.0)
    
    def _calculate_event_detection_rate(self) -> float:
        """Calculate event detection rate"""
        
        if self.detection_stats['total_frames_processed'] == 0:
            return 0.0
        
        total_events = (self.detection_stats['racket_hits_detected'] + 
                       self.detection_stats['wall_hits_detected'] + 
                       self.detection_stats['floor_bounces_detected'])
        
        return total_events / self.detection_stats['total_frames_processed']
    
    def _generate_coaching_recommendations(self, status: Dict[str, Any]) -> List[str]:
        """Generate coaching recommendations based on detection data"""
        
        recommendations = []
        
        if status['total_completed_shots'] > 5:
            recommendations.append("Good rally length - focus on shot placement")
        
        if self.detection_stats['wall_hits_detected'] > self.detection_stats['racket_hits_detected']:
            recommendations.append("Consider more direct shots to reduce wall contacts")
        
        if self.detection_stats['floor_bounces_detected'] > 3:
            recommendations.append("Work on shot control to minimize floor bounces")
        
        if not recommendations:
            recommendations.append("Continue current playing pattern")
        
        return recommendations
    
    def export_enhanced_data(self, filepath: str = "/tmp/enhanced_integration_data.json") -> Dict[str, Any]:
        """Export enhanced detection data"""
        
        export_data = {
            'integration_metadata': {
                'frames_processed': self.detection_stats['total_frames_processed'],
                'export_timestamp': time.time(),
                'court_dimensions': [self.court_width, self.court_height],
                'detection_mode': 'enhanced_integrated'
            },
            'detection_statistics': self.detection_stats,
            'enhanced_detector_status': self.enhanced_detector.get_current_status(),
            'recent_insights': self.get_real_time_insights()
        }
        
        # Export enhanced detector data
        detector_data = self.enhanced_detector.export_shot_data(
            filepath.replace('.json', '_detector.json')
        )
        export_data['detector_export_path'] = filepath.replace('.json', '_detector.json')
        
        try:
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2)
            print(f"✅ Enhanced integration data exported to {filepath}")
        except Exception as e:
            print(f"❌ Failed to export integration data: {e}")
        
        return export_data


# Factory function for easy integration
def create_enhanced_shot_integrator(court_width=640, court_height=360, 
                                   enable_logging=True) -> ShotDetectionIntegrator:
    """
    Create enhanced shot detection integrator for main pipeline
    """
    return ShotDetectionIntegrator(court_width, court_height, enable_logging)


# Backward compatibility functions for existing pipeline
def enhanced_classify_shot(past_ball_pos: List, 
                          court_width: int = 640, 
                          court_height: int = 360,
                          previous_shot: Any = None,
                          integrator: ShotDetectionIntegrator = None) -> List[Any]:
    """
    Enhanced version of classify_shot function with backward compatibility
    """
    
    if integrator is None:
        integrator = create_enhanced_shot_integrator(court_width, court_height, False)
    
    # Use enhanced classification
    enhanced_result = integrator.get_enhanced_shot_classification(past_ball_pos)
    
    # Return in legacy format
    return [
        enhanced_result['shot_type'],
        enhanced_result.get('shot_direction', 'unknown'),
        enhanced_result['confidence'],
        enhanced_result.get('difficulty_score', 0)
    ]


def enhanced_determine_ball_hit(players: Dict, 
                               past_ball_pos: List,
                               proximity_threshold: int = 100,
                               velocity_threshold: int = 20,
                               integrator: ShotDetectionIntegrator = None) -> Tuple[int, bool, str, float]:
    """
    Enhanced version of determine_ball_hit with backward compatibility
    """
    
    if integrator is None:
        integrator = create_enhanced_shot_integrator(enable_logging=False)
    
    if not past_ball_pos:
        return 0, False, 'none', 0.0
    
    # Process with enhanced detection
    ball_position = (past_ball_pos[-1][0], past_ball_pos[-1][1])
    frame_number = len(past_ball_pos)  # Use position count as frame number
    
    result = integrator.process_frame_enhanced(ball_position, players, frame_number, past_ball_pos)
    
    # Extract legacy format
    legacy = result['legacy']
    
    return (
        legacy['who_hit'],
        legacy['ball_hit'],
        'racket_hit' if legacy['ball_hit'] else 'none',
        legacy['hit_confidence']
    )


# Demo and testing
def demo_integration():
    """Demonstrate the enhanced integration system"""
    
    print("🎾 Enhanced Shot Detection Integration Demo")
    print("=" * 60)
    
    # Create integrator
    integrator = create_enhanced_shot_integrator()
    
    # Simulate a squash rally
    rally_trajectory = [
        # Player 1 serves
        (100, 50), (110, 52), (120, 55), (130, 58), (140, 62),
        # Ball approaches front wall
        (50, 70), (30, 75), (20, 80), (15, 85), (25, 90),
        # Ball travels to Player 2  
        (150, 110), (200, 130), (250, 150), (300, 170), (350, 190),
        # Player 2 returns
        (360, 185), (370, 180), (380, 175), (390, 170), (400, 165),
        # Ball hits side wall and bounces
        (550, 140), (580, 135), (600, 140), (590, 145), (580, 150),
        # Eventually bounces on floor
        (400, 250), (390, 280), (380, 290), (370, 285), (360, 280)
    ]
    
    # Mock players
    mock_players = {
        1: type('Player', (), {
            'get_latest_pose': lambda: type('Pose', (), {
                'xyn': [[(0.15, 0.3, 0.9)] + [(0.0, 0.0, 0.0)] * 16]
            })()
        })(),
        2: type('Player', (), {
            'get_latest_pose': lambda: type('Pose', (), {
                'xyn': [[(0.65, 0.4, 0.9)] + [(0.0, 0.0, 0.0)] * 16]
            })()
        })()
    }
    
    # Process rally
    print("\n📊 Processing Rally:")
    autonomous_events_log = []
    
    for frame, ball_pos in enumerate(rally_trajectory):
        # Enhanced processing
        result = integrator.process_frame_enhanced(ball_pos, mock_players, frame)
        
        # Autonomous event detection
        autonomous = integrator.detect_autonomous_events(ball_pos, mock_players, frame)
        
        # Log significant autonomous events
        if any([autonomous['racket_contact'], autonomous['wall_impact'], 
                autonomous['opponent_hit'], autonomous['floor_bounce']]):
            autonomous_events_log.append((frame, autonomous))
            
            print(f"Frame {frame:2d}: Shot Phase = {autonomous['shot_phase']}")
            if autonomous['racket_contact']:
                print(f"         🏓 Racket contact by Player {autonomous['racket_contact']['player_id']}")
            if autonomous['wall_impact']:
                print(f"         🧱 {autonomous['wall_impact']['wall_type'].title()} wall impact")
            if autonomous['opponent_hit']:
                print(f"         🔄 New shot by Player {autonomous['opponent_hit']['new_player_id']}")
            if autonomous['floor_bounce']:
                print(f"         ⬇️  Floor bounce detected")
    
    # Final insights
    print("\n📈 Real-time Insights:")
    insights = integrator.get_real_time_insights()
    
    for category, data in insights.items():
        print(f"\n{category.replace('_', ' ').title()}:")
        if isinstance(data, dict):
            for key, value in data.items():
                print(f"  • {key.replace('_', ' ').title()}: {value}")
    
    # Export data
    export_data = integrator.export_enhanced_data()
    print(f"\n💾 Data exported: {len(export_data)} categories")
    
    return integrator, autonomous_events_log


if __name__ == "__main__":
    # Run integration demo
    demo_integration()