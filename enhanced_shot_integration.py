"""
Integration module for Enhanced Ball Physics into existing squash coaching pipeline

This module provides seamless integration of the new enhanced shot detection system
with the existing ef.py pipeline while maintaining backward compatibility.
"""

import sys
import os
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import json
import time
from collections import defaultdict

# Import the enhanced physics system
from enhanced_ball_physics import (
    EnhancedShotDetector, 
    BallPosition, 
    ShotEvent,
    create_enhanced_shot_detector
)

class EnhancedPipelineIntegrator:
    """
    Integrates enhanced shot detection with existing pipeline
    """
    
    def __init__(self, court_dimensions=(640, 360), enable_legacy_fallback=True):
        self.court_dimensions = court_dimensions
        self.enable_legacy_fallback = enable_legacy_fallback
        
        # Initialize enhanced detector
        self.enhanced_detector = create_enhanced_shot_detector(court_dimensions)
        
        # Integration state
        self.frame_count = 0
        self.last_shot_events = []
        self.performance_stats = {
            'total_frames_processed': 0,
            'enhanced_detections': 0,
            'legacy_fallbacks': 0,
            'total_shots_detected': 0,
            'total_events_detected': 0
        }
        
        # Shot mapping for compatibility with existing system
        self.shot_type_mapping = {
            'direct_shot': 'straight_drive',
            'single_wall_shot': 'crosscourt', 
            'multi_wall_shot': 'boast',
            'unknown': 'defensive'
        }
        
        print("🔥 Enhanced Shot Detection Integration Initialized")
        print(f"   Court dimensions: {court_dimensions}")
        print(f"   Legacy fallback: {'Enabled' if enable_legacy_fallback else 'Disabled'}")
        
    def process_ball_detection(self, 
                             past_ball_pos: List[List],
                             players: Dict,
                             frame_count: int,
                             original_detections: Any = None) -> Dict:
        """
        Enhanced ball detection processing that integrates with existing pipeline
        
        Args:
            past_ball_pos: List of ball positions [[x, y, frame], ...]
            players: Player data dictionary
            frame_count: Current frame number
            original_detections: Original detection results for fallback
            
        Returns:
            Enhanced detection results with backward compatibility
        """
        self.frame_count = frame_count
        self.performance_stats['total_frames_processed'] += 1
        
        try:
            # Convert past_ball_pos to ball detections format
            ball_detections = []
            if past_ball_pos:
                latest_pos = past_ball_pos[-1]
                if len(latest_pos) >= 2:
                    confidence = latest_pos[3] if len(latest_pos) > 3 else 0.8
                    ball_detections = [[latest_pos[0], latest_pos[1], confidence]]
            
            # Process with enhanced detector
            enhanced_results = self.enhanced_detector.process_frame(
                ball_detections, players, frame_count
            )
            
            self.performance_stats['enhanced_detections'] += 1
            
            # Extract events and update stats
            events = enhanced_results.get('events', [])
            self.last_shot_events = events
            self.performance_stats['total_events_detected'] += len(events)
            
            # Create enhanced shot analysis
            shot_analysis = self._create_enhanced_shot_analysis(enhanced_results, past_ball_pos)
            
            # Maintain compatibility with existing pipeline
            compatible_results = self._create_compatible_results(
                enhanced_results, past_ball_pos, players, frame_count
            )
            
            return {
                'enhanced_results': enhanced_results,
                'shot_analysis': shot_analysis,
                'compatible_results': compatible_results,
                'ball_hit_detected': self._check_ball_hit_detected(events),
                'shot_events': events,
                'performance_stats': self.performance_stats.copy()
            }
            
        except Exception as e:
            print(f"Enhanced detection error: {e}")
            if self.enable_legacy_fallback:
                self.performance_stats['legacy_fallbacks'] += 1
                return self._create_fallback_results(original_detections, past_ball_pos)
            else:
                raise e
    
    def _create_enhanced_shot_analysis(self, enhanced_results: Dict, past_ball_pos: List) -> Dict:
        """Create detailed shot analysis from enhanced results"""
        
        ball_position = enhanced_results.get('ball_position')
        events = enhanced_results.get('events', [])
        
        analysis = {
            'ball_tracking': {
                'position': None,
                'velocity': (0, 0),
                'acceleration': (0, 0),
                'confidence': 0.0
            },
            'shot_events': {
                'racket_hits': [],
                'wall_hits': [],
                'floor_bounces': []
            },
            'trajectory_quality': 'unknown',
            'shot_phase': 'none'
        }
        
        # Ball tracking info
        if ball_position:
            analysis['ball_tracking'] = {
                'position': (ball_position.x, ball_position.y),
                'velocity': (ball_position.velocity_x, ball_position.velocity_y),
                'acceleration': (ball_position.acceleration_x, ball_position.acceleration_y),
                'confidence': ball_position.confidence
            }
            
        # Categorize events
        for event in events:
            event_data = {
                'frame': event.frame,
                'position': event.position,
                'confidence': event.confidence,
                'velocity_before': event.velocity_before,
                'velocity_after': event.velocity_after
            }
            
            if event.event_type == 'racket_hit':
                event_data['player_id'] = event.player_id
                event_data['impact_angle'] = event.impact_angle
                analysis['shot_events']['racket_hits'].append(event_data)
                
            elif event.event_type == 'wall_hit':
                event_data['wall_type'] = event.wall_type
                event_data['impact_angle'] = event.impact_angle
                analysis['shot_events']['wall_hits'].append(event_data)
                
            elif event.event_type == 'floor_bounce':
                analysis['shot_events']['floor_bounces'].append(event_data)
        
        # Determine trajectory quality
        trajectory_length = enhanced_results.get('trajectory_length', 0)
        if trajectory_length > 50:
            analysis['trajectory_quality'] = 'excellent'
        elif trajectory_length > 20:
            analysis['trajectory_quality'] = 'good'
        elif trajectory_length > 5:
            analysis['trajectory_quality'] = 'fair'
        else:
            analysis['trajectory_quality'] = 'poor'
            
        # Determine current shot phase
        recent_events = [e for e in events if self.frame_count - e.frame < 10]
        if any(e.event_type == 'racket_hit' for e in recent_events):
            analysis['shot_phase'] = 'start'
        elif any(e.event_type == 'wall_hit' for e in recent_events):
            analysis['shot_phase'] = 'middle'
        elif any(e.event_type == 'floor_bounce' for e in recent_events):
            analysis['shot_phase'] = 'end'
        else:
            analysis['shot_phase'] = 'in_flight'
            
        return analysis
    
    def _create_compatible_results(self, enhanced_results: Dict, past_ball_pos: List, 
                                 players: Dict, frame_count: int) -> Dict:
        """Create results compatible with existing pipeline"""
        
        events = enhanced_results.get('events', [])
        ball_position = enhanced_results.get('ball_position')
        
        # Extract racket hits for compatibility
        racket_hits = [e for e in events if e.event_type == 'racket_hit']
        
        compatible = {
            'ball_hit': len(racket_hits) > 0,
            'who_hit': 0,
            'hit_confidence': 0.0,
            'hit_type': 'none',
            'shot_type': 'unknown',
            'match_in_play': True,
            'ball_position': None,
            'velocity_change': 0.0,
            'direction_change': 0.0
        }
        
        # Ball hit information
        if racket_hits:
            latest_hit = max(racket_hits, key=lambda x: x.frame)
            compatible['who_hit'] = latest_hit.player_id or 0
            compatible['hit_confidence'] = latest_hit.confidence
            compatible['hit_type'] = 'strong_hit' if latest_hit.confidence > 0.7 else 'possible_hit'
            
            # Calculate velocity and direction changes
            if latest_hit.velocity_before and latest_hit.velocity_after:
                vel_before = latest_hit.velocity_before
                vel_after = latest_hit.velocity_after
                
                vel_mag_before = np.sqrt(vel_before[0]**2 + vel_before[1]**2)
                vel_mag_after = np.sqrt(vel_after[0]**2 + vel_after[1]**2)
                
                compatible['velocity_change'] = abs(vel_mag_after - vel_mag_before)
                compatible['direction_change'] = latest_hit.impact_angle or 0.0
        
        # Ball position
        if ball_position:
            compatible['ball_position'] = [ball_position.x, ball_position.y, ball_position.frame]
            
        # Shot type classification
        wall_hits = [e for e in events if e.event_type == 'wall_hit']
        if len(wall_hits) == 0:
            compatible['shot_type'] = 'straight_drive'
        elif len(wall_hits) == 1:
            wall_type = wall_hits[0].wall_type
            if wall_type in ['left', 'right']:
                compatible['shot_type'] = 'boast'
            else:
                compatible['shot_type'] = 'crosscourt'
        else:
            compatible['shot_type'] = 'boast'
            
        # Match state
        compatible['match_in_play'] = {
            'ball_hit': compatible['ball_hit'],
            'movement_detected': len(events) > 0,
            'tracking_quality': enhanced_results.get('trajectory_length', 0) > 10
        }
        
        return compatible
    
    def _check_ball_hit_detected(self, events: List[ShotEvent]) -> bool:
        """Check if ball hit was detected in recent frames"""
        recent_hits = [e for e in events 
                      if e.event_type == 'racket_hit' and 
                      self.frame_count - e.frame < 5]
        return len(recent_hits) > 0
    
    def _create_fallback_results(self, original_detections: Any, past_ball_pos: List) -> Dict:
        """Create fallback results when enhanced detection fails"""
        return {
            'enhanced_results': None,
            'shot_analysis': {'error': 'Enhanced detection failed, using fallback'},
            'compatible_results': {
                'ball_hit': False,
                'who_hit': 0,
                'hit_confidence': 0.0,
                'hit_type': 'none',
                'shot_type': 'unknown',
                'match_in_play': False
            },
            'ball_hit_detected': False,
            'shot_events': [],
            'fallback_used': True
        }
    
    def get_shot_statistics(self) -> Dict:
        """Get comprehensive shot statistics"""
        detector_analysis = self.enhanced_detector.get_shot_analysis()
        
        stats = {
            'system_performance': self.performance_stats.copy(),
            'shot_detection': detector_analysis,
            'frame_processing_rate': (
                self.performance_stats['total_frames_processed'] / 
                max(1, time.time() - getattr(self, 'start_time', time.time()))
            ),
            'detection_success_rate': (
                self.performance_stats['enhanced_detections'] / 
                max(1, self.performance_stats['total_frames_processed'])
            )
        }
        
        return stats
    
    def export_enhanced_data(self, output_dir: str = "output/enhanced_shots"):
        """Export enhanced shot data"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Export shot data
        shot_file = os.path.join(output_dir, f"enhanced_shots_{int(time.time())}.json")
        self.enhanced_detector.export_shot_data(shot_file)
        
        # Export statistics
        stats_file = os.path.join(output_dir, f"shot_statistics_{int(time.time())}.json")
        with open(stats_file, 'w') as f:
            json.dump(self.get_shot_statistics(), f, indent=2)
            
        print(f"Enhanced data exported to {output_dir}")
        
    def reset(self):
        """Reset the integrator state"""
        self.enhanced_detector = create_enhanced_shot_detector(self.court_dimensions)
        self.frame_count = 0
        self.last_shot_events = []
        self.performance_stats = {
            'total_frames_processed': 0,
            'enhanced_detections': 0,
            'legacy_fallbacks': 0,
            'total_shots_detected': 0,
            'total_events_detected': 0
        }

def integrate_enhanced_detection_into_pipeline(frame_width=640, frame_height=360):
    """
    Factory function to create and return an enhanced detection integrator
    for use in the main pipeline
    """
    print("🚀 INTEGRATING ENHANCED BALL PHYSICS INTO COACHING PIPELINE")
    print("=" * 70)
    
    integrator = EnhancedPipelineIntegrator(
        court_dimensions=(frame_width, frame_height),
        enable_legacy_fallback=True
    )
    
    print("✓ Enhanced Physics System: READY")
    print("✓ Kalman Filter Tracking: READY") 
    print("✓ Multi-Modal Event Detection: READY")
    print("✓ Legacy Pipeline Compatibility: READY")
    print("=" * 70)
    
    return integrator

# Wrapper functions for easy integration with existing code
def enhanced_determine_ball_hit(integrator: EnhancedPipelineIntegrator, 
                              players: Dict, past_ball_pos: List, 
                              frame_count: int) -> Tuple[int, float, str]:
    """
    Enhanced version of determine_ball_hit_enhanced with better accuracy
    
    Returns: (player_id, hit_confidence, hit_type)
    """
    try:
        results = integrator.process_ball_detection(past_ball_pos, players, frame_count)
        compatible = results['compatible_results']
        
        return (
            compatible['who_hit'],
            compatible['hit_confidence'], 
            compatible['hit_type']
        )
    except Exception as e:
        print(f"Enhanced ball hit detection failed: {e}")
        return (0, 0.0, "none")

def enhanced_shot_type_detection(integrator: EnhancedPipelineIntegrator,
                               past_ball_pos: List, court_width: int, 
                               court_height: int, threshold: int = 5) -> str:
    """
    Enhanced shot type detection with physics-based analysis
    
    Returns: Shot type string
    """
    try:
        results = integrator.process_ball_detection(past_ball_pos, {}, integrator.frame_count)
        return results['compatible_results']['shot_type']
    except Exception as e:
        print(f"Enhanced shot type detection failed: {e}")
        return "unknown"

def enhanced_match_state_detection(integrator: EnhancedPipelineIntegrator,
                                 players: Dict, past_ball_pos: List, **kwargs) -> Dict:
    """
    Enhanced match state detection with comprehensive ball tracking
    
    Returns: Match state dictionary
    """
    try:
        results = integrator.process_ball_detection(past_ball_pos, players, integrator.frame_count)
        return results['compatible_results']['match_in_play']
    except Exception as e:
        print(f"Enhanced match state detection failed: {e}")
        return {'ball_hit': False, 'movement_detected': False, 'tracking_quality': False}

if __name__ == "__main__":
    # Test integration
    integrator = integrate_enhanced_detection_into_pipeline()
    
    # Simulate some processing
    test_ball_pos = [[100, 200, 1], [105, 205, 2], [110, 210, 3]]
    test_players = {1: {}, 2: {}}
    
    for i in range(3):
        results = integrator.process_ball_detection(test_ball_pos[:i+1], test_players, i)
        print(f"Frame {i} results: {results['compatible_results']}")
    
    stats = integrator.get_shot_statistics()
    print(f"Final statistics: {stats}")
