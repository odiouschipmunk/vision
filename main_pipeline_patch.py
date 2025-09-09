"""
Main Pipeline Integration Patch for Enhanced Shot Detection
This module patches the main.py pipeline to use the enhanced shot detection system.
"""

import sys
import os
import time
import json
from typing import Dict, List, Tuple, Optional, Any

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from enhanced_shot_integration_v2 import (
        create_enhanced_shot_integrator,
        enhanced_classify_shot,
        enhanced_determine_ball_hit
    )
    ENHANCED_AVAILABLE = True
    print("✅ Enhanced shot detection system loaded successfully")
except ImportError as e:
    print(f"⚠️ Enhanced shot detection not available: {e}")
    ENHANCED_AVAILABLE = False

class EnhancedShotTracker:
    """
    Enhanced version of the ShotTracker class with improved autonomous detection
    """
    
    def __init__(self, court_width=640, court_height=360):
        self.court_width = court_width
        self.court_height = court_height
        
        # Initialize enhanced integrator if available
        if ENHANCED_AVAILABLE:
            self.integrator = create_enhanced_shot_integrator(
                court_width, court_height, enable_logging=True
            )
            self.enhanced_mode = True
            print("🎾 Enhanced shot tracking initialized")
        else:
            self.integrator = None
            self.enhanced_mode = False
            print("⚠️ Fallback to legacy shot tracking")
        
        # Tracking state
        self.active_shots = []
        self.completed_shots = []
        self.frame_count = 0
        self.last_events = []
        
        # Performance metrics
        self.detection_metrics = {
            'frames_processed': 0,
            'shots_detected': 0,
            'wall_hits_detected': 0,
            'floor_bounces_detected': 0,
            'player_hits_detected': 0,
            'high_confidence_detections': 0
        }
    
    def process_frame(self, ball_position: Tuple[float, float], 
                     players_data: Dict, 
                     frame_number: int,
                     past_ball_pos: List = None) -> Dict[str, Any]:
        """
        Process frame with enhanced detection if available
        """
        
        self.frame_count = frame_number
        self.detection_metrics['frames_processed'] += 1
        
        if self.enhanced_mode and self.integrator:
            # Use enhanced detection
            result = self.integrator.process_frame_enhanced(
                ball_position, players_data, frame_number, past_ball_pos
            )
            
            # Update metrics
            self._update_metrics_from_enhanced(result)
            
            # Get autonomous events for clear identification
            autonomous = self.integrator.detect_autonomous_events(
                ball_position, players_data, frame_number
            )
            
            result['autonomous_events'] = autonomous
            
            return result
        
        else:
            # Fallback to basic detection
            return self._basic_frame_processing(
                ball_position, players_data, frame_number, past_ball_pos
            )
    
    def get_enhanced_shot_analysis(self, past_ball_pos: List, 
                                  players_data: Dict = None) -> Dict[str, Any]:
        """
        Get enhanced shot analysis with clear event identification
        """
        
        if not past_ball_pos or len(past_ball_pos) < 5:
            return {
                'events_detected': {
                    'racket_hit': False,
                    'front_wall_hit': False,
                    'opponent_hit': False,
                    'floor_bounce': False
                },
                'shot_classification': 'unknown',
                'confidence': 0.0,
                'analysis_method': 'insufficient_data'
            }
        
        if self.enhanced_mode and self.integrator:
            # Enhanced analysis
            classification = self.integrator.get_enhanced_shot_classification(
                past_ball_pos, players_data
            )
            
            # Get current autonomous detection state
            ball_pos = (past_ball_pos[-1][0], past_ball_pos[-1][1])
            autonomous = self.integrator.detect_autonomous_events(
                ball_pos, players_data or {}, self.frame_count
            )
            
            return {
                'events_detected': {
                    'racket_hit': autonomous['racket_contact'] is not None,
                    'front_wall_hit': (autonomous['wall_impact'] is not None and 
                                     autonomous['wall_impact'].get('is_front_wall', False)),
                    'opponent_hit': autonomous['opponent_hit'] is not None,
                    'floor_bounce': autonomous['floor_bounce'] is not None
                },
                'shot_classification': classification['shot_type'],
                'confidence': classification['confidence'],
                'shot_direction': classification.get('shot_direction', 'unknown'),
                'difficulty_score': classification.get('difficulty_score', 0),
                'tactical_intent': classification.get('tactical_intent', 'unknown'),
                'analysis_method': 'enhanced_physics_based',
                'autonomous_events': autonomous,
                'enhanced_features': classification.get('enhanced_features', {})
            }
        
        else:
            # Basic analysis
            return self._basic_shot_analysis(past_ball_pos)
    
    def detect_clear_shot_events(self, ball_position: Tuple[float, float],
                                players_data: Dict,
                                frame_number: int) -> Dict[str, Any]:
        """
        Detect clear shot events as requested:
        1. Ball got hit from the racket
        2. Ball hit the front wall  
        3. Ball got hit by opponent's racket (new shot)
        4. Ball bounced to the ground
        """
        
        if self.enhanced_mode and self.integrator:
            autonomous = self.integrator.detect_autonomous_events(
                ball_position, players_data, frame_number
            )
            
            # Clear event detection with confidence scores
            clear_events = {
                'ball_hit_from_racket': {
                    'detected': autonomous['racket_contact'] is not None,
                    'player_id': (autonomous['racket_contact']['player_id'] 
                                if autonomous['racket_contact'] else None),
                    'confidence': (autonomous['racket_contact']['confidence'] 
                                 if autonomous['racket_contact'] else 0.0),
                    'position': (autonomous['racket_contact']['position'] 
                               if autonomous['racket_contact'] else None),
                    'frame': frame_number
                },
                
                'ball_hit_front_wall': {
                    'detected': (autonomous['wall_impact'] is not None and 
                               autonomous['wall_impact'].get('is_front_wall', False)),
                    'confidence': (autonomous['wall_impact']['confidence'] 
                                 if autonomous['wall_impact'] else 0.0),
                    'position': (autonomous['wall_impact']['position'] 
                               if autonomous['wall_impact'] else None),
                    'impact_angle': (autonomous['wall_impact'].get('impact_angle', 0) 
                                   if autonomous['wall_impact'] else 0),
                    'frame': frame_number
                },
                
                'ball_hit_by_opponent': {
                    'detected': autonomous['opponent_hit'] is not None,
                    'new_player_id': (autonomous['opponent_hit']['new_player_id'] 
                                    if autonomous['opponent_hit'] else None),
                    'confidence': (autonomous['opponent_hit']['confidence'] 
                                 if autonomous['opponent_hit'] else 0.0),
                    'shot_transition': (autonomous['opponent_hit']['shot_transition'] 
                                      if autonomous['opponent_hit'] else False),
                    'frame': frame_number
                },
                
                'ball_bounced_to_ground': {
                    'detected': autonomous['floor_bounce'] is not None,
                    'confidence': (autonomous['floor_bounce']['confidence'] 
                                 if autonomous['floor_bounce'] else 0.0),
                    'position': (autonomous['floor_bounce']['position'] 
                               if autonomous['floor_bounce'] else None),
                    'bounce_quality': (autonomous['floor_bounce']['bounce_quality'] 
                                     if autonomous['floor_bounce'] else 0),
                    'frame': frame_number
                }
            }
            
            # Overall detection summary
            total_detections = sum(1 for event in clear_events.values() if event['detected'])
            avg_confidence = (sum(event['confidence'] for event in clear_events.values() 
                                if event['detected']) / max(total_detections, 1))
            
            clear_events['summary'] = {
                'total_events_detected': total_detections,
                'average_confidence': avg_confidence,
                'shot_phase': autonomous['shot_phase'],
                'recommended_actions': autonomous['recommended_actions'],
                'frame_number': frame_number,
                'autonomous_operation': True
            }
            
            return clear_events
        
        else:
            # Fallback basic detection
            return self._basic_event_detection(ball_position, players_data, frame_number)
    
    def get_real_time_status(self) -> Dict[str, Any]:
        """
        Get real-time status for autonomous operation
        """
        
        if self.enhanced_mode and self.integrator:
            insights = self.integrator.get_real_time_insights()
            
            status = {
                'detection_system': 'enhanced_autonomous',
                'operational_status': 'active',
                'performance_metrics': self.detection_metrics,
                'real_time_insights': insights,
                'capabilities': [
                    'Autonomous racket hit detection',
                    'Physics-based wall impact detection', 
                    'Opponent shot transition detection',
                    'Floor bounce identification',
                    'Real-time confidence scoring',
                    'Tactical analysis'
                ]
            }
            
            return status
        
        else:
            return {
                'detection_system': 'basic_legacy',
                'operational_status': 'limited',
                'performance_metrics': self.detection_metrics,
                'capabilities': [
                    'Basic ball tracking',
                    'Simple hit detection'
                ]
            }
    
    def export_enhanced_data(self, filepath_base: str = "/tmp/enhanced_shot_tracking") -> List[str]:
        """
        Export enhanced detection data
        """
        
        exported_files = []
        
        if self.enhanced_mode and self.integrator:
            # Export enhanced data
            integration_file = f"{filepath_base}_integration.json"
            self.integrator.export_enhanced_data(integration_file)
            exported_files.append(integration_file)
        
        # Export tracking metrics
        metrics_file = f"{filepath_base}_metrics.json"
        try:
            with open(metrics_file, 'w') as f:
                json.dump({
                    'tracking_metrics': self.detection_metrics,
                    'system_info': {
                        'enhanced_mode': self.enhanced_mode,
                        'frames_processed': self.frame_count,
                        'export_timestamp': time.time()
                    }
                }, f, indent=2)
            exported_files.append(metrics_file)
        except Exception as e:
            print(f"⚠️ Failed to export metrics: {e}")
        
        return exported_files
    
    # Helper methods
    
    def _update_metrics_from_enhanced(self, result: Dict[str, Any]):
        """Update metrics from enhanced detection result"""
        
        for event in result['enhanced']['events']:
            if event.confidence > 0.6:
                if event.event_type == 'racket_hit':
                    self.detection_metrics['player_hits_detected'] += 1
                elif event.event_type == 'wall_hit':
                    self.detection_metrics['wall_hits_detected'] += 1
                elif event.event_type == 'floor_bounce':
                    self.detection_metrics['floor_bounces_detected'] += 1
                
                if event.confidence > 0.8:
                    self.detection_metrics['high_confidence_detections'] += 1
        
        if result['enhanced']['shot_updates']['new_shots']:
            self.detection_metrics['shots_detected'] += len(
                result['enhanced']['shot_updates']['new_shots']
            )
    
    def _basic_frame_processing(self, ball_position: Tuple[float, float], 
                               players_data: Dict, 
                               frame_number: int,
                               past_ball_pos: List = None) -> Dict[str, Any]:
        """Basic frame processing fallback"""
        
        return {
            'enhanced': {
                'events': [],
                'active_shots': 0,
                'completed_shots': 0,
                'shot_updates': {'new_shots': [], 'updated_shots': [], 'completed_shots': []}
            },
            'legacy': {
                'who_hit': 0,
                'ball_hit': False,
                'shot_type': 'unknown',
                'hit_confidence': 0.0,
                'match_in_play': True
            },
            'integration': {
                'frame_number': frame_number,
                'detection_mode': 'basic_fallback'
            }
        }
    
    def _basic_shot_analysis(self, past_ball_pos: List) -> Dict[str, Any]:
        """Basic shot analysis fallback"""
        
        return {
            'events_detected': {
                'racket_hit': False,
                'front_wall_hit': False,
                'opponent_hit': False,
                'floor_bounce': False
            },
            'shot_classification': 'unknown',
            'confidence': 0.0,
            'analysis_method': 'basic_fallback'
        }
    
    def _basic_event_detection(self, ball_position: Tuple[float, float],
                              players_data: Dict,
                              frame_number: int) -> Dict[str, Any]:
        """Basic event detection fallback"""
        
        return {
            'ball_hit_from_racket': {
                'detected': False,
                'confidence': 0.0,
                'frame': frame_number
            },
            'ball_hit_front_wall': {
                'detected': False,
                'confidence': 0.0,
                'frame': frame_number
            },
            'ball_hit_by_opponent': {
                'detected': False,
                'confidence': 0.0,
                'frame': frame_number
            },
            'ball_bounced_to_ground': {
                'detected': False,
                'confidence': 0.0,
                'frame': frame_number
            },
            'summary': {
                'total_events_detected': 0,
                'average_confidence': 0.0,
                'autonomous_operation': False,
                'frame_number': frame_number
            }
        }


# Integration functions for main pipeline

def patch_main_pipeline():
    """
    Patch the main pipeline to use enhanced shot detection
    """
    
    print("🔧 Patching main pipeline with enhanced shot detection...")
    
    # Create global enhanced tracker
    global enhanced_shot_tracker
    enhanced_shot_tracker = EnhancedShotTracker()
    
    print("✅ Enhanced shot detection patch applied")
    
    return enhanced_shot_tracker


def enhanced_shot_detection_wrapper(ball_position: Tuple[float, float],
                                   players_data: Dict,
                                   frame_number: int,
                                   past_ball_pos: List = None) -> Dict[str, Any]:
    """
    Wrapper function for enhanced shot detection in main pipeline
    """
    
    global enhanced_shot_tracker
    
    if 'enhanced_shot_tracker' not in globals():
        enhanced_shot_tracker = EnhancedShotTracker()
    
    # Process frame with enhanced detection
    result = enhanced_shot_tracker.process_frame(
        ball_position, players_data, frame_number, past_ball_pos
    )
    
    # Get clear event detection
    clear_events = enhanced_shot_tracker.detect_clear_shot_events(
        ball_position, players_data, frame_number
    )
    
    # Add clear events to result
    result['clear_shot_events'] = clear_events
    
    return result


def get_enhanced_shot_summary() -> Dict[str, Any]:
    """
    Get summary of enhanced shot detection for main pipeline
    """
    
    global enhanced_shot_tracker
    
    if 'enhanced_shot_tracker' not in globals():
        return {'error': 'Enhanced shot tracker not initialized'}
    
    return enhanced_shot_tracker.get_real_time_status()


# Demo function
def demo_main_pipeline_integration():
    """
    Demonstrate integration with main pipeline
    """
    
    print("🎾 Enhanced Shot Detection - Main Pipeline Integration Demo")
    print("=" * 70)
    
    # Initialize enhanced tracker
    tracker = patch_main_pipeline()
    
    # Simulate main pipeline processing
    print("\n📊 Simulating Main Pipeline Processing:")
    
    # Mock data similar to main pipeline
    simulated_frames = [
        # Rally sequence
        {'ball_pos': (100, 50), 'frame': 1},
        {'ball_pos': (110, 52), 'frame': 2},
        {'ball_pos': (120, 55), 'frame': 3},
        {'ball_pos': (25, 80), 'frame': 10},   # Near front wall
        {'ball_pos': (200, 120), 'frame': 20}, # Mid court
        {'ball_pos': (420, 190), 'frame': 30}, # Player 2 area
        {'ball_pos': (430, 185), 'frame': 31}, # Hit by player 2
        {'ball_pos': (500, 300), 'frame': 40}  # Near floor
    ]
    
    # Mock players (simplified)
    mock_players = {
        1: type('MockPlayer', (), {
            'get_latest_pose': lambda: type('MockPose', (), {
                'xyn': [[(0.15, 0.3, 0.9)] + [(0.0, 0.0, 0.0)] * 16]
            })()
        })(),
        2: type('MockPlayer', (), {
            'get_latest_pose': lambda: type('MockPose', (), {
                'xyn': [[(0.65, 0.4, 0.9)] + [(0.0, 0.0, 0.0)] * 16]
            })()
        })()
    }
    
    # Process frames
    all_results = []
    past_positions = []
    
    for frame_data in simulated_frames:
        ball_pos = frame_data['ball_pos']
        frame_num = frame_data['frame']
        
        # Add to position history
        past_positions.append([ball_pos[0], ball_pos[1], frame_num])
        
        # Process with enhanced wrapper
        result = enhanced_shot_detection_wrapper(
            ball_pos, mock_players, frame_num, past_positions
        )
        
        all_results.append(result)
        
        # Print clear event detection results
        clear_events = result['clear_shot_events']
        summary = clear_events['summary']
        
        if summary['total_events_detected'] > 0:
            print(f"\nFrame {frame_num:2d}: {summary['total_events_detected']} events detected")
            print(f"         Shot Phase: {clear_events.get('autonomous_events', {}).get('shot_phase', 'unknown')}")
            
            # Print specific events
            if clear_events['ball_hit_from_racket']['detected']:
                player_id = clear_events['ball_hit_from_racket']['player_id']
                conf = clear_events['ball_hit_from_racket']['confidence']
                print(f"         🏓 Ball hit from racket by Player {player_id} (conf: {conf:.2f})")
            
            if clear_events['ball_hit_front_wall']['detected']:
                conf = clear_events['ball_hit_front_wall']['confidence']
                print(f"         🧱 Ball hit front wall (conf: {conf:.2f})")
            
            if clear_events['ball_hit_by_opponent']['detected']:
                player_id = clear_events['ball_hit_by_opponent']['new_player_id']
                print(f"         🔄 New shot by opponent Player {player_id}")
            
            if clear_events['ball_bounced_to_ground']['detected']:
                conf = clear_events['ball_bounced_to_ground']['confidence']
                print(f"         ⬇️  Ball bounced to ground (conf: {conf:.2f})")
    
    # Get final summary
    print("\n📈 Enhanced Shot Detection Summary:")
    summary = get_enhanced_shot_summary()
    
    print(f"  Detection System: {summary['detection_system']}")
    print(f"  Status: {summary['operational_status']}")
    print(f"  Frames Processed: {summary['performance_metrics']['frames_processed']}")
    print(f"  Events Detected: {sum(v for k, v in summary['performance_metrics'].items() if 'detected' in k)}")
    
    print("\n🎯 Capabilities:")
    for capability in summary['capabilities']:
        print(f"  • {capability}")
    
    # Export data
    exported_files = tracker.export_enhanced_data()
    print(f"\n💾 Exported {len(exported_files)} data files:")
    for file_path in exported_files:
        print(f"  • {file_path}")
    
    print("\n✅ Main pipeline integration demo completed successfully!")
    
    return tracker, all_results


if __name__ == "__main__":
    # Run main pipeline integration demo
    demo_main_pipeline_integration()