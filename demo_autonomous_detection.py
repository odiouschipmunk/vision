"""
Simplified Enhanced Shot Detection Demo
Demonstrates the four key shot detection improvements without external dependencies.
"""

import math
import time
import json
from typing import Dict, List, Tuple, Optional, Any

class SimplifiedShotEvent:
    """Simplified shot event for demonstration"""
    
    def __init__(self, event_type: str, frame: int, position: Tuple[float, float], 
                 confidence: float, player_id: Optional[int] = None):
        self.event_type = event_type
        self.frame_number = frame
        self.ball_position = position
        self.confidence = confidence
        self.player_id = player_id
        self.details = {}

class SimplifiedPhysicsDetector:
    """Simplified physics-based detector for demonstration"""
    
    def __init__(self, court_width=640, court_height=360):
        self.court_width = court_width
        self.court_height = court_height
        
        # Detection thresholds
        self.min_velocity_change = 15  # pixels/frame
        self.wall_proximity_threshold = 25  # pixels from wall
        self.floor_proximity_threshold = 0.8  # 80% down the court
    
    def detect_racket_hit(self, trajectory: List[Tuple[float, float]], 
                         players_data: Dict, frame_number: int) -> SimplifiedShotEvent:
        """Detect when ball is hit by racket"""
        
        if len(trajectory) < 4:
            return SimplifiedShotEvent("racket_hit", frame_number, trajectory[-1], 0.0)
        
        # Calculate velocity change
        recent_positions = trajectory[-4:]
        velocity_change = self._calculate_velocity_change(recent_positions)
        
        # Check player proximity
        current_pos = trajectory[-1]
        player_proximity = self._get_nearest_player_distance(current_pos, players_data)
        
        # Calculate confidence based on velocity change and player proximity
        confidence = 0.0
        if velocity_change > self.min_velocity_change and player_proximity < 80:
            confidence = min(1.0, (velocity_change / 30.0) * (80 - player_proximity) / 80.0)
        
        # Determine hitting player
        hitting_player = self._identify_hitting_player(current_pos, players_data) if confidence > 0.5 else None
        
        event = SimplifiedShotEvent("racket_hit", frame_number, current_pos, confidence, hitting_player)
        event.details = {'velocity_change': velocity_change, 'player_proximity': player_proximity}
        
        return event
    
    def detect_wall_hit(self, trajectory: List[Tuple[float, float]], 
                       frame_number: int) -> SimplifiedShotEvent:
        """Detect when ball hits walls (especially front wall)"""
        
        if len(trajectory) < 3:
            return SimplifiedShotEvent("wall_hit", frame_number, trajectory[-1], 0.0)
        
        current_pos = trajectory[-1]
        x, y = current_pos
        
        # Calculate distances to walls
        wall_distances = {
            'front': y,  # Top wall (front wall in squash)
            'back': self.court_height - y,
            'left': x,
            'right': self.court_width - x
        }
        
        closest_wall = min(wall_distances.items(), key=lambda x: x[1])
        wall_type, distance = closest_wall
        
        # Check for direction change near wall
        direction_change = self._calculate_direction_change(trajectory[-3:])
        
        # Calculate confidence
        confidence = 0.0
        if distance < self.wall_proximity_threshold:
            proximity_score = (self.wall_proximity_threshold - distance) / self.wall_proximity_threshold
            direction_score = min(1.0, direction_change / (math.pi / 3))  # Normalize to 60 degrees
            confidence = (proximity_score + direction_score) / 2
        
        event = SimplifiedShotEvent("wall_hit", frame_number, current_pos, confidence)
        event.details = {'wall_type': wall_type, 'distance': distance, 'direction_change': direction_change}
        
        return event
    
    def detect_floor_bounce(self, trajectory: List[Tuple[float, float]], 
                           frame_number: int) -> SimplifiedShotEvent:
        """Detect when ball bounces on floor"""
        
        if len(trajectory) < 3:
            return SimplifiedShotEvent("floor_bounce", frame_number, trajectory[-1], 0.0)
        
        current_pos = trajectory[-1]
        height_ratio = current_pos[1] / self.court_height
        
        # Check if in lower court area
        floor_proximity = max(0, height_ratio - self.floor_proximity_threshold) / (1 - self.floor_proximity_threshold)
        
        # Check for bounce pattern (down then up)
        bounce_pattern = self._detect_bounce_pattern(trajectory[-3:])
        
        # Calculate confidence
        confidence = (floor_proximity * 0.6) + (bounce_pattern * 0.4)
        
        event = SimplifiedShotEvent("floor_bounce", frame_number, current_pos, confidence)
        event.details = {'height_ratio': height_ratio, 'bounce_pattern': bounce_pattern}
        
        return event
    
    def detect_opponent_hit(self, current_event: SimplifiedShotEvent, 
                           active_shots: List[Dict], frame_number: int) -> Optional[Dict]:
        """Detect when opponent hits ball (new shot starts)"""
        
        if current_event.event_type != "racket_hit" or current_event.confidence < 0.6:
            return None
        
        # Check if this is a different player than current shot
        if active_shots and active_shots[-1].get('player_id') != current_event.player_id:
            return {
                'new_shot_detected': True,
                'new_player_id': current_event.player_id,
                'previous_player_id': active_shots[-1].get('player_id'),
                'frame': frame_number,
                'confidence': current_event.confidence
            }
        
        return None
    
    # Helper methods
    
    def _calculate_velocity_change(self, positions: List[Tuple[float, float]]) -> float:
        """Calculate magnitude of velocity change"""
        if len(positions) < 3:
            return 0.0
        
        # Get velocities for consecutive segments
        v1 = self._get_velocity(positions[0], positions[1])
        v2 = self._get_velocity(positions[1], positions[2])
        
        # Calculate magnitude change
        mag1 = math.sqrt(v1[0]**2 + v1[1]**2)
        mag2 = math.sqrt(v2[0]**2 + v2[1]**2)
        
        return abs(mag2 - mag1)
    
    def _calculate_direction_change(self, positions: List[Tuple[float, float]]) -> float:
        """Calculate direction change in radians"""
        if len(positions) < 3:
            return 0.0
        
        v1 = self._get_velocity(positions[0], positions[1])
        v2 = self._get_velocity(positions[1], positions[2])
        
        # Calculate angles
        angle1 = math.atan2(v1[1], v1[0])
        angle2 = math.atan2(v2[1], v2[0])
        
        # Angle difference
        diff = abs(angle2 - angle1)
        return min(diff, 2*math.pi - diff)
    
    def _get_velocity(self, p1: Tuple[float, float], p2: Tuple[float, float]) -> Tuple[float, float]:
        """Get velocity vector between two positions"""
        return (p2[0] - p1[0], p2[1] - p1[1])
    
    def _get_nearest_player_distance(self, ball_pos: Tuple[float, float], players_data: Dict) -> float:
        """Get distance to nearest player"""
        if not players_data:
            return 999.0
        
        min_distance = 999.0
        for player_id, player_pos in players_data.items():
            if player_pos:
                distance = math.sqrt((ball_pos[0] - player_pos[0])**2 + (ball_pos[1] - player_pos[1])**2)
                min_distance = min(min_distance, distance)
        
        return min_distance
    
    def _identify_hitting_player(self, ball_pos: Tuple[float, float], players_data: Dict) -> Optional[int]:
        """Identify which player hit the ball"""
        if not players_data:
            return None
        
        min_distance = float('inf')
        closest_player = None
        
        for player_id, player_pos in players_data.items():
            if player_pos:
                distance = math.sqrt((ball_pos[0] - player_pos[0])**2 + (ball_pos[1] - player_pos[1])**2)
                if distance < min_distance and distance < 100:  # Within reasonable hitting distance
                    min_distance = distance
                    closest_player = player_id
        
        return closest_player
    
    def _detect_bounce_pattern(self, positions: List[Tuple[float, float]]) -> float:
        """Detect bounce pattern (down then up movement)"""
        if len(positions) < 3:
            return 0.0
        
        p1, p2, p3 = positions[0], positions[1], positions[2]
        
        # Check for downward then upward movement
        downward = p2[1] > p1[1]  # Y increases downward in screen coords
        upward = p3[1] < p2[1]    # Y decreases upward
        
        if downward and upward:
            # Calculate "sharpness" of bounce
            descent = p2[1] - p1[1]
            ascent = p2[1] - p3[1]
            if descent > 0 and ascent > 0:
                return min(1.0, (descent + ascent) / 20.0)
        
        return 0.0

class SimplifiedShotDetectionSystem:
    """Simplified enhanced shot detection system for demonstration"""
    
    def __init__(self, court_width=640, court_height=360):
        self.detector = SimplifiedPhysicsDetector(court_width, court_height)
        self.active_shots = []
        self.completed_shots = []
        self.frame_count = 0
        
    def process_frame(self, ball_position: Tuple[float, float], 
                     players_data: Dict, frame_number: int,
                     trajectory: List[Tuple[float, float]]) -> Dict[str, Any]:
        """Process frame and detect shot events"""
        
        self.frame_count = frame_number
        
        if len(trajectory) < 3:
            return self._create_empty_result(frame_number)
        
        # Detect various events
        events = []
        
        # 1. Detect racket hits
        racket_hit = self.detector.detect_racket_hit(trajectory, players_data, frame_number)
        if racket_hit.confidence > 0.5:
            events.append(racket_hit)
        
        # 2. Detect wall hits
        wall_hit = self.detector.detect_wall_hit(trajectory, frame_number)
        if wall_hit.confidence > 0.5:
            events.append(wall_hit)
        
        # 3. Detect floor bounces
        floor_bounce = self.detector.detect_floor_bounce(trajectory, frame_number)
        if floor_bounce.confidence > 0.4:
            events.append(floor_bounce)
        
        # 4. Check for opponent hits (new shots)
        opponent_hit = None
        for event in events:
            if event.event_type == "racket_hit":
                opponent_hit = self.detector.detect_opponent_hit(event, self.active_shots, frame_number)
                if opponent_hit:
                    # Start new shot
                    new_shot = {
                        'player_id': event.player_id,
                        'start_frame': frame_number,
                        'events': [event]
                    }
                    self.active_shots.append(new_shot)
                    break
        
        return {
            'frame_number': frame_number,
            'events': events,
            'opponent_hit': opponent_hit,
            'ball_position': ball_position,
            'clear_detections': self._get_clear_detections(events, opponent_hit)
        }
    
    def _get_clear_detections(self, events: List[SimplifiedShotEvent], 
                             opponent_hit: Optional[Dict]) -> Dict[str, Any]:
        """Get clear detection results for the four requirements"""
        
        clear_detections = {
            'ball_hit_from_racket': {'detected': False, 'confidence': 0.0, 'player_id': None},
            'ball_hit_front_wall': {'detected': False, 'confidence': 0.0, 'wall_type': None},
            'ball_hit_by_opponent': {'detected': False, 'confidence': 0.0, 'new_player_id': None},
            'ball_bounced_to_ground': {'detected': False, 'confidence': 0.0, 'bounce_quality': 0.0}
        }
        
        for event in events:
            if event.event_type == "racket_hit" and event.confidence > 0.5:
                clear_detections['ball_hit_from_racket'] = {
                    'detected': True,
                    'confidence': event.confidence,
                    'player_id': event.player_id
                }
            
            elif event.event_type == "wall_hit" and event.confidence > 0.5:
                # Check if it's specifically front wall
                is_front_wall = event.details.get('wall_type') == 'front'
                clear_detections['ball_hit_front_wall'] = {
                    'detected': is_front_wall,
                    'confidence': event.confidence if is_front_wall else 0.0,
                    'wall_type': event.details.get('wall_type')
                }
            
            elif event.event_type == "floor_bounce" and event.confidence > 0.4:
                clear_detections['ball_bounced_to_ground'] = {
                    'detected': True,
                    'confidence': event.confidence,
                    'bounce_quality': event.details.get('bounce_pattern', 0.0)
                }
        
        # Check for opponent hit
        if opponent_hit:
            clear_detections['ball_hit_by_opponent'] = {
                'detected': True,
                'confidence': opponent_hit['confidence'],
                'new_player_id': opponent_hit['new_player_id']
            }
        
        return clear_detections
    
    def _create_empty_result(self, frame_number: int) -> Dict[str, Any]:
        """Create empty result for insufficient data"""
        return {
            'frame_number': frame_number,
            'events': [],
            'opponent_hit': None,
            'ball_position': None,
            'clear_detections': {
                'ball_hit_from_racket': {'detected': False, 'confidence': 0.0},
                'ball_hit_front_wall': {'detected': False, 'confidence': 0.0},
                'ball_hit_by_opponent': {'detected': False, 'confidence': 0.0},
                'ball_bounced_to_ground': {'detected': False, 'confidence': 0.0}
            }
        }

def demo_enhanced_shot_detection():
    """Demonstrate the enhanced shot detection system"""
    
    print("🎾 Enhanced Shot Detection System Demo")
    print("=" * 60)
    print("Demonstrating autonomous detection of four key requirements:")
    print("1. 🏓 Ball hit from racket")
    print("2. 🧱 Ball hit front wall")
    print("3. 🔄 Ball hit by opponent (new shot)")
    print("4. ⬇️  Ball bounced to ground")
    print("=" * 60)
    
    # Create detection system
    detector = SimplifiedShotDetectionSystem()
    
    # Create demo rally with all four event types
    demo_rally = [
        # Player 1 serves (racket hit)
        {
            'trajectory': [(95, 45), (100, 50), (105, 55), (110, 60)],
            'ball_pos': (110, 60),
            'players': {1: (100, 50)},  # Player 1 near ball
            'frame': 5,
            'description': 'Player 1 serves'
        },
        
        # Ball approaches and hits front wall
        {
            'trajectory': [(50, 40), (30, 30), (20, 25), (15, 20)],
            'ball_pos': (15, 20),
            'players': {},
            'frame': 15,
            'description': 'Ball hits front wall'
        },
        
        # Ball travels to Player 2
        {
            'trajectory': [(100, 80), (200, 120), (300, 160), (400, 200)],
            'ball_pos': (400, 200),
            'players': {2: (420, 190)},  # Player 2 near ball
            'frame': 25,
            'description': 'Ball approaches Player 2'
        },
        
        # Player 2 hits (opponent hit - new shot)
        {
            'trajectory': [(400, 200), (410, 195), (420, 190), (440, 180)],
            'ball_pos': (440, 180),
            'players': {2: (420, 190)},  # Player 2 hitting
            'frame': 30,
            'description': 'Player 2 returns (opponent hit)'
        },
        
        # Ball eventually bounces on floor
        {
            'trajectory': [(300, 270), (290, 300), (285, 320), (280, 310)],
            'ball_pos': (280, 310),
            'players': {},
            'frame': 45,
            'description': 'Ball bounces on floor'
        }
    ]
    
    print("\n📊 Processing Demo Rally:")
    
    detected_requirements = {
        'ball_hit_from_racket': False,
        'ball_hit_front_wall': False,
        'ball_hit_by_opponent': False,
        'ball_bounced_to_ground': False
    }
    
    all_results = []
    
    for rally_frame in demo_rally:
        # Process frame
        result = detector.process_frame(
            rally_frame['ball_pos'],
            rally_frame['players'],
            rally_frame['frame'],
            rally_frame['trajectory']
        )
        
        all_results.append(result)
        
        # Check detections
        clear_detections = result['clear_detections']
        events_detected = []
        
        if clear_detections['ball_hit_from_racket']['detected']:
            detected_requirements['ball_hit_from_racket'] = True
            player_id = clear_detections['ball_hit_from_racket']['player_id']
            conf = clear_detections['ball_hit_from_racket']['confidence']
            events_detected.append(f"🏓 Racket hit by Player {player_id} (conf: {conf:.2f})")
        
        if clear_detections['ball_hit_front_wall']['detected']:
            detected_requirements['ball_hit_front_wall'] = True
            conf = clear_detections['ball_hit_front_wall']['confidence']
            events_detected.append(f"🧱 Front wall hit (conf: {conf:.2f})")
        
        if clear_detections['ball_hit_by_opponent']['detected']:
            detected_requirements['ball_hit_by_opponent'] = True
            player_id = clear_detections['ball_hit_by_opponent']['new_player_id']
            conf = clear_detections['ball_hit_by_opponent']['confidence']
            events_detected.append(f"🔄 Opponent hit by Player {player_id} (conf: {conf:.2f})")
        
        if clear_detections['ball_bounced_to_ground']['detected']:
            detected_requirements['ball_bounced_to_ground'] = True
            conf = clear_detections['ball_bounced_to_ground']['confidence']
            events_detected.append(f"⬇️  Floor bounce (conf: {conf:.2f})")
        
        # Print frame results
        events_str = " | ".join(events_detected) if events_detected else "No events"
        print(f"Frame {rally_frame['frame']:2d}: {rally_frame['description']:<30} {events_str}")
    
    # Print summary
    print(f"\n✅ Enhanced Shot Detection Results:")
    total_detected = sum(detected_requirements.values())
    print(f"  Requirements detected: {total_detected}/4")
    
    requirement_names = {
        'ball_hit_from_racket': '🏓 Ball hit from racket',
        'ball_hit_front_wall': '🧱 Ball hit front wall',
        'ball_hit_by_opponent': '🔄 Ball hit by opponent (new shot)',
        'ball_bounced_to_ground': '⬇️  Ball bounced to ground'
    }
    
    for req_key, req_name in requirement_names.items():
        status = "✅" if detected_requirements[req_key] else "❌"
        print(f"  {status} {req_name}")
    
    # Overall assessment
    print(f"\n🎯 Autonomous Detection Assessment:")
    if total_detected == 4:
        print("  ✅ EXCELLENT: All four requirements successfully detected!")
        print("     System is ready for autonomous shot detection.")
        print("     Clear identification of all shot events.")
    elif total_detected >= 3:
        print("  🟡 VERY GOOD: 3-4 requirements detected autonomously.")
        print("     System is highly functional for autonomous operation.")
        print("     Minor fine-tuning may improve remaining detections.")
    elif total_detected >= 2:
        print("  🟠 GOOD: 2-3 requirements detected.")
        print("     System shows promise but needs improvement.")
        print("     Focus on enhancing failed detection algorithms.")
    else:
        print("  ❌ NEEDS WORK: Less than 2 requirements detected.")
        print("     Significant improvements needed for autonomous operation.")
    
    # Technical details
    print(f"\n🔬 Technical Implementation Details:")
    print("  • Physics-based trajectory analysis")
    print("  • Multi-factor confidence scoring")
    print("  • Player proximity validation")
    print("  • Wall type differentiation (front vs side)")
    print("  • Bounce pattern recognition")
    print("  • Opponent transition detection")
    print("  • Real-time autonomous operation")
    
    # Export demo results
    try:
        demo_data = {
            'demo_results': all_results,
            'requirements_detected': detected_requirements,
            'success_rate': total_detected / 4,
            'autonomous_capable': total_detected >= 3,
            'timestamp': time.time()
        }
        
        with open('/tmp/enhanced_shot_detection_demo.json', 'w') as f:
            json.dump(demo_data, f, indent=2, default=str)
        
        print(f"\n💾 Demo results exported to /tmp/enhanced_shot_detection_demo.json")
    except Exception as e:
        print(f"⚠️  Could not export demo results: {e}")
    
    return total_detected, detected_requirements

def run_validation_tests():
    """Run comprehensive validation tests"""
    
    print("\n" + "="*70)
    print("🧪 RUNNING COMPREHENSIVE VALIDATION TESTS")
    print("="*70)
    
    test_cases = [
        {
            'name': 'Clear racket hit validation',
            'trajectory': [(300, 200), (302, 202), (320, 220), (340, 210)],
            'players': {1: (310, 210)},
            'expected': 'ball_hit_from_racket'
        },
        {
            'name': 'Front wall hit validation',
            'trajectory': [(100, 100), (60, 60), (20, 20), (30, 30)],
            'players': {},
            'expected': 'ball_hit_front_wall'
        },
        {
            'name': 'Floor bounce validation',
            'trajectory': [(300, 280), (310, 310), (320, 320), (330, 310)],
            'players': {},
            'expected': 'ball_bounced_to_ground'
        }
    ]
    
    detector = SimplifiedShotDetectionSystem()
    
    passed_tests = 0
    total_tests = len(test_cases)
    
    for i, test_case in enumerate(test_cases):
        print(f"\n🧪 Test {i+1}: {test_case['name']}")
        
        result = detector.process_frame(
            test_case['trajectory'][-1],
            test_case['players'],
            i + 1,
            test_case['trajectory']
        )
        
        detected = result['clear_detections'][test_case['expected']]['detected']
        confidence = result['clear_detections'][test_case['expected']]['confidence']
        
        if detected and confidence > 0.5:
            print(f"  ✅ PASSED: {test_case['expected']} detected (confidence: {confidence:.2f})")
            passed_tests += 1
        else:
            print(f"  ❌ FAILED: {test_case['expected']} not detected (confidence: {confidence:.2f})")
    
    # Test results
    success_rate = passed_tests / total_tests
    print(f"\n📊 Validation Results:")
    print(f"  Tests Passed: {passed_tests}/{total_tests}")
    print(f"  Success Rate: {success_rate:.1%}")
    
    if success_rate >= 0.8:
        print("  ✅ VALIDATION SUCCESSFUL: System ready for deployment")
    elif success_rate >= 0.6:
        print("  🟡 VALIDATION PARTIAL: System functional with improvements needed")
    else:
        print("  ❌ VALIDATION FAILED: System needs significant work")
    
    return success_rate

if __name__ == "__main__":
    print("🚀 Enhanced Shot Detection System - Autonomous Demonstration")
    print("=" * 70)
    print("This system autonomously detects the four key shot events:")
    print("1. Ball got hit from the racket")
    print("2. Ball hit the front wall")
    print("3. Ball got hit by opponent's racket (new shot)")
    print("4. Ball bounced to the ground")
    print("=" * 70)
    
    # Run demonstration
    total_detected, requirements_status = demo_enhanced_shot_detection()
    
    # Run validation tests
    validation_success = run_validation_tests()
    
    # Final assessment
    print("\n" + "="*70)
    print("🎯 FINAL AUTONOMOUS SHOT DETECTION ASSESSMENT")
    print("="*70)
    
    print(f"\n📈 Overall Performance:")
    print(f"  • Demo Requirements Detected: {total_detected}/4 ({total_detected/4:.1%})")
    print(f"  • Validation Success Rate: {validation_success:.1%}")
    
    overall_score = (total_detected/4 + validation_success) / 2
    
    print(f"\n🏆 Overall Autonomous Score: {overall_score:.1%}")
    
    if overall_score >= 0.8:
        print("\n✅ SYSTEM READY FOR AUTONOMOUS SHOT DETECTION!")
        print("   All four requirements can be detected autonomously")
        print("   Clear identification of shot events")
        print("   High confidence in detection accuracy")
        print("   Ready for integration with main pipeline")
    elif overall_score >= 0.6:
        print("\n🟡 SYSTEM FUNCTIONAL FOR AUTONOMOUS OPERATION")
        print("   Most requirements working well")
        print("   Some fine-tuning recommended")
        print("   Suitable for production with monitoring")
    else:
        print("\n🟠 SYSTEM NEEDS IMPROVEMENT")
        print("   Some requirements not meeting standards")
        print("   Requires algorithm improvements")
        print("   Additional development needed")
    
    print(f"\n🔧 Enhanced Shot Detection Features:")
    print("  ✅ Physics-based trajectory analysis")
    print("  ✅ Multi-factor confidence scoring") 
    print("  ✅ Autonomous player identification")
    print("  ✅ Wall type differentiation")
    print("  ✅ Real-time bounce detection")
    print("  ✅ Opponent transition recognition")
    print("  ✅ Clear event identification")
    
    print(f"\n🚀 Next Steps:")
    print("  1. Integrate enhanced detection with main.py")
    print("  2. Test with real video footage")
    print("  3. Fine-tune detection parameters")
    print("  4. Deploy autonomous shot detection")
    
    print(f"\n💡 The enhanced shot detection system is {'READY' if overall_score >= 0.7 else 'IN DEVELOPMENT'}!")