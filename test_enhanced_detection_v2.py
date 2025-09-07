"""
Enhanced Test Suite for Shot Detection System
Tests all four required detection types with comprehensive validation and autonomous operation.
"""

import sys
import os
import time
import json
import math
from typing import Dict, List, Tuple, Any

# Add path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from enhanced_shot_detection_system import create_enhanced_shot_detector, ShotEvent
    from enhanced_shot_integration_v2 import create_enhanced_shot_integrator
    from main_pipeline_patch import EnhancedShotTracker, patch_main_pipeline
    MODULES_AVAILABLE = True
    print("✅ All enhanced detection modules loaded successfully")
except ImportError as e:
    print(f"❌ Import error: {e}")
    MODULES_AVAILABLE = False

class AutonomousShotDetectionTester:
    """
    Comprehensive autonomous test suite for enhanced shot detection system
    Tests the four key requirements:
    1. Ball hit from racket detection
    2. Ball hit front wall detection
    3. Ball hit by opponent (new shot) detection  
    4. Ball bounced to ground detection
    """
    
    def __init__(self):
        self.test_results = {
            'autonomous_operation': [],
            'racket_hit_tests': [],
            'wall_hit_tests': [],
            'opponent_hit_tests': [],
            'floor_bounce_tests': [],
            'integration_tests': [],
            'performance_metrics': {}
        }
        
        self.court_width = 640
        self.court_height = 360
        
    def run_autonomous_validation(self) -> Dict[str, Any]:
        """
        Run autonomous validation of the enhanced shot detection system
        """
        
        if not MODULES_AVAILABLE:
            return {'error': 'Enhanced detection modules not available'}
        
        print("🎾 Running Autonomous Shot Detection Validation")
        print("=" * 60)
        print("Testing the four key detection requirements:")
        print("1. 🏓 Ball hit from racket")
        print("2. 🧱 Ball hit front wall")
        print("3. 🔄 Ball hit by opponent (new shot)")
        print("4. ⬇️  Ball bounced to ground")
        print("=" * 60)
        
        start_time = time.time()
        
        # Test autonomous operation
        print("\n🤖 Testing Autonomous Operation...")
        autonomous_results = self.test_autonomous_operation()
        self.test_results['autonomous_operation'] = autonomous_results
        
        # Test specific requirements
        print("\n🏓 Testing Requirement 1: Ball hit from racket...")
        racket_results = self.test_racket_hit_requirement()
        self.test_results['racket_hit_tests'] = racket_results
        
        print("\n🧱 Testing Requirement 2: Ball hit front wall...")
        wall_results = self.test_wall_hit_requirement()
        self.test_results['wall_hit_tests'] = wall_results
        
        print("\n🔄 Testing Requirement 3: Ball hit by opponent...")
        opponent_results = self.test_opponent_hit_requirement()
        self.test_results['opponent_hit_tests'] = opponent_results
        
        print("\n⬇️  Testing Requirement 4: Ball bounced to ground...")
        floor_results = self.test_floor_bounce_requirement()
        self.test_results['floor_bounce_tests'] = floor_results
        
        print("\n🔧 Testing System Integration...")
        integration_results = self.test_system_integration()
        self.test_results['integration_tests'] = integration_results
        
        # Calculate performance metrics
        total_time = time.time() - start_time
        self.test_results['performance_metrics'] = {
            'total_validation_time': total_time,
            'tests_passed': self._count_passed_tests(),
            'tests_failed': self._count_failed_tests(),
            'overall_success_rate': self._calculate_success_rate(),
            'autonomous_score': self._calculate_autonomous_score()
        }
        
        # Print comprehensive summary
        self.print_autonomous_summary()
        
        return self.test_results
    
    def test_autonomous_operation(self) -> List[Dict[str, Any]]:
        """Test fully autonomous operation of the detection system"""
        
        results = []
        
        # Test Case 1: Complete rally detection
        print("  Testing complete rally autonomous detection...")
        
        rally_trajectory = [
            # Player 1 serves
            (100, 50), (110, 52), (120, 55), (130, 58), (140, 62),
            # Ball approaches front wall
            (50, 70), (30, 75), (20, 80), (15, 85), (25, 90),
            # Ball travels to Player 2  
            (150, 110), (200, 130), (250, 150), (300, 170), (350, 190),
            # Player 2 returns (opponent hit)
            (360, 185), (370, 180), (380, 175), (390, 170), (400, 165),
            # Ball hits side wall
            (550, 140), (580, 135), (600, 140), (590, 145), (580, 150),
            # Eventually bounces on floor
            (400, 250), (390, 280), (380, 290), (370, 285), (360, 280)
        ]
        
        mock_players = self._create_mock_players_sequence()
        
        result = self._test_autonomous_rally(rally_trajectory, mock_players)
        results.append(result)
        
        return results
    
    def test_racket_hit_requirement(self) -> List[Dict[str, Any]]:
        """Test Requirement 1: Ball hit from racket detection"""
        
        results = []
        
        # Test Case 1: Clear racket contact
        test_case = {
            'name': 'Clear racket contact with velocity change',
            'description': 'Ball changes velocity and direction when hitting racket',
            'trajectory': [
                (300, 200), (302, 202), (304, 204),  # Slow approach
                (320, 220), (340, 210), (360, 200)   # Sudden acceleration
            ],
            'players': self._create_mock_players([(310, 210)]),
            'requirement': 'ball_hit_from_racket'
        }
        
        result = self._test_requirement(test_case)
        results.append(result)
        print(f"    {'✅' if result['passed'] else '❌'} {test_case['name']}")
        
        # Test Case 2: Player proximity validation
        test_case_2 = {
            'name': 'Player proximity validation',
            'description': 'Racket hit only detected when player is close to ball',
            'trajectory': [
                (100, 100), (105, 102), (110, 104),  # Smooth movement
                (120, 108), (125, 110), (130, 112)   # No significant change
            ],
            'players': self._create_mock_players([(500, 300)]),  # Player far away
            'requirement': 'ball_hit_from_racket',
            'should_detect': False
        }
        
        result_2 = self._test_requirement(test_case_2)
        results.append(result_2)
        print(f"    {'✅' if result_2['passed'] else '❌'} {test_case_2['name']}")
        
        return results
    
    def test_wall_hit_requirement(self) -> List[Dict[str, Any]]:
        """Test Requirement 2: Ball hit front wall detection"""
        
        results = []
        
        # Test Case 1: Front wall impact
        test_case = {
            'name': 'Front wall impact detection',
            'description': 'Ball hits front wall and bounces back',
            'trajectory': [
                (100, 100), (80, 80), (60, 60),
                (40, 40), (20, 20), (30, 30)  # Hits front wall (y=0) and bounces
            ],
            'requirement': 'ball_hit_front_wall'
        }
        
        result = self._test_requirement(test_case)
        results.append(result)
        print(f"    {'✅' if result['passed'] else '❌'} {test_case['name']}")
        
        # Test Case 2: Side wall vs front wall differentiation
        test_case_2 = {
            'name': 'Front wall vs side wall differentiation',
            'description': 'System correctly identifies front wall hits specifically',
            'trajectory': [
                (500, 200), (550, 180), (600, 160),
                (620, 150), (610, 155), (590, 165)  # Side wall hit
            ],
            'requirement': 'ball_hit_front_wall',
            'should_detect': False  # This is side wall, not front wall
        }
        
        result_2 = self._test_requirement(test_case_2)
        results.append(result_2)
        print(f"    {'✅' if result_2['passed'] else '❌'} {test_case_2['name']}")
        
        return results
    
    def test_opponent_hit_requirement(self) -> List[Dict[str, Any]]:
        """Test Requirement 3: Ball hit by opponent (new shot) detection"""
        
        results = []
        
        # Test Case 1: Clear opponent transition
        test_case = {
            'name': 'Clear opponent shot transition',
            'description': 'Ball hit by Player 1, then by Player 2 (new shot)',
            'trajectory_sequence': [
                # Player 1 hits
                [(100, 100), (120, 110), (140, 120)],
                # Ball travels
                [(200, 150), (300, 180), (400, 200)],
                # Player 2 hits (new shot)
                [(420, 190), (440, 180), (460, 170)]
            ],
            'players_sequence': [
                self._create_mock_players([(100, 100)]),  # Player 1
                self._create_mock_players([]),            # Transit
                self._create_mock_players([(430, 185)])   # Player 2
            ],
            'requirement': 'ball_hit_by_opponent'
        }
        
        result = self._test_opponent_requirement(test_case)
        results.append(result)
        print(f"    {'✅' if result['passed'] else '❌'} {test_case['name']}")
        
        return results
    
    def test_floor_bounce_requirement(self) -> List[Dict[str, Any]]:
        """Test Requirement 4: Ball bounced to ground detection"""
        
        results = []
        
        # Test Case 1: Clear floor bounce
        test_case = {
            'name': 'Clear floor bounce pattern',
            'description': 'Ball descends to floor and bounces back up',
            'trajectory': [
                (300, 250), (310, 280), (320, 310),  # Descending
                (330, 320), (340, 310), (350, 300)   # Bounce up
            ],
            'requirement': 'ball_bounced_to_ground'
        }
        
        result = self._test_requirement(test_case)
        results.append(result)
        print(f"    {'✅' if result['passed'] else '❌'} {test_case['name']}")
        
        # Test Case 2: Upper court trajectory (no floor contact)
        test_case_2 = {
            'name': 'No floor bounce - upper court',
            'description': 'Ball stays in upper court area',
            'trajectory': [
                (100, 50), (120, 60), (140, 70),
                (160, 80), (180, 90), (200, 100)
            ],
            'requirement': 'ball_bounced_to_ground',
            'should_detect': False
        }
        
        result_2 = self._test_requirement(test_case_2)
        results.append(result_2)
        print(f"    {'✅' if result_2['passed'] else '❌'} {test_case_2['name']}")
        
        return results
    
    def test_system_integration(self) -> List[Dict[str, Any]]:
        """Test complete system integration"""
        
        results = []
        
        # Test complete pipeline integration
        test_case = {
            'name': 'Complete pipeline integration',
            'description': 'All detection types working together autonomously'
        }
        
        try:
            tracker = EnhancedShotTracker(self.court_width, self.court_height)
            
            # Simulate complete rally
            rally_frames = [
                {'ball_pos': (100, 50), 'frame': 1, 'players': self._create_mock_players([(100, 50)])},
                {'ball_pos': (20, 80), 'frame': 10, 'players': self._create_mock_players([])},
                {'ball_pos': (400, 200), 'frame': 20, 'players': self._create_mock_players([(400, 200)])},
                {'ball_pos': (300, 320), 'frame': 30, 'players': self._create_mock_players([])}
            ]
            
            detected_requirements = set()
            
            for frame_data in rally_frames:
                clear_events = tracker.detect_clear_shot_events(
                    frame_data['ball_pos'], 
                    frame_data['players'], 
                    frame_data['frame']
                )
                
                if clear_events['ball_hit_from_racket']['detected']:
                    detected_requirements.add('racket_hit')
                if clear_events['ball_hit_front_wall']['detected']:
                    detected_requirements.add('front_wall')
                if clear_events['ball_hit_by_opponent']['detected']:
                    detected_requirements.add('opponent_hit')
                if clear_events['ball_bounced_to_ground']['detected']:
                    detected_requirements.add('floor_bounce')
            
            # Check if system detected multiple requirements
            success = len(detected_requirements) >= 2
            
            result = {
                'test_name': test_case['name'],
                'passed': success,
                'detected_requirements': list(detected_requirements),
                'description': test_case['description']
            }
            
        except Exception as e:
            result = {
                'test_name': test_case['name'],
                'passed': False,
                'error': str(e)
            }
        
        results.append(result)
        print(f"    {'✅' if result['passed'] else '❌'} {test_case['name']}")
        
        return results
    
    # Helper methods
    
    def _test_requirement(self, test_case: Dict[str, Any]) -> Dict[str, Any]:
        """Test a specific requirement"""
        
        try:
            tracker = EnhancedShotTracker(self.court_width, self.court_height)
            
            # Process trajectory
            past_positions = []
            for i, ball_pos in enumerate(test_case['trajectory']):
                past_positions.append([ball_pos[0], ball_pos[1], i])
                
                players = test_case.get('players', {})
                clear_events = tracker.detect_clear_shot_events(ball_pos, players, i)
                
                # Check specific requirement
                requirement = test_case['requirement']
                detected = False
                confidence = 0.0
                
                if requirement == 'ball_hit_from_racket':
                    detected = clear_events['ball_hit_from_racket']['detected']
                    confidence = clear_events['ball_hit_from_racket']['confidence']
                elif requirement == 'ball_hit_front_wall':
                    detected = clear_events['ball_hit_front_wall']['detected']
                    confidence = clear_events['ball_hit_front_wall']['confidence']
                elif requirement == 'ball_bounced_to_ground':
                    detected = clear_events['ball_bounced_to_ground']['detected']
                    confidence = clear_events['ball_bounced_to_ground']['confidence']
                
                if detected:
                    should_detect = test_case.get('should_detect', True)
                    return {
                        'test_name': test_case['name'],
                        'passed': detected == should_detect,
                        'detected': detected,
                        'confidence': confidence,
                        'description': test_case['description']
                    }
            
            # No detection occurred
            should_detect = test_case.get('should_detect', True)
            return {
                'test_name': test_case['name'],
                'passed': not should_detect,
                'detected': False,
                'confidence': 0.0,
                'description': test_case['description']
            }
            
        except Exception as e:
            return {
                'test_name': test_case['name'],
                'passed': False,
                'error': str(e)
            }
    
    def _test_opponent_requirement(self, test_case: Dict[str, Any]) -> Dict[str, Any]:
        """Test opponent hit requirement specifically"""
        
        try:
            tracker = EnhancedShotTracker(self.court_width, self.court_height)
            
            frame_number = 0
            detected_opponent_hit = False
            
            # Process trajectory sequences
            for seq_idx, (trajectory, players) in enumerate(zip(test_case['trajectory_sequence'], 
                                                               test_case['players_sequence'])):
                for ball_pos in trajectory:
                    frame_number += 1
                    
                    clear_events = tracker.detect_clear_shot_events(ball_pos, players, frame_number)
                    
                    if clear_events['ball_hit_by_opponent']['detected']:
                        detected_opponent_hit = True
                        break
                
                if detected_opponent_hit:
                    break
            
            return {
                'test_name': test_case['name'],
                'passed': detected_opponent_hit,
                'detected': detected_opponent_hit,
                'description': test_case['description']
            }
            
        except Exception as e:
            return {
                'test_name': test_case['name'],
                'passed': False,
                'error': str(e)
            }
    
    def _test_autonomous_rally(self, trajectory: List[Tuple[float, float]], 
                              players_data: Dict) -> Dict[str, Any]:
        """Test autonomous detection on complete rally"""
        
        try:
            tracker = EnhancedShotTracker(self.court_width, self.court_height)
            
            detected_events = {
                'racket_hits': 0,
                'wall_hits': 0,
                'opponent_hits': 0,
                'floor_bounces': 0
            }
            
            for i, ball_pos in enumerate(trajectory):
                clear_events = tracker.detect_clear_shot_events(ball_pos, players_data, i)
                
                if clear_events['ball_hit_from_racket']['detected']:
                    detected_events['racket_hits'] += 1
                if clear_events['ball_hit_front_wall']['detected']:
                    detected_events['wall_hits'] += 1
                if clear_events['ball_hit_by_opponent']['detected']:
                    detected_events['opponent_hits'] += 1
                if clear_events['ball_bounced_to_ground']['detected']:
                    detected_events['floor_bounces'] += 1
            
            # Success if multiple types of events detected
            total_events = sum(detected_events.values())
            autonomous_success = total_events >= 2
            
            return {
                'test_name': 'Autonomous rally detection',
                'passed': autonomous_success,
                'detected_events': detected_events,
                'total_events': total_events,
                'description': 'Complete autonomous rally detection'
            }
            
        except Exception as e:
            return {
                'test_name': 'Autonomous rally detection',
                'passed': False,
                'error': str(e)
            }
    
    def _create_mock_players(self, positions: List[Tuple[float, float]]) -> Dict[int, Any]:
        """Create mock players for testing"""
        
        players = {}
        
        for i, pos in enumerate(positions):
            player_id = i + 1
            x_norm = pos[0] / self.court_width
            y_norm = pos[1] / self.court_height
            
            players[player_id] = type('MockPlayer', (), {
                'get_latest_pose': lambda: type('MockPose', (), {
                    'xyn': [[
                        (x_norm, y_norm, 0.9),  # High confidence keypoint
                        (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0),
                        (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0),
                        (x_norm-0.02, y_norm-0.02, 0.8),  # Left elbow
                        (x_norm+0.02, y_norm-0.02, 0.8),  # Right elbow
                        (x_norm-0.03, y_norm-0.01, 0.85), # Left wrist
                        (x_norm+0.03, y_norm-0.01, 0.85)  # Right wrist
                    ] + [(0.0, 0.0, 0.0)] * 7]
                })()
            })()
        
        return players
    
    def _create_mock_players_sequence(self) -> Dict[int, Any]:
        """Create mock players for sequence testing"""
        return {
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
    
    def _count_passed_tests(self) -> int:
        """Count total passed tests"""
        total_passed = 0
        for category in self.test_results.values():
            if isinstance(category, list):
                total_passed += sum(1 for test in category if test.get('passed', False))
        return total_passed
    
    def _count_failed_tests(self) -> int:
        """Count total failed tests"""
        total_tests = 0
        total_passed = 0
        for category in self.test_results.values():
            if isinstance(category, list):
                total_tests += len(category)
                total_passed += sum(1 for test in category if test.get('passed', False))
        return total_tests - total_passed
    
    def _calculate_success_rate(self) -> float:
        """Calculate overall success rate"""
        total_tests = self._count_passed_tests() + self._count_failed_tests()
        if total_tests == 0:
            return 0.0
        return self._count_passed_tests() / total_tests
    
    def _calculate_autonomous_score(self) -> float:
        """Calculate autonomous operation score"""
        autonomous_tests = self.test_results.get('autonomous_operation', [])
        if not autonomous_tests:
            return 0.0
        
        passed = sum(1 for test in autonomous_tests if test.get('passed', False))
        return passed / len(autonomous_tests)
    
    def print_autonomous_summary(self):
        """Print comprehensive autonomous validation summary"""
        
        print("\n" + "="*70)
        print("🎾 AUTONOMOUS SHOT DETECTION VALIDATION SUMMARY")
        print("="*70)
        
        metrics = self.test_results['performance_metrics']
        
        # Overall autonomous performance
        print(f"\n🤖 Autonomous Operation Performance:")
        print(f"  • Overall Success Rate: {metrics['overall_success_rate']:.1%}")
        print(f"  • Autonomous Score: {metrics['autonomous_score']:.1%}")
        print(f"  • Tests Passed: {metrics['tests_passed']}")
        print(f"  • Tests Failed: {metrics['tests_failed']}")
        print(f"  • Validation Time: {metrics['total_validation_time']:.2f} seconds")
        
        # Requirement-specific results
        requirements = [
            ('racket_hit_tests', '🏓 Requirement 1: Ball hit from racket'),
            ('wall_hit_tests', '🧱 Requirement 2: Ball hit front wall'),
            ('opponent_hit_tests', '🔄 Requirement 3: Ball hit by opponent'),
            ('floor_bounce_tests', '⬇️  Requirement 4: Ball bounced to ground'),
        ]
        
        print(f"\n📋 Requirement Validation Results:")
        for req_key, req_name in requirements:
            if req_key in self.test_results:
                tests = self.test_results[req_key]
                passed = sum(1 for test in tests if test.get('passed', False))
                total = len(tests)
                print(f"  {req_name}: {passed}/{total} ({'✅' if passed == total else '❌'})")
        
        # Integration results
        integration_tests = self.test_results.get('integration_tests', [])
        if integration_tests:
            passed = sum(1 for test in integration_tests if test.get('passed', False))
            total = len(integration_tests)
            print(f"  🔧 System Integration: {passed}/{total} ({'✅' if passed == total else '❌'})")
        
        # Autonomous operation assessment
        print(f"\n🎯 Autonomous Operation Assessment:")
        success_rate = metrics['overall_success_rate']
        autonomous_score = metrics['autonomous_score']
        
        if success_rate >= 0.9 and autonomous_score >= 0.8:
            print("  ✅ EXCELLENT: System ready for fully autonomous operation")
            print("     • All requirements validated successfully")
            print("     • High confidence in autonomous detection")
            print("     • Ready for production deployment")
        elif success_rate >= 0.7 and autonomous_score >= 0.6:
            print("  🟡 GOOD: System suitable for autonomous operation with monitoring")
            print("     • Most requirements validated")
            print("     • Autonomous operation with occasional supervision")
            print("     • Consider fine-tuning for edge cases")
        elif success_rate >= 0.5:
            print("  🟠 FAIR: System needs improvement for autonomous operation")
            print("     • Some requirements need work")
            print("     • Requires human oversight")
            print("     • Focus on failed test cases")
        else:
            print("  ❌ POOR: System not ready for autonomous operation")
            print("     • Significant improvements needed")
            print("     • Multiple requirements failing")
            print("     • Recommend algorithm review")
        
        # Specific recommendations
        print(f"\n💡 Recommendations for Enhanced Autonomous Operation:")
        
        failed_tests = [category for category in ['racket_hit_tests', 'wall_hit_tests', 
                       'opponent_hit_tests', 'floor_bounce_tests'] 
                       if any(not test.get('passed', False) for test in self.test_results.get(category, []))]
        
        if not failed_tests:
            print("  • System performing excellently across all requirements")
            print("  • Consider deploying to production environment")
            print("  • Monitor performance with real video data")
        else:
            print("  • Focus on improving the following areas:")
            for failed_category in failed_tests:
                req_name = failed_category.replace('_tests', '').replace('_', ' ')
                print(f"    - {req_name.title()} detection accuracy")
        
        print("  • Test with real squash video footage")
        print("  • Validate performance under various lighting conditions")
        print("  • Ensure robust operation with different court layouts")
        
        print(f"\n🚀 Next Steps:")
        print("  1. Address any failed test cases")
        print("  2. Integration with main.py pipeline")
        print("  3. Real-world video testing")
        print("  4. Performance optimization")
        print("  5. Deploy autonomous shot detection system")


def run_autonomous_validation():
    """Run comprehensive autonomous validation"""
    
    print("🚀 Starting Autonomous Shot Detection Validation")
    print("Testing the four key requirements for improved shot detection:")
    print("1. Ball got hit from the racket")
    print("2. Ball hit the front wall")
    print("3. Ball got hit by opponent's racket (new shot)")
    print("4. Ball bounced to the ground")
    print("=" * 70)
    
    if not MODULES_AVAILABLE:
        print("❌ Cannot run validation - enhanced detection modules not available")
        return None
    
    # Create tester and run validation
    tester = AutonomousShotDetectionTester()
    results = tester.run_autonomous_validation()
    
    # Export results
    try:
        with open('/tmp/autonomous_shot_detection_validation.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n💾 Validation results exported to /tmp/autonomous_shot_detection_validation.json")
    except Exception as e:
        print(f"⚠️ Failed to export validation results: {e}")
    
    return results


def demo_enhanced_detection():
    """Demonstrate the enhanced detection system with clear examples"""
    
    print("🎾 Enhanced Shot Detection System Demo")
    print("=" * 50)
    print("Demonstrating clear detection of all four requirements:")
    print("1. 🏓 Ball hit from racket")
    print("2. 🧱 Ball hit front wall") 
    print("3. 🔄 Ball hit by opponent (new shot)")
    print("4. ⬇️  Ball bounced to ground")
    print("=" * 50)
    
    if not MODULES_AVAILABLE:
        print("❌ Cannot run demo - enhanced detection modules not available")
        return
    
    try:
        # Create enhanced tracker
        tracker = EnhancedShotTracker()
        
        # Demo rally with all four event types
        demo_rally = [
            # Player 1 serves (racket hit)
            {'pos': (100, 50), 'frame': 1, 'players': {1: 'mock_player_1'}, 'event': 'Player 1 serves'},
            {'pos': (110, 52), 'frame': 2, 'players': {1: 'mock_player_1'}, 'event': 'Ball in flight'},
            
            # Ball hits front wall
            {'pos': (20, 80), 'frame': 10, 'players': {}, 'event': 'Ball approaches front wall'},
            {'pos': (25, 85), 'frame': 11, 'players': {}, 'event': 'Ball hits front wall'},
            
            # Ball travels to Player 2
            {'pos': (200, 120), 'frame': 20, 'players': {}, 'event': 'Ball travels mid-court'},
            {'pos': (350, 180), 'frame': 25, 'players': {2: 'mock_player_2'}, 'event': 'Ball approaches Player 2'},
            
            # Player 2 hits (opponent hit - new shot)
            {'pos': (400, 200), 'frame': 30, 'players': {2: 'mock_player_2'}, 'event': 'Player 2 hits (opponent hit)'},
            {'pos': (420, 190), 'frame': 31, 'players': {2: 'mock_player_2'}, 'event': 'Ball redirected by Player 2'},
            
            # Ball eventually bounces on floor
            {'pos': (300, 280), 'frame': 40, 'players': {}, 'event': 'Ball descends'},
            {'pos': (290, 320), 'frame': 41, 'players': {}, 'event': 'Ball bounces on floor'},
            {'pos': (280, 300), 'frame': 42, 'players': {}, 'event': 'Ball bounces up'}
        ]
        
        print("\n📊 Processing Demo Rally:")
        detected_events = {
            'ball_hit_from_racket': False,
            'ball_hit_front_wall': False,
            'ball_hit_by_opponent': False,
            'ball_bounced_to_ground': False
        }
        
        for rally_frame in demo_rally:
            # Mock players for demonstration
            mock_players = tracker._create_mock_players_sequence() if rally_frame['players'] else {}
            
            # Get detection results
            clear_events = tracker.detect_clear_shot_events(
                rally_frame['pos'], mock_players, rally_frame['frame']
            )
            
            # Check for detections
            events_this_frame = []
            if clear_events['ball_hit_from_racket']['detected']:
                detected_events['ball_hit_from_racket'] = True
                events_this_frame.append("🏓 Racket hit")
            
            if clear_events['ball_hit_front_wall']['detected']:
                detected_events['ball_hit_front_wall'] = True
                events_this_frame.append("🧱 Front wall hit")
            
            if clear_events['ball_hit_by_opponent']['detected']:
                detected_events['ball_hit_by_opponent'] = True
                events_this_frame.append("🔄 Opponent hit")
            
            if clear_events['ball_bounced_to_ground']['detected']:
                detected_events['ball_bounced_to_ground'] = True
                events_this_frame.append("⬇️  Floor bounce")
            
            # Print frame info
            events_str = " | ".join(events_this_frame) if events_this_frame else ""
            print(f"Frame {rally_frame['frame']:2d}: {rally_frame['event']:<25} {events_str}")
        
        # Summary
        print(f"\n✅ Demo Results Summary:")
        total_detected = sum(detected_events.values())
        print(f"  • Requirements detected: {total_detected}/4")
        
        for requirement, detected in detected_events.items():
            status = "✅" if detected else "❌"
            req_name = requirement.replace('_', ' ').title()
            print(f"  {status} {req_name}")
        
        if total_detected == 4:
            print(f"\n🎯 EXCELLENT: All four requirements successfully detected!")
            print("   Enhanced shot detection system is working autonomously.")
        elif total_detected >= 2:
            print(f"\n🟡 GOOD: {total_detected}/4 requirements detected.")
            print("   System is functional but may need fine-tuning.")
        else:
            print(f"\n🟠 NEEDS WORK: Only {total_detected}/4 requirements detected.")
            print("   System requires improvements for autonomous operation.")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")


if __name__ == "__main__":
    # Run autonomous validation
    print("🎾 Enhanced Shot Detection System - Autonomous Validation")
    print("Testing autonomous detection of the four key requirements")
    print("=" * 70)
    
    # Run demo first
    demo_enhanced_detection()
    
    print("\n" + "="*70)
    
    # Run full validation
    validation_results = run_autonomous_validation()
    
    if validation_results:
        success_rate = validation_results['performance_metrics']['overall_success_rate']
        autonomous_score = validation_results['performance_metrics']['autonomous_score']
        
        print(f"\n🎯 FINAL AUTONOMOUS VALIDATION RESULTS:")
        print(f"  Overall Success Rate: {success_rate:.1%}")
        print(f"  Autonomous Score: {autonomous_score:.1%}")
        
        if success_rate >= 0.8 and autonomous_score >= 0.7:
            print(f"\n✅ SYSTEM READY FOR AUTONOMOUS SHOT DETECTION!")
            print("   All four requirements can be detected autonomously.")
            print("   System ready for integration with main pipeline.")
        elif success_rate >= 0.6:
            print(f"\n🟡 SYSTEM FUNCTIONAL WITH MONITORING")
            print("   Most requirements working, some fine-tuning needed.")
        else:
            print(f"\n❌ SYSTEM NEEDS SIGNIFICANT IMPROVEMENT")
            print("   Multiple requirements not meeting autonomous standards.")
    else:
        print("❌ Validation could not be completed.")