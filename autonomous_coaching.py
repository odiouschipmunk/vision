"""
Autonomous Squash Coaching Pipeline - Clean Implementation
Integrates ball detection, pose estimation, shot classification, and AI coaching using Qwen3
"""

import time
import torch
import cv2
import json
import math
import numpy as np
import logging
import ast
import os
from collections import defaultdict
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
from plotly.subplots import make_subplots
import plotly.offline as pyo

# Import your existing modules
from ultralytics import YOLO
from squash import Referencepoints
from squash.Ball import Ball
from squash.Player import Player

class SquashPatternAnalyzer:
    """Advanced pattern recognition for squash-specific behaviors"""
    
    def __init__(self):
        self.shot_patterns = {
            'attacking_sequence': ['crosscourt', 'straight', 'drop'],
            'defensive_sequence': ['lob', 'deep_drive', 'crosscourt'],
            'pressure_building': ['straight', 'straight', 'crosscourt'],
            'winner_setup': ['crosscourt', 'boast', 'drop']
        }
        
        self.movement_patterns = {
            'good_court_position': {'T_recovery_ratio': 0.7, 'court_coverage': 0.8},
            'poor_positioning': {'T_recovery_ratio': 0.3, 'court_coverage': 0.5},
            'aggressive_movement': {'movement_intensity': 0.8, 'forward_bias': 0.6},
            'defensive_movement': {'movement_intensity': 0.5, 'back_court_bias': 0.7}
        }

    def analyze_shot_sequences(self, shot_sequence):
        """Analyze shot sequences for tactical patterns"""
        if len(shot_sequence) < 3:
            return {'pattern': 'insufficient_data', 'confidence': 0.0}
        
        pattern_scores = {}
        
        for pattern_name, pattern_sequence in self.shot_patterns.items():
            score = self._calculate_sequence_similarity(shot_sequence, pattern_sequence)
            pattern_scores[pattern_name] = score
        
        best_pattern = max(pattern_scores.items(), key=lambda x: x[1])
        
        return {
            'pattern': best_pattern[0],
            'confidence': best_pattern[1],
            'all_scores': pattern_scores,
            'tactical_advice': self._get_tactical_advice(best_pattern[0], best_pattern[1])
        }
    
    def _calculate_sequence_similarity(self, actual_sequence, pattern_sequence):
        """Calculate similarity between actual and expected shot sequences"""
        if not actual_sequence or not pattern_sequence:
            return 0.0
        
        # Use sliding window to find best match
        max_score = 0.0
        pattern_len = len(pattern_sequence)
        
        for i in range(len(actual_sequence) - pattern_len + 1):
            window = actual_sequence[i:i + pattern_len]
            matches = sum(1 for a, p in zip(window, pattern_sequence) if a == p)
            score = matches / pattern_len
            max_score = max(max_score, score)
        
        return max_score
    
    def _get_tactical_advice(self, pattern, confidence):
        """Provide tactical advice based on identified patterns"""
        if confidence < 0.4:
            return "Vary shot selection more to create unpredictability"
        
        advice_map = {
            'attacking_sequence': "Good attacking pattern! Continue with precision and placement",
            'defensive_sequence': "Solid defensive play. Look for opportunities to counter-attack",
            'pressure_building': "Excellent pressure building. Maintain depth and accuracy",
            'winner_setup': "Great setup play! Focus on execution of the finishing shot"
        }
        
        return advice_map.get(pattern, "Continue developing tactical awareness")

class TacticalAnalysisEngine:
    """Advanced tactical analysis engine for squash coaching"""
    
    def __init__(self):
        self.tactical_patterns = {
            'attacking_patterns': {
                'front_court_pressure': ['drop', 'drop', 'drive'],
                'length_then_short': ['drive', 'drive', 'drop'],
                'cross_court_attack': ['crosscourt', 'straight', 'drop'],
                'power_sequence': ['drive', 'drive', 'kill']
            },
            'defensive_patterns': {
                'pressure_relief': ['lob', 'lob', 'drive'],
                'length_game': ['drive', 'drive', 'drive'],
                'defensive_lob': ['lob', 'crosscourt', 'lob'],
                'corner_to_corner': ['crosscourt', 'crosscourt', 'straight']
            },
            'transitional_patterns': {
                'build_and_attack': ['drive', 'crosscourt', 'drop'],
                'counter_attack': ['lob', 'drive', 'kill'],
                'pressure_builder': ['straight', 'straight', 'crosscourt'],
                'deception_sequence': ['crosscourt', 'boast', 'drop']
            }
        }
        
        self.court_zones = {
            'front_court': (0.0, 0.3),  # Y-axis range
            'mid_court': (0.3, 0.7),
            'back_court': (0.7, 1.0)
        }
        
        self.tactical_strengths = {}
        self.tactical_weaknesses = {}

    def analyze_tactical_patterns(self, shot_sequence, position_sequence=None):
        """Analyze tactical patterns in shot sequences"""
        if len(shot_sequence) < 3:
            return {'pattern': 'insufficient_data', 'confidence': 0.0, 'advice': []}
        
        pattern_matches = {}
        tactical_advice = []
        
        # Analyze each pattern type
        for pattern_type, patterns in self.tactical_patterns.items():
            for pattern_name, pattern_shots in patterns.items():
                confidence = self._calculate_pattern_match(shot_sequence, pattern_shots)
                
                if confidence > 0.4:
                    pattern_matches[f"{pattern_type}_{pattern_name}"] = confidence
                    tactical_advice.extend(self._get_pattern_advice(pattern_type, pattern_name, confidence))
        
        # Analyze court positioning if available
        if position_sequence:
            positioning_analysis = self._analyze_court_positioning(position_sequence)
            tactical_advice.extend(positioning_analysis)
        
        # Get the best matching pattern
        best_pattern = max(pattern_matches.items(), key=lambda x: x[1]) if pattern_matches else ('no_pattern', 0.0)
        
        return {
            'pattern': best_pattern[0],
            'confidence': best_pattern[1],
            'all_matches': pattern_matches,
            'tactical_advice': tactical_advice,
            'shot_distribution': self._analyze_shot_distribution(shot_sequence)
        }

    def _calculate_pattern_match(self, shot_sequence, pattern_shots):
        """Calculate how well a shot sequence matches a tactical pattern"""
        if len(shot_sequence) < len(pattern_shots):
            return 0.0
        
        max_match = 0.0
        pattern_len = len(pattern_shots)
        
        # Use sliding window to find best match
        for i in range(len(shot_sequence) - pattern_len + 1):
            window = shot_sequence[i:i + pattern_len]
            matches = sum(1 for a, p in zip(window, pattern_shots) 
                         if self._shots_match(a, p))
            match_score = matches / pattern_len
            max_match = max(max_match, match_score)
        
        return max_match

    def _shots_match(self, actual_shot, pattern_shot):
        """Check if an actual shot matches a pattern shot"""
        if isinstance(actual_shot, list):
            actual_shot = actual_shot[0] if actual_shot else 'unknown'
        
        # Direct match
        if actual_shot == pattern_shot:
            return True
        
        # Semantic matching for similar shots
        shot_groups = {
            'drive': ['straight_drive', 'drive', 'straight'],
            'crosscourt': ['crosscourt', 'angled_crosscourt', 'cross'],
            'drop': ['drop_shot', 'drop', 'nick'],
            'lob': ['lob', 'high_ball', 'defensive_lob'],
            'boast': ['boast', 'three_wall_boast', 'angle'],
            'kill': ['kill_shot', 'nick_shot', 'winner']
        }
        
        for group_name, shots in shot_groups.items():
            if pattern_shot in shots and actual_shot in shots:
                return True
        
        return False

    def _get_pattern_advice(self, pattern_type, pattern_name, confidence):
        """Get tactical advice based on identified patterns"""
        advice = []
        
        if confidence > 0.7:
            if pattern_type == 'attacking_patterns':
                advice.append(f"✅ Excellent {pattern_name.replace('_', ' ')} execution!")
                advice.append("Continue this aggressive approach when opportunities arise.")
            elif pattern_type == 'defensive_patterns':
                advice.append(f"✅ Good {pattern_name.replace('_', ' ')} - solid defensive play.")
                advice.append("Look for chances to transition to attack.")
        elif confidence > 0.4:
            advice.append(f"🔄 {pattern_name.replace('_', ' ')} partially executed.")
            advice.append("Focus on completing the full tactical sequence.")
        
        return advice

    def _analyze_court_positioning(self, position_sequence):
        """Analyze court positioning for tactical insights"""
        advice = []
        
        if not position_sequence:
            return advice
        
        # Calculate time spent in each court zone
        zone_time = {'front_court': 0, 'mid_court': 0, 'back_court': 0}
        
        for pos in position_sequence:
            if len(pos) >= 2:
                y_pos = pos[1]
                for zone, (y_min, y_max) in self.court_zones.items():
                    if y_min <= y_pos <= y_max:
                        zone_time[zone] += 1
                        break
        
        total_positions = sum(zone_time.values())
        if total_positions > 0:
            zone_percentages = {zone: count/total_positions for zone, count in zone_time.items()}
            
            # Analyze positioning patterns
            if zone_percentages['back_court'] > 0.6:
                advice.append("🏃 Positioning: Spending too much time in back court.")
                advice.append("Work on moving forward after good length shots.")
            elif zone_percentages['front_court'] > 0.5:
                advice.append("⚡ Positioning: Aggressive front court positioning.")
                advice.append("Ensure you can recover for attacking shots.")
            
            if zone_percentages['mid_court'] > 0.4:
                advice.append("🎯 Positioning: Good central court control.")
                advice.append("Maintain T-position dominance.")
        
        return advice

    def _analyze_shot_distribution(self, shot_sequence):
        """Analyze distribution of shot types for tactical assessment"""
        shot_counts = {}
        for shot in shot_sequence:
            # Handle different shot formats safely
            if isinstance(shot, str):
                shot_type = shot
            elif isinstance(shot, list) and len(shot) > 0:
                shot_type = shot[0] if isinstance(shot[0], str) else str(shot[0])
            elif shot is None:
                shot_type = 'unknown'
            else:
                shot_type = str(shot)  # Convert to string as fallback
            shot_counts[shot_type] = shot_counts.get(shot_type, 0) + 1
        
        total_shots = len(shot_sequence)
        shot_percentages = {shot: count/total_shots for shot, count in shot_counts.items()}
        
        return {
            'counts': shot_counts,
            'percentages': shot_percentages,
            'variety_score': len(shot_counts),
            'most_used': max(shot_counts.items(), key=lambda x: x[1]) if shot_counts else ('none', 0)
        }

class AdvancedPerformanceAnalyzer:
    """Advanced performance analysis with machine learning-like insights"""
    
    def __init__(self):
        self.performance_history = []
        self.improvement_trends = {}
        self.strength_areas = []
        self.development_areas = []

    def analyze_session_performance(self, coaching_data):
        """Analyze current session performance with trend analysis"""
        session_metrics = self._calculate_session_metrics(coaching_data)
        
        # Add to performance history
        self.performance_history.append({
            'timestamp': time.time(),
            'metrics': session_metrics
        })
        
        # Analyze trends if we have historical data
        trends = self._analyze_trends() if len(self.performance_history) > 1 else {}
        
        # Identify strengths and weaknesses
        strengths, weaknesses = self._identify_performance_areas(session_metrics)
        
        return {
            'current_metrics': session_metrics,
            'trends': trends,
            'strengths': strengths,
            'weaknesses': weaknesses,
            'recommendations': self._generate_performance_recommendations(session_metrics, trends)
        }

    def _calculate_session_metrics(self, coaching_data):
        """Calculate comprehensive session metrics"""
        metrics = {}
        
        if not coaching_data:
            return metrics
        
        # Basic metrics
        total_shots = len(coaching_data)
        successful_shots = sum(1 for data in coaching_data 
                             if data.get('ball_position', [0, 0]) != [0, 0])
        
        metrics['shot_success_rate'] = successful_shots / total_shots if total_shots > 0 else 0
        metrics['total_shots'] = total_shots
        
        # Advanced metrics
        shot_types = []
        for data in coaching_data:
            shot_type = data.get('shot_type', 'unknown')
            # Handle different shot_type formats safely
            if isinstance(shot_type, str):
                shot_types.append(shot_type)
            elif isinstance(shot_type, list) and len(shot_type) > 0:
                shot_types.append(shot_type[0] if isinstance(shot_type[0], str) else str(shot_type[0]))
            else:
                shot_types.append('unknown')
        metrics['shot_variety'] = len(set(shot_types))
        
        # Speed and power metrics
        ball_speeds = [data.get('ball_speed', 0) for data in coaching_data if data.get('ball_speed', 0) > 0]
        if ball_speeds:
            metrics['avg_ball_speed'] = sum(ball_speeds) / len(ball_speeds)
            metrics['max_ball_speed'] = max(ball_speeds)
            metrics['speed_consistency'] = 1.0 - (np.std(ball_speeds) / np.mean(ball_speeds)) if np.mean(ball_speeds) > 0 else 0
        
        # Consistency metrics
        timestamps = [data.get('timestamp', 0) for data in coaching_data]
        if len(timestamps) > 1:
            intervals = [timestamps[i] - timestamps[i-1] for i in range(1, len(timestamps))]
            metrics['rhythm_consistency'] = 1.0 - (np.std(intervals) / np.mean(intervals)) if np.mean(intervals) > 0 else 0
        
        return metrics

    def _analyze_trends(self):
        """Analyze performance trends over time"""
        if len(self.performance_history) < 2:
            return {}
        
        trends = {}
        recent_sessions = self.performance_history[-5:]  # Last 5 sessions
        
        for metric in ['shot_success_rate', 'shot_variety', 'avg_ball_speed']:
            values = [session['metrics'].get(metric, 0) for session in recent_sessions]
            
            if len(values) >= 2:
                # Simple trend calculation
                trend_slope = (values[-1] - values[0]) / len(values)
                trends[metric] = {
                    'direction': 'improving' if trend_slope > 0 else 'declining' if trend_slope < 0 else 'stable',
                    'change_rate': trend_slope,
                    'current_value': values[-1],
                    'previous_value': values[-2] if len(values) >= 2 else values[-1]
                }
        
        return trends

    def _identify_performance_areas(self, metrics):
        """Identify strengths and areas for development"""
        strengths = []
        weaknesses = []
        
        # Benchmarks for good performance
        benchmarks = {
            'shot_success_rate': 0.7,
            'shot_variety': 5,
            'avg_ball_speed': 15,
            'speed_consistency': 0.6,
            'rhythm_consistency': 0.5
        }
        
        for metric, value in metrics.items():
            if metric in benchmarks:
                benchmark = benchmarks[metric]
                
                if value >= benchmark * 1.1:  # 10% above benchmark
                    strengths.append({
                        'area': metric.replace('_', ' ').title(),
                        'value': value,
                        'benchmark': benchmark,
                        'performance': 'excellent'
                    })
                elif value < benchmark * 0.8:  # 20% below benchmark
                    weaknesses.append({
                        'area': metric.replace('_', ' ').title(),
                        'value': value,
                        'benchmark': benchmark,
                        'gap': benchmark - value
                    })
        
        return strengths, weaknesses

    def _generate_performance_recommendations(self, metrics, trends):
        """Generate specific performance recommendations"""
        recommendations = []
        
        # Accuracy recommendations
        if metrics.get('shot_success_rate', 0) < 0.6:
            recommendations.append({
                'priority': 'high',
                'area': 'Accuracy',
                'recommendation': 'Focus on shot accuracy with wall practice and target drills',
                'specific_drill': '15 minutes daily of straight drive targeting specific wall spots',
                'expected_improvement': '10-15% accuracy increase in 2 weeks'
            })
        
        # Variety recommendations
        if metrics.get('shot_variety', 0) < 4:
            recommendations.append({
                'priority': 'medium',
                'area': 'Tactical Variety',
                'recommendation': 'Expand shot repertoire with new shot types',
                'specific_drill': 'Practice 2 new shot types each training session',
                'expected_improvement': 'Increased unpredictability and tactical options'
            })
        
        # Consistency recommendations
        if metrics.get('speed_consistency', 0) < 0.5:
            recommendations.append({
                'priority': 'medium',
                'area': 'Consistency',
                'recommendation': 'Work on shot consistency and rhythm',
                'specific_drill': 'Metronome-based feeding drills for timing',
                'expected_improvement': 'More reliable shot execution under pressure'
            })
        
        # Trend-based recommendations
        for metric, trend_data in trends.items():
            if trend_data['direction'] == 'declining':
                recommendations.append({
                    'priority': 'high',
                    'area': f'Declining {metric.replace("_", " ").title()}',
                    'recommendation': f'Address declining trend in {metric}',
                    'specific_drill': 'Focused practice sessions targeting this specific area',
                    'expected_improvement': 'Reverse declining trend within 1-2 weeks'
                })
        
        return recommendations


class PerformanceTracker:
    """Track and analyze performance improvements over time"""
    
    def __init__(self):
        self.performance_metrics = {
            'shot_accuracy': [],
            'movement_efficiency': [],
            'tactical_awareness': [],
            'endurance_rating': [],
            'court_coverage': []
        }
        
        self.improvement_areas = {
            'technical': ['shot_accuracy', 'shot_variety', 'consistency'],
            'tactical': ['shot_selection', 'court_positioning', 'pattern_recognition'],
            'physical': ['movement_speed', 'endurance', 'court_coverage'],
            'mental': ['concentration', 'pressure_handling', 'match_awareness']
        }

    def update_performance(self, session_data):
        """Update performance metrics with new session data"""
        metrics = self._calculate_session_metrics(session_data)
        
        for key, value in metrics.items():
            if key in self.performance_metrics:
                self.performance_metrics[key].append(value)
        
        return self._generate_performance_report(metrics)
    
    def _calculate_session_metrics(self, session_data):
        """Calculate performance metrics from session data"""
        if not session_data:
            return {}
        
        # Calculate shot accuracy
        total_shots = len(session_data)
        successful_shots = sum(1 for shot in session_data 
                             if shot.get('ball_position', (0, 0)) != (0, 0))
        shot_accuracy = successful_shots / total_shots if total_shots > 0 else 0
        
        # Calculate movement efficiency
        movements = [shot.get('movement_intensity', 0) for shot in session_data]
        movement_efficiency = sum(movements) / len(movements) if movements else 0
        
        # Calculate tactical awareness (shot variety)
        shot_types = set(shot.get('shot_type', 'unknown') for shot in session_data)
        tactical_awareness = len(shot_types) / 10.0  # Normalize to 0-1 scale
        
        return {
            'shot_accuracy': shot_accuracy,
            'movement_efficiency': min(movement_efficiency, 1.0),
            'tactical_awareness': min(tactical_awareness, 1.0),
            'endurance_rating': self._calculate_endurance(session_data),
            'court_coverage': self._calculate_court_coverage(session_data)
        }
    
    def _calculate_endurance(self, session_data):
        """Calculate endurance rating based on performance consistency"""
        if len(session_data) < 10:
            return 0.5
        
        # Analyze performance in first vs last quarter
        quarter_size = len(session_data) // 4
        first_quarter = session_data[:quarter_size]
        last_quarter = session_data[-quarter_size:]
        
        first_performance = self._get_quarter_performance(first_quarter)
        last_performance = self._get_quarter_performance(last_quarter)
        
        # Endurance is maintaining performance (1.0 = no decline, 0.0 = significant decline)
        endurance = last_performance / first_performance if first_performance > 0 else 0.5
        return min(endurance, 1.0)
    
    def _get_quarter_performance(self, quarter_data):
        """Get performance score for a quarter of the session"""
        if not quarter_data:
            return 0.5
        
        successful = sum(1 for shot in quarter_data 
                        if shot.get('ball_position', (0, 0)) != (0, 0))
        return successful / len(quarter_data)
    
    def _calculate_court_coverage(self, session_data):
        """Calculate court coverage efficiency"""
        positions = []
        for shot in session_data:
            pos = shot.get('player1_position', {})
            if pos and 'left_ankle' in pos and 'right_ankle' in pos:
                left = pos['left_ankle']
                right = pos['right_ankle']
                if len(left) >= 2 and len(right) >= 2:
                    center_x = (left[0] + right[0]) / 2
                    center_y = (left[1] + right[1]) / 2
                    positions.append([center_x, center_y])
        
        if len(positions) < 5:
            return 0.5
        
        # Calculate coverage as variance in positions (higher variance = better coverage)
        x_positions = [p[0] for p in positions]
        y_positions = [p[1] for p in positions]
        
        x_range = max(x_positions) - min(x_positions) if x_positions else 0
        y_range = max(y_positions) - min(y_positions) if y_positions else 0
        
        # Normalize coverage score (assumes full court is 1.0 x 1.0)
        coverage = min((x_range + y_range) / 2.0, 1.0)
        return coverage
    
    def _generate_performance_report(self, current_metrics):
        """Generate performance improvement report"""
        report = {
            'current_session': current_metrics,
            'improvements': {},
            'recommendations': []
        }
        
        # Calculate improvements if we have historical data
        for metric, value in current_metrics.items():
            if metric in self.performance_metrics and len(self.performance_metrics[metric]) > 1:
                previous = self.performance_metrics[metric][-2]
                improvement = value - previous
                report['improvements'][metric] = {
                    'change': improvement,
                    'percentage': (improvement / previous * 100) if previous > 0 else 0
                }
        
        # Generate recommendations
        report['recommendations'] = self._generate_recommendations(current_metrics)
        
        return report
    
    def _generate_recommendations(self, metrics):
        """Generate specific recommendations based on performance metrics"""
        recommendations = []
        
        if metrics.get('shot_accuracy', 0) < 0.7:
            recommendations.append({
                'area': 'Technical',
                'issue': 'Shot accuracy below target',
                'suggestion': 'Focus on consistency drills and target practice',
                'priority': 'High'
            })
        
        if metrics.get('movement_efficiency', 0) < 0.6:
            recommendations.append({
                'area': 'Physical',
                'issue': 'Movement efficiency needs improvement',
                'suggestion': 'Work on court movement drills and fitness',
                'priority': 'Medium'
            })
        
        if metrics.get('tactical_awareness', 0) < 0.5:
            recommendations.append({
                'area': 'Tactical',
                'issue': 'Limited shot variety',
                'suggestion': 'Practice different shot types and combinations',
                'priority': 'High'
            })
        
        if metrics.get('court_coverage', 0) < 0.6:
            recommendations.append({
                'area': 'Positioning',
                'issue': 'Poor court coverage',
                'suggestion': 'Focus on returning to T-position and court awareness',
                'priority': 'Medium'
            })
        
        return recommendations

class AutonomousSquashCoach:
    def __init__(self):
        print("🎾 Loading Advanced Squash Coaching AI with Multi-Model Architecture...")
        
        # GPU optimization setup
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🖥️  Coach compute device: {self.device}")
        
        # Advanced squash-specific analysis parameters
        self.analysis_config = {
            'min_rally_length': 3,
            'shot_variety_threshold': 5,
            'movement_intensity_threshold': 15,
            'confidence_threshold': 0.6,
            'bounce_analysis_enabled': True,
            'tactical_depth_analysis': True,
            'court_positioning_weight': 0.3,
            'shot_selection_weight': 0.25,
            'movement_efficiency_weight': 0.25,
            'tactical_awareness_weight': 0.2,
            'fitness_endurance_threshold': 0.7,
            'shot_accuracy_threshold': 0.75,
            'professional_benchmarks': {
                'avg_shot_speed': 45.0,  # m/s professional average
                'avg_rally_length': 12,   # shots per rally
                'court_coverage_efficiency': 0.85,
                'shot_placement_accuracy': 0.80
            }
        }
        
        # Squash-specific coaching knowledge base
        self.squash_knowledge = {
            'shot_techniques': {
                'straight_drive': {
                    'key_points': ['tight to wall', 'good length', 'low trajectory'],
                    'common_errors': ['too wide', 'too short', 'high trajectory'],
                    'improvement_drills': ['wall practice', 'solo hitting', 'target practice']
                },
                'crosscourt': {
                    'key_points': ['wide angle', 'dying length', 'shoulder height'],
                    'common_errors': ['too straight', 'too high', 'interceptable'],
                    'improvement_drills': ['corner targets', 'angle practice', 'deception work']
                },
                'drop_shot': {
                    'key_points': ['soft touch', 'close to front wall', 'dying ball'],
                    'common_errors': ['too high', 'too hard', 'predictable'],
                    'improvement_drills': ['touch practice', 'short game', 'deception drills']
                },
                'lob': {
                    'key_points': ['high trajectory', 'back corner target', 'defensive recovery'],
                    'common_errors': ['too low', 'too short', 'poor placement'],
                    'improvement_drills': ['height practice', 'back corner targets', 'pressure situations']
                },
                'boast': {
                    'key_points': ['side wall first', 'tactical timing', 'element of surprise'],
                    'common_errors': ['too high', 'poor timing', 'overuse'],
                    'improvement_drills': ['side wall practice', 'tactical scenarios', 'combination shots']
                }
            },
            'tactical_patterns': {
                'baseline_rally': 'Maintain pressure with deep shots, look for openings',
                'attacking_sequence': 'Create space, then attack with precision shots',
                'defensive_recovery': 'Use height and time to regain court position',
                'service_tactics': 'Vary pace, placement, and spin to gain advantage'
            },
            'court_positioning': {
                'T_position': 'Central court control for maximum coverage',
                'front_court': 'Quick reactions and soft touch required',
                'back_court': 'Power and accuracy for deep shots',
                'side_walls': 'Angle awareness and tactical shot selection'
            }
        }
        
        # Initialize AI models with fallback options
        self.models = {}
        self.model_priorities = [
            "microsoft/DialoGPT-large",  # Conversation and analysis
            "facebook/blenderbot-400M-distill",  # Coaching dialogue
            "Qwen/Qwen2.5-3B-Instruct"  # Primary analysis model
        ]
        
        self.load_coaching_models()
        
        # Advanced pattern recognition and analysis systems
        self.pattern_analyzer = SquashPatternAnalyzer()
        self.tactical_analyzer = TacticalAnalysisEngine()
        self.performance_analyzer = AdvancedPerformanceAnalyzer()
        
        # Performance tracking and improvement suggestions
        self.performance_tracker = PerformanceTracker()
        
        print(f"🧠 Advanced Squash Coaching AI initialized with {len(self.models)} models loaded")
        print("🎯 Tactical analysis engine: READY")
        print("📊 Performance analysis engine: READY")
        print("🔄 Pattern recognition system: READY")

    def load_coaching_models(self):
        """Load multiple AI models for comprehensive coaching analysis"""
        for model_name in self.model_priorities:
            try:
                print(f"📥 Loading {model_name}...")
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                
                # Handle tokenizer padding token
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    device_map="auto" if torch.cuda.is_available() else "cpu",
                    trust_remote_code=True
                )
                
                self.models[model_name] = {
                    'model': model,
                    'tokenizer': tokenizer,
                    'loaded': True
                }
                
                print(f"✅ {model_name} loaded successfully")
                break  # Use the first successfully loaded model
                
            except Exception as e:
                print(f"❌ Failed to load {model_name}: {e}")
                continue
        
        # Fallback to rule-based system if no models load
        if not self.models:
            print("⚠️  No AI models loaded - using advanced rule-based coaching system")
            self.models['rule_based'] = {'model': None, 'tokenizer': None, 'loaded': False}

    def _get_shot_specific_advice(self, shot_type):
        """Get specific coaching advice for different shot types"""
        advice_map = {
            'straight_drive': "consistent contact point, follow through straight to target",
            'crosscourt': "shoulder turn, hitting across body, wide angle to opposite corner",
            'drop_shot': "soft racket grip, shortened backswing, gentle touch with wrist",
            'lob': "high racket head position, upward swing path, targeting back corners",
            'boast': "side wall contact, deceptive preparation, timing and angle control",
            'volley': "quick preparation, firm wrist, intercepting ball early",
            'kill_shot': "aggressive swing, low contact point, aiming for front wall nick"
        }
        return advice_map.get(shot_type.lower(), "fundamental technique and court positioning")

    def _get_shot_specific_drill(self, shot_type):
        """Get specific drill recommendations for different shot types"""
        drill_map = {
            'straight_drive': "Solo hitting against front wall for 15-20 minutes daily",
            'crosscourt': "Cross-court targets with cones in opposite corners",
            'drop_shot': "Short game practice with partner, focus on touch",
            'lob': "Height and length practice targeting back court corners",
            'boast': "Three-wall boast practice with emphasis on timing",
            'volley': "Wall volley drills to improve hand-eye coordination",
            'kill_shot': "Low target practice with emphasis on accuracy over power"
        }
        return drill_map.get(shot_type.lower(), "General court movement and positioning drills")

    def _calculate_movement_efficiency(self, shot_data):
        """Calculate player movement efficiency based on position data"""
        if not shot_data:
            return 0.5
        
        movement_scores = []
        t_position_returns = 0
        total_movement_opportunities = 0
        
        for shot in shot_data:
            player_pos = shot.get('player1_position', {})
            if player_pos and 'left_ankle' in player_pos and 'right_ankle' in player_pos:
                left = player_pos['left_ankle']
                right = player_pos['right_ankle']
                
                if len(left) >= 2 and len(right) >= 2:
                    center_x = (left[0] + right[0]) / 2
                    center_y = (left[1] + right[1]) / 2
                    
                    # Check if player is near T-position (court center)
                    t_distance = math.sqrt((center_x - 0.5)**2 + (center_y - 0.5)**2)
                    if t_distance < 0.2:  # Within T-position range
                        t_position_returns += 1
                    
                    total_movement_opportunities += 1
        
        if total_movement_opportunities > 0:
            t_recovery_ratio = t_position_returns / total_movement_opportunities
            return min(t_recovery_ratio * 1.2, 1.0)  # Boost slightly for good positioning
        
        return 0.5

    def _calculate_court_coverage(self, shot_data):
        """Calculate court coverage efficiency"""
        positions = []
        
        for shot in shot_data:
            player_pos = shot.get('player1_position', {})
            if player_pos and 'left_ankle' in player_pos and 'right_ankle' in player_pos:
                left = player_pos['left_ankle']
                right = player_pos['right_ankle']
                
                if len(left) >= 2 and len(right) >= 2:
                    center_x = (left[0] + right[0]) / 2
                    center_y = (left[1] + right[1]) / 2
                    positions.append([center_x, center_y])
        
        if len(positions) < 5:
            return 0.5
        
        # Calculate coverage as variance in positions (higher variance = better coverage)
        x_positions = [p[0] for p in positions]
        y_positions = [p[1] for p in positions]
        
        x_range = max(x_positions) - min(x_positions) if x_positions else 0
        y_range = max(y_positions) - min(y_positions) if y_positions else 0
        
        # Normalize coverage score (assumes full court is 1.0 x 1.0)
        coverage = min((x_range + y_range) / 2.0, 1.0)
        return coverage

    def _analyze_strategy(self, shot_data):
        """Analyze strategic patterns and provide tactical insights"""
        strategic_insights = []
        
        if not shot_data:
            return strategic_insights
        
        # Analyze shot variety and tactical awareness
        shot_types = []
        for shot in shot_data:
            shot_type = shot.get('shot_type', 'unknown')
            # Handle different shot_type formats safely
            if isinstance(shot_type, str):
                shot_types.append(shot_type)
            elif isinstance(shot_type, list) and len(shot_type) > 0:
                shot_types.append(shot_type[0] if isinstance(shot_type[0], str) else str(shot_type[0]))
            else:
                shot_types.append('unknown')
        unique_shots = set(shot_types)
        
        if len(unique_shots) < 3:
            strategic_insights.append({
                'type': 'tactical',
                'priority': 'high',
                'message': f"Limited shot variety detected ({len(unique_shots)} types). Expand tactical repertoire.",
                'drill_suggestion': "Practice different shot combinations in structured routines",
                'metrics': {'shot_variety': len(unique_shots)}
            })
        
        # Analyze shot sequences for predictability
        if len(shot_types) > 5:
            sequences = []
            for i in range(len(shot_types) - 2):
                sequences.append((shot_types[i], shot_types[i+1], shot_types[i+2]))
            
            sequence_counts = {}
            for seq in sequences:
                sequence_counts[seq] = sequence_counts.get(seq, 0) + 1
            
            if sequence_counts:
                most_common_sequence = max(sequence_counts.values())
                predictability = most_common_sequence / len(sequences)
                
                if predictability > 0.3:
                    strategic_insights.append({
                        'type': 'tactical',
                        'priority': 'medium',
                        'message': f"Shot patterns may be predictable ({predictability:.1%}). Vary shot sequences.",
                        'drill_suggestion': "Random shot selection drills and pattern breaking exercises",
                        'metrics': {'predictability': predictability}
                    })
        
        # Analyze pressure situations
        pressure_shots = [shot for shot in shot_data if shot.get('ball_speed', 0) > 15]
        if len(pressure_shots) > 0:
            pressure_accuracy = sum(1 for shot in pressure_shots 
                                  if shot.get('ball_position', (0, 0)) != (0, 0)) / len(pressure_shots)
            
            if pressure_accuracy < 0.6:
                strategic_insights.append({
                    'type': 'mental',
                    'priority': 'high',
                    'message': f"Performance under pressure needs improvement ({pressure_accuracy:.1%} accuracy).",
                    'drill_suggestion': "Pressure situation drills and mental training exercises",
                    'metrics': {'pressure_accuracy': pressure_accuracy, 'pressure_shots': len(pressure_shots)}
                })
        
        return strategic_insights

    def _get_rule_based_insights(self, shot_data):
        """Generate comprehensive rule-based coaching insights"""
        insights = []
        
        if not shot_data:
            return insights
        
        # Advanced shot accuracy analysis
        successful_shots = sum(1 for shot in shot_data if shot.get('success', False))
        accuracy = successful_shots / len(shot_data)
        
        # Shot type specific analysis
        shot_types = {}
        for shot in shot_data:
            shot_type = shot.get('shot_type', 'unknown')
            if shot_type not in shot_types:
                shot_types[shot_type] = {'total': 0, 'successful': 0}
            shot_types[shot_type]['total'] += 1
            if shot.get('success', False):
                shot_types[shot_type]['successful'] += 1
        
        # Analyze each shot type
        for shot_type, stats in shot_types.items():
            type_accuracy = stats['successful'] / stats['total'] if stats['total'] > 0 else 0
            
            if type_accuracy < 0.5 and stats['total'] >= 3:
                insights.append({
                    'type': 'technique',
                    'priority': 'high',
                    'shot_type': shot_type,
                    'message': f"{shot_type.title()} accuracy is {type_accuracy:.1%}. Focus on {self._get_shot_specific_advice(shot_type)}",
                    'drill_suggestion': self._get_shot_specific_drill(shot_type),
                    'metrics': {'accuracy': type_accuracy, 'attempts': stats['total']}
                })
        
        # Overall accuracy insights
        if accuracy < 0.6:
            insights.append({
                'type': 'technique',
                'priority': 'high',
                'message': f"Overall shot accuracy is {accuracy:.1%}. Focus on consistent contact point and follow-through.",
                'drill_suggestion': "Practice straight drive repetitions against the front wall",
                'metrics': {'accuracy': accuracy, 'total_shots': len(shot_data)}
            })
        
        # Movement efficiency analysis
        movement_efficiency = self._calculate_movement_efficiency(shot_data)
        court_coverage = self._calculate_court_coverage(shot_data)
        
        if movement_efficiency < 0.7:
            insights.append({
                'type': 'movement',
                'priority': 'medium',
                'message': f"Movement efficiency is {movement_efficiency:.1%}. Work on returning to the T-position after each shot.",
                'drill_suggestion': "T-position recovery drills with ghosting exercises",
                'metrics': {'efficiency': movement_efficiency, 'court_coverage': court_coverage}
            })
        
        # Shot timing analysis
        shot_timings = [shot.get('timing', 0) for shot in shot_data if shot.get('timing')]
        if shot_timings:
            avg_timing = sum(shot_timings) / len(shot_timings)
            timing_variance = np.var(shot_timings) if len(shot_timings) > 1 else 0
            
            if timing_variance > 0.5:
                insights.append({
                    'type': 'timing',
                    'priority': 'medium',
                    'message': f"Inconsistent shot timing detected (variance: {timing_variance:.2f}). Focus on rhythm and preparation.",
                    'drill_suggestion': "Metronome drills and feeding exercises",
                    'metrics': {'avg_timing': avg_timing, 'variance': timing_variance}
                })
        
        # Power and placement analysis
        power_levels = [shot.get('power', 0) for shot in shot_data if shot.get('power')]
        if power_levels:
            avg_power = sum(power_levels) / len(power_levels)
            if avg_power < 0.4:
                insights.append({
                    'type': 'power',
                    'priority': 'low',
                    'message': f"Average shot power is {avg_power:.1%}. Consider incorporating more aggressive shots.",
                    'drill_suggestion': "Power drive drills and length practice",
                    'metrics': {'avg_power': avg_power}
                })
        
        # Strategic analysis
        strategic_insights = self._analyze_strategy(shot_data)
        insights.extend(strategic_insights)
        
        return insights
    
    def analyze_bounce_patterns(self, coaching_data):
        """Enhanced analysis of ball bounce patterns for coaching insights"""
        bounce_analysis = {
            'total_bounces': 0,
            'avg_bounces_per_rally': 0,
            'bounce_locations': [],
            'bounce_frequency': 'low'
        }
        
        try:
            total_bounces = 0
            rallies_with_bounces = 0
            bounce_positions = []
            
            for data_point in coaching_data:
                if isinstance(data_point, dict):                    # Collect wall bounce data
                    wall_bounces = data_point.get('wall_bounce_count', 0)
                    # Ensure wall_bounces is an integer (in case it was incorrectly set as a tuple)
                    if isinstance(wall_bounces, (tuple, list)):
                        wall_bounces = wall_bounces[0] if wall_bounces else 0
                    wall_bounces = int(wall_bounces) if wall_bounces is not None else 0
                    total_bounces += wall_bounces
                    
                    if wall_bounces > 0:
                        rallies_with_bounces += 1
                    
                    # Collect GPU-detected bounces if available
                    gpu_bounces = data_point.get('gpu_detected_bounces', [])
                    if gpu_bounces:
                        bounce_positions.extend(gpu_bounces)
            
            # Calculate statistics
            total_rallies = len(coaching_data)
            bounce_analysis['total_bounces'] = total_bounces
            
            if total_rallies > 0:
                bounce_analysis['avg_bounces_per_rally'] = total_bounces / total_rallies
                bounce_percentage = (rallies_with_bounces / total_rallies) * 100
                
                if bounce_percentage > 60:
                    bounce_analysis['bounce_frequency'] = 'high'
                elif bounce_percentage > 30:
                    bounce_analysis['bounce_frequency'] = 'medium'
                else:
                    bounce_analysis['bounce_frequency'] = 'low'
            
            bounce_analysis['bounce_locations'] = bounce_positions
            
            return bounce_analysis
            
        except Exception as e:
            print(f"Error in bounce pattern analysis: {e}")
            return bounce_analysis

    def generate_enhanced_coaching_insights(self, analysis, bounce_patterns):
        """Generate enhanced coaching insights including bounce analysis"""
        insights = []
        
        # Ball bounce coaching insights
        if bounce_patterns['total_bounces'] > 0:
            insights.append(" **Ball Bounce Analysis:**")
            insights.append(f"   • Total bounces detected: {bounce_patterns['total_bounces']}")
            insights.append(f"   • Average bounces per rally: {bounce_patterns['avg_bounces_per_rally']:.1f}")
            insights.append(f"   • Bounce frequency: {bounce_patterns['bounce_frequency']}")
            
            if bounce_patterns['bounce_frequency'] == 'high':
                insights.append("   •  Consider working on shot placement to reduce unnecessary wall bounces")
                insights.append("   •  Focus on straight drives and court positioning")
            elif bounce_patterns['bounce_frequency'] == 'low':
                insights.append("   • Good shot control with minimal wall bounces")
                insights.append("   • Consider adding tactical boasts when appropriate")
        
        return "\n".join(insights)

    def analyze_match_data(self, coaching_data):
        """Enhanced match data analysis with advanced AI coaching insights"""
        if not coaching_data:
            return "No match data available for analysis"
        
        # Generate comprehensive analysis
        analysis = self.prepare_match_analysis(coaching_data)
        
        # Enhanced bounce pattern analysis
        bounce_patterns = self.analyze_bounce_patterns(coaching_data)
        
        # Get the active model for analysis
        active_model = None
        active_tokenizer = None
        
        for model_name, model_info in self.models.items():
            if model_info.get('loaded', False) and model_info.get('model') is not None:
                active_model = model_info['model']
                active_tokenizer = model_info['tokenizer']
                print(f"🤖 Using {model_name} for advanced coaching analysis")
                break
        
        if not active_model:
            return self.fallback_analysis_with_bounces(analysis, bounce_patterns)

        # Advanced coaching prompt with professional-level analysis
        coaching_prompt = f"""You are an elite squash coach with 20+ years of experience analyzing professional-level matches. 
Provide comprehensive coaching analysis for this squash match data.

MATCH STATISTICS:
{analysis}

BALL BOUNCE PATTERNS:
• Total bounces detected: {bounce_patterns['total_bounces']}
• Average bounces per rally: {bounce_patterns['avg_bounces_per_rally']:.1f}
• Bounce frequency: {bounce_patterns['bounce_frequency']}

PROFESSIONAL BENCHMARKS FOR COMPARISON:
• Professional avg shot speed: 45 m/s
• Professional avg rally length: 12 shots
• Professional court coverage: 85%
• Professional shot accuracy: 80%

Provide expert-level coaching analysis covering:

1. TECHNICAL ANALYSIS:
   - Shot accuracy assessment with specific improvement areas
   - Ball striking technique evaluation
   - Movement patterns and court positioning efficiency
   - Tactical shot selection analysis

2. STRATEGIC INSIGHTS:
   - Pattern recognition in shot sequences
   - Predictability assessment of playing style
   - Court positioning strategy evaluation
   - Pressure situation performance

3. PHYSICAL ASSESSMENT:
   - Movement efficiency and court coverage
   - Endurance indicators throughout the match
   - Speed and agility observations
   - Recovery between points

4. SPECIFIC IMPROVEMENT RECOMMENDATIONS:
   - Priority areas for immediate focus (ranked 1-3)
   - Detailed drill suggestions with durations
   - Training regimen recommendations
   - Mental game and tactical development areas

5. MATCH-SPECIFIC OBSERVATIONS:
   - Key moments of strength and weakness
   - Adaptation strategies during play
   - Consistency patterns over time
   - Areas of competitive advantage

Provide actionable, specific coaching advice that a player could implement immediately."""

        # Replace generation block with GPU-to-CPU fallback
        try:
            inputs = active_tokenizer(coaching_prompt, return_tensors="pt", truncation=True, max_length=3000)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Attempt generation on the designated device
            try:
                with torch.no_grad():
                    outputs = active_model.generate(
                        **inputs,
                        max_new_tokens=800,
                        temperature=0.8,
                        do_sample=True,
                        top_p=0.9,
                        repetition_penalty=1.1,
                        pad_token_id=active_tokenizer.eos_token_id
                    )
            except Exception as gen_e:
                print(f"AI coaching generation error: {gen_e}")
                if self.device.type == 'cuda':
                    print("Falling back to CPU for model generation.")
                    active_model.to('cpu')
                    inputs = {k: v.cpu() for k, v in inputs.items()}
                    with torch.no_grad():
                        outputs = active_model.generate(
                            **inputs,
                            max_new_tokens=800,
                            temperature=0.8,
                            do_sample=True,
                            top_p=0.9,
                            repetition_penalty=1.1,
                            pad_token_id=active_tokenizer.eos_token_id
                        )
                else:
                    raise

            response = active_tokenizer.decode(outputs[0], skip_special_tokens=True)
            generated_text = response[len(coaching_prompt):].strip()

            # Add enhanced rule-based insights
            rule_based_insights = self._compile_advanced_insights(coaching_data, analysis, bounce_patterns)

            # Combine AI analysis with rule-based insights
            final_analysis = f"""🎾 AUTONOMOUS SQUASH COACHING ANALYSIS
{'='*60}

 AI COACHING INSIGHTS:
{generated_text}

 DETAILED TECHNICAL ANALYSIS:
{rule_based_insights}

 PERFORMANCE BENCHMARKING:
{self._generate_performance_benchmarks(coaching_data)}

 IMMEDIATE ACTION ITEMS:
{self._generate_action_items(coaching_data)}

 PROGRESS TRACKING RECOMMENDATIONS:
{self._generate_progress_tracking(coaching_data)}

Note: This analysis combines advanced AI coaching with comprehensive rule-based insights 
for maximum coaching effectiveness."""
            
            return final_analysis
            
        except Exception as e:
            print(f"AI coaching generation error: {e}")
            return self.fallback_analysis_with_bounces(analysis, bounce_patterns)

    # ...existing code...
