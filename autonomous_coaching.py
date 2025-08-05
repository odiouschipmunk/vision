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
            shot_type = shot if isinstance(shot, str) else shot[0] if shot else 'unknown'
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
        shot_types = [data.get('shot_type', 'unknown') for data in coaching_data]
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
        
        # Calculate coverage as variance in positions
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
        shot_types = [shot.get('shot_type', 'unknown') for shot in shot_data]
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

        try:
            inputs = active_tokenizer(coaching_prompt, return_tensors="pt", truncation=True, max_length=3000)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
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

    def _compile_advanced_insights(self, coaching_data, analysis, bounce_patterns):
        """Compile advanced rule-based insights for comprehensive analysis"""
        insights = []
        
        # Generate rule-based insights
        shot_data = self._extract_shot_data(coaching_data)
        rule_insights = self._get_rule_based_insights(shot_data)
        
        # Format insights by category
        categories = {'technique': [], 'tactical': [], 'movement': [], 'timing': [], 'power': [], 'mental': []}
        
        for insight in rule_insights:
            category = insight.get('type', 'general')
            if category in categories:
                categories[category].append(insight)
        
        # Generate formatted insights
        formatted_insights = []
        
        for category, category_insights in categories.items():
            if category_insights:
                formatted_insights.append(f"\n{category.upper()} INSIGHTS:")
                for insight in category_insights:
                    priority_emoji = "🔴" if insight['priority'] == 'high' else "🟡" if insight['priority'] == 'medium' else "🟢"
                    formatted_insights.append(f"  {priority_emoji} {insight['message']}")
                    formatted_insights.append(f"      Drill: {insight['drill_suggestion']}")
                    if 'metrics' in insight:
                        # Ensure all keys are strings to avoid unhashable type errors
                        safe_metrics = {}
                        for k, v in insight['metrics'].items():
                            if isinstance(k, (list, dict, set)):
                                # Convert unhashable types to strings
                                safe_metrics[str(k)] = v
                            else:
                                safe_metrics[k] = v
                        metrics_str = ", ".join([f"{k}: {v}" for k, v in safe_metrics.items()])
                        formatted_insights.append(f"      Metrics: {metrics_str}")
        
        return "\n".join(formatted_insights)

    def _extract_shot_data(self, coaching_data):
        """Extract shot-specific data for analysis"""
        shot_data = []
        
        for data_point in coaching_data:
            if isinstance(data_point, dict):
                # Ensure ball_position is properly formatted to avoid unhashable type errors
                ball_pos = data_point.get('ball_position', [0, 0])
                if isinstance(ball_pos, list):
                    ball_pos = tuple(ball_pos)  # Convert list to tuple for hashability
                
                shot_info = {
                    'shot_type': data_point.get('shot_type', 'unknown'),
                    'success': ball_pos != (0, 0),
                    'ball_position': ball_pos,
                    'ball_speed': data_point.get('ball_speed', 0),
                    'player1_position': data_point.get('player1_position', {}),
                    'timing': data_point.get('timestamp', 0),
                    'power': data_point.get('ball_speed', 0) / 20.0 if data_point.get('ball_speed', 0) > 0 else 0
                }
                shot_data.append(shot_info)
        
        return shot_data

    def _generate_performance_benchmarks(self, coaching_data):
        """Generate performance benchmarks against professional standards"""
        benchmarks = []
        
        # Calculate current performance metrics
        total_shots = len(coaching_data)
        active_shots = sum(1 for data in coaching_data 
                          if data.get('ball_position', (0, 0)) != (0, 0))
        
        if total_shots > 0:
            accuracy = (active_shots / total_shots) * 100
            professional_accuracy = self.analysis_config['professional_benchmarks']['shot_placement_accuracy'] * 100
            
            accuracy_gap = professional_accuracy - accuracy
            
            benchmarks.append(f"• Shot Accuracy: {accuracy:.1f}% (Professional: {professional_accuracy:.1f}%)")
            
            if accuracy_gap > 10:
                benchmarks.append(f"   Focus Area: Accuracy needs {accuracy_gap:.1f}% improvement")
            elif accuracy_gap < -5:
                benchmarks.append(f"   Strength: Accuracy exceeds professional average!")
        
        # Court coverage analysis
        movement_data = [data for data in coaching_data 
                        if data.get('player1_position', {}).get('left_ankle')]
        
        if len(movement_data) > 10:
            coverage_efficiency = self._calculate_court_coverage(self._extract_shot_data(coaching_data))
            professional_coverage = self.analysis_config['professional_benchmarks']['court_coverage_efficiency']
            
            coverage_gap = professional_coverage - coverage_efficiency
            
            benchmarks.append(f"• Court Coverage: {coverage_efficiency:.1%} (Professional: {professional_coverage:.1%})")
            
            if coverage_gap > 0.1:
                benchmarks.append(f"   Focus Area: Coverage needs {coverage_gap:.1%} improvement")
        
        return "\n".join(benchmarks) if benchmarks else "Insufficient data for benchmarking"

    def _generate_action_items(self, coaching_data):
        """Generate immediate actionable items"""
        action_items = []
        
        # Analyze most recent performance patterns
        recent_data = coaching_data[-50:] if len(coaching_data) > 50 else coaching_data
        shot_data = self._extract_shot_data(recent_data)
        
        if shot_data:
            # Check accuracy in recent shots
            recent_accuracy = sum(1 for shot in shot_data if shot['success']) / len(shot_data)
            
            if recent_accuracy < 0.6:
                action_items.append("1.  IMMEDIATE: Focus on shot accuracy - 15 min wall practice daily")
                action_items.append("   - Target specific spots on front wall")
                action_items.append("   - Emphasize consistent contact point")
            
            # Check shot variety
            shot_types = set(shot.get('shot_type', 'unknown') for shot in shot_data)
            if len(shot_types) < 4:
                action_items.append("2.  TACTICAL: Expand shot repertoire - practice 3 new shots weekly")
                action_items.append("   - Incorporate drop shots and lobs")
                action_items.append("   - Work on crosscourt variations")
            
            # Check movement patterns
            movement_efficiency = self._calculate_movement_efficiency(shot_data)
            if movement_efficiency < 0.6:
                action_items.append("3.  MOVEMENT: Improve court positioning - daily ghosting routine")
                action_items.append("   - 10 minutes T-position recovery drills")
                action_items.append("   - Focus on split-step timing")
        
        if not action_items:
            action_items.append(" Performance is strong - maintain current training intensity")
            action_items.append(" Focus on fine-tuning shot placement and consistency")
        
        return "\n".join(action_items)

    def _generate_progress_tracking(self, coaching_data):
        """Generate progress tracking recommendations"""
        tracking_recommendations = []
        
        tracking_recommendations.append(" KEY METRICS TO TRACK:")
        tracking_recommendations.append("• Shot accuracy percentage (target: >75%)")
        tracking_recommendations.append("• Rally length average (target: >8 shots)")
        tracking_recommendations.append("• Court coverage efficiency (target: >70%)")
        tracking_recommendations.append("• Shot variety per session (target: >5 types)")
        
        tracking_recommendations.append("\n RECOMMENDED TRACKING METHODS:")
        tracking_recommendations.append("• Daily training log with video analysis")
        tracking_recommendations.append("• Weekly performance review sessions")
        tracking_recommendations.append("• Monthly benchmark testing")
        tracking_recommendations.append("• Quarterly tactical assessment")
        
        tracking_recommendations.append("\n IMPROVEMENT MILESTONES:")
        tracking_recommendations.append("• Week 1-2: Focus on accuracy improvement")
        tracking_recommendations.append("• Week 3-4: Expand tactical repertoire")
        tracking_recommendations.append("• Month 2: Integrate advanced movement patterns")
        tracking_recommendations.append("• Month 3: Master pressure situation management")
        
        return "\n".join(tracking_recommendations)

    def fallback_analysis_with_bounces(self, analysis, bounce_patterns):
        """Enhanced fallback analysis including bounce patterns"""
        enhanced_insights = self.generate_enhanced_coaching_insights(analysis, bounce_patterns)
        
        return f"""ENHANCED AUTONOMOUS SQUASH COACHING ANALYSIS:

{analysis}

{enhanced_insights}

COACHING RECOMMENDATIONS:
• Continue tracking ball bounces for better shot placement awareness
• Focus on minimizing unnecessary wall bounces through improved accuracy
• Use tactical bounces (boasts) strategically when appropriate
• Work on court positioning to optimize ball trajectory control

Note: This analysis uses enhanced ball bounce detection with GPU acceleration for improved accuracy."""
        
        return enhanced_analysis

    def fallback_analysis(self, analysis):
        """Provide structured analysis when AI model is unavailable"""
        return f"""AUTONOMOUS COACHING ANALYSIS:
{analysis}

COACHING RECOMMENDATIONS:
• Practice shot variety if limited types detected
• Focus on consistent court positioning  
• Work on rally endurance for match fitness
• Develop both crosscourt and straight shot patterns
• Maintain active movement throughout points
• Practice under match-like conditions

TRAINING PRIORITIES:
• Technical: Work on shot consistency and variety
• Tactical: Improve court positioning and shot selection
• Physical: Enhance court movement and endurance
• Mental: Develop match awareness and decision making

Note: Advanced AI coaching model unavailable - providing structured basic analysis."""
    
    def prepare_match_analysis(self, coaching_data):
        """Enhanced match analysis with advanced AI coaching insights and tactical analysis"""
        shot_counts = defaultdict(int)
        activity_frames = 0
        total_frames = len(coaching_data)
        ball_speeds = []
        player_movement_patterns = {'player1': [], 'player2': []}
        shot_sequences = []
        position_sequences = []
        rally_lengths = []
        current_rally_length = 0
        hit_detection_accuracy = []
        movement_intensities = []
        
        # Enhanced data analysis with temporal patterns
        for i, data_point in enumerate(coaching_data):
            if not isinstance(data_point, dict):
                continue
                
            timestamp = data_point.get('timestamp', time.time())
            frame_num = data_point.get('frame', i)
            
            # Activity and rally analysis
            is_active = data_point.get('match_active', False)
            if is_active:
                activity_frames += 1
                current_rally_length += 1
            else:
                if current_rally_length > 0:
                    rally_lengths.append(current_rally_length)
                    current_rally_length = 0
            
            # Enhanced shot analysis with tactical pattern tracking
            shot_type = data_point.get('shot_type', 'unknown')
            if isinstance(shot_type, list) and len(shot_type) > 0:
                # Extract detailed shot components
                direction = shot_type[0] if len(shot_type) > 0 else 'unknown'
                height = shot_type[1] if len(shot_type) > 1 else 'medium'
                style = shot_type[2] if len(shot_type) > 2 else 'standard'
                
                # Create comprehensive shot descriptor
                shot_descriptor = f"{direction}_{height}_{style}"
                shot_counts[shot_descriptor] += 1
                
                # Track shot sequences for tactical pattern analysis
                if len(shot_sequences) == 0 or shot_sequences[-1] != direction:
                    shot_sequences.append(direction)
            elif isinstance(shot_type, str) and shot_type != 'unknown':
                shot_counts[shot_type] += 1
                if len(shot_sequences) == 0 or shot_sequences[-1] != shot_type:
                    shot_sequences.append(shot_type)
            
            # Ball dynamics analysis
            ball_speed = data_point.get('ball_speed', 0)
            if ball_speed > 0:
                ball_speeds.append(ball_speed)
            
            # Hit detection quality tracking
            hit_confidence = data_point.get('hit_confidence', 0)
            if hit_confidence > 0:
                hit_detection_accuracy.append(hit_confidence)
            
            # Enhanced player movement analysis with tactical positioning
            for player_key, player_id in [('player1_position', 'player1'), ('player2_position', 'player2')]:
                if player_key in data_point:
                    player_pos = data_point[player_key]
                    if isinstance(player_pos, dict) and 'left_ankle' in player_pos and 'right_ankle' in player_pos:
                        left_ankle = player_pos['left_ankle']
                        right_ankle = player_pos['right_ankle']
                        
                        if (isinstance(left_ankle, list) and len(left_ankle) >= 2 and 
                            isinstance(right_ankle, list) and len(right_ankle) >= 2):
                            
                            # Calculate player center position
                            center_x = (left_ankle[0] + right_ankle[0]) / 2
                            center_y = (left_ankle[1] + right_ankle[1]) / 2
                            
                            # Add to position sequences for tactical analysis
                            if player_id == 'player1':
                                position_sequences.append([center_x, center_y])
                            
                            # Calculate movement intensity if previous position exists
                            movement_intensity = 0
                            if len(player_movement_patterns[player_id]) > 0:
                                prev_pos = player_movement_patterns[player_id][-1]['position']
                                movement_intensity = math.sqrt(
                                    (center_x - prev_pos[0])**2 + (center_y - prev_pos[1])**2
                                )
                                movement_intensities.append(movement_intensity)
                            
                            player_movement_patterns[player_id].append({
                                'position': [center_x, center_y],
                                'timestamp': timestamp,
                                'frame': frame_num,
                                'movement_intensity': movement_intensity
                            })
        
        # Finalize rally analysis
        if current_rally_length > 0:
            rally_lengths.append(current_rally_length)
        
        # Calculate comprehensive metrics
        activity_percentage = (activity_frames / total_frames * 100) if total_frames > 0 else 0
        most_common_shot = max(shot_counts.items(), key=lambda x: x[1]) if shot_counts else ("unknown", 0)
        shot_variety = len([shot for shot, count in shot_counts.items() if count > 1])
        avg_ball_speed = sum(ball_speeds) / len(ball_speeds) if ball_speeds else 0
        max_ball_speed = max(ball_speeds) if ball_speeds else 0
        avg_rally_length = sum(rally_lengths) / len(rally_lengths) if rally_lengths else 0
        max_rally_length = max(rally_lengths) if rally_lengths else 0
        avg_hit_confidence = sum(hit_detection_accuracy) / len(hit_detection_accuracy) if hit_detection_accuracy else 0
        avg_movement_intensity = sum(movement_intensities) / len(movement_intensities) if movement_intensities else 0
        
        # Advanced tactical analysis
        tactical_analysis = self.tactical_analyzer.analyze_tactical_patterns(shot_sequences, position_sequences)
        
        # Advanced performance analysis
        performance_analysis = self.performance_analyzer.analyze_session_performance(coaching_data)
        
        # Enhanced pattern analysis
        pattern_analysis = self.pattern_analyzer.analyze_shot_sequences(shot_sequences)
        
        # Traditional analysis components
        movement_analysis = self.analyze_player_movement_patterns(player_movement_patterns)
        shot_pattern_analysis = self.analyze_shot_patterns(shot_sequences, shot_counts)
        performance_metrics = self.analyze_performance_metrics(
            activity_percentage, shot_variety, avg_ball_speed, avg_rally_length, avg_hit_confidence
        )
        
        # Generate comprehensive analysis with all new components
        analysis = f"""🎾 COMPREHENSIVE MATCH ANALYSIS WITH ADVANCED AI INSIGHTS
{'='*70}

📊 BASIC STATISTICS:
• Total data points analyzed: {total_frames}
• Active rally time: {activity_percentage:.1f}% of match
• Number of rallies: {len(rally_lengths)}
• Average rally length: {avg_rally_length:.1f} frames
• Longest rally: {max_rally_length} frames

🎯 SHOT ANALYSIS:
• Most frequent shot: {most_common_shot[0]} ({most_common_shot[1]} occurrences)
• Shot variety score: {shot_variety}/10
• Total unique shot types: {len(shot_counts)}
• Top 3 shots: {dict(list(sorted(shot_counts.items(), key=lambda x: x[1], reverse=True)[:3]))}

⚡ BALL DYNAMICS:
• Average ball speed: {avg_ball_speed:.2f} pixels/frame
• Maximum ball speed: {max_ball_speed:.2f} pixels/frame
• Speed consistency: {'High' if ball_speeds and np.std(ball_speeds) < avg_ball_speed * 0.3 else 'Variable'}
• Ball tracking quality: {avg_hit_confidence:.2f}/1.0

🏃 MOVEMENT PATTERNS:
{movement_analysis}

🎪 TACTICAL PATTERN ANALYSIS:
• Identified pattern: {tactical_analysis['pattern']}
• Pattern confidence: {tactical_analysis['confidence']:.1%}
• Shot variety score: {tactical_analysis['shot_distribution']['variety_score']}/10
• Most used shot: {tactical_analysis['shot_distribution']['most_used'][0]} ({tactical_analysis['shot_distribution']['most_used'][1]} times)

📈 ADVANCED PERFORMANCE METRICS:
• Current session accuracy: {performance_analysis['current_metrics'].get('shot_success_rate', 0):.1%}
• Shot variety: {performance_analysis['current_metrics'].get('shot_variety', 0)} types
• Speed consistency: {performance_analysis['current_metrics'].get('speed_consistency', 0):.1%}

🎯 IDENTIFIED STRENGTHS:
{self._format_performance_areas(performance_analysis.get('strengths', []))}

⚠️ DEVELOPMENT AREAS:
{self._format_performance_areas(performance_analysis.get('weaknesses', []))}

🧠 PATTERN RECOGNITION INSIGHTS:
• Strategic pattern: {pattern_analysis['pattern']}
• Pattern confidence: {pattern_analysis['confidence']:.1%}
• Tactical advice: {pattern_analysis['tactical_advice']}

🔄 SHOT PATTERNS & TACTICS:
{shot_pattern_analysis}

📊 PERFORMANCE ASSESSMENT:
{performance_metrics}

📋 DATA QUALITY INDICATORS:
• Ball detection rate: {len(ball_speeds)/total_frames*100:.1f}%
• Movement tracking quality: {'Good' if len(movement_intensities) > total_frames * 0.5 else 'Limited'}
• Average movement intensity: {avg_movement_intensity:.2f} pixels/frame
• Tactical analysis confidence: {tactical_analysis['confidence']:.1%}"""
        
        return analysis

    def _format_performance_areas(self, areas):
        """Format performance areas (strengths/weaknesses) for display"""
        if not areas:
            return "• No significant areas identified"
        
        formatted = []
        for area in areas:
            if 'performance' in area:  # Strength
                formatted.append(f"• {area['area']}: {area['value']:.2f} (Excellent - {area['value']/area['benchmark']:.1%} of benchmark)")
            else:  # Weakness
                formatted.append(f"• {area['area']}: {area['value']:.2f} (Gap: {area['gap']:.2f} from benchmark)")
        
        return "\n".join(formatted)
    
    def analyze_player_movement_patterns(self, movement_patterns):
        """Detailed analysis of player movement patterns"""
        analysis_parts = []
        
        for player_id, movements in movement_patterns.items():
            if len(movements) < 10:
                analysis_parts.append(f"• {player_id.upper()}: Insufficient movement data")
                continue
            
            # Extract movement data
            positions = [m['position'] for m in movements]
            intensities = [m.get('movement_intensity', 0) for m in movements if m.get('movement_intensity', 0) > 0]
            
            # Court coverage analysis
            x_positions = [p[0] for p in positions]
            y_positions = [p[1] for p in positions]
            x_range = (max(x_positions) - min(x_positions)) if x_positions else 0
            y_range = (max(y_positions) - min(y_positions)) if y_positions else 0
            
            # Movement statistics
            avg_intensity = sum(intensities) / len(intensities) if intensities else 0
            max_intensity = max(intensities) if intensities else 0
            
            # Court positioning analysis
            court_center_x = 0.5  # Normalized court center
            court_center_y = 0.5
            
            distances_from_center = []
            for pos in positions:
                distance = math.sqrt((pos[0] - court_center_x)**2 + (pos[1] - court_center_y)**2)
                distances_from_center.append(distance)
            
            avg_distance_from_center = sum(distances_from_center) / len(distances_from_center)
            
            # Generate insights
            coverage_rating = "Excellent" if x_range > 0.4 and y_range > 0.4 else "Good" if x_range > 0.2 and y_range > 0.2 else "Limited"
            mobility_rating = "High" if avg_intensity > 10 else "Medium" if avg_intensity > 5 else "Low"
            positioning = "Central" if avg_distance_from_center < 0.3 else "Peripheral"
            
            analysis_parts.append(
                f"• {player_id.upper()}: Court coverage - {coverage_rating}, "
                f"Mobility - {mobility_rating}, Positioning - {positioning}"
            )
        
        return "\n".join(analysis_parts) if analysis_parts else "• No sufficient movement data for analysis"
    
    def analyze_shot_patterns(self, shot_sequences, shot_counts):
        """Advanced shot pattern and tactical analysis"""
        if len(shot_sequences) < 5:
            return "• Insufficient shot data for meaningful pattern analysis"
        
        # Shot transition analysis
        transitions = defaultdict(int)
        for i in range(len(shot_sequences) - 1):
            current = shot_sequences[i]
            next_shot = shot_sequences[i + 1]
            transitions[f"{current}→{next_shot}"] += 1
        
        # Pattern predictability
        most_common_transition = max(transitions.items(), key=lambda x: x[1]) if transitions else ("none", 0)
        pattern_predictability = (most_common_transition[1] / len(shot_sequences)) * 100
        
        # Shot type distribution
        straight_shots = sum(count for shot, count in shot_counts.items() if 'straight' in shot.lower())
        crosscourt_shots = sum(count for shot, count in shot_counts.items() if 'crosscourt' in shot.lower())
        drive_shots = sum(count for shot, count in shot_counts.items() if 'drive' in shot.lower())
        lob_shots = sum(count for shot, count in shot_counts.items() if 'lob' in shot.lower())
        
        total_directional = straight_shots + crosscourt_shots
        total_height = drive_shots + lob_shots
        
        straight_pct = (straight_shots / total_directional * 100) if total_directional > 0 else 0
        drive_pct = (drive_shots / total_height * 100) if total_height > 0 else 0
        
        return f"""• Shot direction preference: {straight_pct:.1f}% straight, {100-straight_pct:.1f}% crosscourt
• Shot height preference: {drive_pct:.1f}% drives, {100-drive_pct:.1f}% lobs
• Most common transition: {most_common_transition[0]} ({most_common_transition[1]} times)
• Pattern predictability: {pattern_predictability:.1f}% ({'High' if pattern_predictability > 30 else 'Low'})
• Tactical variety: {'High' if len(transitions) > 8 else 'Medium' if len(transitions) > 4 else 'Low'}"""
    
    def analyze_performance_metrics(self, activity_pct, shot_variety, avg_speed, avg_rally, hit_confidence):
        """Performance assessment based on multiple metrics"""
        # Scoring system (0-10 for each metric)
        activity_score = min(10, activity_pct / 10)
        variety_score = min(10, shot_variety)
        speed_score = min(10, avg_speed / 5)  # Assuming 50 pixels/frame is excellent
        rally_score = min(10, avg_rally / 5)  # Assuming 50 frame rallies are excellent
        confidence_score = hit_confidence * 10
        
        overall_score = (activity_score + variety_score + speed_score + rally_score + confidence_score) / 5
        
        # Performance categories
        performance_level = "Excellent" if overall_score >= 8 else "Good" if overall_score >= 6 else "Developing"
        
        return f"""• Overall performance level: {performance_level} ({overall_score:.1f}/10)
• Match intensity: {'High' if activity_pct > 60 else 'Medium' if activity_pct > 30 else 'Low'}
• Shot execution quality: {'Excellent' if hit_confidence > 0.8 else 'Good' if hit_confidence > 0.6 else 'Needs improvement'}
• Rally sustainability: {'Strong' if avg_rally > 15 else 'Moderate' if avg_rally > 8 else 'Weak'}
• Technical diversity: {'High' if shot_variety > 6 else 'Medium' if shot_variety > 3 else 'Limited'}"""

def collect_coaching_data(players, past_ball_pos, type_of_shot, who_hit, match_in_play, frame_count):
    """Collect comprehensive data for autonomous coaching analysis"""
    coaching_data = {
        'frame': frame_count,
        'timestamp': time.time(),
        'shot_type': type_of_shot,
        'player_who_hit': who_hit,
        'match_active': match_in_play is not False,
        'ball_hit_detected': match_in_play[1] if match_in_play is not False else False,
        'player_movement': match_in_play[0] if match_in_play is not False else False,
    }
    
    # Add player position analysis
    if players.get(1) and players.get(2):
        try:
            p1_pose = players[1].get_latest_pose()
            p2_pose = players[2].get_latest_pose()
            
            if p1_pose and p2_pose:
                p1_left_ankle = p1_pose.xyn[0][15] if len(p1_pose.xyn[0]) > 15 else [0, 0]
                p1_right_ankle = p1_pose.xyn[0][16] if len(p1_pose.xyn[0]) > 16 else [0, 0]
                p2_left_ankle = p2_pose.xyn[0][15] if len(p2_pose.xyn[0]) > 15 else [0, 0]
                p2_right_ankle = p2_pose.xyn[0][16] if len(p2_pose.xyn[0]) > 16 else [0, 0]
                
                coaching_data.update({
                    'player1_position': {
                        'left_ankle': [float(p1_left_ankle[0]), float(p1_left_ankle[1])],
                        'right_ankle': [float(p1_right_ankle[0]), float(p1_right_ankle[1])]
                    },
                    'player2_position': {
                        'left_ankle': [float(p2_left_ankle[0]), float(p2_left_ankle[1])],
                        'right_ankle': [float(p2_right_ankle[0]), float(p2_right_ankle[1])]
                    }
                })
        except Exception as e:
            print(f"Error collecting player position data: {e}")
    
    # Add ball trajectory analysis
    if past_ball_pos and len(past_ball_pos) > 0:
        coaching_data.update({
                            'ball_position': (float(past_ball_pos[-1][0]), float(past_ball_pos[-1][1])),
            'ball_trajectory_length': len(past_ball_pos),
            'ball_speed': calculate_ball_speed(past_ball_pos) if len(past_ball_pos) > 1 else 0
        })
    
    return coaching_data

def calculate_ball_speed(ball_positions):
    """Calculate average ball speed from recent positions"""
    if len(ball_positions) < 2:
        return 0
    
    try:
        recent_positions = ball_positions[-5:] if len(ball_positions) >= 5 else ball_positions
        total_distance = 0
        total_time = 0
        
        for i in range(1, len(recent_positions)):
            x1, y1, t1 = recent_positions[i-1]
            x2, y2, t2 = recent_positions[i]
            
            distance = math.sqrt((x2-x1)**2 + (y2-y1)**2)
            time_diff = t2 - t1
            
            if time_diff > 0:
                total_distance += distance
                total_time += time_diff
        
        return total_distance / total_time if total_time > 0 else 0
    except Exception:
        return 0

def generate_coaching_report(coaching_data_collection, path, frame_count):
    """Generate comprehensive coaching report"""
    print("\n" + "="*60)
    print("GENERATING AUTONOMOUS COACHING ANALYSIS...")
    print("="*60)
    
    autonomous_coach = AutonomousSquashCoach()
    
    try:
        coaching_report = autonomous_coach.analyze_match_data(coaching_data_collection)
        # Save comprehensive coaching report
        with open("output/autonomous_coaching_report.txt", "w", encoding='utf-8') as f:
            f.write("AUTONOMOUS SQUASH COACHING ANALYSIS\n")
            f.write("="*50 + "\n\n")
            f.write(f"Analysis Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Video Analyzed: {path}\n")
            f.write(f"Total Frames Processed: {frame_count}\n")
            f.write(f"Enhanced Coaching Data Points: {len(coaching_data_collection)}\n\n")
            f.write("COACHING INSIGHTS:\n")
            f.write("-" * 20 + "\n")
            f.write(coaching_report)
            f.write("\n\n" + "="*50 + "\n")
            f.write("End of Analysis\n")
        # Save detailed coaching data
        with open("output/detailed_coaching_data.json", "w", encoding='utf-8') as f:
            json.dump({
                "match_metadata": {
                    "video_path": path,
                    "total_frames": frame_count,
                    "analysis_date": time.strftime('%Y-%m-%d %H:%M:%S'),
                    "total_data_points": len(coaching_data_collection)
                },
                "coaching_data": coaching_data_collection[-500:] if len(coaching_data_collection) > 500 else coaching_data_collection
            }, f, indent=2)
        
        print(f"✓ Coaching report saved to: output/autonomous_coaching_report.txt")
        print(f"✓ Detailed data saved to: output/detailed_coaching_data.json")
        print(f"✓ Total enhanced data points analyzed: {len(coaching_data_collection)}")
        print("✓ Autonomous coaching analysis complete!")
        
        # Display key insights
        print("\nKEY COACHING INSIGHTS:")
        print("-" * 30)
        print(coaching_report[:400] + "..." if len(coaching_report) > 400 else coaching_report)
        
    except Exception as e:
        print(f"Error generating coaching report: {e}")        # Still save basic data
        with open("output/match_data_summary.json", "w", encoding='utf-8') as f:
            json.dump({
                "total_frames": frame_count,
                "enhanced_coaching_points": len(coaching_data_collection),
                "basic_analysis": "Coaching analysis failed - data preserved"
            }, f, indent=2)
        print("✓ Basic match data saved to: output/match_data_summary.json")

def create_3d_vis(path, frame_width=640, frame_height=360):
    import ast
    import plotly.graph_objects as go
    import plotly.io as pio
    pio.renderers.default = "browser"
    frames = []
    # Create a 3d visualization of the match data
    reference_points_3d = [
            [0, 9.75, 0],  # Top-left corner, 1
            [6.4, 9.75, 0],  # Top-right corner, 2
            [6.4, 0, 0],  # Bottom-right corner, 3
            [0, 0, 0],  # Bottom-left corner, 4
            [3.2, 4.57, 0],  # "T" point, 5
            [0, 2.71, 0],  # Left bottom of the service box, 6
            [6.4, 2.71, 0],  # Right bottom of the service box, 7
            [0, 9.75, 0.48],  # left of tin, 8
            [6.4, 9.75, 0.48],  # right of tin, 9
            [0, 9.75, 1.78],  # Left of the service line, 10
            [6.4, 9.75, 1.78],  # Right of the service line, 11
            [0, 9.75, 4.57],  # Left of the top line of the front court, 12
            [6.4, 9.75, 4.57],  # Right of the top line of the front court, 13
        ]
    reference_points = Referencepoints.get_reference_points(
        path=path, frame_width=frame_width, frame_height=frame_height
    )
    
    # Get data from output/final.csv
    df = pd.read_csv("output/final.csv")
    
    # Parse string representations of lists to actual lists
    def parse_position(pos_str):
        if isinstance(pos_str, str):
            try:
                return ast.literal_eval(pos_str)
            except (ValueError, SyntaxError):
                return [0, 0, 0]  # Default position if parsing fails
        elif isinstance(pos_str, list):
            return pos_str
        else:
            return [0, 0, 0]  # Default position
    
    player1pos = [parse_position(pos) for pos in df['Player 1 RL World Position'].tolist()]
    player2pos = [parse_position(pos) for pos in df['Player 2 RL World Position'].tolist()]
    ballpos = [parse_position(pos) for pos in df['Ball RL World Position'].tolist()]
    
    # Extract coordinates for plotting
    player1_x = [pos[0] for pos in player1pos]
    player1_y = [pos[1] for pos in player1pos]
    player1_z = [pos[2] for pos in player1pos]
    
    player2_x = [pos[0] for pos in player2pos]
    player2_y = [pos[1] for pos in player2pos]
    player2_z = [pos[2] for pos in player2pos]
    
    ball_x = [pos[0] for pos in ballpos]
    ball_y = [pos[1] for pos in ballpos]
    ball_z = [pos[2] for pos in ballpos]
    
    fig = go.Figure()
    num_frames = len(player1pos)
    
    # Create frames for animation
    for t in range(num_frames):
        frame_data = [
            go.Scatter3d(
                x=[player1_x[t]], y=[player1_y[t]], z=[player1_z[t]], 
                mode='markers', 
                marker=dict(size=8, color='blue'),
                name='Player 1'
            ),
            go.Scatter3d(
                x=[player2_x[t]], y=[player2_y[t]], z=[player2_z[t]], 
                mode='markers', 
                marker=dict(size=8, color='red'),
                name='Player 2'
            ),
            go.Scatter3d(
                x=[ball_x[t]], y=[ball_y[t]], z=[ball_z[t]], 
                mode='markers', 
                marker=dict(size=6, color='yellow'),
                name='Ball'
            ),
            go.Scatter3d(
                x=[point[0] for point in reference_points_3d],
                y=[point[1] for point in reference_points_3d],
                z=[point[2] for point in reference_points_3d],
                mode='lines+markers', 
                marker=dict(size=4, color='green'),
                line=dict(color='green', width=2),
                name='Court Layout'
            )
        ]
        frames.append(go.Frame(data=frame_data, name=str(t)))
    
    # Add initial traces
    fig.add_trace(go.Scatter3d(
        x=[player1_x[0]], y=[player1_y[0]], z=[player1_z[0]], 
        mode='markers', 
        marker=dict(size=8, color='blue'),
        name='Player 1'
    ))
    fig.add_trace(go.Scatter3d(
        x=[player2_x[0]], y=[player2_y[0]], z=[player2_z[0]], 
        mode='markers', 
        marker=dict(size=8, color='red'),
        name='Player 2'
    ))
    fig.add_trace(go.Scatter3d(
        x=[ball_x[0]], y=[ball_y[0]], z=[ball_z[0]], 
        mode='markers', 
        marker=dict(size=6, color='yellow'),
        name='Ball'
    ))
    fig.add_trace(go.Scatter3d(
        x=[point[0] for point in reference_points_3d],
        y=[point[1] for point in reference_points_3d],
        z=[point[2] for point in reference_points_3d],
        mode='lines+markers', 
        marker=dict(size=4, color='green'),
        line=dict(color='green', width=2),
        name='Court Layout'
    ))
    
    # Set frames
    fig.frames = frames
    
    # Update layout with controls
    fig.update_layout(
        title="3D Squash Match Visualization",
        scene=dict(
            xaxis_title='Court Width (m)',
            yaxis_title='Court Length (m)',
            zaxis_title='Height (m)',
            aspectmode='manual',
            aspectratio=dict(x=1, y=1.5, z=0.7)
        ),
        updatemenus=[dict(
            type='buttons',
            showactive=False,
            buttons=[
                dict(
                    label='Play',
                    method='animate',
                    args=[None, {"frame": {"duration": 100, "redraw": True},
                                "fromcurrent": True}]
                ),
                dict(
                    label='Pause',
                    method='animate',
                    args=[[None], {"frame": {"duration": 0}, "mode": "immediate",
                                  "transition": {"duration": 0}}]
                )
            ]
        )],
        sliders=[dict(
            steps=[dict(
                method='animate',
                args=[[str(k)], {"frame": {"duration": 100, "redraw": True},
                                "mode": "immediate"}],
                label=str(k)
            ) for k in range(num_frames)],
            transition={"duration": 0},
            x=0.1, y=0, xanchor="left", yanchor="top"
        )]
    )
    
    fig.show()

def create_graphics():
    """Create comprehensive visualizations from squash match data"""
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    import ast
    import re
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    import plotly.offline as pyo
    
    # Set style for better looking plots
    try:
        plt.style.use('seaborn-v0_8')
    except:
        plt.style.use('seaborn')
    sns.set_palette("husl")
    
    df = pd.read_csv("output/final.csv")
    
    print(f"Creating comprehensive visualizations from {len(df)} data points...")
    
    # Enhanced data preprocessing with numpy float64 handling
    def parse_numpy_string(x):
        """Parse strings containing numpy.float64 values"""
        if isinstance(x, str) and x != 'nan':
            try:
                # Handle numpy.float64 strings
                if 'np.float64' in x:
                    # Extract numbers from np.float64(value) format
                    numbers = re.findall(r'np\.float64\(([-+]?[0-9]*\.?[0-9]+)\)', x)
                    return [float(num) for num in numbers]
                else:
                    # Regular list evaluation
                    return ast.literal_eval(x)
            except:
                return [0, 0, 0] if '[' in x else [0, 0]
        return [0, 0, 0] if isinstance(x, str) else [0, 0]
    
    # Parse position data with enhanced handling
    df['Ball_Pos_Parsed'] = df['Ball Position'].apply(parse_numpy_string)
    df['Ball_X'] = df['Ball_Pos_Parsed'].apply(lambda x: x[0] if len(x) > 0 else 0)
    df['Ball_Y'] = df['Ball_Pos_Parsed'].apply(lambda x: x[1] if len(x) > 1 else 0)
    
    # Parse world positions with numpy handling
    df['Player1_World_Parsed'] = df['Player 1 RL World Position'].apply(parse_numpy_string)
    df['Player2_World_Parsed'] = df['Player 2 RL World Position'].apply(parse_numpy_string)
    df['Ball_World_Parsed'] = df['Ball RL World Position'].apply(parse_numpy_string)
    
    df['P1_World_X'] = df['Player1_World_Parsed'].apply(lambda x: x[0] if len(x) > 0 else 0)
    df['P1_World_Y'] = df['Player1_World_Parsed'].apply(lambda x: x[1] if len(x) > 1 else 0)
    df['P2_World_X'] = df['Player2_World_Parsed'].apply(lambda x: x[0] if len(x) > 0 else 0)
    df['P2_World_Y'] = df['Player2_World_Parsed'].apply(lambda x: x[1] if len(x) > 1 else 0)
    df['Ball_World_X'] = df['Ball_World_Parsed'].apply(lambda x: x[0] if len(x) > 0 else 0)
    df['Ball_World_Y'] = df['Ball_World_Parsed'].apply(lambda x: x[1] if len(x) > 1 else 0)
    df['Ball_World_Z'] = df['Ball_World_Parsed'].apply(lambda x: x[2] if len(x) > 2 else 0)
    
    # Filter out zero positions for meaningful analysis
    df_active = df[(df['Ball_X'] > 0) | (df['Ball_Y'] > 0)]
    df_world_active = df[(df['Ball_World_X'] > 0) | (df['Ball_World_Y'] > 0)]
    
    # Create output directory
    os.makedirs("output/graphics", exist_ok=True)
    
    # 1. ENHANCED SHOT TYPE ANALYSIS
    print("Creating enhanced shot type visualizations...")
    
    fig = plt.figure(figsize=(20, 15))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)    # Shot type distribution (enhanced pie chart)
    ax1 = fig.add_subplot(gs[0, 0])
    shot_counts = df['Shot Type'].value_counts()
    colors = plt.cm.get_cmap('tab20')(np.linspace(0, 1, len(shot_counts)))
    pie_result = ax1.pie(shot_counts.values.tolist(), 
                        labels=shot_counts.index.tolist(), 
                        autopct='%1.1f%%', 
                        colors=colors.tolist(), startangle=90, 
                        textprops={'fontsize': 10})
    ax1.set_title('Shot Type Distribution', fontsize=16, fontweight='bold', pad=20)
    
    # Shot type over time (enhanced)
    ax2 = fig.add_subplot(gs[0, 1])
    shot_codes = df['Shot Type'].astype('category').cat.codes
    ax2.scatter(df.index, shot_codes, c=shot_codes, cmap='viridis', alpha=0.7, s=30)
    ax2.set_title('Shot Type Evolution', fontsize=16, fontweight='bold')
    ax2.set_xlabel('Frame Number')
    ax2.set_ylabel('Shot Type')
    ax2.grid(True, alpha=0.3)
    
    # Player performance comparison
    ax3 = fig.add_subplot(gs[0, 2])
    hit_counts = df['Who Hit the Ball'].value_counts()
    bars = ax3.bar(hit_counts.index.tolist(), hit_counts.values.tolist(), 
                   color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'], 
                   edgecolor='black', linewidth=1)
    ax3.set_title('Hits by Player', fontsize=16, fontweight='bold')
    ax3.set_xlabel('Player')
    ax3.set_ylabel('Number of Hits')
    ax3.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{int(height)}', ha='center', va='bottom', fontweight='bold')
    
    # Shot accuracy analysis (enhanced)
    ax4 = fig.add_subplot(gs[1, :])
    window_sizes = [10, 20, 50]
    colors_acc = ['red', 'green', 'blue']
    
    for i, window_size in enumerate(window_sizes):
        df['Valid_Shot'] = (df['Ball_X'] > 0) & (df['Ball_Y'] > 0)
        df[f'Shot_Accuracy_{window_size}'] = df['Valid_Shot'].rolling(window=window_size, min_periods=1).mean() * 100
        ax4.plot(df.index, df[f'Shot_Accuracy_{window_size}'], 
                color=colors_acc[i], linewidth=2, label=f'Window {window_size}', alpha=0.8)
    
    ax4.fill_between(df.index, df['Shot_Accuracy_20'], alpha=0.3, color='green')
    ax4.set_title('Shot Detection Accuracy Analysis (Multiple Windows)', fontsize=16, fontweight='bold')
    ax4.set_xlabel('Frame Number')
    ax4.set_ylabel('Accuracy %')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(0, 100)
    
    # Shot type frequency heatmap
    ax5 = fig.add_subplot(gs[2, 0])
    shot_freq_bins = df.groupby(df.index // 20)['Shot Type'].value_counts().unstack(fill_value=0)
    if len(shot_freq_bins) > 0:
        sns.heatmap(shot_freq_bins.T, ax=ax5, cmap='YlOrRd', cbar_kws={'label': 'Frequency'})
        ax5.set_title('Shot Type Frequency Heatmap', fontsize=14, fontweight='bold')
        ax5.set_xlabel('Time Bins (20 frames each)')
        ax5.set_ylabel('Shot Types')
    
    # Player dominance over time
    ax6 = fig.add_subplot(gs[2, 1])
    player_numeric = df['Who Hit the Ball'].map({'1 hit the ball': 1, '2 hit the ball': 2, '0 hit the ball': 0})
    window = 30
    p1_dominance = (player_numeric == 1).rolling(window=window).mean()
    p2_dominance = (player_numeric == 2).rolling(window=window).mean()
    
    ax6.plot(df.index, p1_dominance, label='Player 1', color='red', linewidth=2)
    ax6.plot(df.index, p2_dominance, label='Player 2', color='blue', linewidth=2)
    ax6.fill_between(df.index, p1_dominance, alpha=0.3, color='red')
    ax6.fill_between(df.index, p2_dominance, alpha=0.3, color='blue')
    ax6.set_title(f'Player Dominance Over Time (Window: {window})', fontsize=14, fontweight='bold')
    ax6.set_xlabel('Frame Number')
    ax6.set_ylabel('Dominance Ratio')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    ax6.set_ylim(0, 1)
      # Rally length distribution
    ax7 = fig.add_subplot(gs[2, 2])
    player_changes = (player_numeric != player_numeric.shift(1)).astype(int)
    rally_starts = df[player_changes == 1].index
    rally_lengths = []
    for i in range(len(rally_starts) - 1):
        rally_lengths.append(rally_starts[i+1] - rally_starts[i])
    
    if rally_lengths:
        ax7.hist(rally_lengths, bins=15, color='orange', alpha=0.7, edgecolor='black')
        ax7.set_title('Rally Length Distribution', fontsize=14, fontweight='bold')
        ax7.set_xlabel('Rally Length (frames)')
        ax7.set_ylabel('Frequency')
        ax7.grid(True, alpha=0.3)
    
    plt.savefig('output/graphics/enhanced_shot_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. DETAILED COURT POSITIONING ANALYSIS
    print("Creating detailed court positioning visualizations...")
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Filter valid world coordinates
    p1_valid = df[(df['P1_World_X'] > 0) & (df['P1_World_Y'] > 0)]
    p2_valid = df[(df['P2_World_X'] > 0) & (df['P2_World_Y'] > 0)]
    ball_valid = df[(df['Ball_World_X'] > 0) & (df['Ball_World_Y'] > 0)]
    
    print(f"Player 1 valid positions: {len(p1_valid)}")
    print(f"Player 2 valid positions: {len(p2_valid)}")
    print(f"Ball valid positions: {len(ball_valid)}")
    
    # Player 1 positioning heatmap with court outline
    if len(p1_valid) > 10:
        h1 = ax1.hist2d(p1_valid['P1_World_X'], p1_valid['P1_World_Y'], 
                       bins=20, cmap='Reds', alpha=0.8, density=True)
        ax1.set_xlim(0, 6.4)
        ax1.set_ylim(0, 9.75)
        ax1.set_title('Player 1 Court Positioning Heatmap', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Court Width (m)')
        ax1.set_ylabel('Court Length (m)')
        
        # Add court lines
        ax1.axhline(y=4.57, color='white', linestyle='--', alpha=0.8, linewidth=2, label='Service Line')
        ax1.axvline(x=3.2, color='white', linestyle='--', alpha=0.8, linewidth=2, label='Center Line')
        ax1.legend()
        plt.colorbar(h1[3], ax=ax1, label='Density')
    else:
        ax1.text(0.5, 0.5, 'Insufficient Player 1 data', transform=ax1.transAxes, 
                ha='center', va='center', fontsize=14)
        ax1.set_title('Player 1 Court Positioning', fontsize=14, fontweight='bold')
    
    # Player 2 positioning heatmap
    if len(p2_valid) > 10:
        h2 = ax2.hist2d(p2_valid['P2_World_X'], p2_valid['P2_World_Y'], 
                       bins=20, cmap='Blues', alpha=0.8, density=True)
        ax2.set_xlim(0, 6.4)
        ax2.set_ylim(0, 9.75)
        ax2.set_title('Player 2 Court Positioning Heatmap', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Court Width (m)')
        ax2.set_ylabel('Court Length (m)')
        
        # Add court lines
        ax2.axhline(y=4.57, color='white', linestyle='--', alpha=0.8, linewidth=2)
        ax2.axvline(x=3.2, color='white', linestyle='--', alpha=0.8, linewidth=2)
        plt.colorbar(h2[3], ax=ax2, label='Density')
    else:
        ax2.text(0.5, 0.5, 'Insufficient Player 2 data', transform=ax2.transAxes, 
                ha='center', va='center', fontsize=14)
        ax2.set_title('Player 2 Court Positioning', fontsize=14, fontweight='bold')
    
    # Ball positioning heatmap with trajectory
    if len(ball_valid) > 10:
        h3 = ax3.hist2d(ball_valid['Ball_World_X'], ball_valid['Ball_World_Y'], 
                       bins=25, cmap='Greens', alpha=0.8, density=True)
        ax3.set_xlim(0, 6.4)
        ax3.set_ylim(0, 9.75)
        ax3.set_title('Ball Positioning Heatmap', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Court Width (m)')
        ax3.set_ylabel('Court Length (m)')
        
        # Add court outline and key lines
        court_outline_x = [0, 6.4, 6.4, 0, 0]
        court_outline_y = [0, 0, 9.75, 9.75, 0]
        ax3.plot(court_outline_x, court_outline_y, 'k-', linewidth=3, alpha=0.8)
        ax3.axhline(y=4.57, color='black', linestyle='--', alpha=0.8, linewidth=2)
        ax3.axvline(x=3.2, color='black', linestyle='--', alpha=0.8, linewidth=2)
        plt.colorbar(h3[3], ax=ax3, label='Density')
    else:
        ax3.text(0.5, 0.5, 'Insufficient Ball data', transform=ax3.transAxes, 
                ha='center', va='center', fontsize=14)
        ax3.set_title('Ball Positioning', fontsize=14, fontweight='bold')
    
    # Combined player movements with trajectories
    if len(p1_valid) > 0 and len(p2_valid) > 0:
        # Sample data for better visualization
        p1_sample = p1_valid.iloc[::5] if len(p1_valid) > 50 else p1_valid
        p2_sample = p2_valid.iloc[::5] if len(p2_valid) > 50 else p2_valid
        
        ax4.scatter(p1_sample['P1_World_X'], p1_sample['P1_World_Y'], 
                   c='red', alpha=0.6, s=30, label='Player 1', edgecolors='darkred')
        ax4.scatter(p2_sample['P2_World_X'], p2_sample['P2_World_Y'], 
                   c='blue', alpha=0.6, s=30, label='Player 2', edgecolors='darkblue')
        
        # Add movement trajectories (last 20 points)
        if len(p1_valid) > 20:
            recent_p1 = p1_valid.tail(20)
            ax4.plot(recent_p1['P1_World_X'], recent_p1['P1_World_Y'], 
                    'r-', alpha=0.7, linewidth=2, label='P1 Recent Path')
        
        if len(p2_valid) > 20:
            recent_p2 = p2_valid.tail(20)
            ax4.plot(recent_p2['P2_World_X'], recent_p2['P2_World_Y'], 
                    'b-', alpha=0.7, linewidth=2, label='P2 Recent Path')
        
        ax4.set_xlim(0, 6.4)
        ax4.set_ylim(0, 9.75)
        ax4.set_title('Combined Player Positioning & Movement', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Court Width (m)')
        ax4.set_ylabel('Court Length (m)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Add court outline
        ax4.plot(court_outline_x, court_outline_y, 'k-', linewidth=2, alpha=0.8)
        ax4.axhline(y=4.57, color='gray', linestyle='--', alpha=0.6)
        ax4.axvline(x=3.2, color='gray', linestyle='--', alpha=0.6)
    else:
        ax4.text(0.5, 0.5, 'Insufficient movement data', transform=ax4.transAxes, 
                ha='center', va='center', fontsize=14)
        ax4.set_title('Player Movement Analysis', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('output/graphics/detailed_court_positioning.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. COMPREHENSIVE BALL ANALYSIS
    print("Creating comprehensive ball analysis...")
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Ball height analysis with enhanced visualization
    if len(ball_valid) > 0:
        # Filter realistic heights (below 5m)
        ball_height_clean = ball_valid[ball_valid['Ball_World_Z'] < 5]
        
        if len(ball_height_clean) > 0:
            # Ball height over time with color coding by shot type
            shot_type_colors = {'straight drive': 'red', 'lob': 'blue', 'crosscourt': 'green', 
                              'tight_straight': 'orange', 'angled_crosscourt': 'purple'}
            
            for shot_type in ball_height_clean['Shot Type'].unique():
                shot_data = ball_height_clean[ball_height_clean['Shot Type'] == shot_type]
                color = shot_type_colors.get(shot_type, 'gray')
                ax1.scatter(shot_data.index, shot_data['Ball_World_Z'], 
                          c=color, alpha=0.7, s=20, label=shot_type[:15])
            
            ax1.set_title('Ball Height by Shot Type', fontsize=14, fontweight='bold')
            ax1.set_xlabel('Frame Number')
            ax1.set_ylabel('Ball Height (m)')
            ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax1.grid(True, alpha=0.3)
            
            # Ball height distribution with statistics
            heights = ball_height_clean['Ball_World_Z']
            ax2.hist(heights, bins=25, color='skyblue', alpha=0.7, edgecolor='black', density=True)
            ax2.axvline(heights.mean(), color='red', linestyle='--', linewidth=2, 
                       label=f'Mean: {heights.mean():.2f}m')
            ax2.axvline(heights.median(), color='green', linestyle='--', linewidth=2, 
                       label=f'Median: {heights.median():.2f}m')
            ax2.set_title('Ball Height Distribution', fontsize=14, fontweight='bold')
            ax2.set_xlabel('Height (m)')
            ax2.set_ylabel('Density')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
    
    # Ball speed analysis with enhanced calculations
    if len(ball_valid) > 1:
        ball_speed_data = ball_valid.copy().sort_index()
        ball_speed_data['Ball_Speed_3D'] = np.sqrt(
            (ball_speed_data['Ball_World_X'].diff())**2 + 
            (ball_speed_data['Ball_World_Y'].diff())**2 +
            (ball_speed_data['Ball_World_Z'].diff())**2
        )
        
        # Filter realistic speeds (below 10 m/frame)
        speed_clean = ball_speed_data[
            (ball_speed_data['Ball_Speed_3D'] > 0) & 
            (ball_speed_data['Ball_Speed_3D'] < 10)
        ]
        
        if len(speed_clean) > 0:
            # Speed over time with smoothing
            window = 5
            speed_clean['Speed_Smooth'] = speed_clean['Ball_Speed_3D'].rolling(window=window).mean()
            
            ax3.plot(speed_clean.index, speed_clean['Ball_Speed_3D'], 
                    alpha=0.4, color='lightblue', linewidth=1, label='Raw Speed')
            ax3.plot(speed_clean.index, speed_clean['Speed_Smooth'], 
                    alpha=0.8, color='darkblue', linewidth=2, label=f'Smoothed (n={window})')
            ax3.set_title('Ball Speed Analysis', fontsize=14, fontweight='bold')
            ax3.set_xlabel('Frame Number')
            ax3.set_ylabel('Speed (m/frame)')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # Speed distribution by shot type
            speed_by_shot = []
            shot_labels = []
            for shot_type in speed_clean['Shot Type'].unique():
                shot_speeds = speed_clean[speed_clean['Shot Type'] == shot_type]['Ball_Speed_3D']
                if len(shot_speeds) > 0:
                    speed_by_shot.append(shot_speeds.tolist())
                    shot_labels.append(shot_type[:15])
            
            if speed_by_shot:
                ax4.boxplot(speed_by_shot, labels=shot_labels)
                ax4.set_title('Speed Distribution by Shot Type', fontsize=14, fontweight='bold')
                ax4.set_xlabel('Shot Type')
                ax4.set_ylabel('Speed (m/frame)')
                ax4.tick_params(axis='x', rotation=45)
                ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('output/graphics/comprehensive_ball_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Generate comprehensive statistics
    stats_summary = {
        'Total Frames': len(df),
        'Active Ball Frames': len(ball_valid),
        'Player 1 Position Frames': len(p1_valid),
        'Player 2 Position Frames': len(p2_valid),
        'Activity Rate': f"{len(ball_valid)/len(df)*100:.1f}%",
        'Unique Shot Types': df['Shot Type'].nunique(),
        'Most Common Shot': df['Shot Type'].mode().iloc[0] if len(df['Shot Type'].mode()) > 0 else 'N/A',
        'Player 1 Hits': (df['Who Hit the Ball'] == '1 hit the ball').sum(),
        'Player 2 Hits': (df['Who Hit the Ball'] == '2 hit the ball').sum(),
        'Average Ball Height': f"{ball_valid['Ball_World_Z'].mean():.2f}m" if len(ball_valid) > 0 else 'N/A',
        'Max Ball Height': f"{ball_valid['Ball_World_Z'].max():.2f}m" if len(ball_valid) > 0 else 'N/A'
    }
    
    # Save comprehensive statistics
    with open('output/graphics/comprehensive_match_statistics.txt', 'w') as f:
        f.write("COMPREHENSIVE SQUASH MATCH ANALYSIS\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Analysis Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        for key, value in stats_summary.items():
            f.write(f"{key}: {value}\n")
        
        f.write(f"\nDETAILED SHOT TYPE BREAKDOWN:\n")
        f.write("-" * 30 + "\n")
        for shot_type, count in df['Shot Type'].value_counts().items():
            percentage = (count / len(df)) * 100
            f.write(f"{shot_type}: {count} ({percentage:.1f}%)\n")
        
        f.write(f"\nPLAYER PERFORMANCE ANALYSIS:\n")
        f.write("-" * 30 + "\n")
        for player, count in df['Who Hit the Ball'].value_counts().items():
            percentage = (count / len(df)) * 100
            f.write(f"{player}: {count} ({percentage:.1f}%)\n")
        
        if len(ball_valid) > 0:
            f.write(f"\nBALL TRAJECTORY ANALYSIS:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Average Height: {ball_valid['Ball_World_Z'].mean():.2f}m\n")
            f.write(f"Maximum Height: {ball_valid['Ball_World_Z'].max():.2f}m\n")
            f.write(f"Height Standard Deviation: {ball_valid['Ball_World_Z'].std():.2f}m\n")
    
    print(f"\n{'='*70}")
    print("COMPREHENSIVE VISUALIZATION SUMMARY")
    print(f"{'='*70}")
    print(f"✓ Created detailed shot type analysis with {len(shot_counts)} different shots")
    print(f"✓ Generated court positioning heatmaps for {len(p1_valid) + len(p2_valid)} player positions")
    print(f"✓ Analyzed ball trajectory data from {len(ball_valid)} valid positions")
    print(f"✓ Total data points processed: {len(df)}")
    print(f"✓ Activity rate: {len(ball_valid)/len(df)*100:.1f}%")
    print(f"\nFiles created in output/graphics/:")
    print("• enhanced_shot_analysis.png - Comprehensive shot type analysis")
    print("• detailed_court_positioning.png - Player positioning heatmaps with trajectories")
    print("• comprehensive_ball_analysis.png - Ball height, speed, and trajectory analysis")
    print("• shot_analysis.png - Basic shot type distribution and player performance comparison")
    print("• court_positioning.png - Player and ball positioning heatmaps")
    print("• ball_analysis.png - Ball height and speed analysis over time")
    print("• shot_player_analysis.png - Shot type preferences and accuracy by individual players")
    print("• match_flow.png - Rally analysis and match dynamics over time")
    print("• comprehensive_match_statistics.txt - Detailed statistical analysis")
    
    # Create advanced analytics
    try:
        create_advanced_analytics()
        print("\n✓ Advanced analytics visualizations also created!")
    except Exception as e:
        print(f"\n⚠ Could not create advanced analytics: {e}")
    
    print(f"{'='*70}")
    
    return stats_summary

def create_advanced_analytics():
    """Create additional advanced analytics visualizations"""
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    import pandas as pd
    
    print("Creating advanced analytics visualizations...")
    
    df = pd.read_csv("output/final.csv")
    
    # Create advanced analytics directory
    os.makedirs("output/graphics/advanced", exist_ok=True)
    
    # 1. Shot Transition Matrix
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Create shot sequence for transition analysis
    shot_sequence = df['Shot Type'].tolist()
    transitions = {}
    for i in range(len(shot_sequence) - 1):
        current = shot_sequence[i]
        next_shot = shot_sequence[i + 1]
        key = f"{current} → {next_shot}"
        transitions[key] = transitions.get(key, 0) + 1
    
    # Plot top transitions
    top_transitions = sorted(transitions.items(), key=lambda x: x[1], reverse=True)[:15]
    if top_transitions:
        trans_names, trans_counts = zip(*top_transitions)
        bars = ax1.barh(range(len(trans_names)), trans_counts, color='skyblue')
        ax1.set_yticks(range(len(trans_names)))
        ax1.set_yticklabels([name.replace(' → ', '\n→ ') for name in trans_names], fontsize=8)
        ax1.set_xlabel('Frequency')
        ax1.set_title('Most Common Shot Transitions', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Add value labels
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax1.text(width + 0.3, bar.get_y() + bar.get_height()/2, 
                    str(int(width)), ha='left', va='center')
    
    # 2. Player Performance Timeline
    df['Player_Numeric'] = df['Who Hit the Ball'].map({
        '1 hit the ball': 1, 
        '2 hit the ball': 2, 
        '0 hit the ball': 0
    })
    
    # Rolling performance analysis
    window = 20
    df['P1_Performance'] = (df['Player_Numeric'] == 1).rolling(window=window).mean()
    df['P2_Performance'] = (df['Player_Numeric'] == 2).rolling(window=window).mean()
    
    ax2.plot(df.index, df['P1_Performance'], label='Player 1', color='red', linewidth=2)
    ax2.plot(df.index, df['P2_Performance'], label='Player 2', color='blue', linewidth=2)
    ax2.fill_between(df.index, df['P1_Performance'], alpha=0.3, color='red')
    ax2.fill_between(df.index, df['P2_Performance'], alpha=0.3, color='blue')
    ax2.set_xlabel('Frame Number')
    ax2.set_ylabel('Performance Ratio')
    ax2.set_title(f'Player Performance Over Time (Rolling Window: {window})', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig('output/graphics/advanced/shot_transitions_performance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Court Zone Analysis
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Parse world coordinates for zone analysis
    def safe_eval(x):
        if isinstance(x, str) and x != 'nan':
            try:
                return ast.literal_eval(x)
            except:
                return [0, 0, 0]
        return [0, 0, 0]
    
    df['Ball_World_Parsed'] = df['Ball RL World Position'].apply(safe_eval)
    df['Ball_World_X'] = df['Ball_World_Parsed'].apply(lambda x: x[0] if len(x) > 0 else 0)
    df['Ball_World_Y'] = df['Ball_World_Parsed'].apply(lambda x: x[1] if len(x) > 1 else 0)
    
    # Define court zones
    def get_court_zone(x, y):
        if x == 0 and y == 0:
            return 'Unknown'
        if y < 3.25:  # Front court
            return 'Front Court'
        elif y < 6.5:  # Mid court
            return 'Mid Court'
        else:  # Back court
            return 'Back Court'
    
    df['Court_Zone'] = df.apply(lambda row: get_court_zone(row['Ball_World_X'], row['Ball_World_Y']), axis=1)
    
    # Zone distribution
    zone_counts = df[df['Court_Zone'] != 'Unknown']['Court_Zone'].value_counts()
    if len(zone_counts) > 0:
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        wedges, texts, autotexts = ax1.pie(zone_counts.values, labels=zone_counts.index, 
                                          autopct='%1.1f%%', colors=colors, startangle=90)
        ax1.set_title('Ball Distribution by Court Zone', fontweight='bold')
    
    # Zone by shot type
    zone_shot_crosstab = pd.crosstab(df['Court_Zone'], df['Shot Type'])
    if len(zone_shot_crosstab) > 0:
        zone_shot_crosstab.plot(kind='bar', ax=ax2, colormap='viridis')
        ax2.set_title('Shot Types by Court Zone', fontweight='bold')
        ax2.set_xlabel('Court Zone')
        ax2.set_ylabel('Count')
        ax2.legend(title='Shot Type', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
      # 4. Shot Accuracy Analysis
    df['Shot_Detected'] = (df['Ball RL World Position'] != '[0, 0, 0]') & (df['Ball Position'] != '[0, 0]')
    
    # Shot accuracy by type
    shot_accuracy = df.groupby('Shot Type')['Shot_Detected'].mean() * 100
    if len(shot_accuracy) > 0:
        bars = ax3.bar(range(len(shot_accuracy)), shot_accuracy.values, color='lightgreen')
        ax3.set_xticks(range(len(shot_accuracy)))
        ax3.set_xticklabels(shot_accuracy.index, rotation=45, ha='right')
        ax3.set_ylabel('Detection Accuracy (%)')
        ax3.set_title('Shot Detection Accuracy by Type', fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # Add value labels
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{height:.1f}%', ha='center', va='bottom')
    
    # 5. Rally Intensity Heatmap
    df['Rally_ID'] = (df['Player_Numeric'].diff() != 0).cumsum()
    rally_stats = df.groupby('Rally_ID').agg({
        'Shot Type': 'count',
        'Ball_World_X': 'std',
        'Ball_World_Y': 'std'
    }).rename(columns={'Shot Type': 'Rally_Length', 'Ball_World_X': 'X_Variation', 'Ball_World_Y': 'Y_Variation'})
    
    rally_stats = rally_stats.dropna()
    if len(rally_stats) > 0:
        # Create intensity matrix
        x_bins = np.linspace(rally_stats['Rally_Length'].min(), rally_stats['Rally_Length'].max(), 10)
        y_bins = np.linspace(rally_stats['X_Variation'].min(), rally_stats['X_Variation'].max(), 10)
        
        H, xedges, yedges = np.histogram2d(rally_stats['Rally_Length'], rally_stats['X_Variation'], bins=[x_bins, y_bins])
        
        im = ax4.imshow(H.T, origin='lower', aspect='auto', cmap='YlOrRd')
        ax4.set_title('Rally Intensity Heatmap\n(Length vs Court Coverage)', fontweight='bold')
        ax4.set_xlabel('Rally Length')
        ax4.set_ylabel('Court Coverage (X-variation)')
        plt.colorbar(im, ax=ax4, label='Frequency')
    
    plt.tight_layout()
    plt.savefig('output/graphics/advanced/court_zone_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 6. Player Movement Patterns
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Player position analysis
    df['P1_World_Parsed'] = df['Player 1 RL World Position'].apply(safe_eval)
    df['P2_World_Parsed'] = df['Player 2 RL World Position'].apply(safe_eval)
    df['P1_World_X'] = df['P1_World_Parsed'].apply(lambda x: x[0] if len(x) > 0 else 0)
    df['P1_World_Y'] = df['P1_World_Parsed'].apply(lambda x: x[1] if len(x) > 1 else 0)
    df['P2_World_X'] = df['P2_World_Parsed'].apply(lambda x: x[0] if len(x) > 0 else 0)
    df['P2_World_Y'] = df['P2_World_Parsed'].apply(lambda x: x[1] if len(x) > 1 else 0)
    
    # Movement distance analysis
    df['P1_Movement'] = np.sqrt((df['P1_World_X'].diff())**2 + (df['P1_World_Y'].diff())**2)
    df['P2_Movement'] = np.sqrt((df['P2_World_X'].diff())**2 + (df['P2_World_Y'].diff())**2)
    
    # Filter outliers
    p1_movement_clean = df['P1_Movement'][(df['P1_Movement'] > 0) & (df['P1_Movement'] < 2)]
    p2_movement_clean = df['P2_Movement'][(df['P2_Movement'] > 0) & (df['P2_Movement'] < 2)]
    
    if len(p1_movement_clean) > 0 and len(p2_movement_clean) > 0:
        # Movement comparison
        ax1.hist(p1_movement_clean, alpha=0.7, label='Player 1', color='red', bins=20)
        ax1.hist(p2_movement_clean, alpha=0.7, label='Player 2', color='blue', bins=20)
        ax1.set_xlabel('Movement Distance (m/frame)')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Player Movement Distribution', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Movement over time
        window = 15
        df['P1_Movement_MA'] = df['P1_Movement'].rolling(window=window).mean()
        df['P2_Movement_MA'] = df['P2_Movement'].rolling(window=window).mean()
        
        ax2.plot(df.index, df['P1_Movement_MA'], label='Player 1', color='red', linewidth=2)
        ax2.plot(df.index, df['P2_Movement_MA'], label='Player 2', color='blue', linewidth=2)
        ax2.fill_between(df.index, df['P1_Movement_MA'], alpha=0.3, color='red')
        ax2.fill_between(df.index, df['P2_Movement_MA'], alpha=0.3, color='blue')
        ax2.set_xlabel('Frame Number')
        ax2.set_ylabel('Movement (m/frame)')
        ax2.set_title(f'Player Movement Over Time (MA: {window})', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('output/graphics/advanced/player_movement_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ Advanced analytics visualizations created successfully!")
    print("Files created in output/graphics/advanced/:")
    print("• shot_transitions_performance.png - Shot transitions and performance")
    print("• court_zone_analysis.png - Court zone usage and rally intensity")
    print("• player_movement_analysis.png - Player movement patterns and distributions")
    
    return True

def create_all_graphics():
    """Main function to create all visualizations and provide a comprehensive summary"""
    print("🎾 SQUASH MATCH ANALYSIS - COMPREHENSIVE VISUALIZATION SUITE")
    print("=" * 70)
    
    # Create main graphics
    print("\n📊 Creating main visualizations...")
    stats = create_graphics()
    
    print("\n📈 Creating advanced analytics...")
    create_advanced_analytics()
    
    # Create a comprehensive summary report
    print("\n📋 Generating comprehensive analysis report...")
    
    # Read the existing statistics
    df = pd.read_csv("output/final.csv")
    
    # Generate comprehensive HTML report
    html_report = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Squash Match Analysis Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }}
            .container {{ max-width: 1200px; margin: 0 auto; background-color: white; padding: 30px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }}
            h1 {{ color: #2c3e50; text-align: center; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
            h2 {{ color: #34495e; border-left: 4px solid #3498db; padding-left: 15px; }}
            .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin: 20px 0; }}
            .stat-card {{ background: #ecf0f1; padding: 20px; border-radius: 8px; text-align: center; }}
            .stat-number {{ font-size: 2em; font-weight: bold; color: #3498db; }}
            .stat-label {{ color: #7f8c8d; margin-top: 5px; }}
            .visualization-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin: 20px 0; }}
            .viz-card {{ background: #f8f9fa; padding: 15px; border-radius: 8px; border: 1px solid #dee2e6; }}
            .viz-card h3 {{ margin-top: 0; color: #495057; }}
            .file-list {{ background: #f1f3f4; padding: 15px; border-radius: 5px; font-family: monospace; }}
            .highlight {{ background: #fff3cd; padding: 10px; border-left: 4px solid #ffc107; margin: 10px 0; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🎾 Squash Match Analysis Report</h1>
            
            <div class="highlight">
                <strong>Analysis Complete:</strong> Generated comprehensive visualizations from {len(df)} data points 
                with {df['Shot Type'].nunique()} unique shot types detected.
            </div>
            
            <h2>📊 Key Statistics</h2>
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-number">{len(df)}</div>
                    <div class="stat-label">Total Frames</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">{(df['Ball Position'] != '[0, 0]').sum()}</div>
                    <div class="stat-label">Active Frames</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">{df['Shot Type'].nunique()}</div>
                    <div class="stat-label">Shot Types</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">{((df['Ball Position'] != '[0, 0]').sum() / len(df) * 100):.1f}%</div>
                    <div class="stat-label">Activity Rate</div>
                </div>
            </div>
            
            <h2>🎯 Shot Type Analysis</h2>
            <div class="visualization-grid">
                <div class="viz-card">
                    <h3>Most Common Shots:</h3>
                    <ul>
    """
    
    # Add top 5 shot types
    for shot_type, count in df['Shot Type'].value_counts().head(5).items():
        percentage = (count / len(df)) * 100
        html_report += f"<li>{shot_type}: {count} shots ({percentage:.1f}%)</li>\n"
    
    html_report += f"""
                    </ul>
                </div>
                <div class="viz-card">
                    <h3>Player Performance:</h3>
                    <ul>
    """
    
    # Add player performance
    for player, count in df['Who Hit the Ball'].value_counts().items():
        percentage = (count / len(df)) * 100
        html_report += f"<li>{player}: {count} hits ({percentage:.1f}%)</li>\n"
    
    html_report += f"""
                    </ul>
                </div>
            </div>
            
            <h2> Generated Visualizations</h2>
            
            <h3>Main Analysis Charts:</h3>
            <div class="file-list">
                • shot_analysis.png - Shot type distribution and accuracy<br>
                • court_positioning.png - Player and ball positioning heatmaps<br>
                • ball_analysis.png - Ball trajectory, height, and speed analysis<br>
                • shot_player_analysis.png - Shot types breakdown by player<br>
                • match_flow.png - Rally analysis and match dynamics<br>
                • match_statistics.txt - Detailed numerical statistics
            </div>
            
            <h3>Advanced Analytics:</h3>
            <div class="file-list">
                • shot_transitions_performance.png - Shot transitions and player performance timeline<br>
                • court_zone_analysis.png - Court zone usage and rally intensity analysis<br>
                • player_movement_analysis.png - Player movement patterns and distributions
            </div>
            
            <h2> Key Insights</h2>
            <div class="visualization-grid">
                <div class="viz-card">
                    <h3>Game Dynamics</h3>
                    <p>The match showed an activity rate of {((df['Ball Position'] != '[0, 0]').sum() / len(df) * 100):.1f}%, 
                    indicating {"high" if ((df['Ball Position'] != '[0, 0]').sum() / len(df) * 100) > 80 else "moderate"} 
                    ball tracking success.</p>
                </div>
                <div class="viz-card">
                    <h3>Shot Variety</h3>
                    <p>Analysis detected {df['Shot Type'].nunique()} unique shot types, with 
                    {df['Shot Type'].mode().iloc[0] if len(df['Shot Type'].mode()) > 0 else 'N/A'} 
                    being the most frequently used shot.</p>
                </div>
                <div class="viz-card">
                    <h3>Player Balance</h3>
                    <p>Player activity shows {"balanced play" if abs((df['Who Hit the Ball'] == '1 hit the ball').sum() - (df['Who Hit the Ball'] == '2 hit the ball').sum()) < len(df) * 0.2 else "one player dominating"}
                    throughout the analyzed sequence.</p>
                </div>
            </div>
            
            <h2> File Locations</h2>
            <p>All visualizations have been saved to: <code>output/graphics/</code></p>
            <p>Advanced analytics saved to: <code>output/graphics/advanced/</code></p>
            
            <div class="highlight">
                <strong>Next Steps:</strong> Review the generated visualizations to gain insights into playing patterns, 
                shot effectiveness, court positioning, and areas for improvement.
            </div>
            
            <hr>
            <p style="text-align: center; color: #7f8c8d; margin-top: 30px;">
                Generated by Autonomous Squash Coaching System | {time.strftime('%Y-%m-%d %H:%M:%S')}
            </p>
        </div>
    </body>
    </html>
    """
    
    # Save HTML report
    with open('output/graphics/comprehensive_analysis_report.html', 'w', encoding='utf-8') as f:
        f.write(html_report)
    
    print("\n" + "=" * 70)
    print(" ANALYSIS COMPLETE!")
    print("=" * 70)
    print(f" Generated {9} comprehensive visualizations")
    print(f" Analyzed {len(df)} data points")
    print(f" Detected {df['Shot Type'].nunique()} unique shot types")
    print(f" Created interactive HTML report")
    print("\n Files created:")
    print("    output/graphics/ - Main visualizations (6 files)")
    print("    output/graphics/advanced/ - Advanced analytics (3 files)")
    print("    comprehensive_analysis_report.html - Interactive summary")
    print("\nOpen the HTML report in your browser for a complete overview!")
    print("=" * 70)
    
    return True

def view_all_graphics(interactive=True):
    """Display a summary of all created graphics and optionally open them"""
    import os
    import webbrowser
    from pathlib import Path
    
    graphics_dir = Path("output/graphics")
    
    print(f"\n{'='*80}")
    print("SQUASH MATCH VISUALIZATION GALLERY")
    print(f"{'='*80}")
    
    if not graphics_dir.exists():
        print(" No graphics directory found. Run create_graphics() first!")
        return
    
    # List all generated files
    png_files = list(graphics_dir.glob("*.png"))
    txt_files = list(graphics_dir.glob("*.txt"))
    html_files = list(graphics_dir.glob("*.html"))
    
    print(f"\nVISUALIZATION FILES CREATED:")
    print("-" * 50)
    
    for png_file in sorted(png_files):
        file_size = png_file.stat().st_size / 1024  # KB
        print(f"  {png_file.name:<40} ({file_size:.1f} KB)")
    
    print(f"\n ANALYSIS REPORTS:")
    print("-" * 50)
    
    for txt_file in sorted(txt_files):
        file_size = txt_file.stat().st_size / 1024  # KB
        print(f" {txt_file.name:<40} ({file_size:.1f} KB)")
    
    for html_file in sorted(html_files):
        file_size = html_file.stat().st_size / 1024  # KB
        print(f" {html_file.name:<40} ({file_size:.1f} KB)")
    
    # Advanced directory
    advanced_dir = graphics_dir / "advanced"
    if advanced_dir.exists():
        advanced_files = list(advanced_dir.glob("*.png"))
        if advanced_files:
            print(f"\n ADVANCED ANALYTICS:")
            print("-" * 50)
            for adv_file in sorted(advanced_files):
                file_size = adv_file.stat().st_size / 1024  # KB
                print(f" advanced/{adv_file.name:<35} ({file_size:.1f} KB)")
    
    print(f"\n📋 VISUALIZATION DESCRIPTIONS:")
    print("-" * 50)
    descriptions = {
        "enhanced_shot_analysis.png": "Comprehensive shot type analysis with distribution, evolution, and accuracy metrics",
        "detailed_court_positioning.png": "Player positioning heatmaps with movement trajectories and court zones",
        "comprehensive_ball_analysis.png": "Ball trajectory analysis including height patterns and speed distributions",
        "shot_analysis.png": "Basic shot type distribution and player performance comparison",
        "court_positioning.png": "Player and ball positioning heatmaps with court coverage analysis",
        "ball_analysis.png": "Ball height and speed analysis over time",
        "shot_player_analysis.png": "Shot type preferences and accuracy by individual players",
        "match_flow.png": "Rally analysis and match dynamics over time"
    }
    
    for file_name, description in descriptions.items():
        if (graphics_dir / file_name).exists():
            print(f"• {file_name}: {description}")
    
    print(f"\n TOTAL FILES: {len(png_files)} images, {len(txt_files)} reports, {len(html_files)} interactive")
    
    # Show key statistics
    try:
        stats_file = graphics_dir / "comprehensive_match_statistics.txt"
        if stats_file.exists():
            with open(stats_file, 'r') as f:
                content = f.read()
                lines = content.split('\n')
                
                print(f"\n KEY MATCH INSIGHTS:")
                print("-" * 50)
                for line in lines[5:15]:  # Get the main stats
                    if ':' in line and line.strip():
                        key, value = line.split(':', 1)
                        print(f"• {key.strip()}: {value.strip()}")
    except Exception as e:
        print(f" Could not read statistics: {e}")
    
    print(f"\n{'='*80}")
    print(" ANALYSIS COMPLETE! All visualizations saved to output/graphics/")
    print(f"{'='*80}")
    
    # Offer to open graphics folder only if interactive
    if interactive:
        response = input("\n Would you like to open the graphics folder? (y/N): ").strip().lower()
        if response in ['y', 'yes']:
            try:
                if os.name == 'nt':  # Windows
                    os.startfile(str(graphics_dir))
                elif os.name == 'posix':  # macOS and Linux
                    os.system(f'open "{graphics_dir}"' if os.uname().sysname == 'Darwin' else f'xdg-open "{graphics_dir}"')
                print(" Graphics folder opened!")
            except Exception as e:
                print(f" Could not open folder: {e}")
                print(f" Manual path: {graphics_dir.absolute()}")
    else:
        print(f" Graphics location: {graphics_dir.absolute()}")
    
    return {
        "png_files": len(png_files),
        "txt_files": len(txt_files), 
        "html_files": len(html_files),
        "total_size_kb": sum(f.stat().st_size for f in graphics_dir.rglob("*") if f.is_file()) / 1024
    }
