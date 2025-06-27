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
from collections import defaultdict
from transformers import AutoModelForCausalLM, AutoTokenizer

# Import your existing modules
from ultralytics import YOLO
from squash import Referencepoints
from squash.Ball import Ball
from squash.Player import Player

class AutonomousSquashCoach:
    def __init__(self):
        print("🚀 Loading enhanced autonomous coaching model with GPU optimization...")
        
        # GPU optimization setup
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🔧 Coach compute device: {self.device}")
        
        # Enhanced analysis parameters
        self.analysis_config = {
            'min_rally_length': 3,
            'shot_variety_threshold': 5,
            'movement_intensity_threshold': 15,
            'confidence_threshold': 0.6,
            'bounce_analysis_enabled': True
        }
        
        try:
            self.model_name = "Qwen/Qwen2.5-3B-Instruct"
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else "cpu"
            )
            
            # GPU memory optimization
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                print("   💾 GPU memory optimized for coaching model")
            
            print(f"✅ Enhanced coach model loaded on {self.device}")
            
        except Exception as e:
            print(f"⚠️  Could not load AI coaching model: {e}")
            print("   📝 Will provide enhanced rule-based analysis instead")
            self.model = None
            self.tokenizer = None
    
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
                if isinstance(data_point, dict):
                    # Collect wall bounce data
                    wall_bounces = data_point.get('wall_bounce_count', 0)
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
            insights.append("🏐 **Ball Bounce Analysis:**")
            insights.append(f"   • Total bounces detected: {bounce_patterns['total_bounces']}")
            insights.append(f"   • Average bounces per rally: {bounce_patterns['avg_bounces_per_rally']:.1f}")
            insights.append(f"   • Bounce frequency: {bounce_patterns['bounce_frequency']}")
            
            if bounce_patterns['bounce_frequency'] == 'high':
                insights.append("   • 🔄 Consider working on shot placement to reduce unnecessary wall bounces")
                insights.append("   • 💡 Focus on straight drives and court positioning")
            elif bounce_patterns['bounce_frequency'] == 'low':
                insights.append("   • ✅ Good shot control with minimal wall bounces")
                insights.append("   • 💡 Consider adding tactical boasts when appropriate")
        
        return "\n".join(insights)

    def analyze_match_data(self, coaching_data):
        """Enhanced match data analysis with bounce pattern insights"""
        if not coaching_data:
            return "No match data available for analysis"
        
        # Generate comprehensive analysis
        analysis = self.prepare_match_analysis(coaching_data)
        
        # Enhanced bounce pattern analysis
        bounce_patterns = self.analyze_bounce_patterns(coaching_data)
        
        if not self.model:
            return self.fallback_analysis_with_bounces(analysis, bounce_patterns)
        
        # Enhanced coaching prompt with bounce analysis
        coaching_prompt = f"""You are an expert squash coach analyzing match data with enhanced ball tracking and bounce detection. 

{analysis}

BALL BOUNCE ANALYSIS:
• Total bounces detected: {bounce_patterns['total_bounces']}
• Average bounces per rally: {bounce_patterns['avg_bounces_per_rally']:.1f}
• Bounce frequency: {bounce_patterns['bounce_frequency']}

Provide enhanced coaching advice including:
1. Technical shot analysis with bounce pattern insights
2. Movement pattern assessment  
3. Specific improvement areas focusing on shot placement
4. Training recommendations for better court control
5. Tactical suggestions including optimal bounce usage

Focus on actionable feedback that addresses both technique and bounce control patterns."""

        try:
            inputs = self.tokenizer(coaching_prompt, return_tensors="pt", truncation=True, max_length=2048)
            if self.device == "cuda":
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the generated response
            generated_text = response[len(coaching_prompt):].strip()
            
            # Add enhanced insights
            enhanced_insights = self.generate_enhanced_coaching_insights(analysis, bounce_patterns)
            
            return f"{generated_text}\n\n{enhanced_insights}"
            
        except Exception as e:
            print(f"AI coaching generation error: {e}")
            return self.fallback_analysis_with_bounces(analysis, bounce_patterns)

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
        """Enhanced match analysis with detailed pattern recognition and advanced metrics"""
        shot_counts = defaultdict(int)
        activity_frames = 0
        total_frames = len(coaching_data)
        ball_speeds = []
        player_movement_patterns = {'player1': [], 'player2': []}
        shot_sequences = []
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
            
            # Enhanced shot analysis
            shot_type = data_point.get('shot_type', 'unknown')
            if isinstance(shot_type, list) and len(shot_type) > 0:
                # Extract detailed shot components
                direction = shot_type[0] if len(shot_type) > 0 else 'unknown'
                height = shot_type[1] if len(shot_type) > 1 else 'medium'
                style = shot_type[2] if len(shot_type) > 2 else 'standard'
                
                # Create comprehensive shot descriptor
                shot_descriptor = f"{direction}_{height}_{style}"
                shot_counts[shot_descriptor] += 1
                
                # Track shot sequences for pattern analysis
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
            
            # Player movement analysis with enhanced metrics
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
        
        # Advanced pattern analysis
        movement_analysis = self.analyze_player_movement_patterns(player_movement_patterns)
        shot_pattern_analysis = self.analyze_shot_patterns(shot_sequences, shot_counts)
        performance_analysis = self.analyze_performance_metrics(
            activity_percentage, shot_variety, avg_ball_speed, avg_rally_length, avg_hit_confidence
        )
        
        # Generate comprehensive analysis
        analysis = f"""COMPREHENSIVE MATCH ANALYSIS:

BASIC STATISTICS:
• Total data points analyzed: {total_frames}
• Active rally time: {activity_percentage:.1f}% of match
• Number of rallies: {len(rally_lengths)}
• Average rally length: {avg_rally_length:.1f} frames
• Longest rally: {max_rally_length} frames

SHOT ANALYSIS:
• Most frequent shot: {most_common_shot[0]} ({most_common_shot[1]} occurrences)
• Shot variety score: {shot_variety}/10
• Total unique shot types: {len(shot_counts)}
• Top 3 shots: {dict(list(sorted(shot_counts.items(), key=lambda x: x[1], reverse=True)[:3]))}

BALL DYNAMICS:
• Average ball speed: {avg_ball_speed:.2f} pixels/frame
• Maximum ball speed: {max_ball_speed:.2f} pixels/frame
• Speed consistency: {'High' if ball_speeds and np.std(ball_speeds) < avg_ball_speed * 0.3 else 'Variable'}
• Ball tracking quality: {avg_hit_confidence:.2f}/1.0

MOVEMENT PATTERNS:
{movement_analysis}

SHOT PATTERNS & TACTICS:
{shot_pattern_analysis}

PERFORMANCE ASSESSMENT:
{performance_analysis}

DATA QUALITY INDICATORS:
• Ball detection rate: {len(ball_speeds)/total_frames*100:.1f}%
• Movement tracking quality: {'Good' if len(movement_intensities) > total_frames * 0.5 else 'Limited'}
• Average movement intensity: {avg_movement_intensity:.2f} pixels/frame"""
        
        return analysis
    
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
            'ball_position': [float(past_ball_pos[-1][0]), float(past_ball_pos[-1][1])],
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
        with open("output/autonomous_coaching_report.txt", "w") as f:
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
        with open("output/detailed_coaching_data.json", "w") as f:
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
        print(f"Error generating coaching report: {e}")
        # Still save basic data
        with open("output/match_data_summary.json", "w") as f:
            json.dump({
                "total_frames": frame_count,
                "enhanced_coaching_points": len(coaching_data_collection),
                "basic_analysis": "Coaching analysis failed - data preserved"
            }, f, indent=2)
        print("✓ Basic match data saved to: output/match_data_summary.json")

if __name__ == "__main__":
    print("Autonomous Squash Coaching System loaded successfully!")
    print("Import this module and use collect_coaching_data() and generate_coaching_report() functions")
