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
        print("Loading autonomous coaching model...")
        try:
            self.model_name = "Qwen/Qwen2.5-3B-Instruct"
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                device_map="auto" if torch.cuda.is_available() else "cpu"
            )
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"✓ Coach model loaded on {self.device}")
        except Exception as e:
            print(f"Warning: Could not load AI coaching model: {e}")
            print("Will provide basic analysis instead")
            self.model = None
            self.tokenizer = None
    
    def analyze_match_data(self, coaching_data):
        """Analyze match data and generate coaching insights"""
        if not coaching_data:
            return "No match data available for analysis"
        
        analysis = self.prepare_match_analysis(coaching_data)
        
        if not self.model:
            return self.fallback_analysis(analysis)
        
        coaching_prompt = f"""You are an expert squash coach. Analyze this match data and provide specific coaching advice:

{analysis}

Provide:
1. Technical shot analysis
2. Movement pattern assessment  
3. Specific improvement areas
4. Training recommendations
5. Tactical suggestions

Keep response focused and actionable, and also make sure to point out the timestamps of the data used."""
        
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
            coaching_advice = response[len(coaching_prompt):].strip()
            return coaching_advice
            
        except Exception as e:
            print(f"Error generating AI coaching advice: {e}")
            return self.fallback_analysis(analysis)
    
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
        """Prepare structured analysis from coaching data"""
        shot_counts = defaultdict(int)
        activity_frames = 0
        total_frames = len(coaching_data)
        ball_speeds = []
        
        for data_point in coaching_data:
            if isinstance(data_point, dict):
                shot_type = data_point.get('shot_type', 'unknown')
                
                # Convert list to string if necessary to avoid unhashable type errors
                if isinstance(shot_type, list):
                    shot_type = ', '.join(map(str, shot_type))

                if shot_type and shot_type != 'unknown':
                    shot_counts[shot_type] += 1
                
                if data_point.get('match_active', False):
                    activity_frames += 1
                
                ball_speed = data_point.get('ball_speed', 0)
                if ball_speed > 0:
                    ball_speeds.append(ball_speed)
        
        activity_percentage = (activity_frames / total_frames * 100) if total_frames > 0 else 0
        most_common_shot = max(shot_counts.items(), key=lambda x: x[1]) if shot_counts else ("unknown", 0)
        shot_variety = len([shot for shot, count in shot_counts.items() if count > 1])
        avg_ball_speed = sum(ball_speeds) / len(ball_speeds) if ball_speeds else 0
        
        return f"""MATCH STATISTICS:
• Total data points analyzed: {total_frames}
• Active rally time: {activity_percentage:.1f}%
• Most frequent shot: {most_common_shot[0]} ({most_common_shot[1]} occurrences)
• Shot variety: {shot_variety} different shot types
• Shot distribution: {dict(shot_counts)}
• Average ball speed: {avg_ball_speed:.2f} pixels/frame

PERFORMANCE INDICATORS:
• Match intensity: {'High' if activity_percentage > 60 else 'Moderate' if activity_percentage > 30 else 'Low'}
• Shot consistency: {'Good' if shot_variety > 3 else 'Needs improvement'}
• Playing style: {most_common_shot[0]}-oriented
• Ball tracking quality: {'Good' if len(ball_speeds) > total_frames * 0.3 else 'Limited'}"""

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
