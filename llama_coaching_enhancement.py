#!/usr/bin/env python3
"""
Llama 3.1-8B-Instruct Integration for Enhanced Squash Coaching
This module provides advanced AI-powered coaching insights using the Llama model
"""

import os
import json
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging
from typing import Dict, List, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LlamaCoachingEnhancer:
    """Enhanced coaching system using Llama 3.1-8B-Instruct model"""
    
    def __init__(self, model_name="meta-llama/Llama-3.1-8B-Instruct", device=None):
        """
        Initialize the Llama coaching enhancer
        
        Args:
            model_name: HuggingFace model name
            device: Device to run the model on (auto-detected if None)
        """
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.model = None
        self.is_initialized = False
        
        logger.info(f"Initializing Llama coaching enhancer on device: {self.device}")
        
    def initialize_model(self):
        """Initialize the Llama model and tokenizer"""
        try:
            # Check available GPU memory first
            if self.device == "cuda" and torch.cuda.is_available():
                available_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)
                available_gb = available_memory / 1e9
                logger.info(f"Available GPU memory: {available_gb:.2f} GB")
                
                # If less than 6GB available, skip loading
                if available_gb < 6.0:
                    logger.warning(f"Insufficient GPU memory ({available_gb:.2f} GB) for Llama model. Need at least 6GB.")
                    self.is_initialized = False
                    return
            
            logger.info("Loading Llama tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            logger.info("Loading Llama model...")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                low_cpu_mem_usage=True
            )
            
            if self.device == "cpu":
                self.model = self.model.to(self.device)
            
            self.is_initialized = True
            logger.info("Llama model initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Llama model: {e}")
            self.is_initialized = False
    
    def generate_coaching_insight(self, prompt: str, max_new_tokens: int = 200) -> str:
        """
        Generate coaching insight using Llama model
        
        Args:
            prompt: The coaching prompt
            max_new_tokens: Maximum number of tokens to generate
            
        Returns:
            Generated coaching insight
        """
        if not self.is_initialized:
            logger.warning("Model not initialized, attempting to initialize...")
            self.initialize_model()
            if not self.is_initialized:
                return "Model initialization failed. Using fallback response."
        
        try:
            messages = [
                {"role": "user", "content": prompt}
            ]
            
            inputs = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to(self.model.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs, 
                    max_new_tokens=max_new_tokens,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            response = self.tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[-1]:], 
                skip_special_tokens=True
            )
            
            return response.strip()
            
        except Exception as e:
            logger.error(f"Error generating coaching insight: {e}")
            return f"Error generating insight: {str(e)}"
    
    def analyze_shot_patterns(self, shot_data: List[Dict]) -> Dict[str, Any]:
        """
        Analyze shot patterns using Llama model
        
        Args:
            shot_data: List of shot dictionaries with trajectory and metadata
            
        Returns:
            Analysis results with insights and recommendations
        """
        if not shot_data:
            return {"insights": "No shot data available for analysis"}
        
        # Prepare shot summary
        shot_types = [shot.get('shot_type', 'unknown') for shot in shot_data]
        shot_counts = {}
        for shot_type in shot_types:
            shot_counts[shot_type] = shot_counts.get(shot_type, 0) + 1
        
        # Create analysis prompt
        prompt = f"""
        As an expert squash coach, analyze the following shot patterns from a player's session:
        
        Shot Distribution:
        {json.dumps(shot_counts, indent=2)}
        
        Total Shots: {len(shot_data)}
        
        Please provide:
        1. Pattern analysis and observations
        2. Strengths and areas for improvement
        3. Specific coaching recommendations
        4. Tactical suggestions for match play
        
        Focus on practical, actionable advice that a player can implement immediately.
        """
        
        insight = self.generate_coaching_insight(prompt, max_new_tokens=300)
        
        return {
            "shot_distribution": shot_counts,
            "total_shots": len(shot_data),
            "ai_analysis": insight,
            "recommendations": self._extract_recommendations(insight)
        }
    
    def analyze_player_movement(self, player_positions: Dict, court_dimensions: tuple) -> Dict[str, Any]:
        """
        Analyze player movement patterns using Llama model
        
        Args:
            player_positions: Dictionary of player position data
            court_dimensions: Court width and height
            
        Returns:
            Movement analysis with insights
        """
        if not player_positions:
            return {"insights": "No player position data available"}
        
        # Calculate movement statistics
        movement_stats = {}
        for player_id, positions in player_positions.items():
            if positions and len(positions) > 1:
                total_distance = 0
                for i in range(1, len(positions)):
                    p1, p2 = positions[i-1], positions[i]
                    distance = ((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)**0.5
                    total_distance += distance
                
                movement_stats[f"player_{player_id}"] = {
                    "total_distance": total_distance,
                    "positions_count": len(positions),
                    "average_speed": total_distance / len(positions) if len(positions) > 0 else 0
                }
        
        prompt = f"""
        As a squash movement specialist, analyze the following player movement data:
        
        Court Dimensions: {court_dimensions}
        Movement Statistics:
        {json.dumps(movement_stats, indent=2)}
        
        Please provide:
        1. Movement efficiency analysis
        2. Court coverage assessment
        3. Positioning recommendations
        4. Fitness and conditioning insights
        5. Specific drills to improve movement
        
        Focus on practical movement coaching advice.
        """
        
        insight = self.generate_coaching_insight(prompt, max_new_tokens=250)
        
        return {
            "movement_statistics": movement_stats,
            "ai_analysis": insight,
            "court_dimensions": court_dimensions
        }
    
    def generate_match_report(self, match_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate comprehensive match report using Llama model
        
        Args:
            match_data: Dictionary containing match statistics and data
            
        Returns:
            Comprehensive match report
        """
        prompt = f"""
        As an expert squash analyst, create a comprehensive match report based on the following data:
        
        Match Data:
        {json.dumps(match_data, indent=2)}
        
        Please provide:
        1. Executive summary of the match
        2. Key performance indicators
        3. Tactical analysis
        4. Technical assessment
        5. Areas for improvement
        6. Positive aspects to maintain
        7. Specific training recommendations
        
        Make the report professional, detailed, and actionable for coaching purposes.
        """
        
        report = self.generate_coaching_insight(prompt, max_new_tokens=400)
        
        return {
            "match_report": report,
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model_used": self.model_name
        }
    
    def analyze_rally_patterns(self, rally_data: List[Dict]) -> Dict[str, Any]:
        """
        Analyze rally patterns and tactics using Llama model
        
        Args:
            rally_data: List of rally dictionaries with shot sequences
            
        Returns:
            Rally analysis with tactical insights
        """
        if not rally_data:
            return {"insights": "No rally data available for analysis"}
        
        # Analyze rally statistics
        rally_lengths = [len(rally.get('shots', [])) for rally in rally_data]
        avg_rally_length = sum(rally_lengths) / len(rally_lengths) if rally_lengths else 0
        
        # Count shot types in rallies
        all_shots = []
        for rally in rally_data:
            all_shots.extend([shot.get('shot_type', 'unknown') for shot in rally.get('shots', [])])
        
        shot_distribution = {}
        for shot_type in all_shots:
            shot_distribution[shot_type] = shot_distribution.get(shot_type, 0) + 1
        
        prompt = f"""
        As a squash tactics expert, analyze the following rally patterns:
        
        Rally Statistics:
        - Total Rallies: {len(rally_data)}
        - Average Rally Length: {avg_rally_length:.1f} shots
        - Shot Distribution: {json.dumps(shot_distribution, indent=2)}
        
        Please provide:
        1. Rally pattern analysis
        2. Tactical effectiveness assessment
        3. Shot selection evaluation
        4. Rally construction insights
        5. Tactical recommendations
        6. Pressure handling analysis
        
        Focus on tactical coaching insights and match strategy.
        """
        
        insight = self.generate_coaching_insight(prompt, max_new_tokens=350)
        
        return {
            "rally_statistics": {
                "total_rallies": len(rally_data),
                "average_length": avg_rally_length,
                "shot_distribution": shot_distribution
            },
            "ai_analysis": insight,
            "tactical_recommendations": self._extract_recommendations(insight)
        }
    
    def _extract_recommendations(self, text: str) -> List[str]:
        """
        Extract specific recommendations from AI-generated text
        
        Args:
            text: AI-generated coaching text
            
        Returns:
            List of specific recommendations
        """
        recommendations = []
        lines = text.split('\n')
        
        for line in lines:
            line = line.strip()
            if any(keyword in line.lower() for keyword in ['recommend', 'suggest', 'should', 'try', 'practice', 'focus']):
                if line and not line.startswith('#'):
                    recommendations.append(line)
        
        return recommendations[:5]  # Return top 5 recommendations
    
    def generate_personalized_coaching_plan(self, player_profile: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate personalized coaching plan using Llama model
        
        Args:
            player_profile: Dictionary containing player data and preferences
            
        Returns:
            Personalized coaching plan
        """
        prompt = f"""
        As a professional squash coach, create a personalized coaching plan for this player:
        
        Player Profile:
        {json.dumps(player_profile, indent=2)}
        
        Please create a comprehensive coaching plan including:
        1. Technical development priorities
        2. Tactical training focus areas
        3. Physical conditioning recommendations
        4. Mental game strategies
        5. Specific drills and exercises
        6. Progress tracking methods
        7. Short-term and long-term goals
        
        Make the plan specific, measurable, and tailored to the player's profile.
        """
        
        plan = self.generate_coaching_insight(prompt, max_new_tokens=500)
        
        return {
            "coaching_plan": plan,
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "player_profile": player_profile
        }

# Global instance for easy access
llama_enhancer = None

def initialize_llama_enhancement():
    """Initialize the global Llama enhancer instance"""
    global llama_enhancer
    if llama_enhancer is None:
        llama_enhancer = LlamaCoachingEnhancer()
        llama_enhancer.initialize_model()
    return llama_enhancer

def get_llama_enhancer():
    """Get the global Llama enhancer instance"""
    global llama_enhancer
    if llama_enhancer is None:
        initialize_llama_enhancement()
    return llama_enhancer
