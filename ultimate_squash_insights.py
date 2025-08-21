#!/usr/bin/env python3
"""
Ultimate Squash Insights Generator
=================================

This script provides a comprehensive summary of all generated squash coaching insights.
Run this after completing your video analysis to get the most complete overview.
"""

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import sys

def display_banner():
    """Display an impressive banner"""
    print("🎾" * 35)
    print("🎾" + " " * 33 + "🎾")
    print("🎾    ULTIMATE SQUASH INSIGHTS     🎾")
    print("🎾     Complete Analysis Report    🎾")
    print("🎾" + " " * 33 + "🎾")
    print("🎾" * 35)
    print()

def check_available_data():
    """Check what analysis data is available"""
    data_files = {
        'Enhanced Coaching Report': 'output/enhanced_autonomous_coaching_report.txt',
        'Ultimate Analysis': 'output/ultimate_coaching_analysis.json',
        'Enhanced Coaching Data': 'output/enhanced_coaching_data.json',
        'Final CSV Data': 'output/final.csv',
        'Shot Analysis Report': 'output/shot_analysis_report.txt',
        'Shot Logs': 'output/shots_log.jsonl',
        'Bounce Analysis': 'output/bounce_analysis.jsonl',
        'Graphics Directory': 'output/graphics/',
        'Enhanced Shots': 'output/enhanced_shots/',
        'ReID Analysis': 'output/reid_analysis_report.txt'
    }
    
    available = {}
    for name, path in data_files.items():
        if os.path.exists(path):
            available[name] = path
            if os.path.isfile(path):
                size = os.path.getsize(path)
                available[name + "_size"] = f"{size:,} bytes"
            else:
                # Count files in directory
                try:
                    file_count = len([f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))])
                    available[name + "_count"] = f"{file_count} files"
                except:
                    available[name + "_count"] = "Directory exists"
    
    return available

def display_data_summary(available_data):
    """Display summary of available data"""
    print("📊 AVAILABLE ANALYSIS DATA:")
    print("=" * 50)
    
    categories = {
        '🤖 AI COACHING REPORTS': ['Enhanced Coaching Report'],
        '📈 ANALYSIS DATA': ['Ultimate Analysis', 'Enhanced Coaching Data', 'Final CSV Data'],
        '🎯 SHOT ANALYSIS': ['Shot Analysis Report', 'Shot Logs', 'Bounce Analysis', 'Enhanced Shots'],
        '🎨 VISUALIZATIONS': ['Graphics Directory'],
        '👥 PLAYER TRACKING': ['ReID Analysis']
    }
    
    for category, files in categories.items():
        print(f"\n{category}")
        print("-" * 30)
        for file_type in files:
            if file_type in available_data:
                path = available_data[file_type]
                size_info = available_data.get(file_type + "_size", available_data.get(file_type + "_count", ""))
                print(f"   ✅ {file_type}: {size_info}")
                print(f"      📁 {path}")
            else:
                print(f"   ❌ {file_type}: Not available")

def display_ultimate_analysis():
    """Display ultimate analysis if available"""
    ultimate_path = 'output/ultimate_coaching_analysis.json'
    if not os.path.exists(ultimate_path):
        print("\n❌ Ultimate analysis not available")
        return
    
    try:
        with open(ultimate_path, 'r') as f:
            ultimate_data = json.load(f)
        
        print("\n🚀 ULTIMATE ANALYSIS INSIGHTS:")
        print("=" * 50)
        
        # Display performance insights
        perf_insights = ultimate_data.get('performance_insights', {})
        if perf_insights:
            overall_rating = perf_insights.get('overall_rating', 0)
            print(f"🏆 Overall Performance Rating: {overall_rating*100:.1f}%")
            
            strengths = perf_insights.get('key_strengths', [])
            if strengths:
                print(f"💪 Key Strengths: {', '.join(strengths)}")
            
            improvements = perf_insights.get('priority_improvements', [])
            if improvements:
                print(f"🎯 Priority Improvements: {', '.join(improvements)}")
        
        # Display improvement roadmap
        roadmap = ultimate_data.get('improvement_roadmap', {})
        if roadmap:
            immediate = roadmap.get('immediate_focus', [])
            if immediate:
                print(f"\n⚡ IMMEDIATE FOCUS AREAS:")
                for item in immediate[:3]:  # Top 3
                    print(f"   • {item.get('area', 'Unknown')}: {item.get('target', 'No target set')}")
        
        # Display advanced patterns
        patterns = ultimate_data.get('advanced_patterns', {})
        if patterns:
            attacking = patterns.get('attacking_sequences', [])
            defensive = patterns.get('defensive_patterns', [])
            print(f"\n🎯 TACTICAL PATTERNS DETECTED:")
            print(f"   • Attacking sequences: {len(attacking)}")
            print(f"   • Defensive patterns: {len(defensive)}")
        
    except Exception as e:
        print(f"\n⚠️ Error reading ultimate analysis: {e}")

def display_shot_statistics():
    """Display shot statistics from CSV data"""
    csv_path = 'output/final.csv'
    if not os.path.exists(csv_path):
        print("\n❌ Shot statistics not available (final.csv missing)")
        return
    
    try:
        df = pd.read_csv(csv_path)
        print("\n📊 SHOT STATISTICS:")
        print("=" * 50)
        print(f"Total data points: {len(df):,}")
        
        if 'shot_type' in df.columns:
            shot_counts = df['shot_type'].value_counts()
            print(f"Shot type distribution:")
            for shot_type, count in shot_counts.head(5).items():
                percentage = (count / len(df)) * 100
                print(f"   • {shot_type}: {count} ({percentage:.1f}%)")
        
        if 'ball_x' in df.columns and 'ball_y' in df.columns:
            ball_positions = df[['ball_x', 'ball_y']].dropna()
            print(f"Ball positions tracked: {len(ball_positions):,}")
            
            if len(ball_positions) > 0:
                court_coverage_x = (ball_positions['ball_x'].max() - ball_positions['ball_x'].min())
                court_coverage_y = (ball_positions['ball_y'].max() - ball_positions['ball_y'].min())
                print(f"Court coverage (X): {court_coverage_x:.1f} pixels")
                print(f"Court coverage (Y): {court_coverage_y:.1f} pixels")
                
    except Exception as e:
        print(f"\n⚠️ Error reading shot statistics: {e}")

def display_graphics_summary():
    """Display summary of generated visualizations"""
    graphics_dir = 'output/graphics'
    if not os.path.exists(graphics_dir):
        print("\n❌ Graphics not available")
        return
    
    try:
        graphics_files = []
        for file in os.listdir(graphics_dir):
            if file.endswith(('.png', '.jpg', '.jpeg', '.html', '.svg')):
                file_path = os.path.join(graphics_dir, file)
                file_size = os.path.getsize(file_path)
                graphics_files.append((file, file_size))
        
        print(f"\n🎨 VISUALIZATIONS GENERATED:")
        print("=" * 50)
        print(f"Total graphics files: {len(graphics_files)}")
        
        for file, size in graphics_files:
            file_type = file.split('.')[-1].upper()
            print(f"   📊 {file} ({file_type}, {size:,} bytes)")
        
    except Exception as e:
        print(f"\n⚠️ Error reading graphics directory: {e}")

def display_coaching_highlights():
    """Display key coaching highlights from reports"""
    enhanced_report_path = 'output/enhanced_autonomous_coaching_report.txt'
    if os.path.exists(enhanced_report_path):
        try:
            with open(enhanced_report_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            print(f"\n🤖 AI COACHING HIGHLIGHTS:")
            print("=" * 50)
            
            # Extract key insights (simple keyword search)
            key_phrases = [
                "TECHNICAL ANALYSIS",
                "STRATEGIC INSIGHTS", 
                "PHYSICAL ASSESSMENT",
                "IMPROVEMENT RECOMMENDATIONS",
                "Priority areas",
                "Focus on"
            ]
            
            lines = content.split('\n')
            highlights = []
            
            for line in lines:
                for phrase in key_phrases:
                    if phrase.lower() in line.lower() and len(line.strip()) > 10:
                        highlights.append(line.strip())
                        break
            
            for highlight in highlights[:8]:  # Top 8 highlights
                if highlight and not highlight.startswith('='):
                    print(f"   • {highlight[:80]}{'...' if len(highlight) > 80 else ''}")
                    
        except Exception as e:
            print(f"\n⚠️ Error reading coaching report: {e}")
    else:
        print("\n❌ Enhanced coaching report not available")

def display_next_steps():
    """Display recommended next steps"""
    print(f"\n🎯 RECOMMENDED NEXT STEPS:")
    print("=" * 50)
    print("1. 📖 Review the enhanced coaching report for detailed insights")
    print("   📁 output/enhanced_autonomous_coaching_report.txt")
    print()
    print("2. 📊 Explore the visual analytics")
    print("   📁 output/graphics/ directory")
    print()
    print("3. 📈 Analyze your shot patterns")
    print("   📁 output/shots_log.jsonl")
    print("   📁 output/shot_analysis_report.txt")
    print()
    print("4. 🔍 Check ultimate analysis for deep insights")
    print("   📁 output/ultimate_coaching_analysis.json")
    print()
    print("5. 🎥 Review annotated video")
    print("   📁 output/annotated.mp4")
    print()
    print("6. 📋 Implement training recommendations")
    print("   • Focus on identified improvement areas")
    print("   • Practice suggested drills")
    print("   • Track progress over time")

def display_performance_metrics():
    """Display key performance metrics"""
    try:
        # Try to load enhanced coaching data
        enhanced_data_path = 'output/enhanced_coaching_data.json'
        if os.path.exists(enhanced_data_path):
            with open(enhanced_data_path, 'r') as f:
                enhanced_data = json.load(f)
            
            print(f"\n📈 KEY PERFORMANCE METRICS:")
            print("=" * 50)
            
            total_points = len(enhanced_data)
            print(f"Total analysis points: {total_points:,}")
            
            # Count different types of analysis
            enhanced_points = sum(1 for point in enhanced_data if 'enhanced_analysis' in point)
            shot_events = sum(len(point.get('shot_event_details', [])) for point in enhanced_data)
            ball_physics = sum(1 for point in enhanced_data if 'ball_physics' in point)
            
            print(f"Enhanced analysis points: {enhanced_points:,}")
            print(f"Shot events detected: {shot_events:,}")
            print(f"Physics analysis points: {ball_physics:,}")
            
            # Calculate percentages
            if total_points > 0:
                enhanced_pct = (enhanced_points / total_points) * 100
                physics_pct = (ball_physics / total_points) * 100
                print(f"Enhanced coverage: {enhanced_pct:.1f}%")
                print(f"Physics coverage: {physics_pct:.1f}%")
                
    except Exception as e:
        print(f"\n⚠️ Error reading performance metrics: {e}")

def main():
    """Main function to display all insights"""
    display_banner()
    
    # Check available data
    available_data = check_available_data()
    
    if not available_data:
        print("❌ No analysis data found!")
        print("Make sure you have run ef.py on a video file first.")
        return
    
    # Display all sections
    display_data_summary(available_data)
    display_performance_metrics()
    display_ultimate_analysis()
    display_shot_statistics()
    display_graphics_summary()
    display_coaching_highlights()
    display_next_steps()
    
    print(f"\n🎾 ANALYSIS COMPLETE! 🎾")
    print("=" * 50)
    print(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("Your comprehensive squash analysis is ready for review!")

if __name__ == "__main__":
    main()
