#!/usr/bin/env python3
"""
Enhanced Heatmap Generator for Squash Analysis
Comprehensive heatmap generation with improved data validation, visualization quality, and error handling.
"""

import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from collections import Counter
import ast
import json
from datetime import datetime
from functools import lru_cache
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnhancedHeatmapGenerator:
    """Enhanced heatmap generator with comprehensive visualization capabilities"""
    
    def __init__(self, output_dir="output", cache_dir="static/cache"):
        self.output_dir = Path(output_dir)
        self.cache_dir = Path(cache_dir)
        self.heatmap_dir = self.output_dir / "heatmaps"
        self.analysis_dir = self.output_dir / "analysis"
        
        # Create all necessary directories
        self._ensure_directories()
        
        # Court dimensions (in meters)
        self.court_length = 9.75
        self.court_width = 6.4
        self.court_height = 4.57
        
        # Setup visualization parameters
        plt.style.use('default')
        sns.set_palette("husl")
        
    def _ensure_directories(self):
        """Create all necessary directories"""
        for directory in [self.output_dir, self.cache_dir, self.heatmap_dir, self.analysis_dir]:
            directory.mkdir(parents=True, exist_ok=True)
            logger.info(f"Directory ensured: {directory}")
    
    def _safe_literal_eval(self, data_str, default=None):
        """Safely evaluate string data with fallback"""
        try:
            if pd.isna(data_str) or data_str == '' or data_str == '[]':
                return default
            result = ast.literal_eval(data_str)
            return result if result and result != [0, 0] and result != [0, 0, 0] else default
        except (ValueError, SyntaxError, TypeError):
            return default
    
    def _validate_position_data(self, positions):
        """Validate and clean position data"""
        if not positions:
            return np.array([])
        
        # Convert to numpy array and remove invalid positions
        positions = np.array(positions)
        if positions.size == 0:
            return np.array([])
        
        # Remove zero positions and outliers
        if positions.ndim == 2:
            valid_mask = ~np.all(positions == 0, axis=1)
            if positions.shape[1] >= 2:
                # Remove positions outside reasonable bounds
                valid_mask &= (positions[:, 0] >= 0) & (positions[:, 0] <= 1)
                valid_mask &= (positions[:, 1] >= 0) & (positions[:, 1] <= 1)
            positions = positions[valid_mask]
        
        return positions
    
    def create_enhanced_court_heatmap(self, df, player_column, title, player_id=1):
        """Create enhanced 2D court heatmap with improved visualization"""
        try:
            logger.info(f"Creating enhanced court heatmap for {title}")
            
            # Extract and validate position data
            positions = []
            for idx, row in df.iterrows():
                try:
                    # Try multiple data sources
                    keypoints = self._safe_literal_eval(row[player_column])
                    if keypoints and len(keypoints) >= 17:
                        # Use ankle positions (indices 15 and 16)
                        ankles = np.array([keypoints[15], keypoints[16]])
                        valid_ankles = ankles[~np.all(ankles == [0, 0], axis=1)]
                        if len(valid_ankles) > 0:
                            avg_pos = np.mean(valid_ankles, axis=0)
                            if avg_pos[0] > 0 and avg_pos[1] > 0:
                                positions.append(avg_pos)
                except Exception as e:
                    continue
            
            positions = self._validate_position_data(positions)
            
            if len(positions) == 0:
                logger.warning(f"No valid position data found for {title}")
                return self._create_fallback_heatmap(title, "No position data available")
            
            # Create figure with subplots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
            
            # Heatmap visualization
            heatmap, xedges, yedges = np.histogram2d(
                positions[:, 0], positions[:, 1], 
                bins=30, range=[[0, 1], [0, 1]]
            )
            
            # Plot heatmap
            im1 = ax1.imshow(heatmap.T, origin="lower", cmap="hot", aspect="auto", 
                           extent=[0, 1, 0, 1], alpha=0.8)
            ax1.set_title(f"{title} - Position Density Heatmap", fontsize=14, fontweight='bold')
            ax1.set_xlabel("Court Width (normalized)", fontsize=12)
            ax1.set_ylabel("Court Length (normalized)", fontsize=12)
            
            # Add court boundaries
            self._add_court_boundaries(ax1)
            
            # Add colorbar
            cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
            cbar1.set_label("Visit Frequency", fontsize=10)
            
            # Scatter plot with density coloring
            scatter = ax2.scatter(positions[:, 0], positions[:, 1], 
                                c=range(len(positions)), cmap='viridis', 
                                alpha=0.6, s=20, edgecolors='white', linewidth=0.5)
            ax2.set_title(f"{title} - Movement Pattern", fontsize=14, fontweight='bold')
            ax2.set_xlabel("Court Width (normalized)", fontsize=12)
            ax2.set_ylabel("Court Length (normalized)", fontsize=12)
            ax2.set_xlim(0, 1)
            ax2.set_ylim(0, 1)
            
            # Add court boundaries
            self._add_court_boundaries(ax2)
            
            # Add colorbar for time progression
            cbar2 = plt.colorbar(scatter, ax=ax2, fraction=0.046, pad=0.04)
            cbar2.set_label("Time Progression", fontsize=10)
            
            # Add statistics
            stats_text = self._generate_position_stats(positions)
            fig.text(0.02, 0.02, stats_text, fontsize=10, 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
            
            plt.tight_layout()
            
            # Save with multiple formats
            base_filename = f"{title.lower().replace(' ', '_')}_enhanced_heatmap"
            png_path = self.heatmap_dir / f"{base_filename}.png"
            plt.savefig(png_path, dpi=300, bbox_inches='tight')
            logger.info(f"Enhanced heatmap saved: {png_path}")
            
            plt.close()
            return str(png_path)
            
        except Exception as e:
            logger.error(f"Error creating enhanced court heatmap for {title}: {e}")
            return self._create_fallback_heatmap(title, f"Error: {str(e)}")
    
    def _add_court_boundaries(self, ax):
        """Add squash court boundary lines to the plot"""
        # Court outline
        court_rect = patches.Rectangle((0, 0), 1, 1, linewidth=2, 
                                     edgecolor='black', facecolor='none')
        ax.add_patch(court_rect)
        
        # Service boxes
        ax.axhline(y=0.3, color='gray', linestyle='--', alpha=0.7, linewidth=1)
        ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.7, linewidth=1)
        
        # T-position (approximate)
        ax.plot(0.5, 0.47, 'ro', markersize=8, label='T-Position')
        
        # Add legend
        ax.legend(loc='upper right', fontsize=10)
    
    def _generate_position_stats(self, positions):
        """Generate statistical summary of position data"""
        if len(positions) == 0:
            return "No position data available"
        
        center_x, center_y = np.mean(positions, axis=0)
        std_x, std_y = np.std(positions, axis=0)
        
        # Calculate court coverage
        coverage_x = (np.max(positions[:, 0]) - np.min(positions[:, 0])) * 100
        coverage_y = (np.max(positions[:, 1]) - np.min(positions[:, 1])) * 100
        
        return (f"Position Statistics:\n"
                f"• Total positions: {len(positions)}\n"
                f"• Center: ({center_x:.3f}, {center_y:.3f})\n"
                f"• Spread: ±{std_x:.3f}, ±{std_y:.3f}\n"
                f"• Court coverage: {coverage_x:.1f}% × {coverage_y:.1f}%")
    
    def create_3d_heatmap(self, df, player_column, title):
        """Create enhanced 3D heatmap with proper court visualization"""
        try:
            logger.info(f"Creating 3D heatmap for {title}")
            
            # Extract 3D position data
            positions = []
            for idx, row in df.iterrows():
                pos = self._safe_literal_eval(row[player_column])
                if pos and len(pos) >= 3 and not all(v == 0 for v in pos):
                    positions.append(pos)
            
            if len(positions) == 0:
                logger.warning(f"No valid 3D position data found for {title}")
                return self._create_fallback_heatmap(title, "No 3D position data available")
            
            positions = np.array(positions)
            
            # Create 3D plot
            fig = plt.figure(figsize=(15, 10))
            ax = fig.add_subplot(111, projection='3d')
            
            # Plot 3D court
            self._plot_3d_court(ax)
            
            # Create 3D heatmap
            scatter = ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
                               c=np.ones(len(positions)), cmap='hot', 
                               alpha=0.6, s=50, edgecolors='white', linewidth=0.5)
            
            # Customize plot
            ax.set_xlabel('Court Width (m)', fontsize=12)
            ax.set_ylabel('Court Length (m)', fontsize=12)
            ax.set_zlabel('Height (m)', fontsize=12)
            ax.set_title(f'3D Position Heatmap - {title}', fontsize=14, fontweight='bold')
            
            # Add colorbar
            cbar = plt.colorbar(scatter, ax=ax, shrink=0.5)
            cbar.set_label('Density', fontsize=10)
            
            # Save plot
            filename = f"3d_heatmap_{title.lower().replace(' ', '_')}.png"
            filepath = self.heatmap_dir / filename
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            logger.info(f"3D heatmap saved: {filepath}")
            
            plt.close()
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Error creating 3D heatmap for {title}: {e}")
            return self._create_fallback_heatmap(title, f"3D Error: {str(e)}")
    
    def _plot_3d_court(self, ax):
        """Plot 3D squash court boundaries"""
        # Court dimensions
        length, width, height = self.court_length, self.court_width, self.court_height
        
        # Floor rectangle
        floor_x = [0, width, width, 0, 0]
        floor_y = [0, 0, length, length, 0]
        floor_z = [0, 0, 0, 0, 0]
        ax.plot(floor_x, floor_y, floor_z, 'k-', linewidth=2)
        
        # Walls
        # Front wall
        ax.plot([0, width], [length, length], [0, 0], 'k-', linewidth=2)
        ax.plot([0, width], [length, length], [height, height], 'k-', linewidth=2)
        ax.plot([0, 0], [length, length], [0, height], 'k-', linewidth=2)
        ax.plot([width, width], [length, length], [0, height], 'k-', linewidth=2)
        
        # Side walls
        for x in [0, width]:
            ax.plot([x, x], [0, length], [0, 0], 'k-', linewidth=1)
            ax.plot([x, x], [0, length], [height, height], 'k-', linewidth=1)
            ax.plot([x, x], [0, 0], [0, height], 'k-', linewidth=1)
        
        # Set axis limits
        ax.set_xlim(0, width)
        ax.set_ylim(0, length)
        ax.set_zlim(0, height)
    
    def create_ball_trajectory_heatmap(self, df):
        """Create ball trajectory heatmap with bounce analysis"""
        try:
            logger.info("Creating ball trajectory heatmap")
            
            # Extract ball positions
            ball_positions = []
            for idx, row in df.iterrows():
                ball_pos = self._safe_literal_eval(row.get('Ball Position', '[]'))
                if ball_pos and len(ball_pos) >= 2:
                    ball_positions.append(ball_pos)
            
            if len(ball_positions) == 0:
                return self._create_fallback_heatmap("Ball Trajectory", "No ball position data")
            
            ball_positions = np.array(ball_positions)
            
            # Create figure with multiple views
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
            
            # 1. Trajectory heatmap
            heatmap, xedges, yedges = np.histogram2d(
                ball_positions[:, 0], ball_positions[:, 1], 
                bins=40, range=[[0, 1], [0, 1]]
            )
            
            im1 = ax1.imshow(heatmap.T, origin="lower", cmap="plasma", aspect="auto",
                           extent=[0, 1, 0, 1], alpha=0.8)
            ax1.set_title("Ball Position Heatmap", fontsize=14, fontweight='bold')
            ax1.set_xlabel("Court Width")
            ax1.set_ylabel("Court Length")
            self._add_court_boundaries(ax1)
            plt.colorbar(im1, ax=ax1)
            
            # 2. Trajectory path
            ax2.plot(ball_positions[:, 0], ball_positions[:, 1], 'o-', 
                    alpha=0.6, markersize=2, linewidth=0.5)
            ax2.set_title("Ball Trajectory Path", fontsize=14, fontweight='bold')
            ax2.set_xlabel("Court Width")
            ax2.set_ylabel("Court Length")
            ax2.set_xlim(0, 1)
            ax2.set_ylim(0, 1)
            self._add_court_boundaries(ax2)
            
            # 3. Speed analysis
            speeds = self._calculate_ball_speeds(ball_positions)
            if len(speeds) > 0:
                ax3.hist(speeds, bins=30, alpha=0.7, color='orange', edgecolor='black')
                ax3.set_title("Ball Speed Distribution", fontsize=14, fontweight='bold')
                ax3.set_xlabel("Speed (pixels/frame)")
                ax3.set_ylabel("Frequency")
                ax3.axvline(np.mean(speeds), color='red', linestyle='--', 
                           label=f'Mean: {np.mean(speeds):.2f}')
                ax3.legend()
            
            # 4. Position density by court zones
            self._plot_court_zone_analysis(ax4, ball_positions)
            
            plt.tight_layout()
            
            # Save plot
            filepath = self.heatmap_dir / "ball_trajectory_comprehensive.png"
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            logger.info(f"Ball trajectory heatmap saved: {filepath}")
            
            plt.close()
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Error creating ball trajectory heatmap: {e}")
            return self._create_fallback_heatmap("Ball Trajectory", f"Error: {str(e)}")
    
    def _calculate_ball_speeds(self, positions):
        """Calculate ball speeds between consecutive positions"""
        if len(positions) < 2:
            return []
        
        speeds = []
        for i in range(1, len(positions)):
            dx = positions[i][0] - positions[i-1][0]
            dy = positions[i][1] - positions[i-1][1]
            speed = np.sqrt(dx**2 + dy**2)
            speeds.append(speed)
        
        return speeds
    
    def _plot_court_zone_analysis(self, ax, positions):
        """Plot ball position analysis by court zones"""
        # Define court zones
        zones = {
            'Front Left': (positions[:, 0] < 0.5) & (positions[:, 1] > 0.7),
            'Front Right': (positions[:, 0] >= 0.5) & (positions[:, 1] > 0.7),
            'Mid Left': (positions[:, 0] < 0.5) & (positions[:, 1] >= 0.3) & (positions[:, 1] <= 0.7),
            'Mid Right': (positions[:, 0] >= 0.5) & (positions[:, 1] >= 0.3) & (positions[:, 1] <= 0.7),
            'Back Left': (positions[:, 0] < 0.5) & (positions[:, 1] < 0.3),
            'Back Right': (positions[:, 0] >= 0.5) & (positions[:, 1] < 0.3),
        }
        
        zone_counts = {zone: np.sum(mask) for zone, mask in zones.items()}
        
        # Create bar plot
        bars = ax.bar(zone_counts.keys(), zone_counts.values(), 
                     color=plt.cm.Set3(np.linspace(0, 1, len(zone_counts))))
        ax.set_title("Ball Distribution by Court Zone", fontsize=14, fontweight='bold')
        ax.set_ylabel("Number of Positions")
        ax.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}', ha='center', va='bottom')
    
    def _create_fallback_heatmap(self, title, error_message):
        """Create a fallback visualization when data is insufficient"""
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, f"Heatmap Generation Failed\n\n{title}\n\n{error_message}", 
                ha='center', va='center', fontsize=14, 
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightcoral", alpha=0.8))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(f"Fallback Visualization - {title}", fontsize=16, fontweight='bold')
        ax.axis('off')
        
        # Save fallback
        filename = f"fallback_{title.lower().replace(' ', '_')}.png"
        filepath = self.heatmap_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.warning(f"Fallback heatmap created: {filepath}")
        return str(filepath)
    
    def generate_comprehensive_heatmap_report(self, csv_path):
        """Generate comprehensive heatmap analysis report"""
        try:
            logger.info("Starting comprehensive heatmap generation")
            
            # Load data
            if not os.path.exists(csv_path):
                logger.error(f"CSV file not found: {csv_path}")
                return None
            
            df = pd.read_csv(csv_path)
            logger.info(f"Loaded {len(df)} rows of data")
            
            # Generate all heatmap types
            generated_files = []
            
            # Player heatmaps
            for player_id in [1, 2]:
                player_col = f"Player {player_id} Keypoints"
                if player_col in df.columns:
                    filepath = self.create_enhanced_court_heatmap(
                        df, player_col, f"Player {player_id}", player_id
                    )
                    generated_files.append(filepath)
            
            # 3D heatmaps
            for player_id in [1, 2]:
                rl_col = f"Player {player_id} RL World Position"
                if rl_col in df.columns:
                    filepath = self.create_3d_heatmap(df, rl_col, f"Player {player_id}")
                    generated_files.append(filepath)
            
            # Ball trajectory heatmap
            if "Ball Position" in df.columns:
                filepath = self.create_ball_trajectory_heatmap(df)
                generated_files.append(filepath)
            
            # Generate summary report
            report_path = self._generate_heatmap_summary_report(generated_files, df)
            
            logger.info(f"Comprehensive heatmap generation completed. Report: {report_path}")
            return report_path
            
        except Exception as e:
            logger.error(f"Error in comprehensive heatmap generation: {e}")
            return None
    
    def _generate_heatmap_summary_report(self, generated_files, df):
        """Generate a summary report of all heatmaps"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        report = f"""
ENHANCED HEATMAP ANALYSIS REPORT
Generated: {timestamp}
=====================================

DATA SUMMARY:
- Total frames analyzed: {len(df)}
- Total heatmaps generated: {len([f for f in generated_files if 'fallback' not in f])}
- Failed visualizations: {len([f for f in generated_files if 'fallback' in f])}

GENERATED HEATMAPS:
"""
        
        for filepath in generated_files:
            filename = os.path.basename(filepath)
            status = "✓ SUCCESS" if 'fallback' not in filename else "⚠ FALLBACK"
            report += f"- {filename} ... {status}\n"
        
        report += f"""

FILES LOCATION:
- Heatmaps directory: {self.heatmap_dir}
- Analysis directory: {self.analysis_dir}

RECOMMENDATIONS:
- Check data quality for failed visualizations
- Review position data validation
- Consider adding more data preprocessing steps
"""
        
        # Save report
        report_path = self.analysis_dir / f"heatmap_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(report_path, 'w') as f:
            f.write(report)
        
        logger.info(f"Summary report saved: {report_path}")
        return str(report_path)

if __name__ == "__main__":
    # Initialize and run comprehensive heatmap generation
    generator = EnhancedHeatmapGenerator()
    csv_path = "output/final.csv"
    
    if os.path.exists(csv_path):
        report_path = generator.generate_comprehensive_heatmap_report(csv_path)
        print(f"Enhanced heatmap generation completed!")
        print(f"Report available at: {report_path}")
    else:
        print(f"CSV file not found: {csv_path}")
        print("Please ensure the data file exists before running heatmap generation.")