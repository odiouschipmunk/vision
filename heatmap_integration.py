#!/usr/bin/env python3
"""
Heatmap Integration Module for get_data.py
Enhanced real-time heatmap generation during video processing
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from collections import deque
import json
from datetime import datetime

class RealTimeHeatmapGenerator:
    """Real-time heatmap generation during video processing"""
    
    def __init__(self, frame_width=640, frame_height=360, output_dir="output"):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.output_dir = output_dir
        
        # Initialize heatmap canvases
        self.player1_heatmap = np.zeros((frame_height, frame_width), dtype=np.float32)
        self.player2_heatmap = np.zeros((frame_height, frame_width), dtype=np.float32)
        self.ball_heatmap = np.zeros((frame_height, frame_width), dtype=np.float32)
        self.combined_heatmap = np.zeros((frame_height, frame_width), dtype=np.float32)
        
        # Position history for trajectory analysis
        self.player1_positions = deque(maxlen=1000)
        self.player2_positions = deque(maxlen=1000)
        self.ball_positions = deque(maxlen=500)
        
        # Statistics tracking
        self.frame_count = 0
        self.heatmap_stats = {
            "player1_coverage": 0,
            "player2_coverage": 0,
            "ball_coverage": 0,
            "total_positions": 0
        }
        
        # Ensure output directories
        self._ensure_directories()
    
    def _ensure_directories(self):
        """Create necessary output directories"""
        dirs = [
            os.path.join(self.output_dir, "heatmaps"),
            os.path.join(self.output_dir, "heatmaps", "real_time"),
            os.path.join(self.output_dir, "heatmaps", "final")
        ]
        for directory in dirs:
            os.makedirs(directory, exist_ok=True)
    
    def update_player_heatmap(self, players, frame_count):
        """Update player heatmaps with current positions"""
        try:
            if players.get(1) and players.get(1).get_latest_pose() is not None:
                # Get player 1 ankle positions
                pose = players.get(1).get_latest_pose()
                if hasattr(pose, 'xyn') and len(pose.xyn[0]) > 16:
                    left_ankle = pose.xyn[0][15]  # Left ankle
                    right_ankle = pose.xyn[0][16]  # Right ankle
                    
                    # Calculate average ankle position
                    if left_ankle[0] > 0 and left_ankle[1] > 0 and right_ankle[0] > 0 and right_ankle[1] > 0:
                        avg_x = int((left_ankle[0] + right_ankle[0]) / 2 * self.frame_width)
                        avg_y = int((left_ankle[1] + right_ankle[1]) / 2 * self.frame_height)
                        
                        # Ensure coordinates are within bounds
                        avg_x = max(0, min(avg_x, self.frame_width - 1))
                        avg_y = max(0, min(avg_y, self.frame_height - 1))
                        
                        # Update heatmap with Gaussian distribution
                        self._add_gaussian_point(self.player1_heatmap, avg_x, avg_y, intensity=1.0)
                        self.player1_positions.append((avg_x, avg_y, frame_count))
            
            if players.get(2) and players.get(2).get_latest_pose() is not None:
                # Get player 2 ankle positions
                pose = players.get(2).get_latest_pose()
                if hasattr(pose, 'xyn') and len(pose.xyn[0]) > 16:
                    left_ankle = pose.xyn[0][15]  # Left ankle
                    right_ankle = pose.xyn[0][16]  # Right ankle
                    
                    # Calculate average ankle position
                    if left_ankle[0] > 0 and left_ankle[1] > 0 and right_ankle[0] > 0 and right_ankle[1] > 0:
                        avg_x = int((left_ankle[0] + right_ankle[0]) / 2 * self.frame_width)
                        avg_y = int((left_ankle[1] + right_ankle[1]) / 2 * self.frame_height)
                        
                        # Ensure coordinates are within bounds
                        avg_x = max(0, min(avg_x, self.frame_width - 1))
                        avg_y = max(0, min(avg_y, self.frame_height - 1))
                        
                        # Update heatmap with Gaussian distribution
                        self._add_gaussian_point(self.player2_heatmap, avg_x, avg_y, intensity=1.0)
                        self.player2_positions.append((avg_x, avg_y, frame_count))
                        
        except Exception as e:
            print(f"Error updating player heatmap: {e}")
    
    def update_ball_heatmap(self, ball_position, frame_count):
        """Update ball heatmap with current position"""
        try:
            if ball_position and len(ball_position) >= 2:
                ball_x, ball_y = ball_position[0], ball_position[1]
                
                if ball_x > 0 and ball_y > 0:
                    # Ensure coordinates are within bounds
                    ball_x = max(0, min(int(ball_x), self.frame_width - 1))
                    ball_y = max(0, min(int(ball_y), self.frame_height - 1))
                    
                    # Update heatmap with Gaussian distribution
                    self._add_gaussian_point(self.ball_heatmap, ball_x, ball_y, intensity=0.5)
                    self.ball_positions.append((ball_x, ball_y, frame_count))
                    
        except Exception as e:
            print(f"Error updating ball heatmap: {e}")
    
    def _add_gaussian_point(self, heatmap, x, y, intensity=1.0, radius=20):
        """Add a Gaussian distribution point to the heatmap"""
        try:
            # Create Gaussian kernel
            kernel_size = radius * 2 + 1
            kernel = cv2.getGaussianKernel(kernel_size, radius / 3)
            kernel_2d = kernel @ kernel.T * intensity
            
            # Calculate bounds
            x_start = max(0, x - radius)
            x_end = min(heatmap.shape[1], x + radius + 1)
            y_start = max(0, y - radius)
            y_end = min(heatmap.shape[0], y + radius + 1)
            
            # Calculate kernel bounds
            kx_start = radius - (x - x_start)
            kx_end = kx_start + (x_end - x_start)
            ky_start = radius - (y - y_start)
            ky_end = ky_start + (y_end - y_start)
            
            # Add Gaussian to heatmap
            if x_end > x_start and y_end > y_start:
                heatmap[y_start:y_end, x_start:x_end] += kernel_2d[ky_start:ky_end, kx_start:kx_end]
                
        except Exception as e:
            print(f"Error adding Gaussian point: {e}")
    
    def generate_enhanced_heatmap_overlay(self, base_frame):
        """Generate enhanced heatmap overlay for the current frame"""
        try:
            # Normalize heatmaps
            p1_norm = cv2.normalize(self.player1_heatmap, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            p2_norm = cv2.normalize(self.player2_heatmap, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            ball_norm = cv2.normalize(self.ball_heatmap, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            
            # Apply colormaps
            p1_colored = cv2.applyColorMap(p1_norm, cv2.COLORMAP_OCEAN)
            p2_colored = cv2.applyColorMap(p2_norm, cv2.COLORMAP_HOT)
            ball_colored = cv2.applyColorMap(ball_norm, cv2.COLORMAP_PLASMA)
            
            # Combine heatmaps
            combined = cv2.addWeighted(p1_colored, 0.3, p2_colored, 0.3, 0)
            combined = cv2.addWeighted(combined, 0.8, ball_colored, 0.2, 0)
            
            # Overlay on base frame
            result = cv2.addWeighted(base_frame, 0.7, combined, 0.3, 0)
            
            # Add legend
            self._add_heatmap_legend(result)
            
            return result
            
        except Exception as e:
            print(f"Error generating heatmap overlay: {e}")
            return base_frame
    
    def _add_heatmap_legend(self, frame):
        """Add legend to heatmap overlay"""
        try:
            # Create legend area
            legend_height = 80
            legend_width = 300
            legend = np.zeros((legend_height, legend_width, 3), dtype=np.uint8)
            
            # Add legend items
            cv2.putText(legend, "Player 1 (Blue)", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            cv2.putText(legend, "Player 2 (Red)", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.putText(legend, "Ball (Purple)", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
            
            # Overlay legend on frame
            y_start = frame.shape[0] - legend_height - 10
            x_start = 10
            frame[y_start:y_start + legend_height, x_start:x_start + legend_width] = \
                cv2.addWeighted(frame[y_start:y_start + legend_height, x_start:x_start + legend_width], 
                              0.3, legend, 0.7, 0)
                              
        except Exception as e:
            print(f"Error adding legend: {e}")
    
    def save_periodic_heatmaps(self, frame_count, save_interval=100):
        """Save heatmaps periodically during processing"""
        if frame_count % save_interval == 0:
            try:
                timestamp = datetime.now().strftime("%H%M%S")
                
                # Save individual heatmaps
                self._save_single_heatmap(
                    self.player1_heatmap, 
                    f"player1_heatmap_frame_{frame_count}_{timestamp}.png"
                )
                self._save_single_heatmap(
                    self.player2_heatmap, 
                    f"player2_heatmap_frame_{frame_count}_{timestamp}.png"
                )
                self._save_single_heatmap(
                    self.ball_heatmap, 
                    f"ball_heatmap_frame_{frame_count}_{timestamp}.png"
                )
                
                # Save statistics
                self._save_heatmap_statistics(frame_count)
                
            except Exception as e:
                print(f"Error saving periodic heatmaps: {e}")
    
    def _save_single_heatmap(self, heatmap, filename):
        """Save a single heatmap with enhanced visualization"""
        try:
            # Normalize and apply colormap
            normalized = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            colored = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
            
            # Save to real-time directory
            filepath = os.path.join(self.output_dir, "heatmaps", "real_time", filename)
            cv2.imwrite(filepath, colored)
            
        except Exception as e:
            print(f"Error saving heatmap {filename}: {e}")
    
    def save_final_comprehensive_heatmaps(self):
        """Save final comprehensive heatmaps at the end of processing"""
        try:
            print("Generating final comprehensive heatmaps...")
            
            # Create matplotlib-based final heatmaps
            self._create_matplotlib_heatmap(
                self.player1_heatmap, 
                "Player 1 Movement Heatmap",
                "player1_final_heatmap.png"
            )
            self._create_matplotlib_heatmap(
                self.player2_heatmap, 
                "Player 2 Movement Heatmap",
                "player2_final_heatmap.png"
            )
            self._create_matplotlib_heatmap(
                self.ball_heatmap, 
                "Ball Trajectory Heatmap",
                "ball_final_heatmap.png"
            )
            
            # Create combined analysis
            self._create_combined_analysis_plot()
            
            # Save trajectory analysis
            self._save_trajectory_analysis()
            
            print("Final heatmaps generated successfully!")
            
        except Exception as e:
            print(f"Error saving final heatmaps: {e}")
    
    def _create_matplotlib_heatmap(self, heatmap_data, title, filename):
        """Create enhanced matplotlib heatmap"""
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
            
            # Heatmap visualization
            im1 = ax1.imshow(heatmap_data, cmap='hot', aspect='auto', origin='lower')
            ax1.set_title(f"{title} - Density", fontsize=14, fontweight='bold')
            ax1.set_xlabel("Court Width (pixels)")
            ax1.set_ylabel("Court Length (pixels)")
            plt.colorbar(im1, ax=ax1, label="Visit Frequency")
            
            # Contour visualization
            contour = ax2.contour(heatmap_data, levels=20, cmap='plasma')
            ax2.clabel(contour, inline=True, fontsize=8)
            ax2.set_title(f"{title} - Contours", fontsize=14, fontweight='bold')
            ax2.set_xlabel("Court Width (pixels)")
            ax2.set_ylabel("Court Length (pixels)")
            
            plt.tight_layout()
            
            # Save
            filepath = os.path.join(self.output_dir, "heatmaps", "final", filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"Error creating matplotlib heatmap: {e}")
    
    def _create_combined_analysis_plot(self):
        """Create comprehensive combined analysis plot"""
        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
            
            # Combined heatmap
            combined = self.player1_heatmap + self.player2_heatmap + self.ball_heatmap
            im1 = ax1.imshow(combined, cmap='hot', aspect='auto', origin='lower')
            ax1.set_title("Combined Activity Heatmap", fontsize=14, fontweight='bold')
            plt.colorbar(im1, ax=ax1, label="Activity Level")
            
            # Player comparison
            comparison = self.player1_heatmap - self.player2_heatmap
            im2 = ax2.imshow(comparison, cmap='RdBu', aspect='auto', origin='lower')
            ax2.set_title("Player Movement Comparison", fontsize=14, fontweight='bold')
            plt.colorbar(im2, ax=ax2, label="P1 (Red) vs P2 (Blue)")
            
            # Movement intensity over time
            if self.player1_positions and self.player2_positions:
                p1_times = [pos[2] for pos in self.player1_positions]
                p2_times = [pos[2] for pos in self.player2_positions]
                
                ax3.hist(p1_times, bins=50, alpha=0.7, label='Player 1', color='blue')
                ax3.hist(p2_times, bins=50, alpha=0.7, label='Player 2', color='red')
                ax3.set_title("Movement Activity Over Time", fontsize=14, fontweight='bold')
                ax3.set_xlabel("Frame Number")
                ax3.set_ylabel("Position Updates")
                ax3.legend()
            
            # Coverage statistics
            self._plot_coverage_statistics(ax4)
            
            plt.tight_layout()
            
            # Save
            filepath = os.path.join(self.output_dir, "heatmaps", "final", "comprehensive_analysis.png")
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"Error creating combined analysis: {e}")
    
    def _plot_coverage_statistics(self, ax):
        """Plot coverage and movement statistics"""
        try:
            # Calculate coverage areas
            p1_coverage = np.sum(self.player1_heatmap > 0) / self.player1_heatmap.size * 100
            p2_coverage = np.sum(self.player2_heatmap > 0) / self.player2_heatmap.size * 100
            ball_coverage = np.sum(self.ball_heatmap > 0) / self.ball_heatmap.size * 100
            
            # Plot coverage comparison
            categories = ['Player 1', 'Player 2', 'Ball']
            coverages = [p1_coverage, p2_coverage, ball_coverage]
            
            bars = ax.bar(categories, coverages, color=['blue', 'red', 'green'], alpha=0.7)
            ax.set_title("Court Coverage Comparison", fontsize=14, fontweight='bold')
            ax.set_ylabel("Coverage Percentage (%)")
            
            # Add value labels on bars
            for bar, coverage in zip(bars, coverages):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{coverage:.1f}%', ha='center', va='bottom')
            
            # Update statistics
            self.heatmap_stats.update({
                "player1_coverage": p1_coverage,
                "player2_coverage": p2_coverage,
                "ball_coverage": ball_coverage,
                "total_positions": len(self.player1_positions) + len(self.player2_positions) + len(self.ball_positions)
            })
            
        except Exception as e:
            print(f"Error plotting coverage statistics: {e}")
    
    def _save_trajectory_analysis(self):
        """Save detailed trajectory analysis"""
        try:
            analysis_data = {
                "timestamp": datetime.now().isoformat(),
                "frame_dimensions": {"width": self.frame_width, "height": self.frame_height},
                "statistics": self.heatmap_stats,
                "player1_trajectory": list(self.player1_positions),
                "player2_trajectory": list(self.player2_positions),
                "ball_trajectory": list(self.ball_positions)
            }
            
            # Save JSON data
            filepath = os.path.join(self.output_dir, "heatmaps", "final", "trajectory_analysis.json")
            with open(filepath, 'w') as f:
                json.dump(analysis_data, f, indent=2)
            
            print(f"Trajectory analysis saved to: {filepath}")
            
        except Exception as e:
            print(f"Error saving trajectory analysis: {e}")
    
    def _save_heatmap_statistics(self, frame_count):
        """Save current heatmap statistics"""
        try:
            stats = {
                "frame_count": frame_count,
                "player1_positions": len(self.player1_positions),
                "player2_positions": len(self.player2_positions),
                "ball_positions": len(self.ball_positions),
                "player1_activity": float(np.sum(self.player1_heatmap)),
                "player2_activity": float(np.sum(self.player2_heatmap)),
                "ball_activity": float(np.sum(self.ball_heatmap))
            }
            
            filepath = os.path.join(self.output_dir, "heatmaps", "real_time", f"stats_frame_{frame_count}.json")
            with open(filepath, 'w') as f:
                json.dump(stats, f, indent=2)
                
        except Exception as e:
            print(f"Error saving statistics: {e}")
    
    def get_heatmap_summary(self):
        """Get summary of current heatmap state"""
        return {
            "frame_count": self.frame_count,
            "player1_positions": len(self.player1_positions),
            "player2_positions": len(self.player2_positions),
            "ball_positions": len(self.ball_positions),
            "heatmap_stats": self.heatmap_stats
        }

# Integration functions for get_data.py
def initialize_heatmap_generator(frame_width=640, frame_height=360, output_dir="output"):
    """Initialize the heatmap generator for integration with get_data.py"""
    return RealTimeHeatmapGenerator(frame_width, frame_height, output_dir)

def update_heatmaps_frame(heatmap_generator, players, ball_position, frame_count):
    """Update heatmaps for a single frame - to be called from get_data.py main loop"""
    heatmap_generator.frame_count = frame_count
    heatmap_generator.update_player_heatmap(players, frame_count)
    heatmap_generator.update_ball_heatmap(ball_position, frame_count)
    
    # Save periodic heatmaps every 200 frames
    if frame_count % 200 == 0:
        heatmap_generator.save_periodic_heatmaps(frame_count, save_interval=200)

def generate_heatmap_overlay(heatmap_generator, base_frame):
    """Generate heatmap overlay for current frame"""
    return heatmap_generator.generate_enhanced_heatmap_overlay(base_frame)

def finalize_heatmaps(heatmap_generator):
    """Finalize and save all heatmaps - to be called at end of processing"""
    heatmap_generator.save_final_comprehensive_heatmaps()
    return heatmap_generator.get_heatmap_summary()