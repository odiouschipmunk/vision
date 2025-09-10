import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import ast
import os
from functools import lru_cache

OUTPUT_DIR = "output_final"  # Define output directory as a constant
CACHE_DIR = "static/cache"  # Directory for cached visualizations


def ensure_directories():
    """Create output and cache directories if they don't exist"""
    for directory in [OUTPUT_DIR, CACHE_DIR]:
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
    
    # Also ensure heatmap subdirectories
    heatmap_dirs = ["heatmaps", "analysis", "visualizations"]
    for subdir in heatmap_dirs:
        full_path = os.path.join(OUTPUT_DIR, subdir)
        if not os.path.exists(full_path):
            os.makedirs(full_path, exist_ok=True)


@lru_cache(maxsize=1)
def load_match_data(csv_path):
    """Load and preprocess match data from CSV"""
    return pd.read_csv(csv_path)


def plot_3d_court():
    """Create a 3D squash court visualization"""
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Court dimensions (in meters)
    length = 9.75
    width = 6.4
    height = 4.57

    # Plot court boundaries
    # Floor
    ax.plot([0, width, width, 0, 0], [0, 0, length, length, 0], [0, 0, 0, 0, 0], "k-")
    # Front wall
    ax.plot(
        [0, width, width, 0, 0],
        [length, length, length, length, length],
        [0, 0, height, height, 0],
        "k-",
    )
    # Side walls
    ax.plot([0, 0, 0, 0], [0, length, length, 0], [0, 0, height, height], "k-")
    ax.plot([width, width, width, width], [0, length, length, 0], [0, 0, height, height], "k-")

    ax.set_xlabel("Width (m)")
    ax.set_ylabel("Length (m)")
    ax.set_zlabel("Height (m)")
    ax.set_title("3D Squash Court")

    return ax


def visualize_3d_positions(df):
    """Create 3D visualization of player positions and ball trajectory"""
    ensure_directories()  # Make sure directories exist
    
    cache_path = os.path.join(OUTPUT_DIR, "visualizations", "3d_match_visualization.png")

    # If cached visualization exists, return early
    if os.path.exists(cache_path):
        return

    try:
        ax = plot_3d_court()

        # Plot player positions
        for player_num in [1, 2]:
            positions = []
            for row in df.iterrows():
                try:
                    pos = ast.literal_eval(row[1][f"Player {player_num} RL World Position"])
                    if pos and not all(v == 0 for v in pos):
                        positions.append(pos)
                except:
                    continue

            if positions:
                positions = np.array(positions)
                color = "blue" if player_num == 1 else "red"
                ax.scatter(
                    positions[:, 0],
                    positions[:, 1],
                    positions[:, 2],
                    c=color,
                    alpha=0.3,
                    label=f"Player {player_num}",
                )

        # Plot ball trajectory
        ball_positions = []
        for row in df.iterrows():
            try:
                pos = ast.literal_eval(row[1]["Ball RL World Position"])
                if pos and not all(v == 0 for v in pos):
                    ball_positions.append(pos)
            except:
                continue

        if ball_positions:
            ball_positions = np.array(ball_positions)
            ax.scatter(
                ball_positions[:, 0],
                ball_positions[:, 1],
                ball_positions[:, 2],
                c="green",
                alpha=0.5,
                s=20,
                label="Ball",
            )

            # Draw lines connecting consecutive ball positions
            for i in range(len(ball_positions) - 1):
                ax.plot(
                    [ball_positions[i, 0], ball_positions[i + 1, 0]],
                    [ball_positions[i, 1], ball_positions[i + 1, 1]],
                    [ball_positions[i, 2], ball_positions[i + 1, 2]],
                    "g-",
                    alpha=0.2,
                )

        # Only add legend if there are labels
        if ax.get_legend_handles_labels()[0]:  # Check if there are legend handles
            ax.legend()
            
        plt.title("3D Match Visualization")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        plt.savefig(cache_path, dpi=150, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        print(f"Warning: Could not create 3D visualization: {e}")
        # Create a simple fallback plot
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, f"3D Visualization Error\n{str(e)[:100]}...", 
                ha='center', va='center', transform=ax.transAxes,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral"))
        ax.set_title("3D Visualization - Error Fallback")
        ax.axis('off')
        
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        plt.savefig(cache_path, dpi=150, bbox_inches='tight')
        plt.close()


def visualize_shot_trajectories(df):
    """Create 3D visualization of shot trajectories by type"""
    # Group shots by type
    shot_types = df["Shot Type"].unique()

    for shot_type in shot_types:
        if pd.isna(shot_type):
            continue

        cache_path = os.path.join(
            CACHE_DIR, f'3d_trajectory_{shot_type.lower().replace(" ", "_")}.png'
        )

        # If cached visualization exists, skip
        if os.path.exists(cache_path):
            continue

        ax = plot_3d_court()
        shot_data = df[df["Shot Type"] == shot_type]

        # Plot ball trajectories for this shot type
        ball_positions = []
        for row in shot_data.iterrows():
            try:
                pos = ast.literal_eval(row[1]["Ball RL World Position"])
                if pos and not all(v == 0 for v in pos):
                    ball_positions.append(pos)
            except:
                continue

        if ball_positions:
            ball_positions = np.array(ball_positions)
            ax.scatter(
                ball_positions[:, 0],
                ball_positions[:, 1],
                ball_positions[:, 2],
                c="green",
                alpha=0.5,
                s=20,
            )

            # Draw lines connecting consecutive ball positions
            for i in range(len(ball_positions) - 1):
                ax.plot(
                    [ball_positions[i, 0], ball_positions[i + 1, 0]],
                    [ball_positions[i, 1], ball_positions[i + 1, 1]],
                    [ball_positions[i, 2], ball_positions[i + 1, 2]],
                    "g-",
                    alpha=0.2,
                )

        plt.title(f"3D Trajectory - {shot_type}")
        plt.savefig(cache_path)
        plt.close()


def create_3d_heatmap(df, player_column, title):
    """Create 3D heatmap of player positions"""
    cache_path = os.path.join(
        CACHE_DIR, f'3d_heatmap_{title.lower().replace(" ", "_")}.png'
    )

    # If cached visualization exists, return early
    if os.path.exists(cache_path):
        return

    ax = plot_3d_court()

    positions = []
    for row in df.iterrows():
        try:
            pos = ast.literal_eval(row[1][player_column])
            if pos and not all(v == 0 for v in pos):
                positions.append(pos)
        except:
            continue

    if positions:
        positions = np.array(positions)
        scatter = ax.scatter(
            positions[:, 0],
            positions[:, 1],
            positions[:, 2],
            c=np.ones(len(positions)),
            cmap="hot",
            alpha=0.5,
        )
        plt.colorbar(scatter, label="Frequency")

    plt.title(f"3D Position Heatmap - {title}")
    plt.savefig(cache_path)
    plt.close()


def analyze_shot_distribution(df):
    """Analyze and visualize shot distribution"""
    cache_path = os.path.join(CACHE_DIR, "shot_distribution.png")

    # Count shot types
    shot_counts = Counter(df["Shot Type"].dropna())

    # Create pie chart if not cached
    if not os.path.exists(cache_path):
        plt.figure(figsize=(10, 6))
        plt.pie(shot_counts.values(), labels=shot_counts.keys(), autopct="%1.1f%%")
        plt.title("Shot Type Distribution")
        plt.savefig(cache_path)
        plt.close()

    return dict(shot_counts)


def create_court_heatmap(df, player_column, title):
    """Create 2D heatmap of player positions on court"""
    ensure_directories()  # Make sure directories exist
    
    cache_path = os.path.join(
        OUTPUT_DIR, "heatmaps", f'{title.lower().replace(" ", "_")}_heatmap.png'
    )

    # If cached visualization exists, return early
    if os.path.exists(cache_path):
        return cache_path

    try:
        positions = []
        for pos in df[player_column]:
            try:
                keypoints = ast.literal_eval(pos)
                if keypoints and len(keypoints) >= 17:
                    ankles = np.array([keypoints[15], keypoints[16]])
                    valid_ankles = ankles[~np.all(ankles == [0, 0], axis=1)]
                    if len(valid_ankles) > 0:
                        avg_pos = np.mean(valid_ankles, axis=0)
                        if avg_pos[0] > 0 and avg_pos[1] > 0:  # Valid position
                            positions.append(avg_pos)
            except:
                continue

        if positions:
            positions = np.array(positions)

            plt.figure(figsize=(12, 8))
            heatmap, xedges, yedges = np.histogram2d(
                positions[:, 0], positions[:, 1], bins=25, range=[[0, 1], [0, 1]]
            )
            
            # Create enhanced heatmap
            im = plt.imshow(heatmap.T, origin="lower", cmap="hot", aspect="auto", 
                           extent=[0, 1, 0, 1], alpha=0.8)
            plt.colorbar(im, label="Frequency")
            plt.title(f"{title} Court Position Heatmap", fontsize=14, fontweight='bold')
            plt.xlabel("Court Width (normalized)", fontsize=12)
            plt.ylabel("Court Length (normalized)", fontsize=12)
            
            # Add court boundaries and markers
            plt.axhline(y=0.3, color='white', linestyle='--', alpha=0.7, linewidth=1)
            plt.axvline(x=0.5, color='white', linestyle='--', alpha=0.7, linewidth=1)
            plt.plot(0.5, 0.47, 'wo', markersize=8, label='T-Position')
            plt.legend()
            
            # Add statistics
            stats_text = f"Total positions: {len(positions)}\nCoverage: {np.ptp(positions[:, 0]):.2f} × {np.ptp(positions[:, 1]):.2f}"
            plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
                    verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            plt.savefig(cache_path, dpi=300, bbox_inches='tight')
            plt.close()
            
        else:
            # Create fallback visualization
            plt.figure(figsize=(10, 6))
            plt.text(0.5, 0.5, f"No valid position data\nfor {title}", 
                    ha='center', va='center', transform=plt.gca().transAxes,
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightcoral", alpha=0.8))
            plt.title(f"{title} Court Position Heatmap - No Data", fontsize=14)
            plt.xlabel("Court Width")
            plt.ylabel("Court Length")
            
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            plt.savefig(cache_path, dpi=300, bbox_inches='tight')
            plt.close()
            
        return cache_path
        
    except Exception as e:
        print(f"Error creating court heatmap for {title}: {e}")
        # Create error fallback
        plt.figure(figsize=(10, 6))
        plt.text(0.5, 0.5, f"Heatmap Generation Failed\n{title}\nError: {str(e)[:50]}...", 
                ha='center', va='center', transform=plt.gca().transAxes,
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightcoral", alpha=0.8))
        plt.title(f"{title} Court Position Heatmap - Error", fontsize=14)
        plt.axis('off')
        
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        plt.savefig(cache_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return cache_path


def analyze_t_position_distance(df):
    """Analyze and visualize players' distance from T position"""
    cache_path = os.path.join(CACHE_DIR, "t_position_distance.png")

    # T position in real-world coordinates (meters)
    t_position = np.array([3.2, 4.57, 0])

    def calculate_distance(pos_str):
        try:
            pos = ast.literal_eval(pos_str)
            if pos and not all(v == 0 for v in pos):
                return np.linalg.norm(np.array(pos) - t_position)
            return None
        except:
            return None

    # Calculate distances for both players
    df["P1_T_Distance"] = df["Player 1 RL World Position"].apply(calculate_distance)
    df["P2_T_Distance"] = df["Player 2 RL World Position"].apply(calculate_distance)

    # Create plot if not cached
    if not os.path.exists(cache_path):
        plt.figure(figsize=(12, 6))
        plt.plot(
            df["Frame count"], df["P1_T_Distance"], "b-", label="Player 1", alpha=0.7
        )
        plt.plot(
            df["Frame count"], df["P2_T_Distance"], "r-", label="Player 2", alpha=0.7
        )
        plt.title("Distance from T Position Over Time")
        plt.xlabel("Frame")
        plt.ylabel("Distance (meters)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(cache_path)
        plt.close()

    # Calculate average distances
    avg_p1 = df["P1_T_Distance"].mean()
    avg_p2 = df["P2_T_Distance"].mean()

    return avg_p1, avg_p2


def analyze_shot_success(df):
    """Analyze success rate of different shot types"""
    cache_path = os.path.join(CACHE_DIR, "shot_success_rate.png")

    shot_success = {}

    for shot_type in df["Shot Type"].unique():
        if pd.isna(shot_type):
            continue
        shots = df[df["Shot Type"] == shot_type]
        valid_shots = 0
        for _, row in shots.iterrows():
            try:
                ball_pos = ast.literal_eval(row["Ball Position"])
                if isinstance(ball_pos, list) and len(ball_pos) == 2:
                    if ball_pos[0] != 0 and ball_pos[1] != 0:
                        valid_shots += 1
            except:
                continue
        success_rate = (valid_shots / len(shots)) * 100 if len(shots) > 0 else 0
        shot_success[shot_type] = success_rate

    # Create plot if not cached
    if not os.path.exists(cache_path):
        plt.figure(figsize=(10, 6))
        plt.bar(shot_success.keys(), shot_success.values())
        plt.title("Shot Success Rate by Type")
        plt.xlabel("Shot Type")
        plt.ylabel("Success Rate (%)")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(cache_path)
        plt.close()

    return shot_success


def generate_match_report(csv_path):
    """Generate comprehensive match analysis report"""
    ensure_directories()  # Make sure output and cache directories exist
    df = load_match_data(csv_path)

    # Generate all analyses
    shot_dist = analyze_shot_distribution(df)
    create_court_heatmap(df, "Player 1 Keypoints", "Player 1")
    create_court_heatmap(df, "Player 2 Keypoints", "Player 2")
    t_distances = analyze_t_position_distance(df)
    shot_success = analyze_shot_success(df)

    # Generate 3D visualizations
    visualize_3d_positions(df)
    visualize_shot_trajectories(df)
    create_3d_heatmap(df, "Player 1 RL World Position", "Player 1")
    create_3d_heatmap(df, "Player 2 RL World Position", "Player 2")

    # Create text report
    report = "Match Analysis Report\n"
    report += "===================\n\n"

    report += "1. Shot Distribution\n"
    report += "-------------------\n"
    for shot, count in shot_dist.items():
        report += f"{shot}: {count} shots ({count/sum(shot_dist.values())*100:.1f}%)\n"

    report += "\n2. Shot Success Rates\n"
    report += "-------------------\n"
    for shot, rate in shot_success.items():
        report += f"{shot}: {rate:.1f}%\n"

    report += "\n3. Average Distance from T Position\n"
    report += "--------------------------------\n"
    report += f"Player 1: {t_distances[0]:.2f} meters\n"
    report += f"Player 2: {t_distances[1]:.2f} meters\n"

    report += "\n4. Visualization Files Generated\n"
    report += "-----------------------------\n"
    report += "- 3D match visualization (3d_match_visualization.png)\n"
    report += "- Shot trajectories by type (3d_trajectory_*.png)\n"
    report += "- Player position heatmaps (3d_heatmap_*.png)\n"
    report += "- Shot distribution pie chart (shot_distribution.png)\n"
    report += "- T position distance graph (t_position_distance.png)\n"
    report += "- Shot success rate bar chart (shot_success_rate.png)\n"

    # Save report
    with open(f"{OUTPUT_DIR}/match_report.txt", "w") as f:
        f.write(report)

    return report


if __name__ == "__main__":
    report = generate_match_report("output/final.csv")
    print(report)
