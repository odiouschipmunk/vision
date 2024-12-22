import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import Counter
import ast
from mpl_toolkits.mplot3d import Axes3D
import os

OUTPUT_DIR = 'output_final'  # Define output directory as a constant

def load_match_data(csv_path):
    """Load and preprocess match data from CSV"""
    df = pd.read_csv(csv_path)
    return df

def ensure_output_dir():
    """Create output directory if it doesn't exist"""
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

def plot_3d_court():
    """Create a 3D squash court visualization"""
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Court dimensions (in meters)
    length = 9.75
    width = 6.4
    height = 4.57
    
    # Plot court boundaries
    # Floor
    ax.plot([0, width, width, 0, 0], [0, 0, length, length, 0], [0, 0, 0, 0, 0], 'k-')
    # Front wall
    ax.plot([0, width, width, 0, 0], [length, length, length, length, length], [0, 0, height, height, 0], 'k-')
    # Side walls
    ax.plot([0, 0], [0, length], [0, 0], 'k-')
    ax.plot([0, 0], [length, length], [0, height], 'k-')
    ax.plot([width, width], [0, length], [0, 0], 'k-')
    ax.plot([width, width], [length, length], [0, height], 'k-')
    
    # Service line and tin
    ax.plot([0, width], [length-4.26, length-4.26], [0, 0], 'k--', alpha=0.5)  # Service line
    ax.plot([0, width], [length, length], [0.48, 0.48], 'r-', linewidth=2)  # Tin line
    
    # Set labels and title
    ax.set_xlabel('Width (m)')
    ax.set_ylabel('Length (m)')
    ax.set_zlabel('Height (m)')
    
    return ax

def visualize_3d_positions(df):
    """Create 3D visualization of player positions and ball trajectory"""
    ax = plot_3d_court()
    
    # Plot player positions
    for player_num in [1, 2]:
        positions = []
        for row in df.iterrows():
            try:
                pos = ast.literal_eval(row[1][f'Player {player_num} RL World Position'])
                if pos and not all(v == 0 for v in pos):
                    positions.append(pos)
            except:
                continue
        
        if positions:
            positions = np.array(positions)
            color = 'blue' if player_num == 1 else 'red'
            ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], 
                      c=color, alpha=0.3, label=f'Player {player_num}')
    
    # Plot ball trajectory
    ball_positions = []
    for row in df.iterrows():
        try:
            pos = ast.literal_eval(row[1]['Ball RL World Position'])
            if pos and not all(v == 0 for v in pos):
                ball_positions.append(pos)
        except:
            continue
    
    if ball_positions:
        ball_positions = np.array(ball_positions)
        ax.scatter(ball_positions[:, 0], ball_positions[:, 1], ball_positions[:, 2],
                  c='green', alpha=0.5, s=20, label='Ball')
        
        # Draw lines connecting consecutive ball positions
        for i in range(len(ball_positions)-1):
            ax.plot([ball_positions[i,0], ball_positions[i+1,0]],
                   [ball_positions[i,1], ball_positions[i+1,1]],
                   [ball_positions[i,2], ball_positions[i+1,2]],
                   'g-', alpha=0.2)
    
    ax.legend()
    plt.title('3D Match Visualization')
    plt.savefig(f'{OUTPUT_DIR}/3d_match_visualization.png')
    plt.close()

def visualize_shot_trajectories(df):
    """Create 3D visualization of shot trajectories by type"""
    # Group shots by type
    shot_types = df['Shot Type'].unique()
    
    for shot_type in shot_types:
        if pd.isna(shot_type):
            continue
            
        ax = plot_3d_court()
        shot_data = df[df['Shot Type'] == shot_type]
        
        # Plot ball trajectories for this shot type
        ball_positions = []
        for row in shot_data.iterrows():
            try:
                pos = ast.literal_eval(row[1]['Ball RL World Position'])
                if pos and not all(v == 0 for v in pos):
                    ball_positions.append(pos)
            except:
                continue
        
        if ball_positions:
            ball_positions = np.array(ball_positions)
            ax.scatter(ball_positions[:, 0], ball_positions[:, 1], ball_positions[:, 2],
                      c='green', alpha=0.5, s=20)
            
            # Draw lines connecting consecutive ball positions
            for i in range(len(ball_positions)-1):
                ax.plot([ball_positions[i,0], ball_positions[i+1,0]],
                       [ball_positions[i,1], ball_positions[i+1,1]],
                       [ball_positions[i,2], ball_positions[i+1,2]],
                       'g-', alpha=0.2)
        
        plt.title(f'3D Trajectory - {shot_type}')
        plt.savefig(f'{OUTPUT_DIR}/3d_trajectory_{shot_type.lower().replace(" ", "_")}.png')
        plt.close()

def create_3d_heatmap(df, player_column, title):
    """Create 3D heatmap of player positions"""
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
        scatter = ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
                           c=np.ones(len(positions)), cmap='hot',
                           alpha=0.5)
        plt.colorbar(scatter, label='Frequency')
    
    plt.title(f'3D Position Heatmap - {title}')
    plt.savefig(f'{OUTPUT_DIR}/3d_heatmap_{title.lower().replace(" ", "_")}.png')
    plt.close()

def analyze_shot_distribution(df):
    """Analyze and visualize shot distribution"""
    # Count shot types
    shot_counts = Counter(df['Shot Type'].dropna())
    
    # Create pie chart
    plt.figure(figsize=(10, 6))
    plt.pie(shot_counts.values(), labels=shot_counts.keys(), autopct='%1.1f%%')
    plt.title('Shot Type Distribution')
    plt.savefig(f'{OUTPUT_DIR}/shot_distribution.png')
    plt.close()
    
    return dict(shot_counts)

def create_court_heatmap(df, player_column, title):
    """Create 2D heatmap of player positions on court"""
    positions = []
    for pos in df[player_column]:
        try:
            keypoints = ast.literal_eval(pos)
            # Use ankle positions (indices 15, 16 for left and right ankle)
            ankles = np.array([keypoints[15], keypoints[16]])
            # Filter out [0,0] positions
            valid_ankles = ankles[~np.all(ankles == [0, 0], axis=1)]
            if len(valid_ankles) > 0:
                avg_pos = np.mean(valid_ankles, axis=0)
                positions.append(avg_pos)
        except:
            continue
    
    if positions:
        positions = np.array(positions)
        
        plt.figure(figsize=(10, 6))
        heatmap, xedges, yedges = np.histogram2d(
            positions[:, 0], 
            positions[:, 1], 
            bins=20,
            range=[[0, 1], [0, 1]]
        )
        plt.imshow(heatmap.T, origin='lower', cmap='hot', aspect='auto')
        plt.colorbar(label='Frequency')
        plt.title(f'{title} Court Position Heatmap')
        plt.xlabel('Court Width')
        plt.ylabel('Court Length')
        plt.savefig(f'{OUTPUT_DIR}/{title.lower().replace(" ", "_")}_heatmap.png')
        plt.close()

def analyze_t_position_distance(df):
    """Analyze and visualize players' distance from T position"""
    # T position in real-world coordinates (meters)
    t_position = np.array([3.2, 4.57, 0])  # Center of court
    
    def calculate_distance(pos_str):
        try:
            pos = ast.literal_eval(pos_str)
            if pos and not all(v == 0 for v in pos):
                return np.linalg.norm(np.array(pos) - t_position)
            return None
        except:
            return None
    
    # Calculate distances for both players
    df['P1_T_Distance'] = df['Player 1 RL World Position'].apply(calculate_distance)
    df['P2_T_Distance'] = df['Player 2 RL World Position'].apply(calculate_distance)
    
    # Plot distances over time
    plt.figure(figsize=(12, 6))
    plt.plot(df['Frame count'], df['P1_T_Distance'], 'b-', label='Player 1', alpha=0.7)
    plt.plot(df['Frame count'], df['P2_T_Distance'], 'r-', label='Player 2', alpha=0.7)
    plt.title('Distance from T Position Over Time')
    plt.xlabel('Frame')
    plt.ylabel('Distance (meters)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f'{OUTPUT_DIR}/t_position_distance.png')
    plt.close()
    
    # Calculate average distances
    avg_p1 = df['P1_T_Distance'].mean()
    avg_p2 = df['P2_T_Distance'].mean()
    
    return avg_p1, avg_p2

def analyze_shot_success(df):
    """Analyze success rate of different shot types"""
    shot_success = {}
    
    for shot_type in df['Shot Type'].unique():
        if pd.isna(shot_type):
            continue
        shots = df[df['Shot Type'] == shot_type]
        # Calculate success based on valid ball positions
        valid_shots = 0
        for _, row in shots.iterrows():
            try:
                ball_pos = ast.literal_eval(row['Ball Position'])
                if isinstance(ball_pos, list) and len(ball_pos) == 2:
                    if ball_pos[0] != 0 and ball_pos[1] != 0:
                        valid_shots += 1
            except:
                continue
        success_rate = (valid_shots / len(shots)) * 100 if len(shots) > 0 else 0
        shot_success[shot_type] = success_rate
    
    plt.figure(figsize=(10, 6))
    plt.bar(shot_success.keys(), shot_success.values())
    plt.title('Shot Success Rate by Type')
    plt.xlabel('Shot Type')
    plt.ylabel('Success Rate (%)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/shot_success_rate.png')
    plt.close()
    
    return shot_success

def generate_match_report(csv_path):
    """Generate comprehensive match analysis report"""
    ensure_output_dir()  # Make sure output directory exists
    df = load_match_data(csv_path)
    
    # Generate all analyses
    shot_dist = analyze_shot_distribution(df)
    create_court_heatmap(df, 'Player 1 Keypoints', 'Player 1')
    create_court_heatmap(df, 'Player 2 Keypoints', 'Player 2')
    t_distances = analyze_t_position_distance(df)
    shot_success = analyze_shot_success(df)
    
    # Generate 3D visualizations
    visualize_3d_positions(df)
    visualize_shot_trajectories(df)
    create_3d_heatmap(df, 'Player 1 RL World Position', 'Player 1')
    create_3d_heatmap(df, 'Player 2 RL World Position', 'Player 2')
    
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
    with open(f'{OUTPUT_DIR}/match_report.txt', 'w') as f:
        f.write(report)
    
    return report

if __name__ == "__main__":
    report = generate_match_report("output/final.csv")
    print(report) 