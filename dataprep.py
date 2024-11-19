import json
import numpy as np

# Load data
with open('output/final.json', 'r') as file:
    data = json.load(file)

# Preprocess data
def preprocess_frame(frame_data):
    # Convert lists to numpy arrays
    player1 = np.array(frame_data['Player 1'])
    player2 = np.array(frame_data['Player 2'])
    ball = np.array(frame_data['Ball'])
    # Handle missing data
    player1[player1 == 0.0] = np.nan
    player2[player2 == 0.0] = np.nan
    return player1, player2, ball

# Process all frames
processed_data = []
for frame in data:
    player1, player2, ball = preprocess_frame(frame)
    processed_data.append({
        'Frame': frame['Frame'],
        'Player1': player1,
        'Player2': player2,
        'Ball': ball,
        'TypeOfShot': frame['Type of shot'],
        'BallHit': frame['Ball hit'],
        'WallsHit': frame['Walls hit'],
    })
    
speeds = []
for i in range(1, len(processed_data)):
    prev = processed_data[i - 1]
    current = processed_data[i]
    time_diff = current['Frame'] - prev['Frame']
    # Player 1 speed
    p1_prev_pos = np.nanmean(prev['Player1'], axis=0)
    p1_current_pos = np.nanmean(current['Player1'], axis=0)
    p1_speed = np.linalg.norm(p1_current_pos - p1_prev_pos) / time_diff
    # Player 2 speed
    p2_prev_pos = np.nanmean(prev['Player2'], axis=0)
    p2_current_pos = np.nanmean(current['Player2'], axis=0)
    p2_speed = np.linalg.norm(p2_current_pos - p2_prev_pos) / time_diff
    speeds.append({
        'Frame': current['Frame'],
        'Player1Speed': p1_speed,
        'Player2Speed': p2_speed,
    })
print(f'Player 1 average speed: {np.mean([speed["Player1Speed"] for speed in speeds])}')
print(f'Player 2 average speed: {np.mean([speed["Player2Speed"] for speed in speeds])}')
print(f'processed_data: {processed_data}')