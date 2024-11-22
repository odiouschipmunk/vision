import csv
with open('output/final.csv', 'r', newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        print(row['Frame count'], row['Player 1 Keypoints'], row['Player 2 Keypoints'], row['Ball Position'], row['Shot Type'])

def getplayer1pos(frame_number):
    with open('output/final.csv', 'r', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row['Frame count'] == frame_number:
                return row['Player 1 Keypoints']
def getplayer2pos(frame_number):
    with open('output/final.csv', 'r', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row['Frame count'] == frame_number:
                return row['Player 2 Keypoints']
def getballpos(frame_number):
    with open('output/final.csv', 'r', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row['Frame count'] == frame_number:
                return row['Ball Position']
def getshottype(frame_number):
    with open('output/final.csv', 'r', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row['Frame count'] == frame_number:
                return row['Shot Type']

def parse_frame(frame_set):
    player1_set=[]
    player2_set=[]
    ball_set=[]
    shot_set=[]
    for frame in frame_set:
        player1_set.append(getplayer1pos(frame))
        player2_set.append(getplayer2pos(frame))
        ball_set.append(getballpos(frame))
        shot_set.append(getshottype(frame))
    #see where the players are currently in the court using the keypoints

def analyze_player_positions(frame_set):
    player1_positions = []
    player2_positions = []
    
    for frame in frame_set:
        player1_pos = getplayer1pos(frame)
        player2_pos = getplayer2pos(frame)
        
        if player1_pos:
            player1_positions.append(player1_pos)
        if player2_pos:
            player2_positions.append(player2_pos)
    
    def get_position_description(positions):
        if not positions:
            return "No data available"
        
        x_positions = [float(pos.split(',')[0]) for pos in positions]
        y_positions = [float(pos.split(',')[1]) for pos in positions]
        
        avg_x = sum(x_positions) / len(x_positions)
        avg_y = sum(y_positions) / len(y_positions)
        
        if avg_x < 0.5:
            x_desc = "left"
        else:
            x_desc = "right"
        
        if avg_y < 0.5:
            y_desc = "front"
        else:
            y_desc = "back"
        
        return f"generally in the {y_desc} {x_desc} of the court"
    
    player1_desc = get_position_description(player1_positions)
    player2_desc = get_position_description(player2_positions)
    
    return f"Player 1 is {player1_desc}. Player 2 is {player2_desc}."
#get the last 5 frames of the final.csv
with open('output/final.csv', 'r', newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    frame_set = []
    for row in reader:
        frame_set.append(row['Frame count'])
    frame_set = frame_set[-5:]
    print(frame_set)
print(analyze_player_positions(frame_set))
# import csv
# import ast
# import numpy as np
# import io

# def parse_ball_position(ball_pos_str):
#     """Parse the ball position string from CSV into coordinates"""
#     try:
#         return ast.literal_eval(ball_pos_str)
#     except Exception as e:
#         print(f"Error parsing ball position: {str(e)}")
#         return [0, 0]

# def parse_keypoints(keypoints_str):
#     """Parse the keypoints string from CSV into a list of coordinates"""
#     try:
#         # Split the string into lines and clean each line
#         lines = keypoints_str.strip().split('\n')
#         keypoints = []
        
#         for line in lines:
#             # Remove brackets and extra spaces
#             line = line.strip(' []')
#             if not line:
#                 continue
                
#             # Split the line into x and y coordinates
#             coords = [float(x.strip()) for x in line.split() if x.strip()]
#             if len(coords) == 2:
#                 keypoints.append(coords)
                
#         return keypoints
#     except Exception as e:
#         print(f"Error parsing keypoints: {str(e)}")
#         return []

# def analyze_court_position(x, y):
#     """Determine which quarter of the court a position is in"""
#     if x < 0.5:
#         return "front left" if y < 0.5 else "back left"
#     else:
#         return "front right" if y < 0.5 else "back right"

# def analyze_player_movement(keypoints):
#     """Analyze player movement from keypoints"""
#     if len(keypoints) < 2:
#         return "is stationary"
    
#     # Get valid keypoints (non-zero)
#     valid_points = [kp for kp in keypoints if kp[0] != 0 and kp[1] != 0]
#     if len(valid_points) < 2:
#         return "has limited movement"
    
#     # Calculate direction and distance
#     start = valid_points[0]
#     end = valid_points[-1]
#     dx = end[0] - start[0]
#     dy = end[1] - start[1]
    
#     # Determine movement direction
#     if abs(dx) > abs(dy):
#         direction = "moving laterally" if abs(dx) > 0.1 else "holding position"
#     else:
#         direction = "moving forward" if dy < -0.1 else "moving back" if dy > 0.1 else "holding position"
    
#     return direction

# def analyze_ball_position(ball_pos):
#     """Analyze the ball's position on the court"""
#     x, y = ball_pos
#     if x < 320:  # Assuming court width of 640
#         return "front left" if y < 180 else "back left"  # Assuming court height of 360
#     else:
#         return "front right" if y < 180 else "back right"

# def analyze_shot(shot_type, ball_pos):
#     """Analyze the shot type and its characteristics"""
#     shot_analysis = {
#         'description': '',
#         'typical_walls': 1,
#         'difficulty': 'medium',
#         'strategic_value': 'moderate'
#     }
    
#     if 'drive' in shot_type.lower():
#         if 'cross' in shot_type.lower():
#             shot_analysis.update({
#                 'description': 'A powerful crosscourt drive',
#                 'typical_walls': 2,
#                 'difficulty': 'high',
#                 'strategic_value': 'high'
#             })
#         else:
#             shot_analysis.update({
#                 'description': 'A straight drive along the wall',
#                 'typical_walls': 1,
#                 'difficulty': 'medium',
#                 'strategic_value': 'moderate'
#             })
    
#     return shot_analysis

# def analyze_csv_data():
#     """Analyze squash game data from CSV file"""
#     analyses = []
    
#     with open('output/final.csv', 'r', newline='') as csvfile:
#         reader = csv.DictReader(csvfile)
        
#         for row_num, row in enumerate(reader, 1):
#             try:
#                 frame_count = int(row['Frame count'])
#                 player1_keypoints = parse_keypoints(row['Player 1 Keypoints'])
#                 player2_keypoints = parse_keypoints(row['Player 2 Keypoints'])
#                 ball_position = eval(row['Ball Position'])  # Safe for this format: [x, y]
#                 shot_type = row['Shot Type'].strip()
                
#                 # Rest of your analysis code remains the same
#                 if not player1_keypoints or not player2_keypoints:
#                     continue
                
#                 p1_final_pos = next((kp for kp in reversed(player1_keypoints) 
#                                    if kp[0] != 0 and kp[1] != 0), None)
#                 p2_final_pos = next((kp for kp in reversed(player2_keypoints) 
#                                    if kp[0] != 0 and kp[1] != 0), None)
                
#                 if p1_final_pos and p2_final_pos:
#                     # Your existing analysis code...
#                     analysis = f"""Frame {frame_count} Analysis:
#     Player 1: {p1_final_pos}
#     Player 2: {p2_final_pos}
#     Ball: {ball_position}
#     Shot: {shot_type}
#     """
#                     analyses.append(analysis)
                    
#             except Exception as e:
#                 print(f"Error processing row {row_num}: {str(e)}")
#                 continue
                
#     return "\n".join(analyses) if analyses else "No valid analyses could be generated."
if __name__ == "__main__":
    # ...existing code...
    frame_set = [1, 2, 3, 4, 5]  # Example frame set
    print(analyze_player_positions(frame_set))
    # ...existing code...
#     print("\nFinal Analysis Output:")
#     print("=" * 80)
#     print(analyze_csv_data())