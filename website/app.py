from flask import Flask, render_template, jsonify, send_from_directory
import pandas as pd
import json
import ast
import os
import numpy as np
from squash_analysis import generate_match_report
from functools import lru_cache
import threading

app = Flask(__name__)

# Global cache for storing data
cache = {
    'match_report': None,
    'positions_data': None,
    'shot_types': None,
    'heatmap_data': None
}

def initialize_data():
    """Initialize all data and visualizations at server startup"""
    print("Initializing data and generating visualizations...")
    
    try:
        # Generate the match report
        cache['match_report'] = generate_match_report("output/final.csv")
        
        # Read and process the CSV data
        df = pd.read_csv("output/final.csv")
        
        # Process player positions for visualization
        player1_positions = []
        player2_positions = []
        ball_positions = []
        
        for _, row in df.iterrows():
            try:
                # Process Player 1 positions
                p1_pos = ast.literal_eval(row['Player 1 RL World Position'])
                if p1_pos and not all(v == 0 for v in p1_pos):
                    player1_positions.append(p1_pos)
                    
                # Process Player 2 positions
                p2_pos = ast.literal_eval(row['Player 2 RL World Position'])
                if p2_pos and not all(v == 0 for v in p2_pos):
                    player2_positions.append(p2_pos)
                    
                # Process Ball positions
                ball_pos = ast.literal_eval(row['Ball RL World Position'])
                if ball_pos and not all(v == 0 for v in ball_pos):
                    ball_positions.append(ball_pos)
            except:
                continue
        
        # Store processed data in cache
        cache['positions_data'] = {
            'player1': player1_positions,
            'player2': player2_positions,
            'ball': ball_positions
        }
        
        # Get and store shot distribution data
        cache['shot_types'] = df['Shot Type'].value_counts().to_dict()
        
        # Process and store heatmap data
        cache['heatmap_data'] = {
            'player1': player1_positions,
            'player2': player2_positions
        }
        
        print("Data initialization complete!")
        
    except Exception as e:
        print(f"Error during data initialization: {str(e)}")
        raise

# Initialize data in a separate thread to not block server startup
threading.Thread(target=initialize_data).start()

@app.route('/')
def index():
    # Wait for data to be initialized if necessary
    if not all(cache.values()):
        return "Data is still being initialized. Please refresh in a few seconds.", 503
    
    return render_template('index.html', 
                         positions=json.dumps(cache['positions_data']),
                         shot_types=json.dumps(cache['shot_types']),
                         match_report=cache['match_report'])

@app.route('/get_heatmap_data')
def get_heatmap_data():
    if not cache['heatmap_data']:
        return jsonify({'error': 'Data not yet initialized'}), 503
    return jsonify(cache['heatmap_data'])

@app.route('/video/<path:filename>')
def serve_video(filename):
    return send_from_directory('../', filename)

if __name__ == '__main__':
    app.run(debug=True)
