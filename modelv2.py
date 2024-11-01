import cv2
import numpy as np
import pandas as pd

def extract_frames(video_path, fps=1):
    video = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = video.read()
        if not ret:
            break
        frames.append(frame)
    video.release()
    return frames[::int(30/fps)]  # Assuming 30fps video, adjust as needed

def load_annotations(annotation_path):
    return pd.read_csv(annotation_path)

def load_positions(positions_path):
    return pd.read_csv(positions_path)

# Example usage
frames = extract_frames('squash_game.mp4')
annotations = load_annotations('annotations.csv')
positions = load_positions('positions.csv')