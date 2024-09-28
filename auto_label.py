import cv2
import os
import numpy as np
import mediapipe as mp
import tensorflow as tf
from multiprocessing import Pool

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logging

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Ensure TensorFlow uses the GPU
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    print("TensorFlow is using the GPU")
else:
    print("No GPU found. TensorFlow will use the CPU.")

def process_video(video_path, output_folder):
    cap = cv2.VideoCapture(video_path)
    output_video_path = os.path.join(output_folder, os.path.basename(video_path))

    # Get video properties
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Output video writer
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    with mp_pose.Pose() as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Convert BGR image to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process the frame to get pose landmarks
            result = pose.process(rgb_frame)

            # Draw pose landmarks on the frame
            if result.pose_landmarks:
                mp_drawing.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Write the processed frame to output video
            out.write(frame)

    cap.release()
    out.release()

def auto_label_videos(input_folder, output_folder):
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    video_files = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith('.mp4') or f.endswith('.avi')]
    
    # Use multiprocessing to process videos in parallel
    with Pool() as pool:
        pool.starmap(process_video, [(video_file, output_folder) for video_file in video_files])

if __name__ == "__main__":
    # Define input and output directories
    input_folders = ["videos"]
    output_folder = "processed_videos"

    for folder in input_folders:
        print(f"Processing folder: {folder}")
        auto_label_videos(folder, output_folder)