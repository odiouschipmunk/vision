import os
import cv2
import subprocess
import json

# Path to OpenPose binary
OPENPOSE_BIN = 'openpose/build/examples/openpose/openpose.bin'

# Path to the folder containing your videos
video_folder = 'videos'

# Path to the folder to store annotations
output_folder = 'annotations'

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Process each video in the 'videos' folder
for video_file in os.listdir(video_folder):
    if video_file.endswith(('.mp4', '.avi', '.mov')):
        video_path = os.path.join(video_folder, video_file)
        output_json_path = os.path.join(output_folder, f"{os.path.splitext(video_file)[0]}_keypoints")

        # Run OpenPose on the video
        try:
            subprocess.run([
                OPENPOSE_BIN,
                '--video', video_path,              # Input video path
                '--write_json', output_json_path,   # Where to save keypoint annotations
                '--display', '0',                   # Disable the display (for headless processing)
                '--render_pose', '0',               # Disable pose rendering for faster processing
                '--model_pose', 'BODY_25',          # Use BODY_25 model for keypoints
                '--hand',                           # Enable hand keypoints
                '--face',                           # Enable face keypoints
                '--write_video', output_folder,     # Optionally save the annotated video
            ])
            print(f"Processed: {video_file}")
        except Exception as e:
            print(f"Error processing {video_file}: {e}")

print("Annotation process completed.")
