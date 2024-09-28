import cv2
import mediapipe as mp
import os
import torch
import warnings
from tqdm import tqdm
# Suppress specific FutureWarnings
warnings.filterwarnings("ignore", category=FutureWarning, module="torch.cuda.amp")

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model.cuda()  # Use GPU

# Path to the videos folder
videos_folder = 'videos'

# Create output folder if it doesn't exist
output_folder = 'labeled_videos'
os.makedirs(output_folder, exist_ok=True)

# Process each video in the folder
i=0
for video_file in os.listdir(videos_folder):
    if video_file.endswith('.mp4'):
        video_path = os.path.join(videos_folder, video_file)
        cap = cv2.VideoCapture(video_path)
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Define the codec and create VideoWriter object
        out = cv2.VideoWriter(os.path.join(output_folder, video_file), 
                              cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
        
        if not out.isOpened():
            print(f"Error: Could not open video writer for {video_file}")
            cap.release()
            continue
        
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Convert the BGR image to RGB
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                
                # Process the image and detect the pose
                results = pose.process(image)
                
                # Convert the image back to BGR for OpenCV
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                
                # Detect objects (ball and racket) using YOLOv5
                with torch.amp.autocast('cuda'):
                    results_yolo = model(image)
                for detection in results_yolo.xyxy[0]:
                    x1, y1, x2, y2, conf, cls = detection
                    label = model.names[int(cls)]
                    if label in ['sports ball', 'tennis racket']:
                        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                        cv2.putText(image, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                
                # Draw the pose annotation on the image
                if results.pose_landmarks:
                    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                
                # Write the frame with the pose and object annotations
                out.write(image)
        except Exception as e:
            print(f"Error processing video {video_file}: {e}")
        finally:
            print(i)
            i=i+1
            cap.release()
            out.release()

pose.close()
print("Processing complete. Labeled videos are saved in the 'labeled_videos' folder.")