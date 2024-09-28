import torch
import cv2
import os

# Load YOLOv5 model
model = torch.hub.load("ultralytics/yolov5", "yolov5s")  # or yolov5n - yolov5x6, custom

# Define the folder containing the videos
video_folder = 'videos'

# Define the classes you are interested in
target_classes = ['person', 'sports ball', 'tennis racket', 'squash racket', 'squash ball']

# Process each video in the folder
for video_file in os.listdir(video_folder):
    if video_file.endswith(".mp4"):
        cap = cv2.VideoCapture(os.path.join(video_folder, video_file))
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Inference
            results = model(frame)

            # Extract bounding boxes and labels
            for *box, conf, cls in results.xyxy[0]:
                label = model.names[int(cls)]
                if label in target_classes:
                    # Draw bounding box
                    cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
                    # Put label text
                    cv2.putText(frame, label, (int(box[0]), int(box[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # Optionally, display the frame with detections
            cv2.imshow("Frame", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
cv2.destroyAllWindows()