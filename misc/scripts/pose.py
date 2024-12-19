import os
from ultralytics import YOLO

# Load the YOLO model
model = YOLO("yolov8s-pose.pt")

video_folder = "videos"

# Process each video in the folder
for video_file in os.listdir(video_folder):
    if video_file.endswith(".mp4"):
        # Run the model on the video file
        results = model(
            source=os.path.join(video_folder, video_file),
            show=True,
            conf=0.5,
            save=True,
        )

        # Extract detected objects
        for result in results:
            for obj in result.boxes:
                if obj.cls in [
                    model.names.index("squash racket"),
                    model.names.index("squash ball"),
                ]:
                    print(
                        f"Detected {model.names[obj.cls]} with confidence {obj.conf:.2f}"
                    )
