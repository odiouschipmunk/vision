import torch
import cv2
import os

# Load pre-trained YOLOv5 model (e.g., yolov5s)
pretrained_model = torch.hub.load("ultralytics/yolov5", "yolov5s")

# Load your custom weights
custom_model = torch.hub.load("ultralytics/yolov5", "custom", path="yolov5/runs/train/exp8/weights/best.pt")

# Get the number of classes in the custom model
num_classes = len(custom_model.names)

# Modify the final layer of the pre-trained model to match the number of classes in the custom model
pretrained_model.model.model[-1] = torch.nn.Conv2d(
    in_channels=pretrained_model.model.model[-1].in_channels,
    out_channels=num_classes * (5 + num_classes),  # 5 is for the bounding box attributes
    kernel_size=pretrained_model.model.model[-1].kernel_size,
    stride=pretrained_model.model.model[-1].stride,
    padding=pretrained_model.model.model[-1].padding
)

# Load the custom weights into the modified pre-trained model
pretrained_model.model.load_state_dict(custom_model.model.state_dict(), strict=False)

# Define the folder containing the videos
video_folder = 'videos'

# Define the classes you are interested in
target_classes = ['person', 'squash_racket', 'squash_ball']

# Confidence threshold
conf_threshold = 0.25

# Process each video in the folder
for video_file in os.listdir(video_folder):
    if video_file.endswith(".mp4"):
        cap = cv2.VideoCapture(os.path.join(video_folder, video_file))
        out = None
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Inference
            results = pretrained_model(frame)

            # Process results
            for *box, conf, cls in results.xyxy[0]:  # xyxy format
                if conf < conf_threshold:
                    continue
                label = pretrained_model.names[int(cls)]
                if label in target_classes:
                    # Draw bounding box
                    cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
                    # Put label near bounding box
                    cv2.putText(frame, f'{label} {conf:.2f}', (int(box[0]), int(box[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # Initialize video writer if not already initialized
            if out is None:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(f'annotated_{video_file}', fourcc, 20.0, (frame.shape[1], frame.shape[0]))

            # Write the annotated frame to the output video
            out.write(frame)

            # Display the frame with bounding boxes
            cv2.imshow('Annotated Video', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        if out is not None:
            out.release()

cv2.destroyAllWindows()