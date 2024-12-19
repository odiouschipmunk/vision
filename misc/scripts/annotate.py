"""
import torch
import cv2
import os
import albumentations as A
from albumentations.pytorch import ToTensorV2
from collections import deque

# Load pre-trained YOLOv5 model (e.g., yolov5s)
pretrained_model = torch.hub.load("ultralytics/yolov5", "yolov5s")

# Load your custom weights
custom_model = torch.hub.load("ultralytics/yolov5", "custom", path="yolov5/runs/train/exp8/weights/best.pt")

# Get the number of classes in the custom model
num_classes = len(custom_model.names)

# Modify the final layer of the pre-trained model to match the number of classes in the custom model
pretrained_model.model.model[-1] = torch.nn.Conv2d(
    in_channels=pretrained_model.model.model[-1].conv.in_channels,
    out_channels=num_classes * (5 + num_classes),  # 5 is for the bounding box attributes
    kernel_size=pretrained_model.model.model[-1].conv.kernel_size,
    stride=pretrained_model.model.model[-1].conv.stride,
    padding=pretrained_model.model.model[-1].conv.padding
)

# Load the custom weights into the modified pre-trained model
pretrained_model.model.load_state_dict(custom_model.model.state_dict(), strict=False)

# Define the folder containing the videos
video_folder = 'videos'

# Define the classes you are interested in
target_classes = ['squash_racket', 'squash_ball', 'person']  # Replace with your actual class names

# Confidence threshold
conf_threshold = 0.25

# Define augmentation pipeline
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.Rotate(limit=15, p=0.5),
    A.MotionBlur(p=0.2),
    A.CLAHE(p=0.2),
    A.ColorJitter(p=0.2),
    A.RandomScale(scale_limit=0.2, p=0.5),
    ToTensorV2()
])

# Helper function: Apply augmentations to a single frame
def apply_augmentation(frame):
    augmented_frame = transform(image=frame)["image"].numpy().transpose(1, 2, 0)
    return augmented_frame

# Helper function: Apply augmentations to images in a folder
def apply_augmentation_to_folder(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for image_file in os.listdir(input_folder):
        if image_file.endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(input_folder, image_file)
            image = cv2.imread(image_path)
            augmented_image = transform(image=image)["image"].numpy().transpose(1, 2, 0)
            output_path = os.path.join(output_folder, image_file)
            cv2.imwrite(output_path, augmented_image)

# Apply augmentations to the images in the specified folders
apply_augmentation_to_folder('dataset/images/train/squash_racket', 'dataset/images/train/squash_racket_augmented')
apply_augmentation_to_folder('dataset/images/train/squash_ball', 'dataset/images/train/squash_ball_augmented')

# Helper function: Detect players using pre-trained model
def detect_person(frame):
    person_results = pretrained_model(frame)
    person_boxes = []
    for *box, conf, cls in person_results.xyxy[0]:
        if pretrained_model.names[int(cls)] == 'person' and conf > conf_threshold:
            person_boxes.append(box)
            cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 0), 2)  # Draw person bbox
    return frame, person_boxes

# Helper function: Detect objects within the player ROI
def detect_within_roi(frame, player_box, model):
    roi = frame[int(player_box[1]):int(player_box[3]), int(player_box[0]):int(player_box[2])]
    results = model(roi)
    for *box, conf, cls in results.xyxy[0]:
        box[0] += player_box[0]
        box[1] += player_box[1]
        box[2] += player_box[0]
        box[3] += player_box[1]
        label = model.names[int(cls)]
        if label in target_classes and conf > conf_threshold:
            cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
            cv2.putText(frame, f'{label} {conf:.2f}', (int(box[0]), int(box[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    return frame

# Helper function: Filter small detections
def filter_small_detections(results, min_size=20):
    filtered_results = []
    for *box, conf, cls in results.xyxy[0]:
        width = box[2] - box[0]
        height = box[3] - box[1]
        if width > min_size and height > min_size:
            filtered_results.append((*box, conf, cls))
    return filtered_results

# Temporal smoothing buffer
detection_buffer = deque(maxlen=10)

# Helper function: Apply temporal smoothing to detections
def temporal_smoothing(detections):
    detection_buffer.append(detections)
    averaged_detections = []
    for detection in zip(*detection_buffer):
        averaged_box = [sum(x) / len(x) for x in zip(*detection)]
        averaged_detections.append(averaged_box)
    return averaged_detections

# Process each video in the folder
for video_file in os.listdir(video_folder):
    if video_file.endswith(".mp4"):
        cap = cv2.VideoCapture(os.path.join(video_folder, video_file))
        out = None

        # Get the video resolution and frame rate
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        # Initialize video writer once
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(f'annotated_{video_file}', fourcc, fps, (frame_width, frame_height))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Apply data augmentation
            frame = apply_augmentation(frame)

            # Detect players first using the person detection model
            frame, person_boxes = detect_person(frame)

            # Loop over detected persons and detect squash-related objects (balls, rackets) within ROIs
            for person_box in person_boxes:
                frame = detect_within_roi(frame, person_box, custom_model)

            # Write the annotated frame to the output video
            out.write(frame)

            # Display the frame with bounding boxes
            cv2.imshow('Annotated Video', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        out.release()

cv2.destroyAllWindows()





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








import torch
import os
from ultralytics import YOLO
model=YOLO('yolov8m-pose.pt')
video_folder = 'videos'


# Process each video in the folder
for video_file in os.listdir(video_folder):
    if video_file.endswith(".mp4"):
        results=model(source=video_folder+"/"+video_file, show=True, conf=0.5, save=True)







import os
from ultralytics import YOLO

# Define paths
video_folder = "C:/Users/default.DESKTOP-7FKFEEG/vision/videos"
model_path = "C:/Users/default.DESKTOP-7FKFEEG/vision/final-train/train/weights/best.pt"

# Load the trained model
model = YOLO(model_path)

# Process each video file in the folder
for video_file in os.listdir(video_folder):
    if video_file.endswith(".mp4"):
        video_path = os.path.join(video_folder, video_file)
        results = model(source=video_path, show=True, conf=0.05, save=True)




import torch
import os
from ultralytics import YOLO
#posemodel=YOLO('yolov8m-pose.pt')
video_folder = 'videos'
segmodel=YOLO('yolov8s-seg.pt')


# Process each video in the folder
for video_file in os.listdir(video_folder):
    if video_file.endswith(".mp4"):
        results=segmodel(source=video_folder+"/"+video_file, show=True, conf=0.685, save=True)






to get squash ball


import cv2
from ultralytics import YOLO
import os
from squash import player
#segmodel = YOLO('yolov8s-seg.pt')
video_folder = 'full-games'
posemodel=YOLO('models/yolo11m-pose.pt')
conf=0.9
for video_file in os.listdir(video_folder):
    if video_file.endswith(".mp4"):
        path = video_folder + "/" + video_file
        cap=cv2.VideoCapture(path)
        while cap.isOpened():
            success, frame=cap.read()
            if success:
                results=posemodel(frame, conf=conf)
                while(results[0].boxes is not None):
                    people=0
                    for class_id in results[0].boxes.cls:
                        if class_id==0:
                            people+=1
                    if people<2:
                        conf*=0.9
                        results=posemodel(frame, conf=conf)
                    else:
                        break
                    if(conf<0.1):
                        break
                annotated_frame=results[0].plot()
                cv2.imshow('Annotated Frame', annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break
cap.release()
cv2.destroyAllWindows()




"""

from ultralytics import YOLO

# Load the trained model
model = YOLO("datasets\\squash_ball\\train7\\weights\\best.pt")
results = model(source="main-video.mp4", show=True, conf=0.3, save=True, stream=True)
