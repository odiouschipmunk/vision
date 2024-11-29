from squash import Functions
from ultralytics import YOLO
import cv2
nposemodel=YOLO("models/yolo11m-pose.pt")
tracker=Functions.setup_keypoint_tracking(pose_model=nposemodel)
cap=cv2.VideoCapture('main.mp4')
while True:
    ret, frame = cap.read()
    if not ret:
        break
        
    tracked_objects, annotated_frame = tracker(frame)
    
    cv2.imshow('Tracked Poses', annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()