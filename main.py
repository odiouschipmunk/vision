from squash import player
from squash import ball
from squash import court
from squash import ball

import cv2
from ultralytics import YOLO
import os
players-{}
video_folder = 'full-games'
posemodel=YOLO('models/yolo11s-pose.pt')
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