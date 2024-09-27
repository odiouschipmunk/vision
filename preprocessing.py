# preprocess_and_annotate.py

import cv2
import os
import numpy as np
from openpose import pyopenpose as op

def preprocess_frame(frame):
    # Example preprocessing: normalize
    normalized_frame = frame / 255.0
    return normalized_frame

def annotate_frame(frame, pose_keypoints):
    # Draw OpenPose keypoints
    for person in pose_keypoints:
        for i in range(len(person)):
            cv2.circle(frame, (int(person[i][0]), int(person[i][1])), 5, (0, 0, 255), -1)
    return frame

def process_videos(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    params = dict()
    params["model_folder"] = "models/"

    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()

    for filename in os.listdir(input_dir):
        if filename.endswith('.mp4'):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)

            cap = cv2.VideoCapture(input_path)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, 20.0, (width, height))

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                datum = op.Datum()
                datum.cvInputData = frame
                opWrapper.emplaceAndPop([datum])

                preprocessed_frame = preprocess_frame(frame)
                annotated_frame = annotate_frame(preprocessed_frame, datum.poseKeypoints)
                out.write((annotated_frame * 255).astype('uint8'))

            cap.release()
            out.release()

if __name__ == "__main__":
    input_dir = 'videos'
    output_dir = 'annotated_videos'

    process_videos(input_dir, output_dir)