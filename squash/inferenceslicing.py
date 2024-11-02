import numpy as np
import torch
from ultralytics.engine.results import Results, Boxes

def inference_slicing(frame, model, num_slices=4):
    """
    Perform inference slicing on a frame and merge detections.

    Parameters:
    - frame: The input frame to be sliced.
    - model: The detection model to be used.
    - num_slices: Number of slices to divide the frame into.

    Returns:
    - final_detections: A list containing the merged detection result.
    """
    height, width, _ = frame.shape
    slice_height = height // num_slices
    detections = []

    # Slice the frame and perform detection on each slice
    for i in range(num_slices):
        y_start = i * slice_height
        y_end = (i + 1) * slice_height if i < num_slices else height
        slice_frame = frame[y_start:y_end, :]
        detection = model(slice_frame)
        detections.append((y_start, detection))

    # Merge detections
    all_boxes = []
    for y_offset, detection in detections:
        if hasattr(detection, 'boxes') and detection.boxes is not None:
            boxes = detection.boxes
            # Adjust box coordinates to the original frame
            boxes.xyxy[:, [1, 3]] += y_offset
            all_boxes.append(boxes)

    # Concatenate all boxes
    if all_boxes:
        merged_boxes_xyxy = torch.cat([b.xyxy for b in all_boxes], dim=0)
        merged_scores = torch.cat([b.conf for b in all_boxes], dim=0)
        merged_cls = torch.cat([b.cls for b in all_boxes], dim=0)
        merged_boxes = Boxes(merged_boxes_xyxy)
        merged_boxes.conf = merged_scores
        merged_boxes.cls = merged_cls
    else:
        merged_boxes = None

    # Create a final detection result
    final_detection = Results(orig_img=frame, path='', names=model.names)
    final_detection.boxes = merged_boxes
    final_detection.orig_shape = frame.shape

    # Return a list containing the final_detection to match your code structure
    return [final_detection]

# Usage in your Functions.py
# merged_result = inference_slicing(frame, ballmodel)
# ball = merged_result
# Then you can access ball[0].boxes as in your code