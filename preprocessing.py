import cv2
import os
import imgaug.augmenters as iaa
from PIL import Image
import numpy as np

# Extract frames from video
def extract_frames(video_path, output_dir, frame_rate=10):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    success = True

    while success:
        success, frame = cap.read()
        if frame_count % frame_rate == 0 and success:
            cv2.imwrite(f"{output_dir}/frame_{frame_count}.jpg", frame)
        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()

# Augmentation sequence
seq = iaa.Sequential([
    iaa.Fliplr(0.5),  # Flip 50% of the images
    iaa.Affine(rotate=(-10, 10)),  # Rotate between -10 to 10 degrees
    iaa.Multiply((0.8, 1.2))  # Brightness change 80-120%
])

# Augment image
def augment_image(image_path):
    image = Image.open(image_path)
    image_np = np.array(image)
    augmented_image = seq(image=image_np)
    return Image.fromarray(augmented_image)

# Example: Extract frames and augment
extract_frames('squash_video.mp4', 'output_frames', frame_rate=10)
augmented_image = augment_image("output_frames/frame_1000.jpg")
augmented_image.save("augmented_frame_1000.jpg")
