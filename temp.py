import os
import random
from PIL import Image
import numpy as np
import shutil

# Define paths
train_images_path = 'dataset/images/train'
val_images_path = 'dataset/images/val'
train_labels_path = 'dataset/labels/train'
val_labels_path = 'dataset/labels/val'

# Class names and their corresponding indices
class_names = ['squash_racket', 'squash_ball']
class_indices = {name: idx for idx, name in enumerate(class_names)}

# Function to load images and generate labels
def load_images_and_generate_labels(image_path, label_path):
    images = []
    labels = []
    for root, dirs, files in os.walk(image_path):
        for file in files:
            if file.endswith('.jpg') or file.endswith('.png'):
                img_path = os.path.join(root, file)
                img = Image.open(img_path)
                images.append(np.array(img))
                # Extract label from directory structure
                label_name = os.path.basename(os.path.dirname(img_path))
                label_index = class_indices[label_name]
                labels.append(label_index)
                # Generate corresponding label file with dummy bounding box values
                label_file_path = os.path.join(label_path, label_name, file.replace('.jpg', '.txt').replace('.png', '.txt'))
                os.makedirs(os.path.dirname(label_file_path), exist_ok=True)
                with open(label_file_path, 'w') as f:
                    # Dummy bounding box values (centered and covering the whole image)
                    f.write(f"{label_index} 0.5 0.5 1.0 1.0")
    return images, labels

# Function to move a percentage of files from one directory to another
def move_files(src_dir, dest_dir, percentage):
    for class_name in class_names:
        src_class_dir = os.path.join(src_dir, class_name)
        dest_class_dir = os.path.join(dest_dir, class_name)
        os.makedirs(dest_class_dir, exist_ok=True)
        
        files = [f for f in os.listdir(src_class_dir) if f.endswith('.jpg') or f.endswith('.png')]
        num_files_to_move = int(len(files) * percentage)
        files_to_move = random.sample(files, num_files_to_move)
        
        for file in files_to_move:
            src_file = os.path.join(src_class_dir, file)
            dest_file = os.path.join(dest_class_dir, file)
            shutil.move(src_file, dest_file)
            
            # Move corresponding label file
            label_file = file.replace('.jpg', '.txt').replace('.png', '.txt')
            src_label_file = os.path.join(train_labels_path, class_name, label_file)
            dest_label_file = os.path.join(val_labels_path, class_name, label_file)
            if os.path.exists(src_label_file):
                shutil.move(src_label_file, dest_label_file)

# Move 5% of the training images and labels to the validation set
move_files(train_images_path, val_images_path, 0.05)

# Load train and validation datasets
train_images, train_image_labels = load_images_and_generate_labels(train_images_path, train_labels_path)
val_images, val_image_labels = load_images_and_generate_labels(val_images_path, val_labels_path)

# Debugging statements
print(f'Train images path: {train_images_path}')
print(f'Validation images path: {val_images_path}')
print(f'Train labels path: {train_labels_path}')
print(f'Validation labels path: {val_labels_path}')
print(f'Loaded {len(train_images)} training images and generated {len(train_image_labels)} training labels.')
print(f'Loaded {len(val_images)} validation images and generated {len(val_image_labels)} validation labels.')

# Check if validation images are loaded correctly
if len(val_images) == 0:
    # Additional debugging to list files in validation directory
    for root, dirs, files in os.walk(val_images_path):
        print(f"Checking directory: {root}")
        for file in files:
            print(f"Found file: {file}")
            if not (file.endswith('.jpg') or file.endswith('.png')):
                print(f"File {file} is not a valid image file.")
    raise Exception(f"No validation images found in {val_images_path}")

# Create a YOLOv5 configuration file
data_yaml = f"""
train: {os.path.abspath(train_images_path)}
val: {os.path.abspath(val_images_path)}

nc: 2
names: ['squash_racket', 'squash_ball']
"""

# Ensure the dataset directory exists
os.makedirs('dataset', exist_ok=True)

with open('dataset/data.yaml', 'w') as f:
    f.write(data_yaml)

# Train YOLOv5 model
os.system('python yolov5/train.py --img 640 --batch 16 --epochs 35 --data dataset/data.yaml --weights yolov5s.pt')