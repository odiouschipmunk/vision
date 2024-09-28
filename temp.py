import os
import shutil
import random

def split_dataset(image_dir, label_dir, train_ratio=0.95):
    images = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
    random.shuffle(images)
    
    train_size = int(len(images) * train_ratio)
    train_images = images[:train_size]
    val_images = images[train_size:]
    
    for img in train_images:
        shutil.move(os.path.join(image_dir, img), os.path.join(image_dir, 'train', img))
        label = img.replace('.jpg', '.txt')
        shutil.move(os.path.join(label_dir, label), os.path.join(label_dir, 'train', label))
    
    for img in val_images:
        shutil.move(os.path.join(image_dir, img), os.path.join(image_dir, 'val', img))
        label = img.replace('.jpg', '.txt')
        shutil.move(os.path.join(label_dir, label), os.path.join(label_dir, 'val', label))

# Example usage
split_dataset('dataset/images/train/squash_ball', 'dataset/labels/squash_ball')
split_dataset('dataset/images/train/squash_racket', 'dataset/labels/squash_racket')