import torchvision.transforms as A
import cv2
import os

# Define augmentation pipeline
transform = A.Compose(
    [
        A.RandomHorizontalFlip(p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    ]
)


# Apply augmentations to the dataset
def apply_augmentation(image_dir, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for image_file in os.listdir(image_dir):
        image_path = os.path.join(image_dir, image_file)
        image = cv2.imread(image_path)
        augmented_image = transform(image=image)["image"]

        save_path = os.path.join(save_dir, image_file)
        cv2.imwrite(save_path, augmented_image.numpy().transpose(1, 2, 0))


# Example usage
apply_augmentation(
    "dataset/images/train/squash_ball", "dataset/images/train/squash_ball"
)
apply_augmentation(
    "dataset/images/train/squash_racket", "dataset/images/train/squash_racket"
)
