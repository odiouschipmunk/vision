import os
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logging

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose

# Define directories
video_dir = 'videos'
output_dir = 'processed_data'
os.makedirs(output_dir, exist_ok=True)

# Parameters
IMG_SIZE = 224
SEQ_LENGTH = 30  # Number of frames per sequence
BATCH_SIZE = 8
EPOCHS = 10

def extract_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    with mp_pose.Pose() as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = pose.process(rgb_frame)
            if result.pose_landmarks:
                frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
                frames.append(frame)
            if len(frames) == SEQ_LENGTH:
                break
    cap.release()
    return frames

def load_data(video_dir):
    X, y = [], []
    for filename in os.listdir(video_dir):
        if filename.endswith('.mp4'):
            video_path = os.path.join(video_dir, filename)
            frames = extract_frames(video_path)
            if len(frames) == SEQ_LENGTH:
                X.append(frames)
                # Assuming labels are encoded in the filename, e.g., 'action_label.mp4'
                label = filename.split('_')[0]
                y.append(label)
    return np.array(X), np.array(y)

def preprocess_data(X, y):
    X = X / 255.0  # Normalize pixel values
    y = tf.keras.utils.to_categorical(y, num_classes=len(set(y)))  # One-hot encode labels
    return X, y

def create_model(input_shape, num_classes):
    model = tf.keras.Sequential([
        tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(32, (3, 3), activation='relu'), input_shape=input_shape),
        tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling2D((2, 2))),
        tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(64, (3, 3), activation='relu')),
        tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling2D((2, 2))),
        tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(128, (3, 3), activation='relu')),
        tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling2D((2, 2))),
        tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten()),
        tf.keras.layers.LSTM(64, return_sequences=False),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def main():
    # Ensure TensorFlow uses the GPU
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if physical_devices:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        print("TensorFlow is using the GPU")
    else:
        print("No GPU found. TensorFlow will use the CPU.")

    # Load and preprocess data
    X, y = load_data(video_dir)
    X, y = preprocess_data(X, y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create model
    input_shape = (SEQ_LENGTH, IMG_SIZE, IMG_SIZE, 3)
    num_classes = y.shape[1]
    model = create_model(input_shape, num_classes)

    # Train model
    model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(X_test, y_test))

    # Save model
    model.save('squash_action_model.h5')

    # Evaluate model
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f'Test accuracy: {accuracy:.2f}')

if __name__ == "__main__":
    main()