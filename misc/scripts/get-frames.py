import os
import cv2
import random
from tqdm import tqdm
def extract_random_frames(video_path, output_folder, num_frames=5000):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if frame_count < num_frames:
        print(f"Video {video_path} has less than {num_frames} frames.")
        return
    
    frame_indices = sorted(random.sample(range(frame_count), num_frames))
    
    count = 0
    for idx in tqdm(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        i=random.randint(0,100000000)
        if ret:
            frame_filename = os.path.join(output_folder, f"{i}_frame_{idx}.jpg")
            i=random.randint(0,100000000)
            cv2.imwrite(frame_filename, frame)
            count += 1
    
    cap.release()
    print(f"Extracted {count} frames from {video_path}")

def process_videos(videos_folder, output_folder, num_frames=5000):
    for filename in tqdm(os.listdir(videos_folder)):
        if filename.endswith(('.mp4', '.avi', '.mov', '.mkv')):
            video_path = os.path.join(videos_folder, filename)
            extract_random_frames(video_path, output_folder, num_frames)

if __name__ == "__main__":
    videos_folder = 'black_ball_games'
    output_folder = 'black_ball_frames'
    process_videos(videos_folder, output_folder)