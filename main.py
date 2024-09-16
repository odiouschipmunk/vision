import os
import av
import numpy as np
import torch
from transformers import LlavaNextVideoProcessor, LlavaNextVideoForConditionalGeneration

# Path to the videos folder
video_folder = '/videos'

# Model and processor initialization
model_id = "llava-hf/LLaVA-NeXT-Video-7B-hf"
model = LlavaNextVideoForConditionalGeneration.from_pretrained(
    model_id, 
    torch_dtype=torch.float16, 
    low_cpu_mem_usage=True, 
).to(0)
from pytube import YouTube
from urllib.error import HTTPError
import logging
import os
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Path to the text file containing YouTube links
input_file = "youtube_links.txt"

# Path to the folder where the videos will be saved
output_folder = "videos"

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Read the YouTube links from the input file
with open(input_file, "r") as file:
    youtube_links = file.readlines()

# Function to download video with retries
def download_video(link, output_folder, index, retries=3):
    for attempt in range(retries):
        try:
            # Log the URL being processed
            logging.info(f"Processing URL: {link}")

            # Validate the URL format
            if not link.startswith("https://www.youtube.com/watch?v="):
                raise ValueError(f"Invalid YouTube URL: {link}")

            # Download the video using pytube
            yt = YouTube(link)
            stream = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first()

            if stream:
                stream.download(output_path=output_folder, filename=f"video_{index+1}.mp4")
                logging.info(f"Downloaded video {index+1} from {link}")
                return True  # Download successful, exit function
            else:
                logging.error(f"No suitable stream found for {link}")
                return False
        except HTTPError as e:
            logging.error(f"HTTPError for {link}: {e}")
            if attempt + 1 < retries:
                logging.info(f"Retrying ({attempt + 1}/{retries})...")
                time.sleep(2)  # Wait for a short period before retrying
            else:
                logging.error(f"Failed to download after {retries} retries: {link}")
                return False
        except ValueError as e:
            logging.error(e)
            return False
        except Exception as e:
            logging.error(f"An error occurred for {link}: {e}")
            return False

# Download and save the videos with retries
for i, link in enumerate(youtube_links):
    link = link.strip()  # Remove any leading/trailing whitespaces or newlines
    download_video(link, output_folder, i)

logging.info("All videos downloaded and saved successfully!")

processor = LlavaNextVideoProcessor.from_pretrained(model_id)

# Function to read video using PyAV
def read_video_pyav(container, indices):
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])

# Function to load and preprocess videos
def load_videos(video_folder):
    video_data = []
    for video_file in os.listdir(video_folder):
        if video_file.endswith('.mp4'):
            container = av.open(os.path.join(video_folder, video_file))
            total_frames = container.streams.video[0].frames
            indices = np.arange(0, total_frames, total_frames / 8).astype(int)
            clip = read_video_pyav(container, indices)
            video_data.append(clip)
    return video_data

# Load and preprocess videos
video_data = load_videos(video_folder)

# Define a chat history and use `apply_chat_template` to get correctly formatted prompt
conversation = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Why is this video funny?"},
            {"type": "video"},
        ],
    },
]

# Process each video and generate output
for clip in video_data:
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs_video = processor(text=prompt, videos=clip, padding=True, return_tensors="pt").to(model.device)
    output = model.generate(**inputs_video, max_new_tokens=100, do_sample=False)
    print(processor.decode(output[0][2:], skip_special_tokens=True))