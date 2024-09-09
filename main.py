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