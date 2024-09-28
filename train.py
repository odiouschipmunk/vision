import os
import av
import torch
import numpy as np
from huggingface_hub import hf_hub_download

model_id = "MCG-NJU/videomae-base"
from transformers import AutoProcessor, AutoModelForCausalLM

processor = AutoProcessor.from_pretrained("lmms-lab/LLaVA-NeXT-Video-7B-DPO")
model = AutoModelForCausalLM.from_pretrained("lmms-lab/LLaVA-NeXT-Video-7B-DPO")


def read_video_pyav(container, indices):
    '''
    Decode the video with PyAV decoder.
    Args:
        container (`av.container.input.InputContainer`): PyAV container.
        indices (`List[int]`): List of frame indices to decode.
    Returns:
        result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
    '''
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

def fine_tune_model(video_dir):
    for video_file in os.listdir(video_dir):
        if video_file.endswith('.mp4'):
            video_path = os.path.join(video_dir, video_file)
            container = av.open(video_path)

            # Sample uniformly 8 frames from the video, can sample more for longer videos
            total_frames = container.streams.video[0].frames
            indices = np.arange(0, total_frames, total_frames / 8).astype(int)
            clip = read_video_pyav(container, indices)

            # Define a chat history and use `apply_chat_template` to get correctly formatted prompt
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What is the player doing wrong in this video?"},
                        {"type": "video"},
                    ],
                },
            ]

            prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
            inputs_video = processor(text=prompt, videos=clip, padding=True, return_tensors="pt").to(model.device)

            # Fine-tune the model
            output = model.generate(**inputs_video, max_new_tokens=100, do_sample=False)
            print(processor.decode(output[0][2:], skip_special_tokens=True))

if __name__ == "__main__":
    video_dir = 'videos'
    fine_tune_model(video_dir)

    # Save the fine-tuned model
    model.save_pretrained("fine_tuned_squash_coach_model")
    processor.save_pretrained("fine_tuned_squash_coach_processor")