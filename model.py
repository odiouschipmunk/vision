import uuid
import requests
import cv2
import torch
from transformers import LlavaNextVideoProcessor, LlavaNextVideoForConditionalGeneration
from PIL import Image
device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "llava-hf/LLaVA-NeXT-Video-7B-hf"

model = LlavaNextVideoForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
).to(device)

processor = LlavaNextVideoProcessor.from_pretrained(model_id)

def sample_frames(url, num_frames):
    response = requests.get(url)
    path_id = str(uuid.uuid4())

    path = f"./{path_id}.mp4"

    with open(path, "wb") as f:
         f.write(response.content)

    video = cv2.VideoCapture(path)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    interval = total_frames // num_frames
    frames = []
    for i in range(total_frames):
        ret, frame = video.read()
        if not ret:
            continue
        if i % interval == 0:
            pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            frames.append(pil_img)
    video.release()
    return frames

conversation = [
    {

        "role": "user",
        "content": [
            {"type": "text", "text": "Why is this video funny?"},
            {"type": "video"},
            ],
    },
]

prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

video_url = "https://huggingface.co/spaces/merve/llava-interleave/resolve/main/cats_1.mp4"
video = sample_frames(video_url, 8)

inputs = processor(text=prompt, videos=video, padding=True, return_tensors="pt").to(model.device)

output = model.generate(**inputs, max_new_tokens=100, do_sample=False)
print(processor.decode(output[0][2:], skip_special_tokens=True))

# Why is this video funny? ASSISTANT: The humor in this video comes from the cat's facial expression and body language. The cat appears to be making a funny face, with its eyes squinted and mouth open, which can be interpreted as a playful or mischievous expression. Cats often make such faces when they are in a good mood or are playful, and this can be amusing to people who are familiar with their behavior. The combination of the cat's expression and the close-
