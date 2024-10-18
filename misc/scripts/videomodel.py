import cv2
import numpy as np
from transformers import VideoTextToTextModel, VideoTextToTextProcessor

class VideoTextModel:
        def __init__(self, video_path, prompt):
            self.video_path = video_path
            self.prompt = prompt
            self.video_capture = cv2.VideoCapture(video_path)
            self.processor = VideoTextToTextProcessor.from_pretrained("huggingface/video-text-to-text")
            self.model = VideoTextToTextModel.from_pretrained("huggingface/video-text-to-text")

        def process_video(self):
            frames = []
            while self.video_capture.isOpened():
                ret, frame = self.video_capture.read()
                if not ret:
                    break
                frames.append(frame)
                cv2.imshow('Frame', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            self.video_capture.release()
            cv2.destroyAllWindows()
            return frames

        def generate_text(self):
            frames = self.process_video()
            inputs = self.processor(text=self.prompt, videos=frames, return_tensors="pt")
            outputs = self.model.generate(**inputs)
            return self.processor.decode(outputs[0], skip_special_tokens=True)

if __name__ == "__main__":
    video_path = "annotated.mp4"
    prompt = "You are a squash coach, coach me as I am the guy in the black shirt(id 2)"
    model = VideoTextModel(video_path, prompt)
    model.process_video()
    print(model.generate_text())
    
    if __name__ == "__main__":
        video_path = "annotated.mp4"
        prompt = "You are a squash coach, coach me as I am the guy in the black shirt(id 2)"
        model = VideoTextModel(video_path, prompt)
        print(model.generate_text())