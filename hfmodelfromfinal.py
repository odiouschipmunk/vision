from dotenv import load_dotenv
load_dotenv()
with open("final.txt", "r", encoding='utf-8') as file:
    text = file.read()
import transformers
import torch

model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)

messages = [
    {"role": "system", "content": "You are a squash coach analyzing data about a squash game. You must think through all your answers step by step. In the end, you are to return what happened in the game and analyze patterns. Make sure to be super specific and DO NOT say stuff like 'the game was strategic'. Instead, say 'the player hit the ball to the left corner of the court to force the opponent to run to the left side of the court'. Remember to be as specific as possible."},
    {"role": "user", "content": text},
]
import time
start=time.time()
outputs = pipeline(
    messages,
    max_new_tokens=8192,
)
print(outputs[0]["generated_text"][-1])
print(time.time()-start)