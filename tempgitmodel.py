import os
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential
from transformers import GPT2Tokenizer


endpoint = "https://models.inference.ai.azure.com"
model_name = "AI21-Jamba-1.5-Large"
token = os.environ["GITHUB_TOKEN"]

client = ChatCompletionsClient(
    endpoint=endpoint,
    credential=AzureKeyCredential(token),
)
import json
squash_data=json.load(open("output/final.json"))
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

messages = [
    "You are a squash coach reading through big data. Tell me exactly what both players are doing based on the following json data.",
    f"{squash_data}",
    "How exactly do you know? Use specific data points from the data."
]

total_tokens = sum(len(tokenizer.encode(message)) for message in messages)

# print(f"Total tokens in request: {total_tokens}")
# response = client.complete(
#     messages=[
#         SystemMessage(content="You are a squash coach reading through big data. Tell me exactly what both players are doing based on the following json data."),
#         UserMessage(content=f"{squash_data}"),
#         UserMessage(content="How exactly do you know? Use specific data points from the data.")
#     ],
    
#     temperature=1.0,
#     top_p=1.0,
#     max_tokens=4000,
#     model=model_name
# )

# print(response.choices[0].message.content)