from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

model = "ai21labs/AI21-Jamba-1.5-Large"

llm = LLM(model=model,
          tensor_parallel_size=8,
          max_model_len=220*1024,
          quantization="experts_int8",
         )

tokenizer = AutoTokenizer.from_pretrained(model)

messages = [
   {"role": "system", "content": "You are an ancient oracle who speaks in cryptic but wise phrases, always hinting at deeper meanings."},
   {"role": "user", "content": "Hello!"},
]

prompts = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

sampling_params = SamplingParams(temperature=0.4, top_p=0.95, max_tokens=100)
outputs = llm.generate(prompts, sampling_params)

generated_text = outputs[0].outputs[0].text
print(generated_text)