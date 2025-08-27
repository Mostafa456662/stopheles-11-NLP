import torch
import dotenv

from transformers import pipeline
from huggingface_hub import login



hf_token = "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"  
login(hf_token)

# Define the model ID
model_id = ""


pipe = pipeline(
    "text-generation",
    model=model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)


messages = [
        {"role": "user", "content": "Hi"},
]

# Generate the response
outputs = pipe(
    messages,
    max_new_tokens=256,
    do_sample=True,
    temperature=0.7,
    top_p=0.9,
)

# Print the generated response (last message in the conversation)
print(outputs[0]["generated_text"][-1]["content"])