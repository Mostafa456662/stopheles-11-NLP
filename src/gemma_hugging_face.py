import torch
import dotenv
import os
from transformers import pipeline
from huggingface_hub import login

# Load environment variables from .env file
dotenv.load_dotenv()

# Get Hugging Face token from environment variable
hf_token = os.getenv("hugging_face_token")
login(hf_token)


model_id = "google/gemma-2b-it"

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