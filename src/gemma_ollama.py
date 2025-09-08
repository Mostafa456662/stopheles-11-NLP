import ollama

# Define the model and conversation
model_id = "gemma3:4b"
messages = [{"role": "user", "content": "Hi"}]

# Generate the response
response = ollama.chat(
    model=model_id,
    messages=messages,
    options={"temperature": 0.7, "top_p": 0.9, "max_tokens": 256},
)

# Extract and print the response
response = response["message"]["content"]
print(response)
