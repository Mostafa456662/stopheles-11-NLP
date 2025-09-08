import ollama
from gemma import generate
import os
from dotenv import load_dotenv
import json

load_dotenv()



# Chat Loop

messages = []

while True:
    user_input = input("You: ").strip()
    if user_input.lower() in ["quit", "exit", "bye"]:
        print("👋 Goodbye!")
        break
    if not user_input:
        continue
    # Add user message
    messages.append({"role": "user", "content": user_input})

    # Stream response from Ollama
    print("Assistant: ", end="", flush=True)
    
    response = generate(messages=messages, prompt=os.getenv("system_prompt"))
    print()

    # Add model response to history
    messages.append({"role": "assistant", "content": response})
