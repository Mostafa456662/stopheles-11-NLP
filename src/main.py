import ollama
from dotenv import load_dotenv


# local imports
from router import route
from tasks.gemma import generate


load_dotenv()


# Chat Loop

messages = []

while True:
    user_input = input("You: ").strip()
    if user_input.lower() in ["quit", "exit", "bye"]:
        print("Goodbye!")
        break
    if not user_input:
        continue
    # Add user message
    messages.append({"role": "user", "content": user_input})

    # Stream response from Ollama
    print("Assistant: ", end="", flush=True)
    response = route(user_input=messages)

    # Add model response to history
    messages.append({"role": "assistant", "content": response})
