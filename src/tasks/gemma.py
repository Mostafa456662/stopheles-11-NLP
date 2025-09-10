import ollama


def generate(
    messages: list[dict[str, str]] | None = None,
    prompt: str = "",
    max_tokens: int = 500,
    verbose=True,
) -> str:
    """Generate response using gemma"""
    if messages:
        full_messages = [{"role": "user", "content": prompt}] + messages
    else:
        full_messages = [{"role": "user", "content": prompt}]

    stream = ollama.chat(
        model="gemma3:4b",
        messages=full_messages,
        stream=True,
        options={"num_predict": max_tokens, "temperature": 0.3},
    )

    response = ""
    if verbose:
        for chunk in stream:
            content = chunk["message"]["content"]
            print(content, end="", flush=True)
            response += content

    print()
    return response
