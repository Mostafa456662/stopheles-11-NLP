import json
import os
from typing import Dict, Any
from gemma import generate
from tasks.summarise import summarise_paper
from tasks.explain import explain
from tasks.classify import classify_paper


function_implementations = {
    "summarise_paper": summarise_paper,
    "explain": explain,
    "classify_paper": classify_paper,
}


available_functions = {
    "summarise_paper": {
        "description": "summarise a machine learning paper",
        "parameters": {"paper_path": "string - path to the paper file"},
    },
    "explain": {
        "description": "Search for and explain a ML concept across papers",
        "parameters": {
            "query": "the query requesting explanation for a certain topic or passage"
        },
    },
    "classify_paper": {
        "description": "Classify a paper into appropriate folders",
        "parameters": {"paper_path": "string - path to the paper to classify"},
    },
}

functions_desc = "\n".join(
    [
        f"- {name}: {info['description']}\n  Parameters: {info['parameters']}"
        for name, info in available_functions.items()
    ]
)


def select_and_call_function(user_input: str) -> str:
    """Use LLM to select appropriate function and extract parameters"""

    # Create function selection prompt
    functions_desc = "\n".join(
        [
            f"- {name}: {info['description']}\n  Parameters: {info['parameters']}"
            for name, info in available_functions.items()
        ]
    )

    prompt = f"""You are a function selector for a ML paper assistant.

    AVAILABLE FUNCTIONS:
    {functions_desc}

    Based on the user's request, select the appropriate function and extract the required parameters.

    Respond with a JSON object in this format:
    {{
        "function": "function_name",
        "parameters": {{
            "param1": "value1",
            "param2": "value2"
        }},
        
    }}



    USER REQUEST: "{user_input}"""
    try:
        # Get function selection from LLM
        llm_response = generate(prompt=prompt)
        # Extract JSON from response (in case there's extra text)
        start = llm_response.find("{")
        end = llm_response.rfind("}") + 1
        json_str = llm_response[start:end]

        function_call = json.loads(json_str)

        # Validate function exists
        function_name = function_call.get("function")
        if function_name not in function_implementations:
            return f"Error: Unknown function '{function_name}'"

        # Extract parameters
        parameters = function_call.get("parameters", {})

        # Call the selected function with the correct parameter format
        selected_function = function_implementations[function_name]

        if function_name == "summarise_paper":
            print("summarise_pape")
            result = selected_function(paper_path=parameters.get("paper_path", ""))
        elif function_name == "explain":
            print("explain")
            result = selected_function(query=parameters.get("query", ""))
        elif function_name == "classify_paper":
            print("classify")
            result = selected_function(paper_path=parameters.get("paper_path", ""))
        else:
            result = selected_function(**parameters)

        return result

    except json.JSONDecodeError as e:
        return f"Error parsing function selection: {str(e)}"
    except Exception as e:
        return f"Error executing function: {str(e)}"


if __name__ == "__main__":
    # Test the function selector
    test_inputs = [
        "Can you summarise the paper at /path/to/paper.pdf?",
        "Explain what is a transformer model",
        "Classify the paper located at /papers/new_research.txt",
    ]

    for user_input in test_inputs:
        print(f"\nUser: {user_input}")
        result = select_and_call_function(user_input)
        print(f"Assistant: {result}")
        print("-" * 50)
