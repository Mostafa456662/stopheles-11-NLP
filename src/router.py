import json
import os
from typing import Dict, Any
from tasks.gemma import generate
from tasks.explain import explain
from tasks.classify import classify
from tasks.summarise import summarise_paper


function_implementations = {
    "summarise_paper": summarise_paper,
    "explain": explain,
    "classify": classify,
}


available_functions = {
    "summarise_paper": {
        "description": "summarise a machine learning paper",
        "parameters": {
            "new_paper": "string - name of the paper",
            "folder_path": "path to the folder which contains the folders of papers to be classified into",
        },
    },
    "explain": {
        "description": "Search for and explain a ML concept across papers",
        "parameters": {
            "query": "the query requesting explanation for a certain topic or passage"
        },
    },
    "classify": {
        "description": "Classify a paper into appropriate folders",
        "parameters": {"new_paper_path": "string - path to the paper to classify"},
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
            print("summarise_paper")
            result = selected_function(paper_path=parameters.get("paper_path", ""))
        elif function_name == "explain":
            print("explain")
            result = selected_function(query=parameters.get("query", ""))
        elif function_name == "classify":
            print("classify")
            result = selected_function(
                new_paper_path=parameters.get("new_paper_path", "")
            )
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
        "Can you classify the paper at doc\papers\Fine_Tune_LViT_for_zero_shot_classifiction[1].pdf",
    ]

    for user_input in test_inputs:
        print(f"\nUser: {user_input}")
        result = select_and_call_function(user_input)

        print("-" * 50)
