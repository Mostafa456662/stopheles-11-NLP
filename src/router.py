import re
import json
from typing import Dict, Any
from tasks.gemma import generate
from tasks.explain import explain
from tasks.classify import classify
from tasks.summarise import summarise_paper
from paper_identifier import identify_paper


def extract_number(text):
    match = re.search(r"[-+]?\d*\.?\d+", text)
    if match:
        return float(match.group())
    return None


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
            "query": "the query requesting explanation for a certain topic or passage, include any additional details that show what the user specifically wants"
        },
    },
    "classify": {
        "description": "Classify a paper into appropriate folders",
        "parameters": {"new_paper": "string - path to the paper to classify"},
    },
}

functions_desc = "\n".join(
    [
        f"- {name}: {info['description']}\n  Parameters: {info['parameters']}"
        for name, info in available_functions.items()
    ]
)


def get_confidence(user_input, functions_desc):
    prompt = f"""You are a function selector for a ML paper assistant.

    AVAILABLE FUNCTIONS:
    summarise_paper: summarises a paper
    explain: explains a certain passage or concept in a paper
    classify: classifies a paper into a folder of similar papers

    Based on the user's request, select the appropriate function and extract the required parameters.
    return just one thing and one thing only, a score of how confident you are in your choice, note that 
    it is entirely possible that the use is not picking a function at all and is just asking about something
    else, if that is the case, output a low get_confidence score, whatever you do dont give me anything other than
    a number between 0 and 1

    USER REQUEST: "{user_input}"""

    confidence_score = float(extract_number(generate(prompt=prompt)))

    return confidence_score


def select_and_call_function(user_input: str, function_desc) -> str:
    """Use LLM to select appropriate function and extract parameters"""

    # Create function selection prompt

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

        if function_name == "summarise_paper" or function_name == "classify":
            print("finding paper")
            paper = identify_paper(paper_name=parameters.get("new_paper", ""))[0]

        if function_name == "summarise_paper":
            print(paper)
            result = selected_function(file_path=paper)
        elif function_name == "classify":
            result = selected_function(new_paper=paper)
        elif function_name == "explain":
            print("explain")
            result = selected_function(query=parameters.get("query", ""))

        return result

    except json.JSONDecodeError as e:
        return f"Error parsing function selection: {str(e)}"
    except Exception as e:
        return f"Error executing function: {str(e)}"


def route(user_input, messages=""):
    functions_desc = "\n".join(
        [
            f"- {name}: {info['description']}\n  Parameters: {info['parameters']}"
            for name, info in available_functions.items()
        ]
    )
    confidence = get_confidence(user_input=user_input, functions_desc=functions_desc)

    if confidence >= 0.8:
        select_and_call_function(user_input=user_input, function_desc=functions_desc)
    else:
        prompt = f"""
        you are an ML assistant tool that helps with explaining concepts
        USER_INPUT: {user_input}

        """
        generate(prompt=prompt, messages=messages)


if __name__ == "__main__":
    user_input = "explain the fine tune lvit paper"

    route(user_input=user_input)
