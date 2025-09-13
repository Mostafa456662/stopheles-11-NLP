import json
from typing import Dict, Any
from tasks.gemma import generate
from tasks.explain import explain
from tasks.classify import classify
from tasks.summarise import summarise_paper
from paper_identifier import identify_paper

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


def confidence(user_input, functions_desc):
    prompt = f"""You are a function selector for a ML paper assistant.

    AVAILABLE FUNCTIONS:
    {functions_desc}

    Based on the user's request, select the appropriate function and extract the required parameters.
    return just one thing and one thing only, a score of how confident you are in your choice, note that 
    it is entirely possible that the use is not picking a function at all and is just asking about something
    else, if that is the case, output a low confidence score, whatever you do dont give me anything other than
    a number between 0 and 1

    USER REQUEST: "{user_input}"""

    confidence_score = float(generate(prompt=prompt))

    return confidence_score
print(confidence("hi"))