#!/usr/bin/env python3
"""
Convert STOP dataset to CATS format for function calling evaluation.

This script extracts audio files, function definitions, and ground truth function calls
from the STOP dataset and converts them to the JSONL format required by CATS.
"""

import argparse
import json
import os
import shutil
from pathlib import Path
from tqdm import tqdm
import sys

# Add parent directory to path for importing STOP parsing functions
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import functions from the evaluation script
from FunctionalCall.eval_function_call import (
    parse_insl, insl_to_function_calls, extract_function_names_from_insl
)

def extract_openai_functions(dataset_path, output_file):
    """
    Extract OpenAI-compatible function definitions from STOP dataset.

    Args:
        dataset_path: Path to the STOP dataset
        output_file: Path to write the function definitions
    """
    # Load the STOP dataset
    function_definitions = {}
    intents_file = os.path.join(dataset_path, "intents.json")

    if not os.path.exists(intents_file):
        print(f"Error: Intent definitions file not found at {intents_file}")
        return None

    with open(intents_file, "r") as f:
        intents_data = json.load(f)

    # Convert STOP intents to OpenAI function definitions
    for intent in intents_data:
        intent_name = intent.get("name")
        if not intent_name:
            continue

        # Extract slot definitions
        slots = intent.get("slots", [])
        parameters = {
            "type": "object",
            "properties": {},
            "required": []
        }

        for slot in slots:
            slot_name = slot.get("name")
            if not slot_name:
                continue

            # Add to parameters
            parameters["properties"][slot_name.lower()] = {
                "type": "string",
                "description": slot.get("description", f"The {slot_name} parameter")
            }

            # Add to required parameters if mandatory
            if slot.get("required", False):
                parameters["required"].append(slot_name.lower())

        # Create function definition
        function_definitions[intent_name] = {
            "name": intent_name,
            "description": intent.get("description", f"Function for {intent_name}"),
            "parameters": parameters
        }

    # Write the function definitions to file
    with open(output_file, "w") as f:
        json.dump(list(function_definitions.values()), f, indent=2)

    print(f"Wrote {len(function_definitions)} function definitions to {output_file}")
    return list(function_definitions.values())

def extract_stop_utterances(dataset_path, audio_input_dir, output_dir, functions):
    """
    Extract utterances and audio from STOP dataset and create CATS-compatible JSONL.

    Args:
        dataset_path: Path to the STOP dataset
        audio_input_dir: Directory containing audio files
        output_dir: Directory to output the JSONL file and copied audio
        functions: List of function definitions
    """
    # Create output directory for audio files
    audio_output_dir = os.path.join(output_dir, "function_calling", "audio")
    os.makedirs(audio_output_dir, exist_ok=True)

    # Prepare to read utterances
    utterances_file = os.path.join(dataset_path, "utterances.json")
    cat_records = []

    if not os.path.exists(utterances_file):
        print(f"Error: Utterances file not found at {utterances_file}")
        return

    with open(utterances_file, "r") as f:
        utterances_data = json.load(f)

    print(f"Processing {len(utterances_data)} utterances...")

    for utterance in tqdm(utterances_data):
        # Check if utterance has an audio file and semantic parse
        audio_filename = utterance.get("audio_filename")
        insl_parse = utterance.get("semanticParse")

        if not audio_filename or not insl_parse:
            continue

        # Check if audio file exists
        audio_file_path = os.path.join(audio_input_dir, audio_filename)
        if not os.path.exists(audio_file_path):
            continue

        try:
            # Parse the INSL representation into a structured format
            parsed_insl = parse_insl(insl_parse)
            function_call = insl_to_function_calls(parsed_insl)

            # Extract function names to filter available functions
            function_names = extract_function_names_from_insl(insl_parse)
            available_funcs = [f for f in functions if f["name"] in function_names]

            # Copy audio file to output directory
            output_audio_path = os.path.join(audio_output_dir, audio_filename)
            shutil.copy2(audio_file_path, output_audio_path)

            # Create a CATS-compatible record
            record = {
                "filename": audio_filename,
                "transcript": utterance.get("text", ""),
                "function_call": function_call,
                "available_functions": function_names
            }

            cat_records.append(record)
        except Exception as e:
            print(f"Error processing utterance {audio_filename}: {e}")

    # Write records to JSONL file
    output_jsonl = os.path.join(output_dir, "function_call_inputs.jsonl")
    with open(output_jsonl, "w") as f:
        for record in cat_records:
            f.write(json.dumps(record) + "\n")

    print(f"Wrote {len(cat_records)} records to {output_jsonl}")
    print(f"Copied audio files to {audio_output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Convert STOP dataset to CATS format")
    parser.add_argument("--stop_dataset", required=True, help="Path to the STOP dataset directory")
    parser.add_argument("--audio_dir", required=True, help="Path to directory containing STOP audio files")
    parser.add_argument("--output_dir", default="data", help="Output directory for CATS data")

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Extract function definitions
    functions_file = os.path.join(args.output_dir, "openai_functions.json")
    functions = extract_openai_functions(args.stop_dataset, functions_file)

    if functions:
        # Extract utterances and create CATS-compatible data
        extract_stop_utterances(args.stop_dataset, args.audio_dir, args.output_dir, functions)

    print("Conversion completed.")

if __name__ == "__main__":
    main()