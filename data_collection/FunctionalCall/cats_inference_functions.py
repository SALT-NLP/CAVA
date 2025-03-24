#!/usr/bin/env python3
"""
Extended CATS inference script with support for loading functions from a JSON file.
This is used for function calling tasks, specifically with the STOP dataset.
"""

import argparse
import json
import os
from pathlib import Path

from cats.inference import main as cats_main
from cats.inference import reset_api_counters
from cats.config import create_task_configs, TaskConfig


def load_functions_from_json(functions_file):
    """
    Load function definitions from a JSON file.

    Args:
        functions_file: Path to the JSON file containing function definitions

    Returns:
        List of function definitions
    """
    if not os.path.exists(functions_file):
        raise FileNotFoundError(f"Functions file not found: {functions_file}")

    with open(functions_file, 'r') as f:
        functions = json.load(f)

    return functions


def main():
    """
    Run CATS inference with support for function calling.
    """
    parser = argparse.ArgumentParser(description="Run CATS inference with function calling support")
    parser.add_argument("--model", default="gpt-4o-audio", help="Model to use for inference")
    parser.add_argument("--task", default="function_call", help="Task to evaluate")
    parser.add_argument("--functions_file", default="data/openai_functions.json", help="Path to functions definition file")
    parser.add_argument("--data_file", default="data/function_call_inputs.jsonl", help="Path to input data file")
    parser.add_argument("--audio_dir", default="function_calling/audio/", help="Path to audio directory")

    args = parser.parse_args()

    # Reset API counters
    reset_api_counters()

    # Get available tasks
    tasks = create_task_configs()

    # Get the function_call task
    if args.task not in tasks:
        raise ValueError(f"Task {args.task} not found in available tasks")

    task_config = tasks[args.task]

    # Create a modified task config with the specified functions file
    if args.task == "function_call":
        # Load functions from file
        functions = load_functions_from_json(args.functions_file)

        # Create a new TaskConfig with the loaded functions
        task_config = TaskConfig(
            name=task_config.name,
            prompt_template=task_config.prompt_template,
            labels=task_config.labels,
            max_new_tokens=task_config.max_new_tokens,
            use_logits_processor=task_config.use_logits_processor,
            output_processor=task_config.output_processor,
            field_name=task_config.field_name,
            audio_dir=args.audio_dir,
            data_file=args.data_file,
            speech_output=task_config.speech_output,
            output_audio_dir=task_config.output_audio_dir,
            template_fields=task_config.template_fields,
            verify_tokenization=task_config.verify_tokenization,
            functions=functions,
            function_field_name=task_config.function_field_name,
            use_functions=True
        )

    # Override tasks dictionary with our modified task
    tasks = {args.task: task_config}

    # Run the main CATS inference with the modified task
    from cats.inference import load_model, run_evaluation

    # Load the model
    model_resources = load_model(args.model)

    # Run evaluation
    accuracy, results = run_evaluation(model_resources, task_config)

    print(f"Evaluation complete for model {args.model} on task {args.task}")
    print(f"Accuracy: {accuracy:.2f}")


if __name__ == "__main__":
    main()