import json
import os
from typing import List, Dict, Any, Optional


def adapt_jsonl_format(
    input_file: str, output_file: str, task_name: str = "emotion", field_name: str = "emotion"
) -> None:
    """
    Adapts the provided JSONL format to match the structure expected by inference.py

    Args:
        input_file: Path to the input JSONL file
        output_file: Path to write the adapted JSONL file
        task_name: Name of the task (default: 'emotion')
        field_name: Name of the field containing ground truth (default: 'emotion')
    """
    adapted_records = []

    # Read the input file
    with open(input_file, "r") as f:
        for line in f:
            try:
                record = json.loads(line.strip())

                # Process each generated audio entry
                for audio_entry in record.get("generated_audio", []):
                    # Create a new record with the expected structure
                    adapted_entry = {
                        "filename": audio_entry.get("filename", ""),
                        field_name: audio_entry.get(field_name, ""),  # Use the specified field name
                        "sentence": record.get("sentence", ""),  # Optional context
                        "voice_gender": record.get("voice_gender", ""),  # Optional metadata
                        "expected_emotions": record.get("emotions", []),  # Keep original emotion list
                    }

                    # Add to the list of adapted records
                    adapted_records.append(adapted_entry)

            except json.JSONDecodeError as e:
                print(f"Error parsing JSON line: {e}")
                continue

    # Write the adapted records to the output file
    with open(output_file, "w") as f:
        for entry in adapted_records:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"Converted {len(adapted_records)} records to format compatible with inference.py")
    print(f"Saved to {output_file}")


def create_task_input_files(input_file: str, output_dir: str, tasks: List[Dict[str, str]]) -> Dict[str, str]:
    """
    Creates separate input files for different tasks based on the same source data

    Args:
        input_file: Path to the input JSONL file
        output_dir: Directory to write the adapted JSONL files
        tasks: List of task configurations with 'name' and 'field_name' keys

    Returns:
        Dictionary mapping task names to output file paths
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    task_files = {}

    for task in tasks:
        task_name = task["name"]
        field_name = task["field_name"]

        # Create output filename
        output_file = os.path.join(output_dir, f"audio_inputs_{task_name}.jsonl")

        # Adapt the format for this task
        adapt_jsonl_format(input_file=input_file, output_file=output_file, task_name=task_name, field_name=field_name)

        task_files[task_name] = output_file

    return task_files


if __name__ == "__main__":
    # Example usage
    input_file = "final_data.jsonl"
    output_file = "audio_inputs.jsonl"

    # For a single task
    adapt_jsonl_format(input_file, output_file, task_name="emotion", field_name="emotion")

    # For multiple tasks
    tasks = [{"name": "emotion", "field_name": "emotion"}, {"name": "transcription", "field_name": "transcript"}]

    task_files = create_task_input_files(input_file, "data", tasks)
    print(f"Created input files for tasks: {list(task_files.keys())}")
