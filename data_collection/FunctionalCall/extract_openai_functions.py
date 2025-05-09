#!/usr/bin/env python3
"""
Convert STOP dataset intents into OpenAI Function Calling compatible format.

This script analyzes the STOP dataset (Spoken Task Oriented Semantic Parsing)
and generates OpenAI function definitions based on the intent patterns found in the data.

The script:
1. Loads a processed STOP dataset (created by fairseq_to_hf_stop.py)
2. Extracts and analyzes all intent patterns
3. Creates OpenAI function definitions for each main intent
4. Exports the function definitions as JSON

Usage:
    python stop_to_openai_functions.py --dataset_path ~/Downloads/hf_stop/stop_dataset
"""

import argparse
import json
import os
import re
import sys
from collections import defaultdict, Counter
from pathlib import Path
from typing import Dict, List, Set, Any, Optional

# Check and install required dependencies
required_packages = ["datasets", "tqdm"]
installed_packages = []

for package in required_packages:
    try:
        __import__(package)
        installed_packages.append(package)
    except ImportError:
        print(f"Installing required package: {package}")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        installed_packages.append(package)

print(f"Successfully installed/verified packages: {', '.join(installed_packages)}")

from datasets import load_from_disk
from tqdm import tqdm

#!/usr/bin/env python3
"""
INSL Format Parser for Audio Task Functional Call.

This module provides parser functionality for the INSL (INtent-SLot) format
used to represent voice commands and their structured semantic representations.
"""

# Regular expressions for parsing INSL frames
INTENT_PATTERN = r'\[IN:([A-Z_]+)'
SLOT_PATTERN = r'\[SL:([A-Z_]+)'
NESTED_INTENT_PATTERN = r'\[SL:[A-Z_]+ \[IN:([A-Z_]+)'

def parse_insl(parse_string: str, output_format: str = "full") -> Dict[str, Any]:
    """
    Parse INSL frame text and extract intents and slots.

    Args:
        parse_string: The INSL frame parse string
        output_format: The output format to return:
            - "full": Complete hierarchical structure (default)
            - "function_call": OpenAI function call format with name and arguments

    Returns:
        Dict containing the parsed information in the requested format or None if parsing fails
    """
    if not parse_string or not parse_string.strip():
        return {} if output_format == "full" else None

    # Extract main intent
    main_intent_match = re.search(INTENT_PATTERN, parse_string)
    if not main_intent_match:
        return {} if output_format == "full" else None

    main_intent = main_intent_match.group(1)

    # Create a token list by parsing the string
    tokens = []
    i = 0
    while i < len(parse_string):
        if i+4 <= len(parse_string) and parse_string[i:i+4] == "[IN:":
            # Find the end of the intent name
            end = parse_string.find(" ", i+4)
            if end == -1:
                # No space found, see if the end bracket is there
                end = parse_string.find("]", i+4)
                if end == -1:
                    # Malformed string, just increment and continue
                    i += 1
                    continue
            intent_name = parse_string[i+4:end]
            tokens.append(("INTENT_START", intent_name, i))  # Store position for tracking
            i = end
        elif i+4 <= len(parse_string) and parse_string[i:i+4] == "[SL:":
            # Find the end of the slot name
            end = parse_string.find(" ", i+4)
            if end == -1:
                # No space found, see if the end bracket is there
                end = parse_string.find("]", i+4)
                if end == -1:
                    # Malformed string, just increment and continue
                    i += 1
                    continue
            slot_name = parse_string[i+4:end]
            tokens.append(("SLOT_START", slot_name, i))  # Store position for tracking
            i = end
        elif i < len(parse_string) and parse_string[i] == "]":
            tokens.append(("END", None, i))
            i += 1
        else:
            i += 1

    # Build a hierarchical structure using a stack-based approach with unique IDs
    stack = []  # Will contain tuples of (type, name, pos, id)
    current_intent_id = None
    intent_counter = {}  # Track occurrence count of each intent type

    # Maps for storing the structure
    intent_instances = {}  # Maps intent_id -> {name, slots, parent_id, pos}
    slot_to_intent = {}  # Maps slot_pos -> intent_id
    slot_values = {}  # Maps slot_name -> slot_value

    # Process tokens to build the structure
    for token_type, token_value, token_pos in tokens:
        if token_type == "INTENT_START":
            # Create a unique ID for this intent instance
            if token_value not in intent_counter:
                intent_counter[token_value] = 0
            count = intent_counter[token_value]
            intent_counter[token_value] += 1
            intent_id = f"{token_value}_{count}"

            # Get the parent intent (if any)
            parent_id = current_intent_id

            # Store this intent instance
            intent_instances[intent_id] = {
                "name": token_value,
                "slots": [],
                "parent_id": parent_id,
                "pos": token_pos,
                "children": []
            }

            # Link this intent as a child of its parent
            if parent_id and parent_id in intent_instances:
                intent_instances[parent_id]["children"].append(intent_id)

            # Push the intent onto the stack
            stack.append(("INTENT", token_value, token_pos, intent_id))
            current_intent_id = intent_id

        elif token_type == "SLOT_START":
            # Add this slot to the current intent
            if current_intent_id and current_intent_id in intent_instances:
                intent_instances[current_intent_id]["slots"].append(token_value)
                slot_to_intent[token_pos] = current_intent_id

                # Extract slot value - find the text between this slot start and the next closing bracket
                slot_end = parse_string.find("]", token_pos)
                if slot_end != -1:
                    # Extract everything after the space after the slot name
                    slot_text_start = parse_string.find(" ", token_pos + 4)
                    if slot_text_start != -1 and slot_text_start < slot_end:
                        slot_value = parse_string[slot_text_start+1:slot_end].strip()
                        slot_values[token_value] = slot_value

            stack.append(("SLOT", token_value, token_pos, None))

        elif token_type == "END":
            # Pop from the stack
            if stack:
                stack.pop()

                # Update current intent if we're back to a previous intent
                current_intent_id = None
                for type_val, name, pos, id_val in reversed(stack):
                    if type_val == "INTENT":
                        current_intent_id = id_val
                        break

    # Find the main intent instance (first occurrence of the main intent)
    main_intent_id = f"{main_intent}_0"

    # If we want the OpenAI function call format, return that simplified structure
    if output_format == "function_call":
        arguments = {}
        if main_intent_id in intent_instances:
            for slot in intent_instances[main_intent_id]["slots"]:
                if slot in slot_values:
                    # Convert slot names to lowercase to match OpenAI function naming convention
                    arguments[slot.lower()] = slot_values[slot]

        return {
            "name": main_intent.lower(),
            "arguments": arguments
        }

    # Otherwise, return the full hierarchical structure
    # Build intent hierarchy (for nested intents of the same type)
    intent_hierarchy = {
        "instances": intent_instances,
        "main_intent": main_intent,
        "main_intent_id": main_intent_id
    }

    # Extract all nested intents (distinct intent types other than main intent)
    all_intent_names = set(instance["name"] for instance in intent_instances.values())
    nested_intent_types = list(all_intent_names - {main_intent})

    # Extract all nested intent instances (all instances except the main one)
    nested_intent_instances = [
        intent_id for intent_id in intent_instances.keys()
        if not intent_id.startswith(f"{main_intent}_")
    ]

    # Build the traditional intent_to_slots mapping (for backward compatibility)
    intent_to_slots = {}
    for intent_id, info in intent_instances.items():
        intent_name = info["name"]
        if intent_name not in intent_to_slots:
            intent_to_slots[intent_name] = []
        for slot in info["slots"]:
            if slot not in intent_to_slots[intent_name]:
                intent_to_slots[intent_name].append(slot)

    # Extract direct slots for the main intent
    direct_slots = intent_instances.get(main_intent_id, {}).get("slots", [])

    # Return both traditional format and the new hierarchical format
    return {
        "intent": main_intent,
        "slots": direct_slots,
        "slot_values": slot_values,
        "nested_intents": nested_intent_types,
        "raw_parse": parse_string,
        "intent_slot_mapping": intent_to_slots,
        "hierarchy": intent_hierarchy,
        "intent_instances": intent_instances,
        "nested_intent_instances": nested_intent_instances
    }

# Define alias for backward compatibility
def extract_intents_and_slots(parse_string: str) -> Dict[str, Any]:
    """
    Extract intents and slots from an INSL frame parse string.
    This is an alias for parse_insl with output_format='full'.

    Args:
        parse_string: The INSL frame parse string

    Returns:
        Dict containing the main intent and its slots and nested intents,
        including hierarchical structure for multiple instances of the same intent type.
    """
    return parse_insl(parse_string, output_format="full")


def analyze_dataset(dataset) -> Dict[str, Any]:
    """
    Analyze the dataset to extract all intent patterns and their associated slots.

    Args:
        dataset: The HuggingFace dataset

    Returns:
        Dict containing intent analysis
    """
    print("Analyzing dataset intent patterns...")

    # Counters and collections
    intent_counter = Counter()
    intent_to_slots = defaultdict(set)
    intent_to_nested_intents = defaultdict(set)
    intent_to_examples = defaultdict(list)

    # Process each split in the dataset
    for split_name, split_data in dataset.items():
        print(f"Processing {split_name} split ({len(split_data)} examples)")

        # Instead of using the dataset iterator (which tries to process audio),
        # directly access the PyArrow table and extract the text columns we need
        table = split_data.data

        # Extract parse and utterance columns as Python lists
        parse_strings = table['parse'].to_pylist()
        utterances = table['utterance'].to_pylist()

        print(f"Extracted {len(parse_strings)} parse strings and {len(utterances)} utterances")

        # Process each example
        for i, (parse_string, utterance) in enumerate(tqdm(zip(parse_strings, utterances), total=len(parse_strings))):
            # Extract intent information
            intent_info = extract_intents_and_slots(parse_string)
            if not intent_info:
                continue

            main_intent = intent_info['intent']
            slots = intent_info['slots']
            nested_intents = intent_info['nested_intents']

            # Update counters and collections
            intent_counter[main_intent] += 1

            # Get the intent-slot mapping for better accuracy
            intent_slot_mapping = intent_info.get('intent_slot_mapping', {})

            # Update slots for the main intent from the mapping
            for slot in slots:  # These are the direct slots for the main intent
                intent_to_slots[main_intent].add(slot)

            # Update nested intents relationships
            for nested_intent in nested_intents:
                # Add the relationship between main intent and nested intent
                intent_to_nested_intents[main_intent].add(nested_intent)

                # If we have slots for nested intents in the mapping, track those too
                if nested_intent in intent_slot_mapping and intent_slot_mapping[nested_intent]:
                    for slot in intent_slot_mapping[nested_intent]:
                        intent_to_slots[nested_intent].add(slot)

            # Store a few examples for each intent
            if len(intent_to_examples[main_intent]) < 5:
                intent_to_examples[main_intent].append({"utterance": utterance, "parse": parse_string})

            # Print progress periodically
            if i % 10000 == 0 and i > 0:
                print(f"Processed {i}/{len(parse_strings)} examples...")

    # Convert sets to lists for JSON serialization
    intent_to_slots = {k: list(v) for k, v in intent_to_slots.items()}
    intent_to_nested_intents = {k: list(v) for k, v in intent_to_nested_intents.items()}

    results = {
        "intent_counts": dict(intent_counter),
        "intent_to_slots": intent_to_slots,
        "intent_to_nested_intents": intent_to_nested_intents,
        "intent_to_examples": intent_to_examples
    }

    return results

def slot_type_to_parameter_type(slot_type: str) -> Dict[str, Any]:
    """
    Convert a STOP slot type to an OpenAI function parameter type.

    Args:
        slot_type: The STOP slot type

    Returns:
        Dict describing the parameter type for OpenAI function calling
    """
    # Default to string type with description
    param_info = {
        "type": "string",
        "description": f"The {slot_type.lower().replace('_', ' ')}"
    }

    # Special handling for common slot types
    if "DATE_TIME" in slot_type:
        param_info["description"] = "Time or date specification (e.g., 'tomorrow at 3pm', 'next Monday')"
    elif "CONTACT" in slot_type or "RECIPIENT" in slot_type or "PERSON" in slot_type:
        param_info["description"] = "Name of the contact or person"
    elif "LOCATION" in slot_type or "DESTINATION" in slot_type:
        param_info["description"] = "The location or place name"
    elif "NUMBER" in slot_type or "AMOUNT" in slot_type:
        param_info["type"] = "number"
        param_info["description"] = f"The {slot_type.lower().replace('_', ' ')}"
    elif "CONTENT" in slot_type:
        param_info["description"] = "The content of the message or note"
    elif "TYPE" in slot_type or "NAME" in slot_type:
        param_info["description"] = f"The {slot_type.lower().replace('_', ' ')}"

    return param_info

def create_openai_function(intent: str, slots: List[str], examples: List[Dict[str, str]],
                        required_slots: List[str] = None) -> Dict[str, Any]:
    """
    Create an OpenAI function definition for an intent.

    Args:
        intent: The intent name
        slots: List of slot types for this intent
        examples: List of example utterances and parses
        required_slots: List of slots that are required (if None, all slots are considered required)

    Returns:
        Dict containing the OpenAI function definition
    """
    # Convert intent name to function name (snake_case)
    function_name = intent.lower()

    # Create a readable description based on the intent name
    description = " ".join(intent.lower().replace('_', ' ').split())
    description = description[0].upper() + description[1:]

    # Add examples to the description if available
    if examples and len(examples) > 0:
        description += "\n\nExamples:"
        for i, example in enumerate(examples[:5]):  # Use up to 5 examples
            if 'utterance' in example:
                description += f"\n- \"{example['utterance']}\""

                # Add parse structure info if available
                if 'intent_structure' in example and 'hierarchy' in example['intent_structure']:
                    hierarchy = example['intent_structure']['hierarchy']
                    if 'instances' in hierarchy:
                        # Count instances of this intent type
                        count = sum(1 for instance in hierarchy['instances'].values()
                                  if instance['name'] == intent)
                        if count > 1:
                            description += f" (Contains {count} instances of {intent})"

    # Create parameters dictionary
    parameters = {
        "type": "object",
        "properties": {},
        "required": []
    }

    # Use the provided list of required slots, or consider all required if not specified
    required_slot_list = required_slots if required_slots is not None else slots

    # Add parameters for each slot
    for slot in slots:
        param_name = slot.lower()
        parameters["properties"][param_name] = slot_type_to_parameter_type(slot)

        # Mark the parameter as required if it's in the required_slots list
        if slot in required_slot_list:
            parameters["required"].append(param_name)

    # Create the function definition
    function_def = {
        "type": "function",
        "function": {
            "name": function_name,
            "description": description,
            "parameters": parameters
        }
    }

    return function_def

def generate_openai_functions(analysis_results: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Generate OpenAI function definitions from the dataset analysis.

    Args:
        analysis_results: Results from the dataset analysis

    Returns:
        List of OpenAI function definitions
    """
    intent_counts = analysis_results["intent_counts"]
    intent_to_slots = analysis_results["intent_to_slots"]
    intent_to_examples = analysis_results["intent_to_examples"]

    # Sort intents by frequency (most common first)
    sorted_intents = sorted(intent_counts.keys(), key=lambda x: intent_counts[x], reverse=True)

    functions = []
    skipped_intents = []

    for intent in sorted_intents:
        # Check if slots exist for this intent
        if intent not in intent_to_slots:
            print(f"Warning: No slots found for intent '{intent}'. Skipping...")
            skipped_intents.append(intent)
            continue

        all_slots = intent_to_slots.get(intent, [])
        examples = intent_to_examples.get(intent, [])

        # Analyze which slots are required vs. optional by checking all examples
        slot_occurrence = {slot: 0 for slot in all_slots}
        total_examples = 0

        # Count occurrences of each slot in examples for this intent
        for example in examples:
            if 'parse' in example:
                # Parse the example if not already parsed
                if 'intent_structure' not in example:
                    example['intent_structure'] = extract_intents_and_slots(example['parse'])

                structure = example['intent_structure']
                if 'intent' in structure and structure['intent'] == intent:
                    total_examples += 1

                    # Check which slots are present in this example
                    # First check direct slots
                    example_slots = set(structure.get('slots', []))

                    # Then check slots from intent_slot_mapping
                    if 'intent_slot_mapping' in structure and intent in structure['intent_slot_mapping']:
                        example_slots.update(structure['intent_slot_mapping'][intent])

                    # Increment count for each slot that appears in this example
                    for slot in all_slots:
                        if slot in example_slots:
                            slot_occurrence[slot] += 1

        # If we didn't find enough examples, fall back to the original method
        if total_examples < 5:
            print(f"Not enough examples ({total_examples}) found for intent '{intent}'. Assuming all slots are required.")
            required_slots = all_slots
        else:
            # Calculate frequency of each slot (percentage of examples it appears in)
            slot_frequency = {slot: count / total_examples for slot, count in slot_occurrence.items()}

            # Consider a slot required if it appears in at least 80% of examples
            required_threshold = 0.8
            required_slots = [slot for slot, freq in slot_frequency.items() if freq >= required_threshold]
            optional_slots = [slot for slot, freq in slot_frequency.items() if freq < required_threshold]

            if optional_slots:
                print(f"Intent '{intent}' ({total_examples} examples): Required slots {required_slots}, Optional slots {optional_slots}")
                print(f"  Slot frequencies: {', '.join([f'{s}:{slot_frequency[s]:.2f}' for s in all_slots])}")

        # Create function definition with proper required fields
        function_def = create_openai_function(intent, all_slots, examples, required_slots)
        functions.append(function_def)

    if skipped_intents:
        print(f"Skipped {len(skipped_intents)} intents due to missing slot information: {', '.join(skipped_intents)}")

    return functions

def main():
    parser = argparse.ArgumentParser(description="Convert STOP dataset intents to OpenAI Function Calling format")
    parser.add_argument('--dataset_path', type=str, default='./stop_dataset',
                        help='Path to the processed STOP dataset')
    parser.add_argument('--output_dir', type=str, default='.',
                        help='Output directory for the OpenAI function definitions')
    parser.add_argument('--min_examples', type=int, default=5,
                        help='Minimum number of examples for an intent to be included')

    args = parser.parse_args()

    # Expand user directory paths
    dataset_path = os.path.expanduser(args.dataset_path)
    output_dir = os.path.expanduser(args.output_dir)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load the dataset
    print(f"Loading dataset from {dataset_path}...")
    try:
        dataset = load_from_disk(dataset_path)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # Analyze the dataset
    analysis_results = analyze_dataset(dataset)

    # Prepare intent analysis output with examples
    analysis_output = {
        "intent_counts": analysis_results["intent_counts"],
        "intents": {}
    }

    # For each intent, include slots, nested intents, and examples
    for intent, count in analysis_results["intent_counts"].items():
        analysis_output["intents"][intent] = {
            "count": count,
            "slots": analysis_results["intent_to_slots"].get(intent, []),
            "nested_intents": analysis_results["intent_to_nested_intents"].get(intent, []),
            "examples": [
                {
                    "utterance": ex["utterance"],
                    "parse": ex["parse"],
                    "intent_structure": extract_intents_and_slots(ex["parse"])
                }
                for ex in analysis_results["intent_to_examples"].get(intent, [])
            ]
        }

    # Save analysis results with the new structure
    with open(os.path.join(output_dir, 'intent_analysis.json'), 'w') as f:
        json.dump(analysis_output, f, indent=2)

    print(f"Found {len(analysis_output['intent_counts'])} distinct intents")
    print(f"Analysis results saved to {os.path.join(output_dir, 'intent_analysis.json')}")

    # Filter out infrequent intents
    filtered_analysis = {
        "intent_counts": {
            intent: count for intent, count in analysis_results["intent_counts"].items()
            if count >= args.min_examples
        },
        "intent_to_slots": {
            intent: slots for intent, slots in analysis_results["intent_to_slots"].items()
            if intent in analysis_results["intent_counts"] and
               analysis_results["intent_counts"][intent] >= args.min_examples
        },
        "intent_to_nested_intents": {
            intent: nested for intent, nested in analysis_results["intent_to_nested_intents"].items()
            if intent in analysis_results["intent_counts"] and
               analysis_results["intent_counts"][intent] >= args.min_examples
        },
        "intent_to_examples": {
            intent: examples for intent, examples in analysis_results["intent_to_examples"].items()
            if intent in analysis_results["intent_counts"] and
               analysis_results["intent_counts"][intent] >= args.min_examples
        }
    }

    print(f"Generating OpenAI functions for {len(filtered_analysis['intent_counts'])} intents with at least {args.min_examples} examples")

    # Generate OpenAI function definitions
    functions = generate_openai_functions(filtered_analysis)

    # Save OpenAI function definitions
    with open(os.path.join(output_dir, 'openai_functions.json'), 'w') as f:
        json.dump(functions, f, indent=2)

    print(f"OpenAI function definitions saved to {os.path.join(output_dir, 'openai_functions.json')}")
    print("Conversion complete!")

def test_parser():
    """Test the parser with some sample parse strings to ensure it correctly handles nested intents."""
    test_cases = [
        # Test case 1: Simple intent with slots
        "[IN:CREATE_ALARM [SL:DATE_TIME TOMORROW ] [SL:TIME SIX_AM ] ] | ",

        # Test case 2: Intent with nested intent
        "[IN:UPDATE_ALARM [SL:ALARM_NAME [IN:GET_TIME [SL:DATE_TIME SEVEN_OCLOCK ] ] ] [SL:DATE_TIME TO_SIX_THIRTY ] ] | ",

        # Test case 3: Multiple levels of nesting
        "[IN:CREATE_REMINDER [SL:PERSON [IN:GET_CONTACT [SL:NAME JOHN ] ] ] [SL:DATE_TIME [IN:GET_TIME [SL:TIME AFTERNOON ] ] ] ] | "
    ]

    for i, test_case in enumerate(test_cases):
        print(f"\nTest Case {i+1}: {test_case}")
        result = extract_intents_and_slots(test_case)
        print(f"Main Intent: {result['intent']}")
        print(f"Direct Slots: {result['slots']}")
        print(f"Nested Intents: {result['nested_intents']}")
        print("Intent to Slot Mapping:")
        for intent, slots in result['intent_slot_mapping'].items():
            print(f"  - {intent}: {slots}")

if __name__ == "__main__":
    # Run tests if argument is --test
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        test_parser()
    else:
        main()
