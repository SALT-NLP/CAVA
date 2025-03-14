#!/usr/bin/env python3

import argparse
import json
import openai
import re
import ast  # for safely parsing function_call arguments if needed
import os
from tqdm import tqdm  # for progress bar
import base64

# -------------------------------------------
# 1. Simple parser for bracket-based INSL
# -------------------------------------------

class ParseError(Exception):
    pass

TOKEN_PATTERN = r"""
    (\[|\])          # bracket tokens
  | (IN:[^\s\[\]]+)  # function name token e.g. IN:GET_EVENT
  | (SL:[^\s\[\]]+)  # slot label token e.g. SL:LOCATION
  | ([^\[\]\s]+)     # bareword fallback token e.g. HOLIDAY, YORK, etc.
"""

def tokenize_insl(input_str):
    """ Break the input string into tokens for bracket-based INSL. """
    input_str = input_str.strip()
    if input_str.endswith("|"):
        input_str = input_str[:-1].rstrip()

    tokens = []
    for match in re.finditer(TOKEN_PATTERN, input_str, re.VERBOSE):
        tokens.append(match.group(0))
    return tokens

def parse_slot_expression(tokens, idx):
    """
    Parses [SL:slotName ... ] => returns a dict {type='SL', name=slotName, value=... }
    The '...' can be bare words or a nested [IN:...] bracket.
    """
    if idx >= len(tokens) or tokens[idx] != '[':
        raise ParseError(f"Expected '[' at {idx}")
    idx += 1  # consume '['

    if idx >= len(tokens) or not tokens[idx].startswith('SL:'):
        raise ParseError(f"Expected 'SL:' but got {tokens[idx]}")
    slot_name = tokens[idx][3:]
    idx += 1

    contents = []
    while idx < len(tokens) and tokens[idx] != ']':
        if tokens[idx] == '[':
            subexpr, idx = parse_insl_expression(tokens, idx)
            contents.append(subexpr)
        else:
            contents.append(tokens[idx])
            idx += 1

    if idx >= len(tokens) or tokens[idx] != ']':
        raise ParseError("Missing closing ']' in slot expression.")
    idx += 1  # consume ']'

    if len(contents) == 1 and isinstance(contents[0], dict) and contents[0].get("type") == "IN":
        slot_value = contents[0]
    else:
        slot_value = " ".join(str(c) for c in contents)

    return {
        "type": "SL",
        "name": slot_name,
        "value": slot_value
    }, idx

def parse_insl_expression(tokens, idx=0):
    """
    Parse an expression like:
      [IN:functionName [SL:slot1 ...] [SL:slot2 [IN:subFunc]] ... ]
    Returns (expr, new_idx)
    """
    if idx >= len(tokens) or tokens[idx] != '[':
        raise ParseError(f"Expected '[' at {idx}")
    idx += 1

    if idx >= len(tokens) or not tokens[idx].startswith('IN:'):
        raise ParseError(f"Expected 'IN:' but got {tokens[idx]}")
    func_name = tokens[idx][3:]
    idx += 1

    expr = {
        "type": "IN",
        "name": func_name,
        "slots": {}
    }

    while idx < len(tokens) and tokens[idx] != ']':
        if tokens[idx] == '[':
            slot_expr, idx = parse_slot_expression(tokens, idx)
            expr["slots"][slot_expr["name"]] = slot_expr["value"]
        else:
            raise ParseError(f"Unexpected token {tokens[idx]}")

    if idx >= len(tokens) or tokens[idx] != ']':
        raise ParseError("Missing closing ']' for function expression.")
    idx += 1  # consume ']'

    return expr, idx

def parse_insl(input_str):
    """ Parse a top-level bracket expression into a nested dictionary. """
    tokens = tokenize_insl(input_str)
    expr, idx = parse_insl_expression(tokens, 0)
    if idx < len(tokens):
        leftover = tokens[idx:]
        if any(t.strip() for t in leftover):
            raise ParseError(f"Extra tokens leftover: {leftover}")
    return expr

# -------------------------------------------
# 2. Convert the parsed INSL to "function_name + nested arguments" (dict)
# -------------------------------------------

def insl_to_function_calls(expr):
    """
    Given a parsed expression from parse_insl, produce a Python dict of the form:
      {
        "name": <functionName>,
        "arguments": { ... }
      }
    where the arguments might be nested if the slot values are themselves an IN: subexpression.
    """
    if expr.get("type") != "IN":
        raise ValueError("Top-level expression must be type=IN.")

    def slot_value_to_arg(val):
        if isinstance(val, dict) and val.get("type") == "IN":
            # Nested function
            return {
                "call_nested": insl_to_function_calls(val)
            }
        else:
            # simple string
            return val

    # Build arguments object
    args = {}
    for slot_name, slot_val in expr["slots"].items():
        args[slot_name.lower()] = slot_value_to_arg(slot_val)
        # e.g. "DATE_TIME" -> "date_time", "CONTENT_EXACT" -> "content_exact"
        # in a real system you'd define a systematic mapping from the gold parse slot names
        # to your function definitions' JSON keys.

    return {
        "name": expr["name"],
        "arguments": args
    }

# -------------------------------------------
# 3. Compare function call structure
# -------------------------------------------

def extract_call_sequence(func_call, sequence=None, parent_map=None, call_index_map=None):
    """
    Extract the sequence of function calls from a nested structure.
    Returns:
    - sequence: list of function calls in order
    - parent_map: dictionary mapping (function_name, index) to parent (function_name, index)
    - call_index_map: dictionary mapping function_name to count of its calls
    """
    if sequence is None:
        sequence = []
    if parent_map is None:
        parent_map = {}
    if call_index_map is None:
        call_index_map = {}

    # Track multiple calls to the same function
    func_name = func_call["name"]
    if func_name not in call_index_map:
        call_index_map[func_name] = 0
    current_index = call_index_map[func_name]
    call_index_map[func_name] += 1

    # Add current function to sequence with its index
    func_call["call_index"] = current_index
    sequence.append(func_call)

    # Check arguments for nested calls
    for arg_name, arg_value in func_call["arguments"].items():
        if isinstance(arg_value, dict) and "call_nested" in arg_value:
            nested_call = arg_value["call_nested"]
            # Record parent-child relationship using (name, index) tuples
            parent_key = (func_name, current_index)
            # The child's index will be determined in the recursive call
            child_name = nested_call["name"]
            child_index = call_index_map.get(child_name, 0)
            parent_map[(child_name, child_index)] = parent_key
            # Recursively process nested call
            extract_call_sequence(nested_call, sequence, parent_map, call_index_map)

    return sequence, parent_map, call_index_map

def compare_function_calls(gold_call, model_calls):
    """
    Compare function calls, ensuring:
    1. All required functions are called with correct arguments
    2. Functions are called in correct order (nested calls before their parents)
    3. Multiple calls to the same function are handled correctly
    Case-insensitive comparison for function names.
    """
    # Extract sequence and relationships from gold standard
    gold_sequence, gold_parent_map, gold_index_map = extract_call_sequence(gold_call)

    # Create a map of model function calls for easy lookup, including multiple calls
    # Use lowercase for keys to make comparison case-insensitive
    model_funcs = {}
    for call in model_calls:
        func_name = call["name"].lower()  # Convert to lowercase for comparison
        if func_name not in model_funcs:
            model_funcs[func_name] = []
        model_funcs[func_name].append(call)

    # Check 1: All required functions are called the correct number of times
    for func_name, count in gold_index_map.items():
        func_name_lower = func_name.lower()  # Convert to lowercase for comparison
        if func_name_lower not in model_funcs or len(model_funcs[func_name_lower]) < count:
            print(f"Missing or insufficient calls to {func_name} (expected {count}, got {len(model_funcs.get(func_name_lower, []))}")
            return False

    # Check 2: Order is correct (nested calls before their parents)
    for i, model_call in enumerate(model_calls):
        func_name = model_call["name"].lower()  # Convert to lowercase for comparison
        # Find which instance of this function call this is
        current_index = next(
            (idx for idx, call in enumerate(model_funcs[func_name]) if call is model_call),
            0
        )

        # Convert function name to lowercase for parent map lookup
        if (func_name, current_index) in gold_parent_map:
            parent_key = gold_parent_map[(func_name, current_index)]
            parent_name, parent_index = parent_key

            # Find parent's position in model calls (case-insensitive)
            parent_positions = [
                j for j, call in enumerate(model_calls)
                if call["name"].lower() == parent_name.lower()
            ]

            if not parent_positions or min(parent_positions) <= i:
                print(f"Order violation: {func_name}[{current_index}] should appear before its parent {parent_name}[{parent_index}]")
                return False

    # Check 3: Arguments match for each function call instance
    for gold_func in gold_sequence:
        func_name = gold_func["name"].lower()  # Convert to lowercase for comparison
        call_index = gold_func["call_index"]

        # Find corresponding model call
        if call_index >= len(model_funcs[func_name]):
            return False

        model_func = model_funcs[func_name][call_index]

        # Compare arguments
        if not compare_arguments(gold_func["arguments"], model_func["arguments"], model_funcs):
            print(f"Argument mismatch for {func_name}[{call_index}]:")
            print(f"  Expected: {gold_func['arguments']}")
            print(f"  Got: {model_func['arguments']}")
            return False

    return True

def compare_arguments(gold_args, model_args, model_funcs):
    """
    Compare arguments, ensuring required arguments are present and have correct structure.
    For nested calls, only verify that the required function was called.
    Allows additional arguments in model_args.
    """
    # Check if all required keys from gold_args are present in model_args
    if not set(gold_args.keys()).issubset(set(model_args.keys())):
        return False

    # Only check the arguments that appear in gold_args
    for key, gold_value in gold_args.items():
        model_value = model_args[key]

        if isinstance(gold_value, dict) and "call_nested" in gold_value:
            # This is a nested call reference
            nested_func_name = gold_value["call_nested"]["name"]

            # Only verify that the nested function was called
            if nested_func_name not in model_funcs:
                return False

            # For the model value, we expect either:
            # 1. A direct result from the nested function call (any non-empty value), or
            # 2. A reference to the nested call
            if isinstance(model_value, dict) and "call_nested" in model_value:
                # Case 2: It's also a nested call reference
                if model_value["call_nested"]["name"].lower() != nested_func_name.lower():  # Case-insensitive comparison
                    return False
            else:
                # Case 1: Just verify that there's a value
                if not model_value:
                    return False
        else:
            # For direct values, just verify that a non-empty value is provided
            if not model_value:
                return False

    return True

# -------------------------------------------
# 4. Main: read JSON, call OpenAI with "functions", compare
# -------------------------------------------

def gen_mock_function_result(func_name, arguments, mock_model, insl_string=None):
    """Return mock results for function calls using a model with INSL context"""
    # Construct a prompt that includes both the function call and INSL context
    context = ""
    if insl_string:
        context = f"INSL Parse: {insl_string}\n\n"

    prompt = (
        f"You are a mock function that generates realistic results. Generate a plausible result for this function call:\n\n"
        f"{context}"
        f"Function: {func_name}\n"
        f"Arguments: {json.dumps(arguments)}\n\n"
        f"Generate a brief, realistic response as if you were this function."
    )

    try:
        response = openai.chat.completions.create(
            model=mock_model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that generates realistic mock function results. Keep responses concise and relevant to the function's purpose."
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )

        result = response.choices[0].message.content.strip()
        return result

    except Exception as e:
        print(f"Warning: Mock generation failed ({e}), using default mock result")
        return f"Mock result for {func_name} with {json.dumps(arguments)}"

def load_functions_from_json(json_file):
    """Load function definitions from a JSON file"""
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            functions = json.load(f)
        if not isinstance(functions, list):
            raise ValueError("Functions JSON file must contain a list of function definitions")
        return functions
    except Exception as e:
        print(f"Error loading functions from {json_file}: {e}")
        print("Falling back to default functions")
        return []

def extract_function_names_from_insl(insl_string):
    """Extract all function names (IN:*) from an INSL string"""
    if not insl_string:
        return set()
    return set(re.findall(r'IN:(\w+)', insl_string))

def filter_functions_for_insl(functions, insl_string):
    """Filter the functions list to only include functions mentioned in the INSL string"""
    if not insl_string:
        return functions

    required_functions = extract_function_names_from_insl(insl_string)
    # Case-insensitive comparison of function names
    return [
        f for f in functions
        if any(f["function"]["name"].lower() == func_name.lower() for func_name in required_functions)
    ]

def process_function_calls(audio_file, model, mock_model, insl_string=None, available_functions=[]):
    """Helper function to process potentially nested function calls"""
    all_function_calls = []

    # Filter functions based on INSL string if provided
    filtered_functions = filter_functions_for_insl(available_functions, insl_string)

    # Construct messages
    messages = [
        {"role": "system", "content": "You are a helpful AI that uses the provided functions when appropriate."}
    ]

    # Read and encode audio file
    with open(audio_file, "rb") as f:
        encoded_audio = base64.b64encode(f.read()).decode("utf-8")

    # Add audio input to messages
    messages.append({
        "role": "user",
        "content": [
            {
                "type": "input_audio",
                "input_audio": {"data": encoded_audio, "format": "wav"}
            }
        ]
    })

    while True:
        try:
            response = openai.chat.completions.create(
                model=model,
                modalities=["text"],
                messages=messages,
                tools=filtered_functions,
                tool_choice="auto",
                temperature=0.0
            )
        except Exception as e:
            raise Exception(f"API call failed: {e}")

        choice = response.choices[0]
        if not choice.message.tool_calls:
            # No more function calls to make
            return all_function_calls

        # Add the assistant's message with the function call
        messages.append(choice.message)

        # Process each tool call in the message
        for tool_call in choice.message.tool_calls:
            func_call_data = tool_call.function

            # Parse the arguments
            arguments = json.loads(func_call_data.arguments)

            # Store the function call
            function_call = {
                "name": func_call_data.name,
                "arguments": arguments
            }
            all_function_calls.append(function_call)

            # Get mock result for the function using the mock model and INSL context
            result = gen_mock_function_result(
                func_call_data.name,
                arguments,
                mock_model,
                insl_string
            )

            # Add the function result to the conversation
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": str(result)
            })

    return all_function_calls

def save_results(output_file, results, total, correct, args, is_final=False):
    """Helper function to save results to a JSON file"""
    output = {
        "summary": {
            "total": total,
            "correct": correct,
            "accuracy": (correct / total * 100) if total > 0 else 0,
            "model": args.model,
            "mock_model": args.mock_model
        },
        "results": results
    }

    # For intermediate saves, append a timestamp to avoid overwriting
    if not is_final and output_file:
        base, ext = os.path.splitext(output_file)
        output_file = f"{base}_part{total}{ext}"

    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        if not is_final:
            print(f"\nIntermediate results saved to {output_file}")
    except Exception as e:
        print(f"\nError saving results to {output_file}: {e}")

def merge_result_files(base_path, final_path):
    """Merge all partial result files into a final one"""
    base, ext = os.path.splitext(base_path)
    all_results = []
    total = 0
    correct = 0
    model = None
    mock_model = None

    # Find and process all partial files
    for file in os.listdir(os.path.dirname(base_path) or '.'):
        if file.startswith(os.path.basename(base)) and '_part' in file:
            try:
                with open(os.path.join(os.path.dirname(base_path) or '.', file), 'r') as f:
                    data = json.load(f)
                    all_results.extend(data['results'])
                    total = data['summary']['total']  # Will get the latest total
                    correct = data['summary']['correct']  # Will get the latest correct count
                    model = data['summary']['model']
                    mock_model = data['summary']['mock_model']
                # Remove the partial file
                os.remove(os.path.join(os.path.dirname(base_path) or '.', file))
            except Exception as e:
                print(f"Error processing partial file {file}: {e}")

    # Save merged results
    output = {
        "summary": {
            "total": total,
            "correct": correct,
            "accuracy": (correct / total * 100) if total > 0 else 0,
            "model": model,
            "mock_model": mock_model
        },
        "results": all_results
    }

    try:
        with open(final_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"\nFinal merged results saved to {final_path}")
    except Exception as e:
        print(f"\nError saving final results to {final_path}: {e}")

def main():
    parser = argparse.ArgumentParser("Evaluate LLM function calling.")
    parser.add_argument("--utterances_file", required=True, help="JSON file with [ {utterance, parse}, ... ]")
    parser.add_argument("--functions_file", required=True, help="JSON file containing function definitions")
    parser.add_argument("--audio_dir", required=True, help="Root directory containing audio files")
    parser.add_argument("--api_key", help="OpenAI API key (if not set, will use OPENAI_API_KEY environment variable)")
    parser.add_argument("--api_base", help="OpenAI API base URL (optional)")
    parser.add_argument("--model", default="gpt-4o", help="Model to evaluate, e.g. gpt-4o, gpt-3.5-turbo")
    parser.add_argument("--mock_model", default="gpt-4o-mini", help="Model for mock generation")
    parser.add_argument("--max_examples", type=int, default=0, help="If >0, limit the number of examples processed")
    parser.add_argument("--output_file", help="JSON file to save test results (optional)")

    args = parser.parse_args()

    # Set OpenAI configuration
    if args.api_key:
        openai.api_key = args.api_key
    else:
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OpenAI API key must be provided either via --api_key argument or OPENAI_API_KEY environment variable")
        openai.api_key = api_key

    if args.api_base:
        openai.base_url = args.api_base

    # Load functions
    functions = load_functions_from_json(args.functions_file)

    # Load data
    with open(args.utterances_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    if args.max_examples > 0:
        data = data[:args.max_examples]

    total = 0
    correct = 0
    results = []  # Store detailed results for each example
    DUMP_FREQUENCY = 100  # Dump results every 100 examples

    # Initialize progress bar
    pbar = tqdm(enumerate(data, start=1), total=len(data), desc="Evaluating examples")

    for i, example in pbar:
        utterance = example.get("utterance", "").strip()
        gold_parse = example.get("parse", "").strip()
        audio_path = example.get("audio_path", "").strip()

        if not utterance or not gold_parse or not audio_path:
            pbar.write(f"[Warning] Example {i} missing fields. Skipping.")
            continue

        # Construct full audio file path
        full_audio_path = os.path.join(args.audio_dir, audio_path)

        result = {
            "example_id": i,
            "utterance": utterance,
            "gold_parse": gold_parse,
            "audio_path": full_audio_path,
            "success": False,
            "error": None,
            "model_calls": None,
            "gold_calls": None
        }

        # Parse gold expression -> function call
        try:
            insl_expr = parse_insl(gold_parse)
            gold_calls = insl_to_function_calls(insl_expr)
            result["gold_calls"] = gold_calls
        except Exception as e:
            error_msg = f"Parsing gold parse for example {i}: {e}"
            pbar.write(f"[Error] {error_msg}")
            result["error"] = error_msg
            results.append(result)
            continue

        try:
            # Process potentially nested function calls with INSL context
            model_calls = process_function_calls(
                full_audio_path,
                args.model,
                args.mock_model,
                gold_parse,
                functions
            )
            total += 1
            result["model_calls"] = model_calls

            if not model_calls:
                pbar.write(f"[Example {i}] ✗ No function call made")
                pbar.write("  Utterance: " + utterance)
                pbar.write("  Gold calls: " + str(gold_calls))
                result["error"] = "No function calls made"
            elif compare_function_calls(gold_calls, model_calls):
                correct += 1
                result["success"] = True
                pbar.write(f"[Example {i}] ✓ Correct function call")
                if len(model_calls) > 1:
                    pbar.write(f"    (Made {len(model_calls)} function calls in total)")
            else:
                pbar.write(f"[Example {i}] ✗ Mismatch")
                pbar.write("  Utterance: " + utterance)
                pbar.write("  Gold calls: " + str(gold_calls))
                pbar.write("  Model calls:")
                for j, call in enumerate(model_calls):
                    pbar.write(f"    {j}. {call['name']}: {json.dumps(call['arguments'])}")
                result["error"] = "Function call mismatch"

        except Exception as e:
            error_msg = f"Processing example {i}: {e}"
            pbar.write(f"[Error] {error_msg}")
            result["error"] = error_msg

        results.append(result)
        # Update progress bar description with current accuracy
        if total > 0:
            pbar.set_description(f"Evaluating examples (Accuracy: {(correct/total*100):.2f}%)")

        # Dump intermediate results every DUMP_FREQUENCY examples
        if args.output_file and len(results) >= DUMP_FREQUENCY:
            save_results(args.output_file, results, total, correct, args)
            results = []  # Clear results list after saving

    pbar.close()

    # Calculate and display summary
    if total > 0:
        accuracy = 100.0 * correct / total
        print(f"\nOverall results: {correct}/{total} correct ({accuracy:.2f}%)")
    else:
        print("No examples were processed.")

    # Save final results if output file is specified
    if args.output_file:
        # Save any remaining results
        if results:  # If there are any unsaved results
            save_results(args.output_file, results, total, correct, args)

        # Merge all partial results into final file
        merge_result_files(args.output_file, args.output_file)

if __name__ == "__main__":
    main()
