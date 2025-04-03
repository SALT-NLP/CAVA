import json
import re
from typing import Dict, List, Any, Optional, Tuple


class ParseError(Exception):
    """Exception raised for errors during parsing function calls."""

    pass


def compare_function_calls(gold_call: Dict[str, Any], model_calls: List[Dict[str, Any]]) -> Tuple[bool, str]:
    """
    Compare function calls, ensuring:
    1. All required functions are called with correct arguments
    2. Functions are called in correct order (nested calls before their parents)
    3. Multiple calls to the same function are handled correctly

    Args:
        gold_call: The expected function call structure (gold standard)
        model_calls: The function calls made by the model

    Returns:
        Tuple of (success: bool, error_message: str)
    """
    # Extract sequence and relationships from gold standard
    gold_sequence, gold_parent_map, gold_index_map = extract_call_sequence(gold_call)

    # If no model calls were made
    if not model_calls:
        return False, "No function calls made by the model"

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
            error_msg = (
                f"Missing or insufficient calls to {func_name} (expected {count}, got"
                f" {len(model_funcs.get(func_name_lower, []))})"
            )
            return False, error_msg

    # Check 2: Order is correct (nested calls before their parents)
    for i, model_call in enumerate(model_calls):
        func_name = model_call["name"].lower()  # Convert to lowercase for comparison
        # Find which instance of this function call this is
        current_index = next((idx for idx, call in enumerate(model_funcs[func_name]) if call is model_call), 0)

        # Convert function name to lowercase for parent map lookup
        if (func_name, current_index) in gold_parent_map:
            parent_key = gold_parent_map[(func_name, current_index)]
            parent_name, parent_index = parent_key

            # Find parent's position in model calls (case-insensitive)
            parent_positions = [j for j, call in enumerate(model_calls) if call["name"].lower() == parent_name.lower()]

            if not parent_positions or min(parent_positions) <= i:
                error_msg = (
                    f"Order violation: {func_name}[{current_index}] should appear before its parent"
                    f" {parent_name}[{parent_index}]"
                )
                return False, error_msg

    # Check 3: Arguments match for each function call instance
    for gold_func in gold_sequence:
        func_name = gold_func["name"].lower()  # Convert to lowercase for comparison
        call_index = gold_func["call_index"]

        # Find corresponding model call
        if call_index >= len(model_funcs[func_name]):
            return False, f"Missing function call: {func_name}[{call_index}]"

        model_func = model_funcs[func_name][call_index]

        # Compare arguments
        args_match, error_msg = compare_arguments(gold_func["arguments"], model_func["arguments"], model_funcs)
        if not args_match:
            return False, f"Argument mismatch for {func_name}[{call_index}]: {error_msg}"

    return True, ""


def extract_call_sequence(func_call: Dict[str, Any], sequence=None, parent_map=None, call_index_map=None):
    """
    Extract the sequence of function calls from a nested structure.

    Args:
        func_call: The function call to process
        sequence: Accumulator for the sequence of calls
        parent_map: Dictionary mapping (function_name, index) to parent (function_name, index)
        call_index_map: Dictionary mapping function_name to count of its calls

    Returns:
        Tuple of (sequence, parent_map, call_index_map)
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


def compare_arguments(
    gold_args: Dict[str, Any], model_args: Dict[str, Any], model_funcs: Dict[str, List[Dict[str, Any]]]
) -> Tuple[bool, str]:
    """
    Compare arguments, ensuring required arguments are present and have correct structure.
    For nested calls, only verify that the required function was called.

    Args:
        gold_args: Expected arguments
        model_args: Model's arguments
        model_funcs: Map of model function calls

    Returns:
        Tuple of (success: bool, error_message: str)
    """
    # Check if all required keys from gold_args are present in model_args
    missing_keys = set(gold_args.keys()) - set(model_args.keys())
    if missing_keys:
        return False, f"Missing required arguments: {', '.join(missing_keys)}"

    # Only check the arguments that appear in gold_args
    for key, gold_value in gold_args.items():
        model_value = model_args[key]

        if isinstance(gold_value, dict) and "call_nested" in gold_value:
            # This is a nested call reference
            nested_func_name = gold_value["call_nested"]["name"].lower()

            # Only verify that the nested function was called
            if nested_func_name not in model_funcs:
                return False, f"Missing nested function call: {nested_func_name}"

            # For the model value, we expect either:
            # 1. A direct result from the nested function call (any non-empty value), or
            # 2. A reference to the nested call
            if isinstance(model_value, dict) and "call_nested" in model_value:
                # Case 2: It's also a nested call reference
                if model_value["call_nested"]["name"].lower() != nested_func_name:  # Case-insensitive comparison
                    return (
                        False,
                        f"Expected nested call to {nested_func_name}, got {model_value['call_nested']['name']}",
                    )
            else:
                # Case 1: Just verify that there's a value
                if not model_value:
                    return False, f"Empty value for argument '{key}' which should reference {nested_func_name}"
        else:
            # For direct values, just verify that a non-empty value is provided
            if not model_value and gold_value:  # Check if gold value is non-empty and model value is empty
                return False, f"Empty value for argument '{key}'"

    return True, ""


def parse_insl_to_function_call(insl_string: str) -> Dict[str, Any]:
    """
    Parse an INSL (Intent and Slot) string into a function call structure.

    Args:
        insl_string: The INSL string to parse

    Returns:
        A dictionary representing the function call
    """
    # Parse the INSL string
    expr = parse_insl(insl_string)

    # Convert to function call format
    return insl_to_function_calls(expr)


def tokenize_insl(input_str: str) -> List[str]:
    """
    Break the input string into tokens for bracket-based INSL.

    Args:
        input_str: The input INSL string

    Returns:
        List of tokens
    """
    input_str = input_str.strip()
    if input_str.endswith("|"):
        input_str = input_str[:-1].rstrip()

    TOKEN_PATTERN = r"""
        (\[|\])          # bracket tokens
      | (IN:[^\s\[\]]+)  # function name token e.g. IN:GET_EVENT
      | (SL:[^\s\[\]]+)  # slot label token e.g. SL:LOCATION
      | ([^\[\]\s]+)     # bareword fallback token e.g. HOLIDAY, YORK, etc.
    """

    tokens = []
    for match in re.finditer(TOKEN_PATTERN, input_str, re.VERBOSE):
        tokens.append(match.group(0))
    return tokens


def parse_slot_expression(tokens: List[str], idx: int) -> Tuple[Dict[str, Any], int]:
    """
    Parses [SL:slotName ... ] => returns a dict {type='SL', name=slotName, value=... }
    The '...' can be bare words or a nested [IN:...] bracket.

    Args:
        tokens: List of tokens
        idx: Current index in the tokens list

    Returns:
        Tuple of (slot_expression, new_index)
    """
    if idx >= len(tokens) or tokens[idx] != "[":
        raise ParseError(f"Expected '[' at {idx}")
    idx += 1  # consume '['

    if idx >= len(tokens) or not tokens[idx].startswith("SL:"):
        raise ParseError(f"Expected 'SL:' but got {tokens[idx]}")
    slot_name = tokens[idx][3:]
    idx += 1

    contents = []
    while idx < len(tokens) and tokens[idx] != "]":
        if tokens[idx] == "[":
            subexpr, idx = parse_insl_expression(tokens, idx)
            contents.append(subexpr)
        else:
            contents.append(tokens[idx])
            idx += 1

    if idx >= len(tokens) or tokens[idx] != "]":
        raise ParseError("Missing closing ']' in slot expression.")
    idx += 1  # consume ']'

    if len(contents) == 1 and isinstance(contents[0], dict) and contents[0].get("type") == "IN":
        slot_value = contents[0]
    else:
        slot_value = " ".join(str(c) for c in contents)

    return {"type": "SL", "name": slot_name, "value": slot_value}, idx


def parse_insl_expression(tokens: List[str], idx: int = 0) -> Tuple[Dict[str, Any], int]:
    """
    Parse an expression like:
      [IN:functionName [SL:slot1 ...] [SL:slot2 [IN:subFunc]] ... ]

    Args:
        tokens: List of tokens
        idx: Current index in the tokens list

    Returns:
        Tuple of (insl_expression, new_index)
    """
    if idx >= len(tokens) or tokens[idx] != "[":
        raise ParseError(f"Expected '[' at {idx}")
    idx += 1

    if idx >= len(tokens) or not tokens[idx].startswith("IN:"):
        raise ParseError(f"Expected 'IN:' but got {tokens[idx]}")
    func_name = tokens[idx][3:]
    idx += 1

    expr = {"type": "IN", "name": func_name, "slots": {}}

    while idx < len(tokens) and tokens[idx] != "]":
        if tokens[idx] == "[":
            slot_expr, idx = parse_slot_expression(tokens, idx)
            expr["slots"][slot_expr["name"]] = slot_expr["value"]
        else:
            raise ParseError(f"Unexpected token {tokens[idx]}")

    if idx >= len(tokens) or tokens[idx] != "]":
        raise ParseError("Missing closing ']' for function expression.")
    idx += 1  # consume ']'

    return expr, idx


def parse_insl(input_str: str) -> Dict[str, Any]:
    """
    Parse a top-level bracket expression into a nested dictionary.

    Args:
        input_str: The INSL string to parse

    Returns:
        Parsed INSL expression
    """
    tokens = tokenize_insl(input_str)
    expr, idx = parse_insl_expression(tokens, 0)
    if idx < len(tokens):
        leftover = tokens[idx:]
        if any(t.strip() for t in leftover):
            raise ParseError(f"Extra tokens leftover: {leftover}")
    return expr


def insl_to_function_calls(expr: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert a parsed INSL expression to a function call structure.

    Args:
        expr: Parsed INSL expression

    Returns:
        Function call structure
    """
    if expr.get("type") != "IN":
        raise ValueError("Top-level expression must be type=IN.")

    def slot_value_to_arg(val):
        if isinstance(val, dict) and val.get("type") == "IN":
            # Nested function
            return {"call_nested": insl_to_function_calls(val)}
        else:
            # simple string
            return val

    # Build arguments object
    args = {}
    for slot_name, slot_val in expr["slots"].items():
        args[slot_name.lower()] = slot_value_to_arg(slot_val)
        # e.g. "DATE_TIME" -> "date_time", "CONTENT_EXACT" -> "content_exact"

    return {"name": expr["name"], "arguments": args}


def evaluate_function_calling(model_calls: List[Dict[str, Any]], gold_parse: str) -> Tuple[bool, str]:
    """
    Evaluate function calling by comparing model's function calls against a gold standard parse.

    Args:
        model_calls: List of function calls made by the model
        gold_parse: The expected INSL parse string

    Returns:
        Tuple of (success: bool, error_message: str)
    """
    try:
        # Parse gold standard into a function call structure
        gold_call = parse_insl_to_function_call(gold_parse)

        # Compare function calls
        return compare_function_calls(gold_call, model_calls)

    except Exception as e:
        return False, f"Error evaluating function calls: {str(e)}"


def extract_function_calls_from_model_response(response_text: str) -> List[Dict[str, Any]]:
    """
    Extract function calls from model's text response.
    This implementation is a placeholder and should be adapted based on how your model formats function calls.

    Args:
        response_text: The model's text response

    Returns:
        List of extracted function calls
    """
    # This is a placeholder - you'll need to implement the actual extraction logic
    # based on your model's output format
    try:
        # Try to parse the response as JSON
        if response_text.strip().startswith("{") and response_text.strip().endswith("}"):
            # Single function call
            parsed = json.loads(response_text)
            if "name" in parsed and "arguments" in parsed:
                return [parsed]
        elif response_text.strip().startswith("[") and response_text.strip().endswith("]"):
            # List of function calls
            parsed = json.loads(response_text)
            if all("name" in call and "arguments" in call for call in parsed):
                return parsed

        # If we couldn't parse as JSON, try to extract function call using regex
        # This is a simple example and may need to be adapted
        function_matches = re.findall(
            r"function:\s*(\w+)\s*arguments:\s*({.*?})", response_text, re.DOTALL | re.IGNORECASE
        )
        if function_matches:
            calls = []
            for name, args_str in function_matches:
                try:
                    args = json.loads(args_str)
                    calls.append({"name": name, "arguments": args})
                except:
                    pass
            if calls:
                return calls

        # If all else fails, return an empty list
        return []

    except Exception as e:
        print(f"Error extracting function calls: {e}")
        return []
