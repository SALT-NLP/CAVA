# -*- coding: utf-8 -*-
"""Intent Evaluation Utilities

Utilities for comparing TOP intent structures with function calls.
Evaluates whether intents match function calls and slots match function arguments.
"""

import re
from typing import Dict, List, Tuple, Any, Optional, Set


def parse_top(text):
    """
    Parse a TOP format string into a nested structure
    Example input: [IN:GET_DIRECTIONS Directions to [SL:DESTINATION [IN:GET_EVENT the [SL:NAME_EVENT Eagles ] [SL:CAT_EVENT game ] ] ] ]

    This function is adapted from the top_parsing.py file
    """

    # Helper function for recursive parsing
    def parse_recursive(text, start_idx):
        current_node = {"type": None, "value": None, "children": []}
        i = start_idx
        text_buffer = ""

        while i < len(text):
            char = text[i]

            # Start of a new node
            if char == "[":
                # Save any text we've accumulated
                if text_buffer.strip():
                    current_node["children"].append({"type": "TEXT", "value": text_buffer.strip(), "children": []})
                    text_buffer = ""

                # Find the node type and its end
                end_of_type = text.find(" ", i)
                if end_of_type == -1:
                    raise ValueError(f"Invalid TOP format at position {i}, missing space after node type")

                node_type = text[i + 1 : end_of_type]

                # Parse the nested node
                i = end_of_type + 1  # Move past the space
                child_node, i = parse_recursive(text, i)
                child_node["type"] = node_type
                current_node["children"].append(child_node)
                continue

            # End of current node
            elif char == "]":
                # Save any remaining text
                if text_buffer.strip():
                    current_node["children"].append({"type": "TEXT", "value": text_buffer.strip(), "children": []})

                return current_node, i + 1

            # Regular text - add to buffer
            else:
                text_buffer += char

            i += 1

        # If we get here in top-level call, it means no closing bracket was found
        if start_idx == 0:
            # Just return what we have for text without brackets
            if text_buffer.strip():
                current_node["value"] = text_buffer.strip()
            return current_node, i
        else:
            raise ValueError(f"Unclosed bracket in TOP format")

    try:
        result, _ = parse_recursive(text, 0)
        # If there's only one child and no value in root, return that child
        if len(result["children"]) == 1 and not result["value"] and not result["type"]:
            return result["children"][0]
        return result
    except Exception as e:
        raise ValueError(f"Failed to parse TOP format: {e}")


def extract_intent_hierarchy(parsed_intent):
    """
    Extract the hierarchy of intents with their slots from a parsed TOP structure.

    Args:
        parsed_intent: A parsed TOP structure (dictionary)

    Returns:
        A list of dictionaries representing the intents with their slots and nested intents
    """
    intents = []

    def traverse(node, parent_intent=None):
        # If this is an intent node
        if node["type"] and node["type"].startswith("IN:"):
            intent_name = node["type"].replace("IN:", "").lower()
            current_intent = {"name": intent_name, "arguments": {}, "nested_intents": []}

            # Add this intent to its parent if it has one
            if parent_intent:
                parent_intent["nested_intents"].append(current_intent)
            else:
                intents.append(current_intent)

            # Process all children
            for child in node["children"]:
                traverse(child, current_intent)

            return current_intent

        # If this is a slot node
        elif node["type"] and node["type"].startswith("SL:"):
            slot_name = node["type"].replace("SL:", "").lower()
            slot_value = ""
            has_nested_intent = False

            # Extract the slot value or process nested intents
            for child in node["children"]:
                if child["type"] and child["type"].startswith("IN:"):
                    # This slot contains a nested intent
                    has_nested_intent = True
                    traverse(child, parent_intent)
                elif child["value"]:
                    # This is a text value for the slot
                    slot_value += child["value"] + " "
                else:
                    # Process any other children of the slot
                    traverse(child, parent_intent)

            # Only add the slot to the parent intent if it has a value
            # and doesn't contain a nested intent
            if parent_intent and not has_nested_intent and slot_value.strip():
                parent_intent["arguments"][slot_name] = slot_value.strip()

        # Process any other node's children
        elif "children" in node:
            for child in node["children"]:
                traverse(child, parent_intent)

    # Start traversal from the root
    traverse(parsed_intent)
    return intents


def extract_function_calls(function_calls_json):
    """
    Extract function calls from JSON representation

    Args:
        function_calls_json: A list of function call dictionaries

    Returns:
        A list of standardized function call dictionaries
    """
    function_calls = []

    for call in function_calls_json:
        function_call = {
            "name": call["name"],
            "arguments": call["arguments"],
            "nested_intents": [],  # Function calls don't have nested intents in the representation
        }
        function_calls.append(function_call)

    return function_calls


def extract_intent_hierarchy_with_order(parsed_intent):
    """
    Extract the hierarchy of intents in depth-first order

    Args:
        parsed_intent: The parsed TOP intent

    Returns:
        List of intent names in depth-first order
    """
    intent_sequence = []

    def traverse(node):
        if node["type"] and node["type"].startswith("IN:"):
            intent_name = node["type"].replace("IN:", "").lower()
            intent_sequence.append(intent_name)

        if "children" in node:
            for child in node["children"]:
                traverse(child)

    traverse(parsed_intent)
    return intent_sequence


def compare_root_intent(parsed_intent, function_calls):
    """
    Check if the first intent matches the final function call

    Args:
        parsed_intent: The parsed TOP intent
        function_calls: List of function call dictionaries

    Returns:
        Tuple: (bool: is_match, str: message)
    """
    # Find the root intent
    root_intent = None

    def find_root(node):
        if node["type"] and node["type"].startswith("IN:"):
            return node["type"].replace("IN:", "").lower()
        return None

    root_intent = find_root(parsed_intent)

    if not root_intent:
        return False, "No intent found in the parsed structure"

    if not function_calls:
        return False, "No function calls found"

    last_function = function_calls[-1]["name"]

    if root_intent == last_function:
        return True, f"Root intent '{root_intent}' matches final function call"
    else:
        return False, f"Root intent '{root_intent}' does not match final function call '{last_function}'"


def count_intents(parsed_intent):
    """
    Count all intents in the parsed TOP structure

    Args:
        parsed_intent: The parsed TOP intent

    Returns:
        Dictionary: mapping intent names to their counts
    """
    intent_counts = {}

    def traverse(node):
        if node["type"] and node["type"].startswith("IN:"):
            intent_name = node["type"].replace("IN:", "").lower()
            intent_counts[intent_name] = intent_counts.get(intent_name, 0) + 1

        if "children" in node:
            for child in node["children"]:
                traverse(child)

    traverse(parsed_intent)
    return intent_counts


def count_function_calls(function_calls):
    """
    Count all function calls

    Args:
        function_calls: List of function call dictionaries

    Returns:
        Dictionary: mapping function names to their counts
    """
    call_counts = {}

    for call in function_calls:
        call_counts[call["name"]] = call_counts.get(call["name"], 0) + 1

    return call_counts


def compare_function_call_counts(parsed_intent, function_calls):
    """
    Check if all functions are called the correct number of times

    Args:
        parsed_intent: The parsed TOP intent
        function_calls: List of function call dictionaries

    Returns:
        Tuple: (bool: is_match, str: message, dict: detailed_results)
    """
    intent_counts = count_intents(parsed_intent)
    function_counts = count_function_calls(function_calls)

    # Check if all intents have a corresponding function call
    all_match = True
    detailed_results = {}

    for intent_name, count in intent_counts.items():
        if intent_name not in function_counts:
            all_match = False
            detailed_results[intent_name] = {
                "intent_count": count,
                "function_count": 0,
                "status": "Missing function call",
            }
        elif function_counts[intent_name] != count:
            all_match = False
            detailed_results[intent_name] = {
                "intent_count": count,
                "function_count": function_counts[intent_name],
                "status": "Count mismatch",
            }
        else:
            detailed_results[intent_name] = {
                "intent_count": count,
                "function_count": function_counts[intent_name],
                "status": "Match",
            }

    # Check if there are any extra function calls not in the intents
    for func_name in function_counts:
        if func_name not in intent_counts:
            all_match = False
            detailed_results[func_name] = {
                "intent_count": 0,
                "function_count": function_counts[func_name],
                "status": "Extra function call",
            }

    if all_match:
        return True, "All functions are called the correct number of times", detailed_results
    else:
        return False, "Function call counts do not match intent counts", detailed_results


def extract_slot_values(parsed_intent):
    """
    Extract slot values from the parsed intent structure

    Args:
        parsed_intent: The parsed TOP intent tree

    Returns:
        Dictionary mapping intent names to their slot values
    """
    intent_slots = {}

    def traverse(node, current_intent=None, slot_path=None):
        # If this is an intent node
        if node["type"] and node["type"].startswith("IN:"):
            intent_name = node["type"].replace("IN:", "").lower()

            # Initialize intent in the result dictionary
            if intent_name not in intent_slots:
                intent_slots[intent_name] = {"slots": {}, "text_value": ""}

            # Set as current intent for children
            current_intent = intent_name
            slot_path = None  # Reset slot path when entering new intent

        # If this is a slot node
        elif node["type"] and node["type"].startswith("SL:"):
            slot_name = node["type"].replace("SL:", "").lower()

            # Update the slot path
            slot_path = slot_name

            # Initialize slot in the intent if we have an intent context
            if current_intent and slot_path:
                if slot_path not in intent_slots[current_intent]["slots"]:
                    intent_slots[current_intent]["slots"][slot_path] = ""

        # If this is a text node and we're in a slot
        elif node["value"] and slot_path and current_intent:
            # Append text to the slot value
            intent_slots[current_intent]["slots"][slot_path] += " " + node["value"]
            intent_slots[current_intent]["slots"][slot_path] = intent_slots[current_intent]["slots"][slot_path].strip()

        # If this is a text node directly under an intent (not in a slot)
        elif node["value"] and current_intent and not slot_path:
            # Append text to the intent's text value
            intent_slots[current_intent]["text_value"] += " " + node["value"]
            intent_slots[current_intent]["text_value"] = intent_slots[current_intent]["text_value"].strip()

        # Process all children with current context
        if "children" in node:
            for child in node["children"]:
                # Pass current context to children
                traverse(child, current_intent, slot_path)

    # Start traversal from the root
    traverse(parsed_intent)

    # Clean up slot values (strip whitespace)
    for intent_name, intent_data in intent_slots.items():
        for slot_name, slot_value in intent_data["slots"].items():
            intent_data["slots"][slot_name] = slot_value.strip()

    return intent_slots


def check_hierarchical_order(parsed_intent, function_calls):
    """
    Check if functions are called in a valid depth-first traversal order

    Args:
        parsed_intent: The parsed TOP intent
        function_calls: List of function call dictionaries

    Returns:
        Tuple: (bool: is_match, str: message, dict: details)
    """
    # Extract intent names in depth-first order
    intent_sequence = extract_intent_hierarchy_with_order(parsed_intent)

    # Extract function call names
    function_names = [call["name"] for call in function_calls]

    # For comparing sets of intents and functions
    intent_set = set(intent_sequence)
    function_set = set(function_names)

    # Check if all intents have corresponding function calls
    missing_intents = [name for name in intent_set if name not in function_set]
    extra_functions = [name for name in function_set if name not in intent_set]

    # Build a list of parent-child relationships in the intent tree
    intent_hierarchy = {}
    intent_children = {}
    intent_parents = {}
    all_intents = set()

    def build_hierarchy(node, parent=None):
        if node["type"] and node["type"].startswith("IN:"):
            intent_name = node["type"].replace("IN:", "").lower()
            all_intents.add(intent_name)

            # Initialize hierarchy entry if needed
            if intent_name not in intent_hierarchy:
                intent_hierarchy[intent_name] = []
                intent_children[intent_name] = set()

            # Record parent-child relationship
            if parent:
                if parent not in intent_hierarchy:
                    intent_hierarchy[parent] = []
                    intent_children[parent] = set()

                intent_hierarchy[parent].append(intent_name)
                intent_children[parent].add(intent_name)
                intent_parents[intent_name] = parent

            current_parent = intent_name
        else:
            current_parent = parent

        if "children" in node:
            for child in node["children"]:
                build_hierarchy(child, current_parent)

    build_hierarchy(parsed_intent)

    # Prepare validation details structure
    validation_details = {
        "is_valid_traversal": True,
        "missing_intents": missing_intents,
        "extra_functions": extra_functions,
        "intent_sequence": intent_sequence,
        "function_sequence": function_names,
        "intent_hierarchy": intent_hierarchy,
        "issues": [],
    }

    # If there are missing intents or extra functions, it's not a valid traversal
    if missing_intents or extra_functions:
        validation_details["is_valid_traversal"] = False
        if missing_intents:
            validation_details["issues"].append(f"Missing intents: {missing_intents}")
        if extra_functions:
            validation_details["issues"].append(f"Extra functions: {extra_functions}")
        return False, "Function calls don't match all intents", validation_details

    # Check if the root intent is last in the function calls
    # This is the key constraint in TOP - the root intent must be processed last
    root_intents = [intent for intent in all_intents if intent not in intent_parents]

    if root_intents and function_names[-1] not in root_intents:
        validation_details["is_valid_traversal"] = False
        validation_details["issues"].append(
            f"Root intent(s) {root_intents} should be the last function call, but found {function_names[-1]}"
        )
        return False, "Root intent must be the last function call", validation_details

    # Check that parents are processed after all their children
    for i, func in enumerate(function_names):
        # If this function has children
        if func in intent_children and intent_children[func]:
            # All children must appear before this function
            remaining_functions = set(function_names[i + 1 :])
            children_after = [child for child in intent_children[func] if child in remaining_functions]

            if children_after:
                validation_details["is_valid_traversal"] = False
                validation_details["issues"].append(f"Function {func} has unprocessed children: {children_after}")
                return False, "Functions must be processed after all their children", validation_details

    # All checks passed
    return True, "All functions are called in a valid hierarchical order", validation_details


def is_value_match(slot_value, func_value):
    """
    Determine if a slot value matches a function value.

    Args:
        slot_value: Value from the intent slot
        func_value: Value from the function argument

    Returns:
        True if values match, False otherwise
    """
    # Normalize values for comparison
    slot_norm = str(slot_value).lower().strip()
    func_norm = str(func_value).lower().strip()

    # Exact match
    if slot_norm == func_norm:
        return True

    # Substring match
    if slot_norm in func_norm or func_norm in slot_norm:
        return True

    return False


def compare_slot_values(parsed_intent, function_calls):
    """
    Compare slot values in the intent with arguments in function calls

    Args:
        parsed_intent: The parsed TOP intent
        function_calls: List of function call dictionaries

    Returns:
        Tuple: (bool: all_match, str: message, dict: detailed_results)
    """
    # Extract slot values from the parsed intent
    intent_slots = extract_slot_values(parsed_intent)

    if not intent_slots:
        return False, "No intent slots found in the parsed structure", {}

    if not function_calls:
        return False, "No function calls found", {}

    # Map function calls by name (use the last occurrence if multiple)
    function_map = {}
    for call in function_calls:
        function_map[call["name"]] = call

    all_match = True
    detailed_results = {}

    # Check each intent against its corresponding function call
    for intent_name, intent_data in intent_slots.items():
        slot_results = {}

        # Skip intents with no slots
        if not intent_data["slots"]:
            detailed_results[intent_name] = {"status": "no_slots", "slots": {}}
            continue

        # Check if this intent has a corresponding function call
        if intent_name not in function_map:
            all_match = False
            detailed_results[intent_name] = {"status": "missing_function", "slots": {}}
            continue

        function_call = function_map[intent_name]
        intent_match = True

        # Check each slot against function arguments
        for slot_name, slot_value in intent_data["slots"].items():
            # Standard slot name matching
            if slot_name not in function_call["arguments"]:
                # No matching argument found
                intent_match = False
                slot_results[slot_name] = {"status": "missing", "intent_value": slot_value, "function_value": None}
                continue

            # Normal case - matching slot name found
            func_value = function_call["arguments"][slot_name]

            # Simple comparison using is_value_match helper
            if not is_value_match(slot_value, func_value):
                intent_match = False
                slot_results[slot_name] = {
                    "status": "value_mismatch",
                    "intent_value": slot_value,
                    "function_value": func_value,
                }
            else:
                slot_results[slot_name] = {"status": "match", "intent_value": slot_value, "function_value": func_value}

        # Check for extra arguments in function call
        for arg_name in function_call["arguments"]:
            if arg_name not in intent_data["slots"]:
                intent_match = False
                slot_results[arg_name] = {
                    "status": "extra",
                    "intent_value": None,
                    "function_value": function_call["arguments"][arg_name],
                }

        # Update overall match status
        if not intent_match:
            all_match = False

        detailed_results[intent_name] = {"status": "match" if intent_match else "slot_mismatch", "slots": slot_results}

    if all_match:
        return True, "All slot values match function arguments", detailed_results
    else:
        return False, "Some slot values do not match function arguments", detailed_results


def evaluate_intent_to_function_mapping(intent_str, function_calls_json):
    """
    Main evaluation function that runs all comparisons

    Args:
        intent_str: The TOP format intent string
        function_calls_json: The function calls JSON

    Returns:
        Dictionary with evaluation results
    """
    try:
        parsed_intent = parse_top(intent_str)

        # Run all evaluations
        root_match, root_message = compare_root_intent(parsed_intent, function_calls_json)

        counts_match, counts_message, count_details = compare_function_call_counts(parsed_intent, function_calls_json)

        # Check hierarchical order
        hierarchy_match, hierarchy_message, hierarchy_details = check_hierarchical_order(
            parsed_intent, function_calls_json
        )

        # Compare slot values
        slots_match, slots_message, slot_details = compare_slot_values(parsed_intent, function_calls_json)

        return {
            "overall_success": root_match and counts_match and hierarchy_match and slots_match,
            "checks": {
                "root_intent_match": {"success": root_match, "message": root_message},
                "function_counts_match": {
                    "success": counts_match,
                    "message": counts_message,
                    "details": count_details,
                },
                "hierarchical_order_match": {
                    "success": hierarchy_match,
                    "message": hierarchy_message,
                    "details": hierarchy_details,
                },
                "slot_values_match": {"success": slots_match, "message": slots_message, "details": slot_details},
            },
            "intent_structure": extract_intent_hierarchy(parsed_intent),
            "parsed_intent": parsed_intent,
        }

    except Exception as e:
        return {"overall_success": False, "error": str(e)}


# Example usage
if __name__ == "__main__":
    # Example 1: Simple nested intent structure with one function call
    intent_str1 = "[IN:GET_WEATHER what 's the [SL:LOCATION [IN:GET_LOCATION [SL:LOCATION_MODIFIER local ] ] ] radar ]"

    function_calls_json1 = [{"name": "get_weather", "arguments": {"weather_attribute": "radar"}}]

    # Example 2: More complex intent structure with multiple nested intents
    intent_str2 = "[IN:GET_DIRECTIONS Directions to [SL:DESTINATION [IN:GET_EVENT the [SL:NAME_EVENT Eagles ] [SL:CAT_EVENT game ] ] ] ]"

    # Valid function calls (in correct depth-first traversal order)
    function_calls_json2 = [
        {"name": "get_event", "arguments": {"name_event": "Eagles", "cat_event": "game"}},
        {"name": "get_directions", "arguments": {"destination": "the Eagles game"}},
    ]

    # Invalid function calls (wrong order)
    function_calls_json2_invalid = [
        {"name": "get_directions", "arguments": {"destination": "the Eagles game"}},
        {"name": "get_event", "arguments": {"name_event": "Eagles", "cat_event": "game"}},
    ]

    # Example 3: Multiple different valid depth-first traversals
    intent_str3 = """[IN:COMPLEX_REQUEST I need to [IN:SCHEDULE_MEETING schedule a meeting ] and [IN:SEND_MESSAGE send a message ] about the [IN:GET_DOCUMENT project document ] ]"""

    # Valid traversal 1
    function_calls_json3_valid1 = [
        {"name": "schedule_meeting", "arguments": {}},
        {"name": "send_message", "arguments": {}},
        {"name": "get_document", "arguments": {}},
        {"name": "complex_request", "arguments": {}},
    ]

    # Valid traversal 2
    function_calls_json3_valid2 = [
        {"name": "schedule_meeting", "arguments": {}},
        {"name": "get_document", "arguments": {}},
        {"name": "send_message", "arguments": {}},
        {"name": "complex_request", "arguments": {}},
    ]

    # Invalid traversal
    function_calls_json3_invalid = [
        {"name": "complex_request", "arguments": {}},
        {"name": "schedule_meeting", "arguments": {}},
        {"name": "send_message", "arguments": {}},
        {"name": "get_document", "arguments": {}},
    ]

    # Run evaluations
    print("Example 1:")
    results1 = evaluate_intent_to_function_mapping(intent_str1, function_calls_json1)
    print(f"Overall success: {results1['overall_success']}")
    for check_name, check_result in results1["checks"].items():
        print(f"  {check_name}: {check_result['success']}")

    print("\nExample 2 (Valid):")
    results2 = evaluate_intent_to_function_mapping(intent_str2, function_calls_json2)
    print(f"Overall success: {results2['overall_success']}")
    for check_name, check_result in results2["checks"].items():
        print(f"  {check_name}: {check_result['success']}")

    print("\nExample 2 (Invalid):")
    results2_invalid = evaluate_intent_to_function_mapping(intent_str2, function_calls_json2_invalid)
    print(f"Overall success: {results2_invalid['overall_success']}")
    for check_name, check_result in results2_invalid["checks"].items():
        print(f"  {check_name}: {check_result['success']}")
        if not check_result["success"]:
            print(f"    Message: {check_result['message']}")

    print("\nExample 3 (Valid Traversal 1):")
    results3_valid1 = evaluate_intent_to_function_mapping(intent_str3, function_calls_json3_valid1)
    print(f"Overall success: {results3_valid1['overall_success']}")
    for check_name, check_result in results3_valid1["checks"].items():
        print(f"  {check_name}: {check_result['success']}")

    print("\nExample 3 (Valid Traversal 2):")
    results3_valid2 = evaluate_intent_to_function_mapping(intent_str3, function_calls_json3_valid2)
    print(f"Overall success: {results3_valid2['overall_success']}")
    for check_name, check_result in results3_valid2["checks"].items():
        print(f"  {check_name}: {check_result['success']}")

    print("\nExample 3 (Invalid Traversal):")
    results3_invalid = evaluate_intent_to_function_mapping(intent_str3, function_calls_json3_invalid)
    print(f"Overall success: {results3_invalid['overall_success']}")
    for check_name, check_result in results3_invalid["checks"].items():
        print(f"  {check_name}: {check_result['success']}")
        if not check_result["success"]:
            print(f"    Message: {check_result['message']}")

    # For more detailed output
    # import json
    # print(json.dumps(results1, indent=2))
