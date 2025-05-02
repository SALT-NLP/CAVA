#!/usr/bin/env python3
import os
import base64
import json
import time
import hashlib
import openai
import diskcache as dc  # pip install diskcache


# ---------------------------
# Helper for Logging Sanitization
# ---------------------------
def sanitize_for_log(obj, max_length=100):
    """
    Recursively sanitize an object for logging.
    - For dictionaries, processes each key/value.
    - For lists, processes each item.
    - For strings longer than max_length, truncates them and appends the full length.
    - For keys named 'data' inside a dict (e.g. in input_audio), replaces the value with a summary.
    """
    if isinstance(obj, dict):
        new_obj = {}
        for k, v in obj.items():
            if k == "data" and isinstance(v, str) and len(v) > max_length:
                new_obj[k] = f"<audio data, length {len(v)} chars>"
            else:
                new_obj[k] = sanitize_for_log(v, max_length)
        return new_obj
    elif isinstance(obj, list):
        return [sanitize_for_log(x, max_length) for x in obj]
    elif isinstance(obj, str):
        if len(obj) > max_length:
            return obj[:max_length] + f"...(len={len(obj)})"
        else:
            return obj
    else:
        return obj


# ---------------------------
# Prompt & Result Parser Setup
# ---------------------------
def pronunciation_constructor(audio1_encoded, audio2_encoded):
    """
    Constructs prompts for comparing pronunciations.

    Returns:
      (system_prompt, user_prompt) where:
        - system_prompt: a string of instructions.
        - user_prompt: a list of dicts with text and audio inputs.
    """
    system_prompt = (
        "You are an expert linguist tasked with comparing two audio recordings solely for their pronunciation. "
        "Focus on the precise sequence of phonemes, the number of syllables, and the stress/emphasis patterns. "
        "Differences due only to regional accent (e.g., British vs. American) should be ignored. "
        "For example, if two speakers say 'tomato' as 'toh-MAH-toh' (even if their accents differ), they match; "
        "if one says 'toh-MAY-toh', then they do not match.\n\n"
        "IMPORTANT: Respond in text only (do not include any audio output) and output valid JSON with exactly two keys: "
        "'reasoning' (a detailed chain-of-thought explanation) and 'match' (a boolean verdict)."
    )
    user_prompt = [
        {"type": "text", "text": "Here is the first audio clip:"},
        {"type": "input_audio", "input_audio": {"data": audio1_encoded, "format": "wav"}},
        {"type": "text", "text": "Here is the second audio clip:"},
        {"type": "input_audio", "input_audio": {"data": audio2_encoded, "format": "wav"}},
        {
            "type": "text",
            "text": (
                "Please analyze these recordings strictly for pronunciation details (phonemes, syllables, stress, emphasis). "
                "Ignore differences solely due to accent. Respond ONLY in text and output valid JSON with keys 'reasoning' and 'match' (boolean)."
            ),
        },
    ]
    return system_prompt, user_prompt


def default_result_parser(response_text):
    """
    Expects a JSON string with keys 'reasoning' and 'match'. Returns a dictionary.
    """
    try:
        parsed = json.loads(response_text)
        return {
            "reasoning": parsed.get("reasoning", "No reasoning provided."),
            "match": bool(parsed.get("match", False)),
        }
    except Exception as e:
        return {"reasoning": f"Error parsing response: {str(e)}", "match": False}


# ---------------------------
# Helper: Create a dummy resources object
# ---------------------------
def get_resources(model_id):
    """
    Returns a dummy resources object that wraps the openai.ChatCompletion API in the new SDK style.
    """

    class DummyModelChatCompletions:
        def create(self, **kwargs):
            # Forward the call to openai.ChatCompletion.create
            return openai.ChatCompletion.create(**kwargs)

    class DummyModelChat:
        completions = DummyModelChatCompletions()

    class DummyModel:
        chat = DummyModelChat()

    class DummyResources:
        model_name = model_id
        model = DummyModel()

    return DummyResources()


# ---------------------------
# Modern compare_speech function using new API/SDK style with improved debug logging
# ---------------------------
def compare_speech(
    resources,
    prompt_constructor=pronunciation_constructor,
    result_parser=default_result_parser,
    audio_path_1=None,
    audio_path_2=None,
    force_refresh=False,
    cache_dir=os.path.join(os.environ.get("CAVA_CACHE_DIR", ".cava_cache"), "speech_judge"),
    debug=False,
):
    """
    Compare two audio files for pronunciation matching using the modern OpenAI API style.

    Parameters:
      resources: A resources object (e.g. from get_resources) with attributes model_name and model.chat.completions.create.
      prompt_constructor: Function that builds (system_prompt, user_prompt) from base64 audio strings.
      result_parser: Function to parse the API response text.
      audio_path_1, audio_path_2: Paths to the two audio files.
      force_refresh: If True, bypasses the cache.
      cache_dir: Directory for disk caching.
      debug: If True, prints debug information immediately.

    Returns:
      Parsed result as a dict.
    """
    if audio_path_1 is None or audio_path_2 is None:
        raise ValueError("Both audio_path_1 and audio_path_2 must be provided.")

    # Read and encode audio files
    try:
        with open(audio_path_1, "rb") as f:
            audio1_data = f.read()
        with open(audio_path_2, "rb") as f:
            audio2_data = f.read()
    except Exception as e:
        if debug:
            print(f"Error reading audio files: {e}")
        return {"reasoning": f"Error reading audio files: {str(e)}", "match": False}

    audio1_encoded = base64.b64encode(audio1_data).decode("utf-8")
    audio2_encoded = base64.b64encode(audio2_data).decode("utf-8")

    # Construct the prompts
    system_prompt, user_prompt = prompt_constructor(audio1_encoded, audio2_encoded)

    # Create a cache key based on inputs and parameters
    parser_key = getattr(result_parser, "__name__", str(result_parser))
    hash_input = (
        audio1_data
        + audio2_data
        + resources.model_name.encode()
        + system_prompt.encode()
        + json.dumps(user_prompt).encode()
        + parser_key.encode()
    )
    key = hashlib.md5(hash_input).hexdigest()
    CACHE_EXPIRE_SECONDS = int(os.environ.get("CAVA_CACHE_EXPIRE", 60 * 60 * 24 * 30))
    cache = dc.Cache(cache_dir)

    if not force_refresh:
        cached_result = cache.get(key, default=None)
        if cached_result is not None:
            if debug:
                print(f"Cache hit for openai API call: {resources.model_name}")
                print("Cached result:", sanitize_for_log(cached_result))
            return cached_result

    # Build messages according to the new API's format:
    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {"role": "user", "content": user_prompt},
    ]

    # Prepare API call arguments using the new SDK style with modalities set to ["text"]
    api_args = {
        "model": resources.model_name,
        "modalities": ["text"],
        "temperature": 0,
        "messages": messages,
    }

    if debug:
        print("Making API call with arguments:", sanitize_for_log(api_args))

    # Retry logic
    max_retries = 5
    sleep_time = 0.1
    response_text = None
    for attempt in range(max_retries):
        try:
            if debug:
                print(f"API call attempt {attempt+1} for model: {resources.model_name}")
            start_time = time.time()
            completion = resources.model.chat.completions.create(**api_args)
            elapsed = time.time() - start_time
            if debug:
                print(f"API call took {elapsed:.2f} seconds (attempt {attempt+1})")
            msg = completion.choices[0].message
            # Use attribute access to get the text content.
            response_text = msg.content
            if response_text is None:
                if debug:
                    print("No text output received, retrying...")
                time.sleep(sleep_time)
                sleep_time *= 2
                continue
            if debug:
                print("Received response text:", sanitize_for_log(response_text))
            break
        except Exception as e:
            if debug:
                print(f"Error during API call on attempt {attempt+1}: {e}")
            time.sleep(sleep_time)
            sleep_time *= 2

    if response_text is None:
        response_text = json.dumps({"reasoning": "No valid text output received from the model.", "match": False})
        if debug:
            print("Final API response: No valid text output received from the model.")

    # Parse the response
    result = result_parser(response_text)
    if debug:
        print("Parsed result:", sanitize_for_log(result))

    # Only cache the result if it does not indicate an error
    reasoning_lower = result.get("reasoning", "").lower()
    if not (reasoning_lower.startswith("error") or "no valid text output" in reasoning_lower):
        cache.set(key, result, expire=CACHE_EXPIRE_SECONDS)
        if debug:
            print("Result cached successfully.")
    elif debug:
        print("Error result not cached.")

    return result


# ---------------------------
# Example usage (if running directly)
# ---------------------------
if __name__ == "__main__":
    # Ensure your environment has OPENAI_API_KEY set.
    # Create a resources object for the desired model.
    model_id = "gpt-4o-audio-preview"  # Update as needed.
    resources = get_resources(model_id)

    # Replace these with your actual audio file paths.
    audio_file_1 = "path/to/audio1.wav"
    audio_file_2 = "path/to/audio2.wav"

    res = compare_speech(resources=resources, audio_path_1=audio_file_1, audio_path_2=audio_file_2, debug=True)
    print("Final Result:", sanitize_for_log(res))
