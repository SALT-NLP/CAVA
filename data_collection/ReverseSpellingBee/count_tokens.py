#!/usr/bin/env python3
import tiktoken

def count_tokens(text: str, encoding) -> int:
    """Return the number of tokens for a given text using the specified encoding."""
    return len(encoding.encode(text))

def main():
    # Try to get the encoding for the model.
    try:
        encoding = tiktoken.encoding_for_model("gpt-4")
    except AttributeError:
        print("tiktoken.encoding_for_model not available; using fallback encoding 'cl100k_base'.")
        encoding = tiktoken.get_encoding("cl100k_base")
    
    # Define the system prompt.
    system_prompt = (
        "You are an expert linguist tasked with comparing two audio recordings solely for their pronunciation. "
        "Focus on the precise sequence of phonemes, the number of syllables, and the stress/emphasis patterns. "
        "Differences due only to regional accent (e.g., British vs. American) should be ignored. "
        "For example, if two speakers say 'tomato' as 'toh-MAH-toh' (even if their accents differ), they match; "
        "if one says 'toh-MAY-toh', then they do not match.\n\n"
        "IMPORTANT: Respond in text only (do not include any audio output) and output valid JSON with exactly two keys: "
        "'reasoning' (a detailed chain-of-thought explanation) and 'match' (a boolean verdict)."
    )
    
    # Define the list of user messages.
    user_content = [
        {"type": "text", "text": "Here is the first audio clip:"},
        {"type": "input_audio", "input_audio": {"data": "audio1_encoded", "format": "wav"}},
        {"type": "text", "text": "Here is the second audio clip:"},
        {"type": "input_audio", "input_audio": {"data": "audio2_encoded", "format": "wav"}},
        {"type": "text", "text": (
            "Please analyze these recordings strictly for pronunciation details (phonemes, syllables, stress, emphasis). "
            "Ignore differences solely due to accent. Respond ONLY in text and output valid JSON with keys 'reasoning' and 'match' (boolean)."
        )}
    ]
    
    # Define a sample output text.
    output_text = (
        "The first recording pronounces the word 'percent' with a clear emphasis on the second syllable, following the phoneme sequence /pərˈsɛnt/. "
        "The second recording also pronounces 'percent' with the same emphasis on the second syllable and the same phoneme sequence. "
        "Both recordings have the same number of syllables and stress pattern."
    )
    
    # Count tokens for the system prompt.
    system_token_count = count_tokens(system_prompt, encoding)
    print("System prompt token count:", system_token_count)
    
    # Count tokens for each user message.
    total_user_tokens = 0
    for idx, message in enumerate(user_content, start=1):
        if message["type"] == "text":
            text = message["text"]
        elif message["type"] == "input_audio":
            # Use a placeholder text for audio inputs.
            text = "[audio clip]"
        token_count = count_tokens(text, encoding)
        total_user_tokens += token_count
        print(f"User message {idx} ({message['type']}) token count:", token_count)
    
    print("Total user prompt token count:", total_user_tokens)
    
    print("Input text token count:", system_token_count + total_user_tokens)


    # Count tokens for the sample output.
    output_token_count = count_tokens(output_text, encoding)
    print("Output text token count:", output_token_count)
    
    # Sum up the total tokens (system + user + output).
    total_tokens = system_token_count + total_user_tokens + output_token_count
    print("Total tokens (system + user + output):", total_tokens)

if __name__ == '__main__':
    main()