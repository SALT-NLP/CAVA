def parse_next_speaker_response(response_text):
    '''
    This function will help to parse the output from LM which contains both reasoning and speaker label to the speaker label like "A" or "B"'''
    try:
        parts = response_text.split('\n')
        
        reasoning = ""
        speaker = ""
        
        # Parse each part
        for part in parts:
            if part.startswith("Reasoning:"):
                reasoning = part.replace("Reasoning:", "").strip()
            elif part.startswith("Speaker:"):
                speaker = part.replace("Speaker:", "").strip()
                speaker=speaker.replace(".", "")
        if not speaker or not reasoning:
            raise ValueError("Invalid response, can't parse speaker or reasoning")
        elif "A" not in speaker and "B" not in speaker and "C" not in speaker and "D" not in speaker and "E" not in speaker:
            raise ValueError(f"Invalid speaker: {speaker}") 
        print(speaker)       
        return speaker
    except Exception as e:
        print(f"Error parsing response: {e}")
        return None