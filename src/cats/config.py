from typing import Any, Callable, Dict, List, NamedTuple, Optional, Tuple, Union

from utils import parse_speaker_label_response


# Define task configuration using NamedTuple for immutability
class TaskConfig(NamedTuple):
    """Immutable configuration for a task"""

    name: str
    prompt_template: str
    labels: Optional[List[str]] = None
    max_new_tokens: int = 1
    use_logits_processor: bool = True
    output_processor: Callable = lambda x: x.strip() if x else None  # Default to identity function
    field_name: str = "response"  # Name of the field for API schema generation
    audio_dir: str = "generated_audio/"
    data_file: str = "audio_inputs.jsonl"
    speech_output: bool = False  # Flag for tasks that evaluate model's speech output
    output_audio_dir: Optional[str] = None  # Directory to save model's speech output
    template_fields: Optional[Dict[str, Any]] = None  # Fields to populate in prompt template
    verify_tokenization: bool = False  # Flag to verify label tokenization
    audio_input: bool = True  # Flag to indicate if audio input is required
    process_function_calls: bool = False  # Flag to process function calls


def create_task_configs() -> Dict[str, TaskConfig]:
    """
    Create a dictionary of available task configurations

    Returns:
        Dictionary mapping task names to TaskConfig objects
    """
    return {
        "emotion": TaskConfig(
            name="emotion",
            prompt_template=(
                "Respond in a single word what emotion the input exhibits.\nIf there is no clear emotion, respond"
                " 'Neutral'."
            ),
            labels=["joy", "surprise", "anger", "sadness", "neutral"],
            max_new_tokens=1,
            field_name="emotion",
            audio_dir="generated_audio/",
            data_file="audio_inputs.jsonl",
            verify_tokenization=True,  # Enable tokenization verification for labels
        ),
        "transcription": TaskConfig(
            name="transcription",
            prompt_template="Transcribe the audio content exactly as spoken.",
            use_logits_processor=False,
            max_new_tokens=100,
            field_name="transcript",
            audio_dir="transcription_test/",
            data_file="audio_inputs.jsonl",
        ),
        "pronunciation_oed": TaskConfig(
            name="pronunciation_oed",
            prompt_template="""You are a text-to-speech model. Your task is to generate a high-quality audio pronunciation of the given word in English. The audio should be clear and easy to understand. Please ensure that the pronunciation is accurate and natural-sounding.

The word to read aloud is: "{word}", using the {region} pronunciation: "{OED}".

Please respond with only the word read aloud clearly, do not say anything else or include any additional text or formatting.""",
            max_new_tokens=10,
            audio_dir="ReverseSpellingBee/",
            data_file="audio_inputs.jsonl",
            speech_output=True,
            output_audio_dir="output_pronunciation_oed/",
            audio_input=False,
            template_fields={"word": "word", "region": "region", "OED": "OED"},  # Map placeholders to record fields
        ),
        "pronunciation_audio": TaskConfig(
            name="pronunciation_audio",
            prompt_template="""You are a text-to-speech model. Your task is to generate a high-quality audio pronunciation of the given word in English. The audio should be clear and easy to understand. Please ensure that the pronunciation is accurate and natural-sounding.

The word to read aloud is: "{word}", using the {region} pronunciation.  I have attached an example audio file for you to use as a reference for the pronunciation.

Please respond with only the word read aloud clearly, do not say anything else or include any additional text or formatting.""",
            max_new_tokens=10,
            audio_dir="ReverseSpellingBee/",
            data_file="audio_inputs.jsonl",
            speech_output=True,
            output_audio_dir="output_pronunciation_audio/",
            audio_input=True,
            template_fields={"word": "word", "region": "region", "OED": "OED"},  # Map placeholders to record fields
        "deception_detection": TaskConfig(
            name="deception_detection",
            prompt_template=(
                "Listen to the following audio of a One Night Ultimate Werewolf game being played. Some of the players"
                " may be the Werewolf. Name which player you think is most likely the Werewolf."
                " Respond using a single word by stating the name of the player you think is the Werewolf."
                " If you think nobody is the Werewolf, respond with: None."
            ),
            max_new_tokens=100,
            field_name="werewolf",
            audio_dir="Werewolf/",
            data_file="werewolf.jsonl",
        ),
        "deception_vote_prediction": TaskConfig(
            name="deception_vote_prediction",
            prompt_template=(
                "Listen to the following audio of a One Night Ultimate Werewolf game being played. Guess who each"
                " player will vote for as the Werewolf. Note that the Werewolf must be one of the players in the game."
                " Respond with a list of names (one word per player) corresponding to the votes, seperated by newline"
                " symbol."
            ),
            max_new_tokens=100,
            field_name="votingOutcome",
            audio_dir="Werewolf/",
            data_file="werewolf.jsonl",
            output_processor=clean_prediction,
        ),
        "function_calling": TaskConfig(
            name="function_calling",
            prompt_template=(
                "Execute all necessary function calls before responding. Treat mock responses as successful, valid"
                " data and use them as needed. If information in the user's query is relative or personalized, such as"
                " their current location, use function calls to gather more information. If nested function calls are"
                " required, execute them in a logical order, starting with lower-level functions. Respond to the user"
                " until after you have made all the function calls you need."
            ),
            field_name="intent",
            audio_dir="function_calling_test/",
            data_file="audio_inputs.jsonl",
            process_function_calls=True,
        ),
        "speaker_diarization": TaskConfig(
            name="speaker_diarization",
            data_file="audio_inputs.jsonl",
            audio_dir="SpeakerDiarization/",
            field_name="speaker_order",
            prompt_template="""
              ### **Task: Speaker Diarization**

              You will analyze the following **meeting audio** and its transcript to distinguish different speakers.

              ### **Context:**

              - The meeting consists of **{num_speakers}** distinct speakers.
              - Below is the transcript of the meeting, **without speaker labels**:
              {unlabeled_transcript}

              ### **Instructions:**
              - **Your goal is to differentiate between speakers based on the structure and flow of the conversation, as well as the voice characteristics of different speakers.**  
              - Assign speakers sequentially, starting from **Speaker 1** up to **Speaker {num_speakers}**.
              - Maintain consistency in assigning speaker labels for different parts of the conversation.

              ### **Output Format:**
              Your output should follow this strict format:
              ```
              Sentence 1: Speaker 1
              Sentence 2: Speaker 2
              Sentence 3: Speaker 3
              Sentence 4: Speaker 1
              ```
            """,
            template_fields={
                "num_speakers": "num_speakers",
                "unlabeled_transcript": "transcript_without_speaker",
            },  # Template fields to replace
            labels=[],  # No fixed labels for this task
            use_logits_processor=False,  # Free-form output
            verify_tokenization=False,
            max_new_tokens=1000,  # Allow longer responses for transcripts
            output_processor=lambda x: parse_speaker_label_response(x),
            output_audio_dir=None,
            speech_output=False,
        ),
        "jeopardy": TaskConfig(
            name="jeopardy",
            prompt_template=(
                "You are a contestant on Jeopardy. the answer must be worded in the form of a question, beginning with"
                " “What is” or “Who are,” for example."
            ),
            use_logits_processor=False,
            max_new_tokens=100,
            field_name="question",
            audio_dir="jeopardy/",
            data_file="audio_inputs.jsonl",
        ),
    }


def format_prompt_template(
    template: str, record: Dict[str, Any], template_fields: Optional[Dict[str, Any]] = None
) -> str:
    """
    Format a prompt template with values from the record

    Args:
        template: The prompt template with placeholders
        record: The record containing field values
        template_fields: Optional mapping from template placeholders to record fields

    Returns:
        Formatted prompt
    """
    if not template_fields:
        return template

    # Create a dictionary of values to replace in the template
    replacements = {}
    for template_key, record_field in template_fields.items():
        if record_field in record:
            replacements[template_key] = record[record_field]

    # Format the template with the replacements
    return template.format(**replacements)


def clean_prediction(prediction_str: str) -> List[str]:
    """
    Cleans and parses the prediction field which is a multiline string of predicted vote names.
    It removes introductory header lines (if any) and strips out markdown asterisks.
    """
    lines = prediction_str.strip().splitlines()
    votes = []
    for line in lines:
        clean_line = line.strip()
        # If the line contains keywords that indicate it's header text, skip it.
        if any(keyword in clean_line.lower() for keyword in ["predicted", "votes", "based on the provided audio"]):
            continue
        # Remove markdown asterisks if present.
        clean_line = clean_line.replace("*", "").strip()
        votes.append(clean_line)
    return votes
