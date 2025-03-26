from typing import Any, Callable, Dict, List, NamedTuple, Optional, Tuple, Union


# Define task configuration using NamedTuple for immutability
class TaskConfig(NamedTuple):
    """Immutable configuration for a task"""

    name: str
    prompt_template: str
    labels: Optional[List[str]] = None
    max_new_tokens: int = 1
    use_logits_processor: bool = True
    output_processor: Callable = lambda x: x.strip()
    field_name: str = "response"  # Name of the field for API schema generation
    audio_dir: str = "generated_audio/"
    data_file: str = "audio_inputs.jsonl"
    speech_output: bool = False  # Flag for tasks that evaluate model's speech output
    output_audio_dir: Optional[str] = None  # Directory to save model's speech output
    template_fields: Optional[Dict[str, Any]] = None  # Fields to populate in prompt template
    verify_tokenization: bool = False  # Flag to verify label tokenization


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
        "deception_detection": TaskConfig(
            name="deception_detection",
            prompt_template="Listen to the following audio of a One Night Ultimate Werewolf game being played. Some of the players may be the Werewolf. Name a player who is the Werewolf. Response using a single word by stating the name of the player who is the Werewolf",
            max_new_tokens = 100,
            field_name="werewolf",
            audio_dir="Werewolf/",
            data_file="werewolf.jsonl",
        ),
        "deception_character_prediction": TaskConfig(
            name="deception_character_prediction",
            prompt_template= "Listen to the following audio of a One Night Ultimate Werewolf game being played. Determine who each player will vote for as the Werewolf. Respond with a JSON-formatted list of names (one word per player) corresponding to the votes.",
            max_new_tokens = 100,
            field_name="votingOutcome",
            audio_dir="Werewolf/",
            data_file="werewolf.jsonl",
        )
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
