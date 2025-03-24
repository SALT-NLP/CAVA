from typing import Any, Callable, Dict, List, NamedTuple, Optional, Tuple, Union
from utils import parse_next_speaker_response

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
        "nextSpeaker":TaskConfig(
        name="next_speaker",
        data_file="audio_inputs.jsonl",
        audio_dir="NextSpeaker/",
        field_name="speaker_answer",  
        prompt_template="""
You will analyze the following **meeting audio** to determine **who will speak next**.

**Context:**
- The meeting has multiple speakers: {formatted_speaker_list}.
- Below is the transcription of the audio context so far:
{transcription}

**Task:**
- Based on the transcription and audio, predict who will speak next after the current audio ends.
- Only consider meaningful contributions as "next speaker" - defined as utterances that:
* Are not minor or filler phrases
* Contain at least 5 words
- You must **only choose from the following list of speakers**: 
                {formatted_speaker_list}
Please answer in the following format: \nReasoning: [Your reasoning here]. \nSpeaker: [The speaker's label here(e.g., \"A\",\"B\")].
""",
        template_fields={"transcription": "context_transcription", "formatted_speaker_list":"formatted_speaker_list"},  # Template fields to replace
        labels=[],  
        use_logits_processor=False,  #considering COT is important for this task, i would like the model to give reason first and then the speaker label
        verify_tokenization=False,
        max_new_tokens=1000, 
        output_processor=lambda x: parse_next_speaker_response(x),  
        output_audio_dir=None,
        speech_output=False
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
