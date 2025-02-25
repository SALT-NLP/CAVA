import base64
import json
import os
import time
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Tuple, Type, Union

import librosa
import numpy as np
import soundfile as sf
import torch
from dotenv import load_dotenv
from pydantic import BaseModel
from tqdm import tqdm
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
    GenerationConfig,
    PrefixConstrainedLogitsProcessor,
    Qwen2AudioForConditionalGeneration,
)

from meau.config import TaskConfig, create_task_configs


# Global API call counters for rate limiting
API_CALL_COUNTERS = {"gemini": 0, "openai": 0}

# Maximum API calls allowed per run
MAX_API_CALLS = 10000

load_dotenv()


# Dynamic schema generation functions
def create_enum_from_labels(labels: List[str], enum_name: str = "DynamicEnum") -> Type[Enum]:
    """
    Dynamically create an Enum class from a list of labels

    Args:
        labels: List of label strings
        enum_name: Name for the generated Enum class

    Returns:
        Dynamically created Enum class
    """
    if not labels:
        raise ValueError("Cannot create Enum from empty labels list")

    return Enum(enum_name, {label.upper(): label.lower() for label in labels})


def create_schema_wrapper(field_name: str, enum_type=None) -> Type[BaseModel]:
    """
    Dynamically create a Pydantic model for API response parsing

    Args:
        field_name: Name of the field to create
        enum_type: Optional enum type for validation

    Returns:
        Dynamically created Pydantic model class
    """
    if enum_type:
        # Create a model with an enum field
        return type(f"{field_name.capitalize()}Wrapper", (BaseModel,), {field_name: enum_type})
    else:
        # Create a model with a string field
        return type(f"{field_name.capitalize()}Wrapper", (BaseModel,), {field_name: (str, ...)})


# Define model configuration
class ModelResources(NamedTuple):
    """Immutable container for model resources"""

    tokenizer: Any
    processor: Any
    model: Any
    model_name: str
    model_type: str  # Type indicator: 'transformers', 'gemini', 'openai'


def load_model(model_name: str) -> ModelResources:
    """
    Load model, tokenizer and processor based on model type

    Args:
        model_name: Model name/path

    Returns:
        ModelResources containing loaded components
    """
    try:
        # Transformers-based models
        if "Qwen2" in model_name:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            processor = AutoProcessor.from_pretrained(model_name)
            model = Qwen2AudioForConditionalGeneration.from_pretrained(model_name, device_map="auto")
            model_type = "transformers"
        elif "diva" in model_name.lower():
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            processor = None
            model = AutoModel.from_pretrained(model_name, device_map="balanced_low_0", trust_remote_code=True).eval()
            model_type = "transformers"

        # API-based models
        elif "gemini" in model_name.lower():
            # Import here to avoid dependency issues if not using Gemini
            import google.generativeai as genai

            api_key = os.environ.get("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("GEMINI_API_KEY environment variable is required for Gemini models")

            genai.configure(api_key=api_key)
            model = genai.GenerativeModel(model_name)
            tokenizer = None
            processor = None
            model_type = "gemini"

        elif "gpt" in model_name.lower():
            # Import here to avoid dependency issues if not using OpenAI
            from openai import OpenAI

            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable is required for OpenAI models")

            model = OpenAI(api_key=api_key)
            tokenizer = None
            processor = None
            model_type = "openai"

        else:
            # Generic transformers model fallback
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            processor = AutoProcessor.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", trust_remote_code=True)
            model_type = "transformers"

        return ModelResources(tokenizer, processor, model, model_name, model_type)

    except Exception as e:
        raise ValueError(f"Error loading model {model_name}: {str(e)}")


def process_audio(audio_file: str, audio_dir: str = "") -> Dict[str, Any]:
    """
    Process audio file into model-compatible format

    Args:
        audio_file: Path to audio file
        audio_dir: Optional directory containing audio files

    Returns:
        Dictionary with processed audio data
    """

    full_path = Path("data") / (audio_dir + audio_file if audio_dir else audio_file)
    audio, sr = librosa.load(str(full_path), sr=16000)
    return {"array": audio, "sampling_rate": sr}


def create_logits_processor(tokenizer: Any, labels: List[str]) -> PrefixConstrainedLogitsProcessor:
    """
    Create a logits processor for constrained generation

    Args:
        tokenizer: Model tokenizer
        labels: List of allowed labels

    Returns:
        Configured logits processor
    """
    return PrefixConstrainedLogitsProcessor(
        prefix_allowed_tokens_fn=lambda batch_id, input_ids: [
            tokenizer.convert_tokens_to_ids(tokenizer.tokenize(label))[0] for label in labels
        ],
        num_beams=1,
    )


def save_temp_audio(audio: Dict[str, Any], task_name: str, model_type: str) -> str:
    """
    Save audio to temporary file

    Args:
        audio: Processed audio data
        task_name: Name of the current task
        model_type: Type of model (for filename)

    Returns:
        Path to saved temporary file
    """
    import tempfile

    # Create a temporary file with proper suffix
    tmp_dir = tempfile.gettempdir()
    temp_audio_path = Path(tmp_dir) / f"meau_{task_name}_{model_type}_{int(time.time())}.wav"

    sf.write(str(temp_audio_path), audio["array"], audio["sampling_rate"], format="wav")
    return str(temp_audio_path)


def cleanup_temp_audio(temp_audio_path: str) -> None:
    """
    Remove temporary audio file

    Args:
        temp_audio_path: Path to temporary file
    """
    try:
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)
    except Exception as e:
        print(f"Warning: Failed to remove temporary file {temp_audio_path}: {e}")


def check_api_call_limit(model_type: str) -> bool:
    """
    Check if API call limit has been reached

    Args:
        model_type: Type of API model ('gemini' or 'openai')

    Returns:
        True if limit is not reached, False otherwise
    """
    if model_type in API_CALL_COUNTERS:
        current_count = API_CALL_COUNTERS[model_type]

        if current_count >= MAX_API_CALLS:
            print(f"WARNING: Maximum API call limit ({MAX_API_CALLS}) reached for {model_type}.")
            return False

        # Increment counter
        API_CALL_COUNTERS[model_type] += 1

        # Print warning when approaching limit
        if current_count % 1000 == 0:
            print(f"API calls to {model_type}: {current_count}/{MAX_API_CALLS}")

        return True

    return True  # Not an API model


@torch.no_grad()
def process_with_qwen(
    resources: ModelResources, audio: Dict[str, Any], text_prompt: str, task_config: TaskConfig
) -> str:
    """
    Process audio with Qwen model

    Args:
        resources: Model resources
        audio: Processed audio data
        text_prompt: Text prompt for the model
        task_config: Task configuration

    Returns:
        Model output
    """
    # Save audio to temporary file
    temp_audio_path = save_temp_audio(audio, task_config.name, "qwen2")

    try:
        # Create conversation input
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio_url": temp_audio_path},
                    {"type": "text", "text": text_prompt},
                ],
            },
        ]

        # Apply chat template
        text_input = resources.processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)

        # Tokenize input
        inputs = resources.tokenizer(text=text_input, return_tensors="pt").to("cuda")

        # Prepare generation kwargs
        gen_kwargs = {**inputs, "max_new_tokens": task_config.max_new_tokens}

        # Add logits processor if needed
        if task_config.use_logits_processor and task_config.labels:
            logits_processor = create_logits_processor(resources.tokenizer, task_config.labels)
            gen_kwargs["logits_processor"] = [logits_processor]

        # Generate response
        outputs = resources.model.generate(**gen_kwargs)
        response = resources.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # More robust extraction of assistant's response
        if "assistant" in response:
            response_parts = response.split("assistant")
            # Take the last part after "assistant"
            response = response_parts[-1].strip()

        return response
    finally:
        # Clean up temporary file
        cleanup_temp_audio(temp_audio_path)


@torch.no_grad()
def process_with_diva(
    resources: ModelResources, audio: Dict[str, Any], text_prompt: str, task_config: TaskConfig
) -> str:
    """
    Process audio with DiVA model

    Args:
        resources: Model resources
        audio: Processed audio data
        text_prompt: Text prompt for the model
        task_config: Task configuration

    Returns:
        Model output
    """
    # Save audio to temporary file - although not directly used by the model
    # it can be helpful for debugging or logging purposes
    temp_audio_path = save_temp_audio(audio, task_config.name, "diva")

    try:
        # Prepare generation kwargs
        gen_kwargs = {
            "audio": [audio["array"]],
            "text_prompt": ["\n" + text_prompt],
            "max_new_tokens": task_config.max_new_tokens,
        }

        # Add logits processor if needed
        if task_config.use_logits_processor and task_config.labels:
            logits_processor = create_logits_processor(resources.tokenizer, task_config.labels)
            gen_kwargs["logits_processor"] = logits_processor

        # Generate response
        with torch.cuda.amp.autocast(dtype=torch.float16):
            llm_message = resources.model.generate(**gen_kwargs)

        return llm_message[0]
    finally:
        # Clean up temporary file
        cleanup_temp_audio(temp_audio_path)


def process_with_gemini(
    resources: ModelResources, audio: Dict[str, Any], text_prompt: str, task_config: TaskConfig
) -> str:
    """
    Process audio with Gemini model

    Args:
        resources: Model resources
        audio: Processed audio data
        text_prompt: Text prompt for the model
        task_config: Task configuration

    Returns:
        Model output
    """
    import google.generativeai as genai

    # Check API call limit
    if not check_api_call_limit("gemini"):
        # Return default value if limit reached
        return task_config.labels[0] if task_config.labels else "API limit reached"

    # Save audio to temporary file
    temp_audio_path = save_temp_audio(audio, task_config.name, "gemini")

    try:
        # Create inputs for the model
        prompt = text_prompt
        inputs = [
            prompt,
            {"mime_type": "audio/wav", "data": Path(temp_audio_path).read_bytes()},
        ]

        # Set up retry logic
        max_retries = 5
        sleep_time = 1

        # Try to generate content with retries for API rate limits
        for attempt in range(max_retries):
            try:
                # If we have labels for constrained generation
                if task_config.labels and task_config.use_logits_processor:
                    # Dynamically create enum type from labels
                    DynamicEnum = create_enum_from_labels(task_config.labels, f"{task_config.name.capitalize()}Enum")

                    response = resources.model.generate_content(
                        inputs,
                        generation_config=genai.GenerationConfig(
                            response_mime_type="text/x.enum", response_schema=DynamicEnum
                        ),
                    )
                else:
                    response = resources.model.generate_content(inputs)

                response_text = response.candidates[0].content.parts[0].text
                return response_text

            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"Gemini API error: {e}. Retrying after {sleep_time}s...")
                    time.sleep(sleep_time)
                    sleep_time *= 2
                else:
                    print(f"Failed after {max_retries} attempts: {e}")
                    # Default to first label if available, otherwise empty string
                    return task_config.labels[0] if task_config.labels else ""
    finally:
        # Clean up temporary file
        cleanup_temp_audio(temp_audio_path)


def process_with_openai(
    resources: ModelResources, audio: Dict[str, Any], text_prompt: str, task_config: TaskConfig
) -> str:
    """
    Process audio with OpenAI model

    Args:
        resources: Model resources
        audio: Processed audio data
        text_prompt: Text prompt for the model
        task_config: Task configuration

    Returns:
        Model output
    """
    # Check API call limit
    if not check_api_call_limit("openai"):
        # Return default value if limit reached
        return task_config.labels[0] if task_config.labels else "API limit reached"

    # Save audio to temporary file
    temp_audio_path = save_temp_audio(audio, task_config.name, "gpt")

    try:
        # Encode audio to base64
        with open(temp_audio_path, "rb") as audio_file:
            encoded_audio = base64.b64encode(audio_file.read()).decode("utf-8")

        # Prepare content
        messages_content = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": text_prompt},
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": encoded_audio,
                            "format": "wav",
                        },
                    },
                ],
            },
        ]

        # If we have labels for constrained generation, use schema
        if task_config.labels and task_config.use_logits_processor:
            # Dynamically create a pydantic model for structured output
            SchemaWrapper = create_schema_wrapper(task_config.field_name)

            # Add schema format guidance
            messages_content[0]["content"][0]["text"] += "\nFormat: " + json.dumps(SchemaWrapper.model_json_schema())

        # Set up retry logic
        max_retries = 5
        sleep_time = 0.1

        # Try to generate content with retries for API rate limits
        for attempt in range(max_retries):
            try:
                completion = resources.model.chat.completions.create(
                    model=resources.model_name,
                    modalities=["text"],
                    temperature=0,
                    messages=messages_content,
                )

                response = completion.choices[0].message.content

                # Try to parse structured response if using schema
                if task_config.labels and task_config.use_logits_processor:
                    try:
                        # First try to parse as JSON
                        response_json = json.loads(response)
                        response = response_json[task_config.field_name]
                    except:
                        # Fall back to checking for label terms in response
                        if task_config.labels:
                            # Check if any of the labels appear in the response
                            response_vec = [int(label.lower() in response.lower()) for label in task_config.labels]

                            if np.sum(response_vec) == 1:
                                # Find which label was detected
                                response = [
                                    label.lower() for label, pred in zip(task_config.labels, response_vec) if pred == 1
                                ][0]

                return response

            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"OpenAI API error: {e}. Retrying after {sleep_time}s...")
                    time.sleep(sleep_time)
                    sleep_time *= 2
                else:
                    print(f"Failed after {max_retries} attempts: {e}")
                    # Default to first label if available, otherwise empty string
                    return task_config.labels[0] if task_config.labels else ""
    finally:
        # Clean up temporary file
        cleanup_temp_audio(temp_audio_path)


def process_sample(resources: ModelResources, audio: Dict[str, Any], text_prompt: str, task_config: TaskConfig) -> str:
    """
    Process a single audio sample based on model type

    Args:
        resources: Model resources
        audio: Processed audio data
        text_prompt: Text prompt for the model
        task_config: Task configuration

    Returns:
        Processed model output
    """
    # Choose processing function based on model type
    if resources.model_type == "transformers":
        if "Qwen2" in resources.model_name:
            response = process_with_qwen(resources, audio, text_prompt, task_config)
        elif "diva" in resources.model_name.lower():
            response = process_with_diva(resources, audio, text_prompt, task_config)
        else:
            raise ValueError(f"Transformers model {resources.model_name} processing not implemented")
    elif resources.model_type == "gemini":
        response = process_with_gemini(resources, audio, text_prompt, task_config)
    elif resources.model_type == "openai":
        response = process_with_openai(resources, audio, text_prompt, task_config)
    else:
        raise ValueError(f"Model type {resources.model_type} processing not implemented")

    # Apply task-specific output processing
    return task_config.output_processor(response)


def process_record(
    resources: ModelResources, record: Dict[str, Any], task_config: TaskConfig
) -> Tuple[Dict[str, Any], int, int]:
    """
    Process a single record from the dataset

    Args:
        resources: Model resources
        record: Record data
        task_config: Task configuration

    Returns:
        Tuple of (processed record, correct count, total count)
    """
    expected_value = record.get(task_config.field_name)
    audio_file = record.get("filename")

    if not audio_file:
        return record, 0, 0

    # Process audio
    audio = process_audio(audio_file, task_config.audio_dir)

    # Get model prediction
    predicted_value = process_sample(resources, audio, task_config.prompt_template, task_config)

    # Add prediction to record
    record["prediction"] = predicted_value

    # Check if prediction is correct
    correct = 0
    if expected_value and predicted_value:
        if predicted_value.lower() == expected_value.lower():
            correct = 1

    return record, correct, 1


def run_evaluation(resources: ModelResources, task_config: TaskConfig) -> Tuple[float, List[Dict[str, Any]]]:
    """
    Run evaluation on a dataset

    Args:
        resources: Model resources
        task_config: Task configuration

    Returns:
        Tuple of (accuracy score, processed records)
    """
    correct = 0
    total = 0
    records_with_preds = []

    data_path = Path("data") / task_config.audio_dir / task_config.data_file

    # Ensure data_path exists
    if not data_path.exists():
        # Try relative to current directory as fallback
        data_path = Path(task_config.audio_dir) / task_config.data_file

    # Log path being used for debugging
    print(f"Loading data from: {data_path}")

    with open(data_path, "r") as f:
        pbar = tqdm(f)
        for line in pbar:
            json_data = json.loads(line)
            processed_samples = []

            processed_record, sample_correct, sample_total = process_record(resources, json_data, task_config)
            processed_samples.append(processed_record)
            correct += sample_correct
            total += sample_total

            # Update progress bar
            if total > 0:
                pbar.set_description(f"{task_config.name}: {100*(correct/total):.2f}% (N={total})")

            records_with_preds.append(json_data)

    # Maintain relative path for output file but use the same directory as input file
    output_path = f"{data_path}_{resources.model_name.split('/')[-1]}_{task_config.name}"
    with open(output_path, "w") as f:
        for entry in records_with_preds:
            json.dump(entry, ensure_ascii=False, fp=f)
            f.write("\n")

    # Calculate and return accuracy
    accuracy = correct / total if total > 0 else 0
    print(f"Model: {resources.model_name}, Task: {task_config.name}, Accuracy: {accuracy:.2%}")
    return accuracy, records_with_preds


def reset_api_counters():
    """Reset all API call counters to zero"""
    for key in API_CALL_COUNTERS:
        API_CALL_COUNTERS[key] = 0


def main():
    """Entry point for the evaluation pipeline"""
    # Reset API counters at the start of a run
    reset_api_counters()

    # Get available tasks
    tasks = create_task_configs()

    # Define task to run
    task_name = "transcription"  # Change this to run different tasks
    task_config = tasks[task_name]

    # Model names to evaluate - now including API-based models
    model_names = [
        # "Qwen/Qwen2-Audio-7B-Instruct",
        # "WillHeld/DiVA-llama-3-v0-8b",
        "models/gemini-2.0-flash-exp",
        "gpt-4o-audio-preview",
    ]

    # Run evaluations for each model
    for model_name in model_names:
        print(f"Evaluating model: {model_name}")

        # Load model resources
        resources = load_model(model_name)

        # Run evaluation
        run_evaluation(
            resources=resources,
            task_config=task_config,
        )

        # Print final API usage for this model if applicable
        if resources.model_type in API_CALL_COUNTERS:
            print(f"Total {resources.model_type} API calls: {API_CALL_COUNTERS[resources.model_type]}")

    # Print final API usage summary
    print("\nAPI Usage Summary:")
    for api_type, count in API_CALL_COUNTERS.items():
        print(f"  {api_type.capitalize()}: {count}/{MAX_API_CALLS} calls")


if __name__ == "__main__":
    main()
