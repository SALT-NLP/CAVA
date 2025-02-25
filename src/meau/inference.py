import json
import torch
import soundfile as sf
import librosa
import os
import time
import base64
import numpy as np
from pathlib import Path
from enum import Enum
from typing import Dict, List, Callable, Tuple, Any, Optional, NamedTuple, Union
from pydantic import BaseModel
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModel,
    AutoProcessor,
    Qwen2AudioForConditionalGeneration,
    PrefixConstrainedLogitsProcessor,
    GenerationConfig,
)

from meau.config import TaskConfig, create_task_configs


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
    full_path = os.path.join(audio_dir, audio_file) if audio_dir else audio_file
    audio, sr = librosa.load(full_path, sr=16000)
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
    temp_audio_path = f"tmp_{task_name}_{model_type}.wav"
    sf.write(temp_audio_path, audio["array"], audio["sampling_rate"], format="wav")
    return temp_audio_path


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
    response = response.split("assistant")[-1]

    return response


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
    # Save audio to temporary file
    temp_audio_path = save_temp_audio(audio, task_config.name, "diva")

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

    # Save audio to temporary file
    temp_audio_path = save_temp_audio(audio, task_config.name, "gemini")

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
    # Save audio to temporary file
    temp_audio_path = save_temp_audio(audio, task_config.name, "gpt")

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
        expected_key: Key containing expected output
        audio_dir: Directory containing audio files

    Returns:
        Tuple of (processed record, correct count, total count)
    """
    expected_value = record.get(task.field_name)
    audio_file = record.get("filename")

    if not audio_file:
        return record, 0, 0

    # Process audio
    audio = process_audio(audio_file, task.audio_dir)

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
        data_file: JSONL file with dataset
        task_config: Task configuration

    Returns:
        Tuple of (accuracy score, processed records)
    """
    correct = 0
    total = 0
    records_with_preds = []

    with open(data_file, "r") as f:
        pbar = tqdm(f)
        for line in pbar:
            json_data = json.loads(line)
            processed_samples = []

            # Process each audio sample in the record
            for record in json_data.get("generated_audio", [json_data]):
                processed_record, sample_correct, sample_total = process_record(resources, record, task_config)
                processed_samples.append(processed_record)
                correct += sample_correct
                total += sample_total

                # Update progress bar
                if total > 0:
                    pbar.set_description(f"{task_config.name}: {100*(correct/total):.2f}% (N={total})")

            # Store predictions
            json_data["processed_samples"] = processed_samples
            records_with_preds.append(json_data)

    # Write results to file
    output_file = f"{data_file}_{resources.model_name.split('/')[-1]}_{task_config.name}"
    with open(output_file, "w") as f:
        for entry in records_with_preds:
            json.dump(entry, ensure_ascii=False, fp=f)
            f.write("\n")

    # Calculate and return accuracy
    accuracy = correct / total if total > 0 else 0
    print(f"Model: {resources.model_name}, Task: {task_config.name}, Accuracy: {accuracy:.2%}")
    return accuracy, records_with_preds


def main():
    """Entry point for the evaluation pipeline"""
    # Directory containing audio files
    audio_dir = "generated_audio/"

    # Input data file
    data_file = os.path.join(audio_dir, "audio_inputs.jsonl")

    # Model names to evaluate - now including API-based models
    model_names = [
        "Qwen/Qwen2-Audio-7B-Instruct",
        "WillHeld/DiVA-llama-3-v0-8b",
        "models/gemini-2.0-flash-exp",
        "gpt-4o-audio-preview",
    ]

    # Get available tasks
    tasks = create_task_configs()

    # Define task to run
    task_name = "emotion"  # Change this to run different tasks
    task_config = tasks[task_name]

    # Run evaluations for each model
    for model_name in model_names:
        try:
            print(f"Evaluating model: {model_name}")

            # Load model resources
            resources = load_model(model_name)

            # Run evaluation
            run_evaluation(
                resources=resources,
                task_config=task_config,
            )
        except Exception as e:
            print(f"Error evaluating model {model_name}: {str(e)}")
            print("Continuing with next model...")
            continue


if __name__ == "__main__":
    main()
