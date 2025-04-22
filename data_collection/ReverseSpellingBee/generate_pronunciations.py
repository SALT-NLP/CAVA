#!/usr/bin/env python3

import argparse
import os
import base64
import concurrent.futures
from dotenv import load_dotenv
from tqdm import tqdm

# Load environment variables from .env file.
load_dotenv()

import openai
# Set OpenAI API key from the .env file.
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY not found in your .env file.")
openai.api_key = openai_api_key

# Import OpenAI client class.
from openai import OpenAI

# Import Hugging Face dataset and hub utilities.
from datasets import load_dataset, Audio, Dataset, DatasetDict
from huggingface_hub import login

PROMPT = """You are a text-to-speech model. Your task is to generate a high-quality audio pronunciation of the given word in English. The audio should be clear and easy to understand. Please ensure that the pronunciation is accurate and natural-sounding.

The word to read aloud is: "{word}"

Please respond with only the word read aloud clearly, do not say anything else or include any additional text or formatting."""

# -------------------------------------------------------------------
# Function to generate pronunciation audio using GPT4o voice API.
# -------------------------------------------------------------------
def generate_pronunciation(word, model_id, openai_client, voice="alloy"):
    """
    Calls the GPT4o voice API to produce an audio pronunciation for `word`.
    Returns the raw WAV bytes.
    """
    completion = openai_client.chat.completions.create(
        model=model_id,
        modalities=["text", "audio"],
        audio={"voice": voice, "format": "wav"},
        messages=[{"role": "user", "content": PROMPT.format(word=word)}],
    )
    # Decode the base64 audio data.
    audio_data = completion.choices[0].message.audio.data
    wav_bytes = base64.b64decode(audio_data)
    return wav_bytes

# -------------------------------------------------------------------
# Helper function to process a single word.
# -------------------------------------------------------------------
def process_word(word, args, openai_client):
    """
    For a given word, check if the corresponding WAV file exists locally.
    If not, generate it using GPT4o and save it.
    Returns a tuple (wav_path, wav_bytes).
    """
    wav_path = os.path.join(args.output_directory, f"{word}.wav")
    if os.path.exists(wav_path):
        with open(wav_path, "rb") as f:
            wav_bytes = f.read()
    else:
        print(f"Generating pronunciation for '{word}'...")
        wav_bytes = generate_pronunciation(
            word=word,
            model_id=args.model_id,
            openai_client=openai_client,
            voice=args.voice
        )
        os.makedirs(os.path.dirname(wav_path), exist_ok=True)
        with open(wav_path, "wb") as f:
            f.write(wav_bytes)
    return wav_path, wav_bytes

# -------------------------------------------------------------------
# Function for words list mode.
# -------------------------------------------------------------------
def process_single_word(word, args, openai_client):
    """
    Processes a single word and returns a dict containing the word and its audio data.
    """
    wav_path, wav_bytes = process_word(word, args, openai_client)
    return {
        "word": word,
        "GPT4o_pronunciation": {"path": wav_path, "bytes": wav_bytes}
    }

# -------------------------------------------------------------------
# Function for processing an example in a Hugging Face dataset.
# -------------------------------------------------------------------
def process_example(example, args, openai_client):
    """
    Processes a dataset example (expects a "word" key) and adds the GPT4o_pronunciation field.
    """
    word = example["word"]
    wav_path, wav_bytes = process_word(word, args, openai_client)
    example["GPT4o_pronunciation"] = {"path": wav_path, "bytes": wav_bytes}
    return example

# -------------------------------------------------------------------
# Main script logic with argparse.
# -------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Generate GPT4o audio pronunciations for words from a list or a Hugging Face dataset."
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="gpt-4o-audio-preview",
        help="Model ID for GPT4o voice API (default: gpt-4o-audio-preview)",
    )
    parser.add_argument(
        "--output_directory",
        type=str,
        default="generated_pronunciations",
        help="Local directory to save generated WAV files (default: generated_pronunciations).",
    )
    parser.add_argument(
        "--input_words",
        type=str,
        default=None,
        help="Comma-separated list of words for pronunciation (used if --huggingface_repo is not provided).",
    )
    parser.add_argument(
        "--huggingface_repo",
        type=str,
        default=None,
        help="Hugging Face dataset repo ID (e.g. username/my_dataset). Overrides --input_words if provided.",
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="If set and using a Hugging Face dataset, push the updated dataset back to the Hub.",
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        default=None,
        help="(Optional) Your Hugging Face token, if needed for private repos or pushing changes.",
    )
    parser.add_argument(
        "--voice",
        type=str,
        default="alloy",
        help="Voice setting for GPT4o audio generation (default: alloy).",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="Number of concurrent workers for processing requests (default: 8).",
    )
    args = parser.parse_args()

    # Optionally log in to Hugging Face Hub.
    if args.hf_token:
        login(token=args.hf_token)
    else:
        hf_token = os.getenv("HF_TOKEN")
        if hf_token:
            login(token=hf_token)

    # Ensure the local output directory exists.
    os.makedirs(args.output_directory, exist_ok=True)

    # Prepare the GPT4o client.
    openai_client = OpenAI()

    # -------------------------------------------------------------------
    # Hugging Face dataset mode.
    # -------------------------------------------------------------------
    if args.huggingface_repo:
        print(f"Loading dataset '{args.huggingface_repo}'...")
        dataset_dict = load_dataset(args.huggingface_repo)
        if not isinstance(dataset_dict, DatasetDict):
            dataset_dict = DatasetDict({"train": dataset_dict})
        if "word" not in dataset_dict["train"].column_names:
            raise ValueError("The loaded dataset does not contain a 'word' column.")

        updated_splits = {}
        for split_name, split_dataset in dataset_dict.items():
            print(f"Processing split '{split_name}' with {len(split_dataset)} examples...")
            examples = list(split_dataset)
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_workers) as executor:
                processed_examples = list(tqdm(
                    executor.map(lambda ex: process_example(ex, args, openai_client), examples),
                    total=len(examples),
                    desc=f"Processing {split_name}"
                ))
            new_dataset = Dataset.from_list(processed_examples)
            new_dataset = new_dataset.cast_column("GPT4o_pronunciation", Audio())
            new_dataset = new_dataset.cast_column("audio", Audio())
            updated_splits[split_name] = new_dataset

        updated_dataset = DatasetDict(updated_splits)

        if args.push_to_hub:
            print(f"Pushing updated dataset back to '{args.huggingface_repo}' on Hugging Face...")
            updated_dataset.push_to_hub(args.huggingface_repo)
            print("Push complete!")
        print("Dataset processing complete!")

    # -------------------------------------------------------------------
    # Words list mode.
    # -------------------------------------------------------------------
    elif args.input_words:
        words = [word.strip() for word in args.input_words.split(",") if word.strip()]
        print(f"Processing {len(words)} words from the input_words list...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            results = list(tqdm(
                executor.map(lambda word: process_single_word(word, args, openai_client), words),
                total=len(words),
                desc="Processing words"
            ))
        dataset = Dataset.from_list(results)
        dataset = dataset.cast_column("GPT4o_pronunciation", Audio())
        print("Words list processing complete!")
        # Optionally, you can save this dataset locally.
        # For example: dataset.save_to_disk("local_audio_dataset")
    else:
        raise ValueError("Please provide either --huggingface_repo or --input_words.")

    print("All done!")

if __name__ == "__main__":
    main()