#!/usr/bin/env python3
"""
judge_match.py

This script loads a Hugging Face audio dataset, compares two audio columns using the GPT-4o-audio model,
and optionally pushes the results back to the hub using custom multithreaded processing.

Example Usage:
  # Process entire dataset and push updated results back using 32 threads:
  python judge_match.py --repo_id MichaelR207/wiktionary_pronunciations --model_id gpt-4o-audio-preview --push_to_hub --num_proc 32

  # Process 5 samples (simple mode, no push) using 4 threads:
  python judge_match.py --repo_id MichaelR207/wiktionary_pronunciations --model_id gpt-4o-audio-preview --limit 5 --num_proc 4

  # Test mode: process only the first 500 examples and push to a test repo:
  python judge_match.py --repo_id MichaelR207/wiktionary_pronunciations --model_id gpt-4o-audio-preview --test_mode --push_to_hub --num_proc 32

Ensure you have a .env file with HF_TOKEN and OPENAI_API_KEY defined.

---
Azure Inference Server Setup:
To use an Azure-hosted inference server, set the following environment variables (or update your .env file):
    OPENAI_API_BASE="https://YOUR-AZURE-OPENAI-ENDPOINT/"
    OPENAI_API_KEY="YOUR-AZURE-API-KEY"
Also, ensure the deployment name (model_id) is configured correctly for your Azure endpoint.
"""

import argparse
import base64
import os
import json
import tempfile
import wave
import time
import logging
import traceback
import hashlib
import atexit
import signal
import concurrent.futures
import sys

import soundfile as sf
from tqdm import tqdm

from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env

import numpy as np
import openai  # Requires version >=1.0.0
from joblib import Memory
from datasets import load_dataset, Dataset
from huggingface_hub import login
from datasets import Audio

# Suppress extra logs from openai, urllib3, and requests.
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("requests").setLevel(logging.WARNING)

# Global list to track temporary files.
TEMP_FILES = []

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
def setup_logging(debug=False):
    log_level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler("judge_match.log"),
            logging.StreamHandler()
        ]
    )

# ---------------------------------------------------------------------------
# Cleanup temporary files (runs at exit or on SIGINT)
# ---------------------------------------------------------------------------
def cleanup_temp_files():
    logging.info("Cleaning up temporary files...")
    for path in TEMP_FILES:
        if os.path.exists(path):
            try:
                os.remove(path)
                logging.debug(f"Removed: {path}")
            except Exception as e:
                logging.debug(f"Failed to remove {path}: {e}")

# Register cleanup on exit and SIGINT
atexit.register(cleanup_temp_files)
def sigint_handler(signal, frame):
    cleanup_temp_files()
    sys.exit(0)
signal.signal(signal.SIGINT, sigint_handler)

# ---------------------------------------------------------------------------
# Helper: get_audio_file_path creates a temporary .wav file from audio_info if needed.
# ---------------------------------------------------------------------------
def get_audio_file_path(audio_info):
    path = audio_info.get("path")
    if path and os.path.exists(path):
        return path, False  # Existing file provided.
    if "array" in audio_info and audio_info["array"] is not None and "sampling_rate" in audio_info:
        try:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            wav_path = tmp.name
            tmp.close()
            with wave.open(wav_path, 'wb') as wf:
                wf.setnchannels(1)      # mono
                wf.setsampwidth(2)      # int16
                wf.setframerate(audio_info["sampling_rate"])
                audio_array = audio_info["array"]
                if not isinstance(audio_array, np.ndarray):
                    audio_array = np.array(audio_array)
                int16_array = (np.clip(audio_array, -1, 1) * 32767).astype(np.int16)
                wf.writeframes(int16_array.tobytes())
            TEMP_FILES.append(wav_path)
            return wav_path, True
        except Exception as e:
            logging.error(f"Error creating temporary audio file: {e}")
            raise
    raise ValueError("Missing valid audio data in audio_info.")

# ---------------------------------------------------------------------------
# Argument Parsing
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare two audio columns in a Hugging Face dataset using GPT-4o-audio."
    )
    parser.add_argument("--repo_id", type=str, default="MichaelR207/wiktionary_pronunciations",
                        help="Hugging Face dataset repo (default: %(default)s)")
    parser.add_argument("--model_id", type=str, default="gpt-4o-audio-preview",
                        help="OpenAI model to call (default: %(default)s)")
    parser.add_argument("--push_to_hub", action="store_true",
                        help="Push updated dataset back to Hugging Face")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only this many examples (no push)")
    parser.add_argument("--num_proc", type=int, default=1,
                        help="Number of threads for parallel processing (default: %(default)s)")
    parser.add_argument("--cache_dir", type=str, default="judge_match_cache",
                        help="Directory for joblib Memory (default: %(default)s)")
    parser.add_argument("--hf_token", type=str, default=os.getenv("HF_TOKEN"),
                        help="Hugging Face token (default from .env: HF_TOKEN)")
    parser.add_argument("--openai_api_key", type=str, default=os.getenv("OPENAI_API_KEY"),
                        help="OpenAI API key (default from .env: HF_TOKEN)")
    parser.add_argument("--openai_api_base", type=str, default=os.getenv("OPENAI_API_BASE"),
                        help="Azure OpenAI endpoint (default from .env: OPENAI_API_BASE)")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug logging")
    parser.add_argument("--force_refresh", action="store_true",
                        help="Force API call and ignore cached results")
    parser.add_argument("--test_mode", action="store_true",
                        help="Test mode: process only the first 500 examples.")
    parser.add_argument("--test_repo_id", type=str, default=None,
                        help="Hugging Face repo id for test mode push (if not provided, will use primary repo id with a '-test' suffix).")
    return parser.parse_args()

# ---------------------------------------------------------------------------
# Create system prompt.
# ---------------------------------------------------------------------------
def create_system_prompt():
    return (
        "You are an expert linguist tasked with comparing two audio recordings solely for their pronunciation. "
        "Focus on the precise sequence of phonemes, the number of syllables, and the stress/emphasis patterns. "
        "Differences due only to regional accent (e.g., British vs. American) should be ignored. "
        "For example, if two speakers say 'tomato' as 'toh-MAH-toh' (even if their accents differ), they match; "
        "if one says 'toh-MAY-toh', then they do not match.\n\n"
        "IMPORTANT: Respond in text only (do not include any audio output) and output valid JSON with exactly two keys: "
        "'reasoning' (a detailed chain-of-thought explanation) and 'match' (a boolean verdict)."
    )

# ---------------------------------------------------------------------------
# The core API call without caching.
# ---------------------------------------------------------------------------
def compare_pronunciations_uncached(custom_key, audio1_path, audio2_path, model_id, debug=False):
    try:
        with open(audio1_path, "rb") as f:
            audio1_data = f.read()
        with open(audio2_path, "rb") as f:
            audio2_data = f.read()
    except Exception as e:
        return {"reasoning": f"Error reading audio files: {str(e)}", "match": False}

    audio1_encoded = base64.b64encode(audio1_data).decode("utf-8")
    audio2_encoded = base64.b64encode(audio2_data).decode("utf-8")

    user_content = [
        {"type": "text", "text": "Here is the first audio clip:"},
        {"type": "input_audio", "input_audio": {"data": audio1_encoded, "format": "wav"}},
        {"type": "text", "text": "Here is the second audio clip:"},
        {"type": "input_audio", "input_audio": {"data": audio2_encoded, "format": "wav"}},
        {"type": "text", "text": (
            "Please analyze these recordings strictly for pronunciation details (phonemes, syllables, stress, emphasis). "
            "Ignore differences solely due to accent. Respond ONLY in text and output valid JSON with keys 'reasoning' and 'match' (boolean)."
        )}
    ]

    retries = 2
    result = None
    for attempt in range(retries + 1):
        try:
            client = openai.Client(api_key=os.getenv("OPENAI_API_KEY"))
            start_time = time.time()
            completion = client.chat.completions.create(
                model=model_id,
                modalities=["text"],
                messages=[
                    {"role": "system", "content": create_system_prompt()},
                    {"role": "user", "content": user_content}
                ],
                temperature=0.0
            )
            elapsed = time.time() - start_time
            if debug:
                logging.debug(f"API call took {elapsed:.2f} seconds (attempt {attempt+1})")
            msg = completion.choices[0].message
            content = msg.content
            if content is None and hasattr(msg, "audio") and msg.audio is not None:
                content = msg.audio.transcript
            if content is None:
                if debug:
                    logging.debug("No text output received, retrying...")
                continue
            if debug:
                logging.debug("Received content: " + content)
            try:
                parsed = json.loads(content)
                result = {
                    "reasoning": parsed.get("reasoning", "No reasoning provided."),
                    "match": bool(parsed.get("match", False))
                }
                break
            except Exception as parse_err:
                if debug:
                    logging.debug(f"Error parsing JSON output on attempt {attempt+1}: {parse_err}")
                result = {"reasoning": f"Error parsing JSON output: {str(parse_err)}", "match": False}
        except Exception as e:
            if debug:
                logging.debug(f"Error calling API on attempt {attempt+1}: {traceback.format_exc()}")
            result = {"reasoning": f"Error calling GPT-4o-audio: {str(e)}", "match": False}
        time.sleep(2)
    if result is None:
        result = {"reasoning": "No valid text output received from the model.", "match": False}
    return result

# ---------------------------------------------------------------------------
# Caching wrapper for thread-safe caching using joblib.Memory.
# Computes an MD5 hash of (custom_key, model_id, debug) for a stable key.
# ---------------------------------------------------------------------------
def get_cached_compare_pronunciations(custom_key, audio1_path, audio2_path, model_id, debug=False, force_refresh=False, memory=None):
    if custom_key is None:
        key_hash = None
    else:
        key_hash = hashlib.md5(f"{custom_key}||{model_id}||{debug}".encode()).hexdigest()
    if force_refresh:
        return compare_pronunciations_uncached(key_hash, audio1_path, audio2_path, model_id, debug)
    cached_func = memory.cache(compare_pronunciations_uncached, ignore=['audio1_path', 'audio2_path'])
    return cached_func(key_hash, audio1_path, audio2_path, model_id, debug)

# ---------------------------------------------------------------------------
# process_example: process one example.
# Store the original index for later restoration.
# ---------------------------------------------------------------------------
def process_example(example, idx, memory, args):
    audio1_path = None
    audio2_path = None
    temp1 = False
    temp2 = False
    try:
        # If already processed and in expected format, return it.
        if ("gpt4o_reasoning" in example and 
            not example["gpt4o_reasoning"].startswith("Error") and 
            isinstance(example.get("GPT4o_pronunciation"), dict) and 
            "array" in example["GPT4o_pronunciation"] and 
            "sampling_rate" in example["GPT4o_pronunciation"]):
            return example

        try:
            word = example.get("word", "")
            region = example.get("region", "")
            oed = example.get("OED", "")
            custom_key = f"{word}||{region}||{oed}"
            logging.debug(f"Custom cache key for index {idx}: {custom_key}")
        except Exception:
            logging.error(f"Error creating custom cache key for index {idx}: {traceback.format_exc()}")
            custom_key = None

        audio1_info = example["audio"]
        audio2_info = example["GPT4o_pronunciation"]

        audio1_path, temp1 = get_audio_file_path(audio1_info)
        audio2_path, temp2 = get_audio_file_path(audio2_info)
        logging.debug(f"Audio paths for index {idx}: audio1: {audio1_path}, audio2: {audio2_path}")

        verdict = get_cached_compare_pronunciations(custom_key, audio1_path, audio2_path,
                                                    args.model_id, args.debug, args.force_refresh, memory)
        logging.debug(f"Verdict for index {idx}: {verdict}")

        example["gpt4o_reasoning"] = verdict["reasoning"]
        example["gpt4o_correct"] = verdict["match"]
        example["orig_index"] = idx

        try:
            data, rate = sf.read(audio2_path)
            logging.debug(f"Before reshape at index {idx}: type: {type(data)}; shape: {np.array(data).shape}; rate: {rate}")
            if np.array(data).ndim == 1:
                logging.debug(f"Reshaping audio data from {np.array(data).shape} to 2D for index {idx}")
                data = data.reshape(-1, 1)
            flattened = data.flatten()
            logging.debug(f"After flatten at index {idx}: shape: {flattened.shape}")
            # Instead of storing the full array, you could choose to store just the file path.
            # Here we keep the processed version (as array) for internal use.
            example["GPT4o_pronunciation"] = {"array": flattened.tolist(), "sampling_rate": rate}
        except Exception as e:
            logging.error(f"Error reading audio from {audio2_path} at index {idx}: {e}")
            return None

        # Remove the original "audio" column from the processed example.
        if "audio" in example:
            del example["audio"]

        return example
    except Exception as e:
        logging.error(f"Error processing index {idx}: {traceback.format_exc()}")
        return None
    finally:
        # Clean up temporary files for this example.
        for path, is_temp in [(audio1_path, temp1), (audio2_path, temp2)]:
            if path is not None and is_temp and os.path.exists(path):
                try:
                    os.remove(path)
                    # Also remove from TEMP_FILES if present.
                    if path in TEMP_FILES:
                        TEMP_FILES.remove(path)
                except Exception as cleanup_err:
                    logging.debug(f"Error cleaning up temp file {path}: {cleanup_err}")

# ---------------------------------------------------------------------------
# Custom parallel processing using ThreadPoolExecutor with tqdm.
# Maintains original ordering by storing results in a list at their original index.
# ---------------------------------------------------------------------------
def process_dataset_in_threads(dataset, memory, args):
    total = len(dataset)
    results = [None] * total

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_proc) as executor:
        future_to_idx = {executor.submit(process_example, ex, idx, memory, args): idx for idx, ex in enumerate(dataset)}
        for future in tqdm(concurrent.futures.as_completed(future_to_idx), total=total, desc="Processing"):
            idx = future_to_idx[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                logging.error(f"Exception in thread for index {idx}: {e}")
                results[idx] = None

    processed_examples = [ex for ex in results if ex is not None]
    logging.info(f"Processed {len(processed_examples)} examples; dropped {total - len(processed_examples)} out of {total}.")
    return processed_examples

# ---------------------------------------------------------------------------
# After processing, restore the original "audio" and "GPT4o_pronunciation" columns.
# ---------------------------------------------------------------------------
def restore_original_columns(processed_dataset, original_dataset):
    # Build a mapping from original index to original columns.
    orig_mapping = {i: {"audio": ex["audio"], "GPT4o_pronunciation": ex["GPT4o_pronunciation"]} 
                    for i, ex in enumerate(original_dataset)}
    def restore(example, idx):
        orig_idx = example.get("orig_index")
        if orig_idx is not None and orig_idx in orig_mapping:
            example["audio"] = orig_mapping[orig_idx]["audio"]
            example["GPT4o_pronunciation"] = orig_mapping[orig_idx]["GPT4o_pronunciation"]
        return example
    return processed_dataset.map(restore, with_indices=True)

# ---------------------------------------------------------------------------
# Main processing.
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    setup_logging(debug=args.debug)
    logging.info("Starting processing.")

    if args.openai_api_key:
        os.environ["OPENAI_API_KEY"] = args.openai_api_key
        openai.api_key = args.openai_api_key
    if args.openai_api_base:
        os.environ["OPENAI_API_BASE"] = args.openai_api_base
    if args.hf_token:
        login(token=args.hf_token)

    try:
        orig_dataset = load_dataset(args.repo_id, split="train")
    except Exception as e:
        logging.error(f"Error loading dataset: {e}")
        return

    if args.test_mode:
        orig_dataset = orig_dataset.select(range(min(len(orig_dataset), 500)))
    elif args.limit is not None:
        orig_dataset = orig_dataset.select(range(args.limit))

    memory = Memory(location=args.cache_dir, verbose=0)
    logging.info("Starting custom threaded processing...")
    processed_examples = process_dataset_in_threads(orig_dataset, memory, args)

    try:
        new_dataset = Dataset.from_list(processed_examples)
    except Exception as e:
        logging.error(f"Error creating dataset from list: {e}")
        raise

    # Restore the original "audio" and "GPT4o_pronunciation" columns.
    new_dataset = restore_original_columns(new_dataset, orig_dataset)

    try:
        # Cast columns to Audio features.
        # Note: We do NOT use Audio(decode=False) here because we want the original data to be used.
        # However, if you store file paths rather than arrays, using decode=False would instruct the HF UI
        # to treat the value as a file path. If your original dataset has actual audio arrays, you can omit decode=False.
        new_dataset = new_dataset.cast_column("audio", Audio())
        new_dataset = new_dataset.cast_column("GPT4o_pronunciation", Audio())
    except Exception as e:
        logging.error(f"Error casting columns: {e}")
        raise

    for i in range(min(10, new_dataset.num_rows)):
        row = new_dataset[i]
        print(row.get("word", f"Example {i}"), row["gpt4o_correct"], row["gpt4o_reasoning"], sep="\t")

    logging.info("Processing complete.")
    if args.push_to_hub:
        repo = (args.test_repo_id if args.test_mode and args.test_repo_id 
                else (args.repo_id + "-test" if args.test_mode else args.repo_id + '-backup'))
        new_dataset.push_to_hub(repo)
        logging.info(f"Dataset pushed to {repo} with updated columns: 'gpt4o_reasoning' and 'gpt4o_correct'.")

if __name__ == "__main__":
    main()