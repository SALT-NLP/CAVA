#!/usr/bin/env python
import argparse
import csv
import json
import os
import re
import shutil
import tempfile
import wave

import numpy as np
from datasets import load_dataset
from tqdm import tqdm

def sanitize(s):
    """Replace non-alphanumeric characters with underscore."""
    return re.sub(r'[^A-Za-z0-9]', '_', s.strip())

def get_audio_file_path(audio_info):
    """
    Given an audio_info dict from the HF dataset, returns a tuple (path, is_temp).
    - If the audio_info contains a "path" and that file exists, returns that path with is_temp=False.
    - Otherwise, if the audio_info contains an "array" and "sampling_rate", writes a temporary WAV file,
      then returns its path with is_temp=True.
    """
    if not isinstance(audio_info, dict):
        raise ValueError("Invalid audio_info format: expected a dict.")
    
    # Try to get the file path directly.
    path = audio_info.get("path")
    if path is not None and os.path.exists(path):
        return path, False

    # If no file path exists, try to create a temporary WAV file from the array.
    if "array" in audio_info and audio_info["array"] is not None and "sampling_rate" in audio_info:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        tmp_name = tmp.name
        tmp.close()
        try:
            with wave.open(tmp_name, 'wb') as wf:
                wf.setnchannels(1)      # mono
                wf.setsampwidth(2)      # 16-bit PCM
                wf.setframerate(audio_info["sampling_rate"])
                audio_array = audio_info["array"]
                if not isinstance(audio_array, np.ndarray):
                    audio_array = np.array(audio_array)
                int16_array = (np.clip(audio_array, -1, 1) * 32767).astype(np.int16)
                wf.writeframes(int16_array.tobytes())
        except Exception as e:
            os.remove(tmp_name)
            raise ValueError(f"Error writing temporary WAV file: {e}")
        return tmp_name, True

    raise ValueError("Missing valid audio data in audio_info.")

def main():
    parser = argparse.ArgumentParser(
        description="Download a HF dataset, filter by input labels, save audio files, and output a deduplicated JSONL file."
    )
    parser.add_argument(
        "--hf_repo",
        type=str,
        default="MichaelR207/wiktionary_pronunciations-final",
        help="Hugging Face repository to pull from."
    )
    parser.add_argument(
        "--input_labels",
        type=str,
        default="combined_labels.csv",
        help="Path to CSV file with input labels."
    )
    parser.add_argument(
        "--audio_dir",
        type=str,
        default="audio",
        help="Directory to save audio files."
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="audio_inputs.jsonl",
        help="Output JSONL file (deduplicated on word, region, and OED)."
    )
    args = parser.parse_args()

    # Ensure the audio directory exists.
    os.makedirs(args.audio_dir, exist_ok=True)

    # Load the input labels CSV and filter for rows with label "Different".
    matching_labels = set()
    with open(args.input_labels, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row["label"].strip() == "Different":
                # Build a tuple key of (word, region, OED) for matching.
                key = (row["word"].strip(), row["region"].strip(), row["OED"].strip())
                matching_labels.add(key)
    print(f"Found {len(matching_labels)} unique keys in CSV marked as 'Different'.")

    # Download the dataset from Hugging Face (train split)
    ds = load_dataset(args.hf_repo, split="train")

    # Set to track keys already processed to deduplicate.
    processed_keys = set()
    match_count = 0

    # Open the output JSONL file.
    with open(args.output_file, "w", encoding="utf-8") as fout:
        # Use enumerate to get an index (for unique filename generation).
        for idx, row in enumerate(tqdm(ds, desc="Processing dataset")):
            word = row.get("word", "").strip()
            region = row.get("region", "").strip()
            oed = row.get("OED", "").strip()
            key = (word, region, oed)
            # Only consider rows with keys marked as "Different" in the CSV.
            if key not in matching_labels:
                continue

            # Deduplicate: if this key has been processed before, skip it.
            if key in processed_keys:
                continue
            processed_keys.add(key)
            match_count += 1

            # Get a valid audio file path.
            try:
                audio_info = row.get("audio")
                audio_path, is_temp = get_audio_file_path(audio_info)
            except Exception as e:
                print(f"Warning: Could not get audio file for word '{word}' ({region}): {e}")
                continue

            # Create a custom, interpretable filename using word, region, and the row index.
            custom_filename = f"{sanitize(word)}_{sanitize(region)}_{idx}.wav"
            dest_path = os.path.join(args.audio_dir, custom_filename)
            try:
                shutil.copy2(audio_path, dest_path)
            except Exception as e:
                print(f"Warning: Could not copy file {audio_path} to {dest_path}: {e}")
                if is_temp and os.path.exists(audio_path):
                    os.remove(audio_path)
                continue

            # Clean up the temporary file if one was created.
            if is_temp and os.path.exists(audio_path):
                os.remove(audio_path)

            # Write a JSONL entry that includes the filename.
            out_dict = {
                "filename": dest_path,
                "OED": oed,
                "word": word,
                "region": region
            }
            fout.write(json.dumps(out_dict) + "\n")
    
    print(f"Total unique matching dataset rows processed: {match_count}")

if __name__ == "__main__":
    main()