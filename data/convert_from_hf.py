#!/usr/bin/env python3
"""
Convert HuggingFace audio datasets to CATS-compatible format

This script converts audio datasets from HuggingFace into the JSONL and audio file
structure expected by the CATS library. It downloads the audio data, saves it to a
specified directory, and creates a JSONL file with the required metadata.

Example JSONL structure for Werewolf dataset:
{"filename": "0.wav", "werewolf": ["Justin", "Mike"], "PlayerNames": ["Justin", "Caitlynn", "Mitchell", "James", "Mike"], "endRoles": ["Werewolf", "Tanner", "Seer", "Robber", "Werewolf"], "votingOutcome": [3, 0, 3, 0, 0]}

Usage:
    python hf_to_cats.py --dataset WillHeld/werewolf --split train --audio-dir werewolf_data --output audio_inputs.jsonl

Author: Will Held
"""

import os
import json
import argparse
from pathlib import Path
from tqdm import tqdm
import soundfile as sf
import numpy as np
from datasets import load_dataset


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Convert HuggingFace datasets to CATS format")
    parser.add_argument(
        "--dataset", type=str, required=True, help="HuggingFace dataset name (e.g., 'WillHeld/werewolf')"
    )
    parser.add_argument("--config", type=str, default=None, help="HuggingFace dataset config")
    parser.add_argument("--split", type=str, default="train", help="Dataset split to use")
    parser.add_argument("--audio-dir", type=str, required=True, help="Directory to save audio files")
    parser.add_argument("--output", type=str, required=True, help="Output JSONL file name")
    parser.add_argument("--audio-column", type=str, default="audio", help="Column containing audio data")
    parser.add_argument("--preserve-columns", action="store_true", help="Preserve all columns from original dataset")
    parser.add_argument("--exclude-columns", type=str, nargs="*", default=[], help="Columns to exclude from output")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of examples to convert")
    parser.add_argument("--data-dir", type=str, default="data", help="Base data directory")
    parser.add_argument(
        "--audio-format", type=str, default="wav", choices=["wav", "flac", "ogg"], help="Audio file format"
    )
    parser.add_argument("--sample-rate", type=int, default=16000, help="Sample rate for saved audio")
    parser.add_argument("--filename-prefix", type=str, default="", help="Prefix for generated filenames")

    return parser.parse_args()


def ensure_dir(directory):
    """Create directory if it doesn't exist"""
    Path(directory).mkdir(parents=True, exist_ok=True)


def save_audio(audio_data, filepath, sample_rate, audio_format):
    """Save audio data to file"""
    sf.write(filepath, audio_data["array"], sample_rate, format=audio_format.upper())


def convert_dataset(args):
    """Convert HuggingFace dataset to CATS format"""
    print(f"Loading dataset: {args.dataset}")
    dataset = load_dataset(args.dataset, args.config, split=args.split, trust_remote_code=True)

    # Limit dataset size if specified
    if args.limit and args.limit > 0:
        dataset = dataset.select(range(min(args.limit, len(dataset))))

    print(f"Dataset size: {len(dataset)} examples")
    print(f"Dataset features: {list(dataset.features.keys())}")

    # Verify that required columns exist
    if args.audio_column not in dataset.features:
        raise ValueError(f"Audio column '{args.audio_column}' not found in dataset features")

    # Create output directory
    audio_path = Path(args.data_dir) / args.audio_dir
    ensure_dir(audio_path)

    # Process and save each example
    records = []
    for i, example in enumerate(tqdm(dataset, desc="Converting examples")):
        # Create a unique filename for the audio
        if args.filename_prefix:
            filename = f"{args.filename_prefix}{i}.{args.audio_format}"
        else:
            # Use index as filename if no prefix
            filename = f"{i}.{args.audio_format}"

        # Save audio data
        audio_filepath = audio_path / filename
        save_audio(example[args.audio_column], audio_filepath, args.sample_rate, args.audio_format)

        # Create the record
        record = {"filename": filename}

        # Add other fields from the dataset
        if args.preserve_columns:
            for key, value in example.items():
                if key not in args.exclude_columns and key != args.audio_column:
                    # Skip audio data and excluded columns
                    # Convert numpy arrays to lists for JSON serialization
                    if isinstance(value, np.ndarray):
                        record[key] = value.tolist()
                    elif not isinstance(value, dict) or key == args.audio_column:
                        # Skip complex objects but include the audio column
                        record[key] = value

        records.append(record)

    # Save records to JSONL file
    output_path = Path(args.data_dir) / args.audio_dir / args.output
    with open(output_path, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")

    print(f"Conversion complete. JSONL file saved to {output_path}")
    print(f"Audio files saved to {audio_path}")
    print(f"To use this dataset with CATS, configure these settings:")
    print(f"  - audio_dir: '{args.audio_dir}'")
    print(f"  - data_file: '{args.output}'")


def main():
    args = parse_args()
    convert_dataset(args)


if __name__ == "__main__":
    main()
