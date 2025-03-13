#!/usr/bin/env python

import argparse
import os
import json
import random
from pathlib import Path
from datasets import load_from_disk
from tqdm import tqdm

def extract_utterances(dataset_path, output_file, split="all", max_samples=None, include_parse=False, filter_domain=None):
    """
    Extract utterances from the STOP dataset in HuggingFace format.

    Args:
        dataset_path: Path to the processed STOP dataset directory
        output_file: Path to save the extracted utterances
        split: Dataset split to use (train, validation, test, or "all" for all splits)
        max_samples: Maximum number of samples to extract (if None, extracts all)
        include_parse: Whether to include parse information in the output
        filter_domain: Filter by domain (e.g., "music", "alarm", etc.)
    """
    try:
        # Load the dataset
        dataset = load_from_disk(dataset_path)

        # Determine available splits
        available_splits = list(dataset.keys())

        # Determine which splits to process
        splits_to_process = []
        if split.lower() == "all":
            splits_to_process = available_splits
            print(f"Processing all available splits: {', '.join(splits_to_process)}")
        elif split in available_splits:
            splits_to_process = [split]
            print(f"Processing split: {split}")
        else:
            print(f"Split '{split}' not found. Available splits: {available_splits}")
            # Default to all available splits
            splits_to_process = available_splits
            print(f"Using all available splits instead")

        # Extract all utterances
        all_items = []
        total_collected = 0

        # Process each split
        for current_split in splits_to_process:
            print(f"Extracting utterances from '{current_split}' split")

            total_in_split = len(dataset[current_split])
            print(f"Found {total_in_split} examples in this split")

            # For each split, determine how many samples to take
            samples_from_this_split = None
            if max_samples:
                # If processing multiple splits and have a max_samples limit,
                # distribute samples proportionally across splits
                if len(splits_to_process) > 1:
                    # Calculate remaining samples needed
                    remaining_samples = max_samples - total_collected
                    if remaining_samples <= 0:
                        # Skip this split if we already have enough samples
                        print(f"Skipping {current_split} split (already collected {total_collected} samples)")
                        continue

                    # Calculate proportion based on split sizes
                    splits_remaining = len(splits_to_process) - splits_to_process.index(current_split)
                    samples_from_this_split = min(remaining_samples, total_in_split)

                    print(f"Taking up to {samples_from_this_split} samples from this split")
                else:
                    # If only processing one split, use the full max_samples
                    samples_from_this_split = min(max_samples, total_in_split)

            # Select examples from this split (all or random sample)
            if samples_from_this_split and samples_from_this_split < total_in_split:
                print(f"Selecting {samples_from_this_split} random samples from {total_in_split} examples")
                random.seed(42 + splits_to_process.index(current_split))  # Different seed for each split
                sample_indices = random.sample(range(total_in_split), samples_from_this_split)
                examples_iterator = [dataset[current_split][i] for i in sample_indices]
            else:
                print(f"Processing all {total_in_split} examples in this split")
                examples_iterator = dataset[current_split]

            # Process examples in this split
            split_items = []
            for item in tqdm(examples_iterator,
                             desc=f"Processing {current_split}",
                             unit="example"):
                # Get the domain from the file_id if available
                domain = None
                if "file_id" in item:
                    file_id = item["file_id"]
                    # Try to extract domain (e.g., from "test_0/music_test/00003470.wav")
                    parts = file_id.split("/")
                    if len(parts) > 1:
                        domain_part = parts[1].split("_")[0]  # Extract "music" from "music_test"
                        domain = domain_part

                # Apply domain filter if specified
                if filter_domain and domain and filter_domain.lower() not in domain.lower():
                    continue

                # Create output item
                output_item = {
                    "utterance": item["utterance"],
                    "split": current_split
                }

                # Add optional fields
                if include_parse:
                    output_item["parse"] = item["parse"]
                if domain:
                    output_item["domain"] = domain
                if "file_id" in item:
                    output_item["audio_path"] = item["file_id"]

                split_items.append(output_item)

            print(f"Collected {len(split_items)} items from {current_split} split")
            all_items.extend(split_items)
            total_collected += len(split_items)

            # If we've reached our sample limit, stop processing more splits
            if max_samples and total_collected >= max_samples:
                print(f"Reached sample limit of {max_samples}, stopping")
                break

        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_file)
        if not output_dir:
            output_dir = "."
        os.makedirs(output_dir, exist_ok=True)

        # Save the utterances
        if output_file.endswith(".json"):
            # Save as JSON
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(all_items, f, indent=2, ensure_ascii=False)
        else:
            # Save as plain text (one utterance per line)
            with open(output_file, 'w', encoding='utf-8') as f:
                for item in all_items:
                    f.write(f"{item['utterance']}\n")

        print(f"Extracted {len(all_items)} utterances to {output_file}")

        # If JSON, also create a plain text version for convenience
        if output_file.endswith(".json"):
            txt_file = output_file.replace(".json", ".txt")
            with open(txt_file, 'w', encoding='utf-8') as f:
                for item in all_items:
                    f.write(f"{item['utterance']}\n")
            print(f"Also saved plain text version to {txt_file}")

        return len(all_items)

    except Exception as e:
        print(f"Error extracting utterances: {e}")
        return 0

def main():
    parser = argparse.ArgumentParser(description="Extract utterances from the STOP dataset")
    parser.add_argument("--dataset", default="stop_dataset",
                        help="Path to the processed STOP dataset directory")
    parser.add_argument("--output", default="./stop_utterances.json",
                        help="Path to save the extracted utterances")
    parser.add_argument("--split", default="all",
                        help="Dataset split to use (train, validation, test, or 'all' for all splits)")
    parser.add_argument("--max-samples", type=int,
                        help="Maximum number of samples to extract")
    parser.add_argument("--include-parse", action="store_true", default=True,
                        help="Include parse information in the output")
    parser.add_argument("--filter-domain",
                        help="Filter by domain (e.g., 'music', 'alarm', etc.)")
    args = parser.parse_args()

    extract_utterances(
        args.dataset,
        args.output,
        args.split,
        args.max_samples,
        args.include_parse,
        args.filter_domain
    )

if __name__ == "__main__":
    main()