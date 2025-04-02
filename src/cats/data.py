import os
import json
import tarfile
import urllib.request
from pathlib import Path
from typing import Dict, Any, List, Optional
from tqdm import tqdm
import heapq
import shutil


def download_file(url: str, destination: str):
    """
    Download a file from a URL with progress bar
    """

    class DownloadProgressBar(tqdm):
        def update_to(self, b=1, bsize=1, tsize=None):
            if tsize is not None:
                self.total = tsize
            self.update(b * bsize - self.n)

    os.makedirs(os.path.dirname(destination), exist_ok=True)

    with DownloadProgressBar(unit="B", unit_scale=True, miniters=1, desc=url.split("/")[-1]) as t:
        urllib.request.urlretrieve(url, filename=destination, reporthook=t.update_to)


def prep_data_function_calling():
    """
    Prepare function calling dataset by:
    1. Downloading STOP dataset if not already downloaded
    2. Extracting the dataset
    3. Converting the data to JSONL format with metadata
    """
    # Define paths
    data_dir = Path(__file__).parents[2] / "data"
    fc_data_dir = data_dir / "function_calling_test"
    dataset_url = "https://dl.fbaipublicfiles.com/stop/stop.tar.gz"
    tar_path = fc_data_dir / "stop.tar.gz"
    extract_dir = fc_data_dir / "stop_data"
    output_file = fc_data_dir / "audio_inputs.jsonl"

    # Create directories if they don't exist
    os.makedirs(fc_data_dir, exist_ok=True)

    # Download the dataset if it doesn't exist
    if not tar_path.exists():
        print(f"Downloading STOP dataset from {dataset_url}...")
        download_file(dataset_url, str(tar_path))

    # Extract the dataset if the extraction directory doesn't exist
    if not extract_dir.exists():
        print(f"Extracting dataset to {extract_dir}...")
        os.makedirs(extract_dir, exist_ok=True)
        with tarfile.open(tar_path, "r:gz") as tar:
            members = tar.getmembers()
            print(f"Found {len(members)} files to extract")
            for member in tqdm(members, desc="Extracting files", unit="file"):
                tar.extract(member, path=extract_dir)

    # If the output file already exists, skip processing
    if output_file.exists():
        print(f"Output file {output_file} already exists. Skipping processing.")
        return

    # Process the dataset
    print("Processing dataset into JSONL format...")

    # Read manifests
    manifest_files = {
        "test": extract_dir / "stop" / "manifests" / "test.tsv",
    }

    dataset_entries = []

    # Function to process a manifest file
    def process_manifest(file_path, split):
        entries = []
        # Count lines in the file for progress bar
        line_count = 0
        with open(file_path, "r", encoding="utf-8") as f:
            for _ in f:
                line_count += 1

        with open(file_path, "r", encoding="utf-8") as f:
            # Skip header line
            header = f.readline().strip().split("\t")
            line_count -= 1  # Adjust for header

            for line in tqdm(f, total=line_count, desc=f"Processing {split}", unit="line"):
                parts = line.strip().split("\t")
                if len(parts) < len(header):
                    continue

                data = dict(zip(header, parts))

                # Extract filename from file_id and create a relative path
                file_id = data.get("file_id", "")
                if "test_0" in file_id or "eval_0" in file_id:
                    top, domain, filen = file_id.split("/")
                    domain = domain.replace("_0", "")
                    file_id = "/".join([top, domain, filen])
                filename = f"stop_data/stop/{file_id}"

                # Parse seqlogical to extract function call data
                seqlogical = data.get("seqlogical", "")
                intent_parts = seqlogical.split("(", 1)

                intent_name = ""
                params = {}

                # Try to parse as a function call if it has the format "function(param=value)"
                if len(intent_parts) > 1:
                    intent_name = intent_parts[0].strip()
                    # Extract parameters by parsing the string between parentheses
                    params_str = intent_parts[1].rsplit(")", 1)[0]

                    # Simple parsing for parameters (this could be made more robust)
                    param_parts = params_str.split(",")
                    for part in param_parts:
                        if "=" in part:
                            key, value = part.split("=", 1)
                            params[key.strip()] = value.strip().strip("\"'")
                else:
                    # If it's not a function call format, just use the seqlogical as intent
                    intent_name = seqlogical.strip()

                entry = {
                    "filename": filename,
                    "split": split,
                    "intent": intent_name,
                    "parameters": params,
                    "sentence": data.get("utterance", ""),
                    "domain": data.get("domain", ""),
                    "gender": data.get("gender", ""),
                    "native": data.get("native", ""),
                    "normalized_utterance": data.get("normalized_utterance", ""),
                    "normalized_seqlogical": data.get("normalized_seqlogical", ""),
                    "raw_data": data,  # Include all raw data from TSV
                }

                entries.append(entry)

        return entries

    # Process each manifest file
    for split, file_path in manifest_files.items():
        if file_path.exists():
            print(f"Processing {split} data...")
            dataset_entries.extend(process_manifest(file_path, split))

    # Write to JSONL file
    print(f"Writing {len(dataset_entries)} entries to output file...")
    with open(output_file, "w", encoding="utf-8") as f:
        for entry in tqdm(dataset_entries, desc="Writing entries", unit="entry"):
            f.write(json.dumps(entry) + "\n")

    print(f"Processed {len(dataset_entries)} entries. Output saved to {output_file}")


import json
import os
import random
import re
import math
from collections import defaultdict
from typing import List, Dict, Any
import argparse


def calculate_intent_tree_depth(json_obj: Dict) -> int:
    """
    Calculate the depth of the parse tree in the 'intent' field.

    Args:
        json_obj: A JSON object that contains an 'intent' field

    Returns:
        The depth of the parse tree
    """
    # Get the intent string
    intent_str = json_obj.get("intent", "")

    # Calculate depth based on nested brackets
    return calculate_bracket_depth(intent_str)


def calculate_bracket_depth(s: str) -> int:
    """
    Calculate the maximum depth of nested brackets in a string.

    For example: "[A [B] [C [D]]]" has a depth of 3
    """
    max_depth = 0
    current_depth = 0

    for char in s:
        if char == "[":
            current_depth += 1
            max_depth = max(max_depth, current_depth)
        elif char == "]":
            current_depth -= 1

    return max_depth


def stratified_complex_sampling(
    jsonl_path: str,
    num_samples: int = 1000,
    complexity_bias: float = 2.0,  # Higher values give more weight to complex trees
) -> List[Dict]:
    """
    Sample parse trees using stratified sampling across domains and genders,
    with a bias towards more complex queries.

    Args:
        jsonl_path: Path to the JSONL file
        num_samples: Total number of samples to return
        complexity_bias: How much to weight complexity in sampling

    Returns:
        List of stratified sampled parse trees
    """
    # First pass: collect domain and gender information and calculate depths
    trees_by_domain_gender = defaultdict(list)
    domain_gender_counts = defaultdict(int)

    print("First pass: collecting information about trees...")

    # Read all records
    all_trees = []
    with open(jsonl_path, "r") as f:
        for line_num, line in enumerate(f):
            if line_num % 10000 == 0 and line_num > 0:
                print(f"  Processed {line_num} lines...")

            try:
                # Parse JSON object from the line
                tree = json.loads(line.strip())

                # Calculate the depth of the intent tree
                depth = calculate_intent_tree_depth(tree)

                # Skip trees with no intent or depth 0
                if depth == 0:
                    continue

                # Extract domain and gender
                domain = tree.get("domain", "unknown")
                gender = tree.get("gender", "unknown")

                # Add complexity information to the tree
                tree["_complexity"] = depth

                # Store by domain and gender combination
                domain_gender_key = f"{domain}_{gender}"
                trees_by_domain_gender[domain_gender_key].append(tree)
                domain_gender_counts[domain_gender_key] += 1

                all_trees.append(tree)

            except json.JSONDecodeError:
                print(f"Warning: Could not parse JSON at line {line_num+1}. Skipping.")
            except Exception as e:
                print(f"Error processing line {line_num+1}: {str(e)}. Skipping.")

    print(f"Found {len(all_trees)} valid trees")
    print(f"Found {len(trees_by_domain_gender)} domain-gender combinations")

    # Calculate how many samples to take from each domain-gender combination
    total_records = len(all_trees)
    samples_per_domain_gender = {}

    for domain_gender, count in domain_gender_counts.items():
        # Calculate proportional allocation
        proportion = 1 / len(domain_gender_counts)
        samples_per_domain_gender[domain_gender] = min(count, int(num_samples * proportion))

    # Adjust to ensure we get exactly num_samples
    total_allocated = sum(samples_per_domain_gender.values())
    if total_allocated < num_samples:
        # Distribute remaining samples to largest categories
        remaining = num_samples - total_allocated
        sorted_domain_genders = sorted(domain_gender_counts.items(), key=lambda x: x[1], reverse=True)
        for i in range(min(remaining, len(sorted_domain_genders))):
            domain_gender = sorted_domain_genders[i][0]
            samples_per_domain_gender[domain_gender] += 1
    elif total_allocated > num_samples:
        # Remove samples from smallest categories
        excess = total_allocated - num_samples
        sorted_domain_genders = sorted(domain_gender_counts.items(), key=lambda x: x[1])
        for i in range(min(excess, len(sorted_domain_genders))):
            domain_gender = sorted_domain_genders[i][0]
            if samples_per_domain_gender[domain_gender] > 1:  # Ensure at least 1 sample per category
                samples_per_domain_gender[domain_gender] -= 1

    # Second pass: select samples from each stratum with complexity bias
    print("Second pass: selecting samples with complexity bias...")
    selected_samples = []

    for domain_gender, trees in trees_by_domain_gender.items():
        num_to_select = samples_per_domain_gender.get(domain_gender, 0)
        if num_to_select == 0:
            continue

        if num_to_select >= len(trees):
            # Take all trees in this stratum
            selected_samples.extend(trees)
        else:
            # Weight by complexity for sampling
            weights = [math.pow(tree["_complexity"], complexity_bias) for tree in trees]
            total_weight = sum(weights)
            probabilities = [w / total_weight for w in weights]

            # Sample without replacement
            sampled_indices = random.choices(
                population=list(range(len(trees))),
                k=num_to_select,
                weights=probabilities,  # But we're using weighted sampling here
            )

            # Add the selected trees
            for idx in sampled_indices:
                selected_samples.append(trees[idx])

    # Clean up any temporary fields we added
    for tree in selected_samples:
        if "_complexity" in tree:
            del tree["_complexity"]

    print(f"Selected {len(selected_samples)} samples")

    # Print statistics about the domains and genders
    domain_counts = defaultdict(int)
    gender_counts = defaultdict(int)
    depths = []

    for tree in selected_samples:
        domain = tree.get("domain", "unknown")
        gender = tree.get("gender", "unknown")
        depth = calculate_intent_tree_depth(tree)

        domain_counts[domain] += 1
        gender_counts[gender] += 1
        depths.append(depth)

    print("\nSampling Statistics:")
    print("\nDomain Distribution:")
    for domain, count in sorted(domain_counts.items()):
        print(f"  {domain}: {count} ({count/len(selected_samples)*100:.1f}%)")

    print("\nGender Distribution:")
    for gender, count in sorted(gender_counts.items()):
        print(f"  {gender}: {count} ({count/len(selected_samples)*100:.1f}%)")

    print("\nComplexity Statistics:")
    print(f"  Min depth: {min(depths)}")
    print(f"  Max depth: {max(depths)}")
    print(f"  Avg depth: {sum(depths)/len(depths):.2f}")

    return selected_samples


def save_samples(samples: List[Dict], output_path: str):
    """Save sampled trees to a new JSONL file."""
    with open(output_path, "w") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")


def filter_top_calls(json_file, num_samples=1000, bias=100, seed=42):
    random.seed(seed)
    # Verify the input path exists
    if not os.path.exists(json_file):
        print(f"Error: Input file {json_file} does not exist.")
        return

    print(f"Processing {json_file} to find parse trees with the most complex intent fields...")

    # Sample trees with stratification and complexity bias
    samples = stratified_complex_sampling(jsonl_path=json_file, num_samples=num_samples, complexity_bias=bias)

    # Save samples to output file
    save_samples(samples, json_file)

    print(f"Successfully saved {len(samples)} samples to {json_file}")
