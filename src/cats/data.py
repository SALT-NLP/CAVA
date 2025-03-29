import os
import json
import tarfile
import urllib.request
from pathlib import Path
from typing import Dict, Any, List, Optional
from tqdm import tqdm
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

    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=destination, reporthook=t.update_to)

def prep_function_calling_data():
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
        "train": extract_dir / "stop" / "manifests" / "train.tsv",
        "test": extract_dir / "stop" / "manifests" / "test.tsv",
        "eval": extract_dir / "stop" / "manifests" / "eval.tsv"
    }

    dataset_entries = []

    # Function to process a manifest file
    def process_manifest(file_path, split):
        entries = []
        # Count lines in the file for progress bar
        line_count = 0
        with open(file_path, 'r', encoding='utf-8') as f:
            for _ in f:
                line_count += 1

        with open(file_path, 'r', encoding='utf-8') as f:
            # Skip header line
            header = f.readline().strip().split('\t')
            line_count -= 1  # Adjust for header

            for line in tqdm(f, total=line_count, desc=f"Processing {split}", unit="line"):
                parts = line.strip().split('\t')
                if len(parts) < len(header):
                    continue

                data = dict(zip(header, parts))

                # Extract filename from file_id and create a relative path
                file_id = data.get('file_id', '')
                filename = f"stop_data/stop/{file_id}"

                # Parse seqlogical to extract function call data
                seqlogical = data.get('seqlogical', '')
                intent_parts = seqlogical.split('(', 1)

                intent_name = ""
                params = {}

                # Try to parse as a function call if it has the format "function(param=value)"
                if len(intent_parts) > 1:
                    intent_name = intent_parts[0].strip()
                    # Extract parameters by parsing the string between parentheses
                    params_str = intent_parts[1].rsplit(')', 1)[0]

                    # Simple parsing for parameters (this could be made more robust)
                    param_parts = params_str.split(',')
                    for part in param_parts:
                        if '=' in part:
                            key, value = part.split('=', 1)
                            params[key.strip()] = value.strip().strip('"\'')
                else:
                    # If it's not a function call format, just use the seqlogical as intent
                    intent_name = seqlogical.strip()

                entry = {
                    "filename": filename,
                    "split": split,
                    "intent": intent_name,
                    "parameters": params,
                    "sentence": data.get('utterance', ''),
                    "domain": data.get('domain', ''),
                    "gender": data.get('gender', ''),
                    "native": data.get('native', ''),
                    "normalized_utterance": data.get('normalized_utterance', ''),
                    "normalized_seqlogical": data.get('normalized_seqlogical', ''),
                    "raw_data": data  # Include all raw data from TSV
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
    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in tqdm(dataset_entries, desc="Writing entries", unit="entry"):
            f.write(json.dumps(entry) + '\n')

    print(f"Processed {len(dataset_entries)} entries. Output saved to {output_file}")
