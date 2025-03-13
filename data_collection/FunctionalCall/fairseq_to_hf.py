#!/usr/bin/env python3
import argparse
import os
import json
import sys
import subprocess
import shutil
from pathlib import Path

# Check and install required dependencies
required_packages = ["datasets", "pandas", "tqdm", "soundfile"]
installed_packages = []

for package in required_packages:
    try:
        __import__(package)
        installed_packages.append(package)
    except ImportError:
        print(f"Installing required package: {package}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        installed_packages.append(package)

print(f"Successfully installed/verified packages: {', '.join(installed_packages)}")

from tqdm import tqdm
import pandas as pd
import soundfile as sf
from datasets import Dataset, Audio, Features, Value, DatasetDict

def get_insl_frame(parse):
    """Convert a parse string to the format required by fairseq."""
    res = []
    x = []
    for tok in parse.split():
        if tok[0] in ["[", "]"]:
            if x:
                res.append('_'.join(x))
                x = []
            res.append(tok.upper())
        else:
            x.append(tok.upper())
    if x:  # Handle any remaining tokens
        res.append('_'.join(x))
    return " ".join(res) + ' | '

def load_stop_manifest(manifest_path):
    """Load the STOP manifest file."""
    with open(manifest_path, 'r') as f:
        lines = f.readlines()

    keys = lines[0].strip().split('\t')
    data = []

    for line in lines[1:]:
        values = line.strip().split('\t')
        item = {k: v for k, v in zip(keys, values)}
        data.append(item)

    return data


def convert_fairseq_to_hf(stop_root, output_dir, splits=['train', 'eval', 'test'], max_items=None, copy_audio=True):
    """Convert fairseq STOP dataset to HuggingFace format.

    Args:
        stop_root: Path to the STOP dataset root directory
        output_dir: Path where to save the HuggingFace dataset
        splits: List of splits to process (defaults to ['train', 'eval', 'test'])
        max_items: Maximum number of items to process per split (for testing)
        copy_audio: Whether to copy audio files to make the dataset self-contained
    """
    stop_root = Path(stop_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Create audio directory in output folder if needed
    audio_output_dir = output_dir / 'audio'
    if copy_audio:
        audio_output_dir.mkdir(exist_ok=True, parents=True)

    # Print directories to help debug
    print(f"STOP root directory: {stop_root}")
    print(f"Output directory: {output_dir}")
    if copy_audio:
        print(f"Audio output directory: {audio_output_dir}")
    print(f"Processing splits: {splits}")
    print(f"Max items per split: {max_items if max_items else 'No limit'}")
    print(f"Copy audio files: {copy_audio}")

    # Check if manifest directory exists
    manifest_dir = stop_root / 'manifests'
    if not manifest_dir.exists():
        print(f"Error: Manifest directory not found at {manifest_dir}")
        print("Checking for available files in the root directory:")
        print(list(stop_root.glob("*")))
        return

    # Use default splits if none provided
    if splits is None:
        splits = ['train', 'eval', 'test']

    hf_dataset_dict = {}

    for split in splits:
        print(f"Processing {split} split...")
        manifest_path = stop_root / 'manifests' / f"{split}.tsv"

        # Load the manifest
        data = load_stop_manifest(manifest_path)

        # Limit number of items if requested (useful for testing)
        if max_items is not None:
            print(f"Limiting to {max_items} items for {split} split.")
            data = data[:max_items]

        # Create split directory in audio output directory if copying
        if copy_audio:
            split_audio_dir = audio_output_dir / split
            split_audio_dir.mkdir(exist_ok=True, parents=True)

        # Extract the required information
        examples = []
        for item in tqdm(data):
            # Correct audio path structure
            # The manifest uses paths like test_0/alarm_test_0/00003470.wav
            # But the actual structure is test_0/alarm_test/00003470.wav
            file_id = item['file_id']
            file_id_parts = file_id.split('/')

            if len(file_id_parts) == 3:
                # Extract split type, domain, and file name
                split_type, domain_with_suffix, filename = file_id_parts

                # Simpler solution: Just remove the last part after the second underscore if it exists
                if '_' in domain_with_suffix:
                    domain_parts = domain_with_suffix.split('_')
                    if len(domain_parts) >= 3:
                        # It's a domain_type_splittype_number format (e.g., music_test_0)
                        # Just take the first two parts
                        corrected_domain = f"{domain_parts[0]}_{domain_parts[1]}"
                        corrected_path = f"{split_type}/{corrected_domain}/{filename}"
                        audio_path = stop_root / corrected_path
                    else:
                        # Already in the correct format
                        audio_path = stop_root / file_id
                else:
                    audio_path = stop_root / file_id
            else:
                audio_path = stop_root / file_id

            try:
                # Get audio info
                audio_info = sf.info(audio_path)

                # Extract domain from file_id
                domain = file_id_parts[1].split('_')[0] if len(file_id_parts) >= 2 else 'unknown'

                if copy_audio:
                    # Copy audio file to output directory
                    # Create a new path that preserves domain information
                    if len(file_id_parts) == 3:
                        _, domain, filename = file_id_parts
                        # Create domain directory if it doesn't exist
                        domain_dir = split_audio_dir / domain
                        domain_dir.mkdir(exist_ok=True, parents=True)

                        # New destination path for the audio file
                        new_audio_path = domain_dir / filename
                    else:
                        # Fallback: just use filename
                        filename = file_id_parts[-1]
                        new_audio_path = split_audio_dir / filename

                    # Copy the audio file
                    shutil.copy2(audio_path, new_audio_path)

                    # Use the absolute path in the dataset to ensure it can be found regardless of current directory
                    dataset_audio_path = str(new_audio_path.absolute())
                else:
                    # Use the original path as absolute path
                    dataset_audio_path = str(audio_path.absolute())

                insl_frame = item.get('decoupled_normalized_seqlogical', item.get('normalized_utterance', item.get('utterance', '')))

                example = {
                    'file_id': file_id,
                    'audio_path': dataset_audio_path,
                    'sampling_rate': audio_info.samplerate,
                    'duration': audio_info.duration,
                    'utterance': item.get('normalized_utterance', item.get('utterance', '')),  # Fall back to 'utterance' if normalized not available
                    'parse': get_insl_frame(insl_frame),
                    'domain': domain.split('_')[0],
                    'gender': item.get('gender', 'unknown'),
                    'native': item.get('native', 'unknown')
                }

                # Store original path as a reference only if we're copying
                if copy_audio:
                    example['original_audio_path'] = str(audio_path.absolute())

            except Exception as e:
                print(f"Error processing {audio_path}: {e}")
                continue
            examples.append(example)

        # Create dataset if there are examples
        if examples:
            df = pd.DataFrame(examples)
            hf_dataset = Dataset.from_pandas(df)

            # Add audio feature
            hf_dataset = hf_dataset.cast_column('audio_path', Audio())

            # Rename the split 'eval' to 'validation' for HF compatibility
            split_name = 'validation' if split == 'eval' else split
            hf_dataset_dict[split_name] = hf_dataset
        else:
            print(f"Warning: No valid examples found for {split} split. Skipping.")
            continue

    # Create and save the DatasetDict if we have at least one dataset
    if hf_dataset_dict:
        dataset_dict = DatasetDict(hf_dataset_dict)

        # Save the dataset
        dataset_dict.save_to_disk(output_dir / 'stop_dataset')
        print(f"Dataset saved to {output_dir / 'stop_dataset'}")
    else:
        print("Error: No valid datasets found for any split. Cannot create dataset.")

    # Also create a dataset card (README.md) if we found valid data
    if hf_dataset_dict:
        readme_content = """---
language:
- en
license: cc-by-4.0
task_categories:
- automatic-speech-recognition
- spoken-language-understanding
---

# STOP: A dataset for Spoken Task Oriented Semantic Parsing

This dataset is derived from the STOP dataset presented in the paper [STOP: A dataset for Spoken Task Oriented Semantic Parsing](https://arxiv.org/abs/2207.10643).

The dataset contains audio recordings of spoken commands and their corresponding semantic parses.

"""

        if copy_audio:
            readme_content += """## Dataset Structure

This is a self-contained version of the STOP dataset with all audio files included in the `audio` directory.

The dataset contains:
- Audio files organized by split (train/validation/test) and domain (music, alarm, etc.)
- Normalized utterance text
- Semantic parse representations in the INSL frame format

## Directory Structure

```
output_dir/
├── audio/
│   ├── train/
│   │   ├── music/
│   │   ├── alarm/
│   │   └── ...
│   ├── validation/
│   │   ├── music/
│   │   ├── alarm/
│   │   └── ...
│   └── test/
│       ├── music/
│       ├── alarm/
│       └── ...
└── stop_dataset/  # The HuggingFace dataset files
```
"""
        else:
            readme_content += """## Dataset Structure

The dataset contains:
- References to the original audio files
- Normalized utterance text
- Semantic parse representations in the INSL frame format

Note: This version of the dataset references the original audio files and is not self-contained.
You need to have access to the original STOP dataset files.
"""

        readme_content += """
## Usage

```python
from datasets import load_from_disk

dataset = load_from_disk("path/to/stop_dataset")

# Access a sample
sample = dataset["train"][0]
print(sample["utterance"])
print(sample["parse"])

# Play audio
import IPython.display as ipd
ipd.Audio(sample["audio_path"])

# Access additional fields
print(sample["domain"])
print(sample["gender"])
print(sample["native"])
```

## Citation

```bibtex
@article{stop2022,
  title={{STOP}: A dataset for Spoken Task Oriented Semantic Parsing},
  author={Rao, Paden and Popuri, Sravya and Bapna, Ankur and Arivazhagan, Naveen and Freitag, Markus},
  journal={arXiv preprint arXiv:2207.10643},
  year={2022}
}
```
"""

        with open(output_dir / 'stop_dataset/README.md', 'w') as f:
            f.write(readme_content)

        print("Dataset card created.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert STOP dataset from fairseq to HuggingFace format')
    parser.add_argument('--stop_root', type=str, default='./stop',
                    help='Path to STOP root directory')
    parser.add_argument('--output_dir', type=str, default='.',
                    help='Output directory for HuggingFace dataset')
    parser.add_argument('--splits', type=str, default='train,eval,test',
                    help='Comma-separated list of splits to process (train, eval, test)')
    parser.add_argument('--max_items', type=int, default=None,
                    help='Maximum number of items to process per split (for testing)')
    parser.add_argument('--no_copy_audio', action='store_true',
                    help='Do not copy audio files to the output directory (dataset will reference original files)')

    args = parser.parse_args()

    # Expand user directory paths
    stop_root = os.path.expanduser(args.stop_root)
    output_dir = os.path.expanduser(args.output_dir)

    # Convert splits string to list
    splits = [s.strip() for s in args.splits.split(',')]

    # Convert (copy_audio is True by default, set to False if --no_copy_audio is specified)
    convert_fairseq_to_hf(stop_root, output_dir, splits, args.max_items, not args.no_copy_audio)