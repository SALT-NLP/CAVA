import os
import argparse
import pandas as pd
import json
from datasets import Dataset, Audio

def main(target_repo: str):
    # Load the CSV file
    df = pd.read_csv("output.csv")
    # Convert the IPAs column from string to a Python list
    df["IPAs"] = df["IPAs"].apply(json.loads)
    
    # Rename the file_path column to "audio" so that we can cast it as an Audio feature
    df = df.rename(columns={"file_path": "audio"})
    
    # Create a Hugging Face dataset from the pandas DataFrame
    ds = Dataset.from_pandas(df)
    
    # Cast the "audio" column to an Audio feature (this will handle loading the audio files)
    ds = ds.cast_column("audio", Audio())
    
    # Retrieve the Hugging Face token from the environment
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token is None:
        raise ValueError("Environment variable HF_TOKEN not set.")
    
    # Push the dataset to the Hub as a public repository
    ds.push_to_hub(target_repo, token=hf_token, private=False)
    print(f"Dataset successfully pushed to https://huggingface.co/datasets/{target_repo}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload an audio dataset to the Hugging Face Hub.")
    parser.add_argument(
        "--target_repo",
        type=str,
        default="MichaelR207/wiktionary_pronunciations",
        help="Target repository name on the Hugging Face Hub (default: MichaelR207/wiktionary_pronunciations)"
    )
    args = parser.parse_args()
    main(args.target_repo)