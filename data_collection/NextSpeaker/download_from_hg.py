import argparse
import os
import json
import base64
from typing import List, Dict, Any
import time
import gc

from tqdm import tqdm  # For explicit garbage collection
from datasets import load_dataset
import traceback

def chunked_base64_decode(base64_string, chunk_size=1000000):
    """
    Decode a base64 string in chunks to avoid memory issues with very large strings
    
    Args:
        base64_string: The base64 encoded string
        chunk_size: Size of chunks to process at once
        
    Returns:
        bytes: The decoded data
    """
    # Determine padding - base64 strings should have length multiple of 4
    padding_needed = len(base64_string) % 4
    if padding_needed:
        base64_string += '=' * (4 - padding_needed)
    
    # Process in chunks to avoid memory issues
    chunks = []
    for i in range(0, len(base64_string), chunk_size):
        chunk = base64_string[i:i+chunk_size]
        if i + chunk_size >= len(base64_string):
            # Last chunk might need padding
            while len(chunk) % 4 != 0:
                chunk += '='
        decoded_chunk = base64.b64decode(chunk)
        chunks.append(decoded_chunk)
    
    # Combine all chunks
    return b''.join(chunks)

def download_from_huggingface(output_dir: str, local_audio_dir: str, max_samples: int = None) -> bool:
    """
    Download dataset from Hugging Face and save in local format using streaming
    
    Args:
        repo_name: Name of the Hugging Face repository
        output_dir: Directory to save the JSONL file
        local_audio_dir: Directory to save the audio files
        max_samples: Maximum number of samples to download (None for all)
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        print(f"Loading dataset from Hugging Face repository: rma9248/CATS-ami-next-speaker...")
        
        # Load dataset from Hugging Face with streaming
        dataset = load_dataset(
            "rma9248/CATS-ami-next-speaker", 
            split='train',
            streaming=True
        )
        max_samples = min(max_samples, 1000) if max_samples is not None else 1000
        print(f"Successfully loaded dataset")
        
        # Create output directories
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(local_audio_dir, exist_ok=True)
        
        # Path for the output JSONL file
        output_jsonl = os.path.join(output_dir, "audio_inputs.jsonl")
        
        # Clear the output file if it exists
        with open(output_jsonl, 'w') as f:
            pass
        
        # Process examples as they stream in
        example_count = 0
        
        for example in tqdm(dataset, total=max_samples, desc="Downloading examples"):
            # Skip if we've reached max samples
            if max_samples is not None and example_count >= max_samples:
                break
                
            example_count += 1
            
            # Update progress regularly
            if example_count % 10 == 0:
                print(f"Processing example {example_count}")
            
            try:
                # Extract datapoint ID and create a unique label
                datapoint_id = example.get('datapoint_id', f"example_{example_count}")
                meeting_id = example.get('meeting_id', 'unknown')
                label = f"{example_count}_{datapoint_id}"
                
                audio_file_path = os.path.join(local_audio_dir, f"{label}.wav")
                
                # Save audio file from base64 using chunked decoding
                audio_data = example['audio_data']
                
                # Use chunked decoding to handle large files
                audio_bytes = chunked_base64_decode(audio_data)
                
                # Write to file
                with open(audio_file_path, 'wb') as f:
                    f.write(audio_bytes)
                
                # Create relative path for the audio file (for JSONL)
                rel_audio_path = os.path.relpath(audio_file_path, output_dir)
                
                # Create entry for JSONL
                local_entry = {
                    "meeting_id": meeting_id,
                    "label": label,
                    "start_time": example.get('start_time', 0),
                    "context_end_time": example.get('context_end_time', 0),
                    "filename": rel_audio_path
                }
                
                # Add other fields if they exist in the example
                for field in ['context_transcription', 'test_statement', 'transcript_without_speaker', 
                             'valid_speakers', 'speaker_order', 'num_speakers', 'speaker_answer']:
                    if field in example:
                        local_entry[field] = example[field]
                speaker_list = example["valid_speakers"]
                sorted_speaker_list = sorted(speaker_list)  # Sort the list alphabetically
                formatted_speaker_list = ", ".join(sorted_speaker_list)
                local_entry["formatted_speaker_list"] = formatted_speaker_list
                # Write entry to JSONL file immediately to save memory
                with open(output_jsonl, 'a') as f:
                    json.dump(local_entry, f)
                    f.write('\n')
                
                # Free memory
                del audio_bytes
                gc.collect()
                
            except Exception as e:
                error_msg = str(e)
                traceback_str = traceback.format_exc()
                
                print(f"Error processing example {example_count}: {error_msg}")
                print(traceback_str)
                
        print(f"Successfully downloaded and converted {example_count} examples")
        print(f"JSONL file saved at: {output_jsonl}")
        print(f"Audio files saved in: {local_audio_dir}")
        
        return True
        
    except ImportError:
        print("Make sure you have installed the required packages:")
        print("pip install datasets huggingface_hub")
        return False
    except Exception as e:
        print(f"Error downloading from Hugging Face: {str(e)}")
        print(traceback.format_exc())
        return False

def main(args):
    # Download from Hugging Face and save locally
    download_from_huggingface(
        output_dir=args.output_dir,
        local_audio_dir=args.audio_dir,
        max_samples=args.max_samples
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Download dataset from Hugging Face and save locally using streaming')
    
    parser.add_argument('--output_dir', type=str, default="../../data/NextSpeaker/",
                        help='Directory to save the JSONL file (default: ../../data/NextSpeaker/)')
    parser.add_argument('--audio_dir', type=str, default="../../data/NextSpeaker/audio",
                        help='Directory to save audio files (default: ../../data/NextSpeaker/audio)')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Maximum number of samples to download (default: all)')
    
    args = parser.parse_args()
    main(args)