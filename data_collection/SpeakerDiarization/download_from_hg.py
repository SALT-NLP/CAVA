import argparse
import os
import json
import base64
from typing import List, Dict, Any
import time
import gc  # For explicit garbage collection
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

def download_from_huggingface(output_dir: str, local_audio_dir: str) -> bool:
    """
    Download dataset from Hugging Face and save in local format using streaming
    
    Args:
        output_dir: Directory to save the JSONL file
        local_audio_dir: Directory to save the audio files
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        print(f"Loading dataset from Hugging Face repository: rma9248/CATS-ami-speaker-diarization...")
        
        # Load dataset from Hugging Face with streaming
        dataset = load_dataset(
            "rma9248/CATS-ami-speaker-diarization", 
            split='train',
            streaming=True
        )
        
        print(f"Successfully loaded dataset")
        
        # Create output directories
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(local_audio_dir, exist_ok=True)
        
        # Path for the output JSONL file
        output_jsonl = os.path.join(output_dir, "audio_inputs.jsonl")
        

        
        # Process examples as they stream in
        local_data = []
        example_count = 0
        
        for example in dataset:
            example_count += 1
            
            # Update progress regularly
            if example_count % 10 == 0:
                print(f"Processing example {example_count}")
            
            try:
                meeting_id = example['meeting_id']
                label = example['label']
                
                # Create directory structure for audio
                sentences = 20  # Default value for data from huggingface
                meeting_audio_dir = os.path.join(local_audio_dir, meeting_id, f"#sentences={sentences}", str(label))
                os.makedirs(meeting_audio_dir, exist_ok=True)
                
                # Path for local audio file
                audio_file_path = os.path.join(meeting_audio_dir, "meeting_cut.wav")
                
                # Save audio file from base64 using chunked decoding
                audio_data = example['audio_data']
                
                # Use chunked decoding to handle large files
                audio_bytes = chunked_base64_decode(audio_data)
                
                # Write to file
                with open(audio_file_path, 'wb') as f:
                    f.write(audio_bytes)
                
                # Create relative path for the audio file (for JSONL)
                rel_audio_path = os.path.join("audio", meeting_id, f"#sentences={sentences}", str(label), "meeting_cut.wav")
                
                # Create entry for JSONL
                local_entry = {
                    "meeting_id": meeting_id,
                    "label": label,
                    "start_time": example['start_time'],
                    "end_time": example['end_time'],
                    "transcript_without_speaker": example['transcript_without_speaker'],
                    "filename": rel_audio_path,
                    "valid_speakers": example['valid_speakers'],
                    "speaker_order": example['speaker_order'],
                    "num_speakers": example['num_speakers']
                }
                
                # Add to local data list
                local_data.append(local_entry)
                
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
                
        
        print(f"Successfully downloaded and converted {example_count} examples")
        print(f"JSONL file saved at: {output_jsonl}")
        print(f"Audio files saved in: {local_audio_dir}")
        
        return True
        
    except ImportError:
        return False
    except Exception as e:
        print(f"Error downloading from Hugging Face: {str(e)}")
        print(traceback.format_exc())
        return False

def main(args):
    # Download from Hugging Face and save locally
    download_from_huggingface(
        output_dir=args.output_dir or os.path.join("../../data/SpeakerDiarization/"),
        local_audio_dir=args.audio_dir or os.path.join("../../data/SpeakerDiarization/audio")
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Download dataset from Hugging Face and save locally using streaming')
    
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to save the JSONL file (default: ../../data/SpeakerDiarization/)')
    parser.add_argument('--audio_dir', type=str, default=None,
                        help='Directory to save audio files (default: ../../data/SpeakerDiarization/audio)')
    
    args = parser.parse_args()
    main(args)