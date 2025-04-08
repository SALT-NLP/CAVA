"""
process_dataset.py loads the werewolf dataset from huggingface and prepares it in the correct format for inference.py
"""

import os
import json
import datasets
import soundfile as sf  

werewolf_dataset = datasets.load_dataset("mikesun26card/werewolf_audio_dataset", split="train")

jsonl_path = "werewolf.jsonl" 

with open(jsonl_path, "w", encoding="utf-8") as f_jsonl:
    for i, sample in enumerate(werewolf_dataset):
        wave_array = sample["file_name"]["array"]
        sampling_rate = sample["file_name"]["sampling_rate"]
        
        filename = f"{i}.wav"
        local_audio_path = os.path.join("WereWolf", "data", filename)
        
        sf.write(local_audio_path, wave_array, sampling_rate)
        
        metadata = {
            "filename": filename,
            "werewolf": json.loads(sample["werewolfNames"]) if isinstance(sample["werewolfNames"], str) else sample["werewolfNames"],
            "PlayerNames": json.loads(sample["playerName"]) if isinstance(sample["playerName"], str) else sample["playerName"],
            "endRoles": json.loads(sample["endRoles"]) if isinstance(sample["endRoles"], str) else sample["endRoles"],
            "votingOutcome": json.loads(sample["votingOutcome"]) if isinstance(sample["votingOutcome"], str) else sample["votingOutcome"],
        }       
        
        f_jsonl.write(json.dumps(metadata) + "\n")

print("Export completed.")
