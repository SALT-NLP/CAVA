"""
load_hf.py contains the code to upload the audio dataset to huggingface.
"""

from datasets import load_dataset, DatasetDict
from datasets import Audio

dataset = load_dataset("audiofolder", data_dir = "hf_audio_dataset")
dataset.push_to_hub("mikesun26card/werewolf_audio_dataset")


# print(dataset)
# sample = dataset['train'][0]
# print(sample)
# print(sample["audio"].keys())  # might show "array", "sampling_rate", "path"
# print(type(sample["audio"]["array"]))
# print(sample["audio"]["array"].shape)



### Below code is for debugging
# full_ds = load_dataset("audiofolder", data_dir="hf_audio_dataset")["train"]

# preview_ds = full_ds.select(range(10))

# def reduce_audio(example):
#     return {"audio": example["audio"]["path"]}

# preview_ds = preview_ds.map(reduce_audio, remove_columns=["audio"])

# combined_ds = DatasetDict({
#     "train": full_ds,       # full dataset with heavy arrays
#     "preview": preview_ds   # lightweight preview with only file paths
# })

# for i in range(len(full_ds)):
#     sample = full_ds[i]
#     audio_info = sample["audio"]
#     file_path = audio_info["path"]
#     array = audio_info["array"]
#     size_mb = array.nbytes / (1024 * 1024)  # Convert bytes to MB
#     print(f"Row {i}:")
#     print(f"  Audio path: {file_path}")
#     print(f"  Array shape: {array.shape}")
#     print(f"  Array size: {size_mb:.2f} MB")
#     print(f"  Sampling rate: {audio_info['sampling_rate']}")

# max_size = 0
# max_index = -1

# for i in range(len(full_ds)):
#     sample = full_ds[i]
#     array_size = sample["audio"]["array"].nbytes  # size in bytes
#     if array_size > max_size:
#         max_size = array_size
#         max_index = i

# print(f"Largest array is in row {max_index} with size: {max_size / (1024 * 1024):.2f} MB")



#dataset.push_to_hub("mikesun26card/werewolf_audio_dataset")
