# Next Speaker Data Collection Pipeline
## You can directly download and process the data from huggingface
Please cd back to the home directory of CATS first, then run this script:
```sh
python data/convert_from_hf.py --dataset rma9248/CATS-ami-next-speaker-audio --split train --audio-dir NextSpeaker --output audio_inputs.jsonl --preserve-columns
```