export CAVA_CACHE_SEED=0

python data/convert_from_hf.py --dataset SALT-NLP/MultiModalInstructionFollowing --split train --audio-dir MultimodalInstructionFollowing --output audio_inputs.jsonl --limit 1000
python src/cava/inference.py --task multimodal_instruction_following
