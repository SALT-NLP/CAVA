export CAVA_CACHE_SEED=0

# Get model names from arguments or use default
MODELS=""
if [ $# -gt 0 ]; then
    MODELS="--models $@"
fi

python data/convert_from_hf.py --dataset SALT-NLP/MultiModalInstructionFollowing --split train --audio-dir MultimodalInstructionFollowing --output audio_inputs.jsonl --limit 1000
python src/cava/inference.py --task multimodal_instruction_following  $MODELS
