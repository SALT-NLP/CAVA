export CAVA_CACHE_SEED=0

# Get model names from arguments or use default
MODELS=""
if [ $# -gt 0 ]; then
    MODELS="--models $@"
fi

python data/convert_from_hf.py --dataset WillHeld/werewolf --split train --audio-dir Werewolf --output audio_inputs.jsonl
python src/cava/inference.py --task deception_detection $MODELS
python src/cava/inference.py --task deception_vote_prediction $MODELS
