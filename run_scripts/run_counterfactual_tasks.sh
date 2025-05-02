export CAVA_CACHE_SEED=2

# Get model names from arguments or use default
MODELS=""
if [ $# -gt 0 ]; then
    MODELS="--models $@"
fi

python data/convert_from_hf.py --dataset WillHeld/EmoCF --split train --audio-dir EmoCF --output audio_inputs.jsonl
python src/cava/inference.py --task emotion $MODELS
python src/cava/inference.py --task tone_aware_reply $MODELS
