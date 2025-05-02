export CAVA_CACHE_SEED=0

# Get model names from arguments or use default
MODELS=""
if [ $# -gt 0 ]; then
    MODELS="--models $@"
fi

python data/convert_from_hf.py --dataset MichaelR207/pronunciation_control --split train --audio-dir ReverseSpellingBee --output audio_inputs.jsonl
python src/cava/inference.py --task pronunciation_oed $MODELS
python src/cava/inference.py --task pronunciation_audio $MODELS
