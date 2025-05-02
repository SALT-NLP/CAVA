export CAVA_CACHE_SEED=7

# Get model names from arguments or use default
MODELS=""
if [ $# -gt 0 ]; then
    MODELS="--models $@"
fi

python data/convert_from_hf.py --dataset WillHeld/CartesiaJeopardy --split combined --audio-dir jeopardy --output audio_inputs.jsonl --exclude-columns filename
# Occasionally Futures Seem to Cause Hangs which matters a lot given the latency metrics
python src/cava/inference.py --task jeopardy --workers 1 $MODELS
