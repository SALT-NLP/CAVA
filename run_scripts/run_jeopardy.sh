export CATS_CACHE_SEED=7

#python data/convert_from_hf.py --dataset WillHeld/CartesiaJeopardy --split combined --audio-dir jeopardy --output audio_inputs.jsonl --exclude-columns filename
# Occasionally Futures Seem to Cause Hangs which matters a lot given the latency metrics
python src/cats/inference.py --task jeopardy --workers 1
