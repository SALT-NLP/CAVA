export CAVA_CACHE_SEED=2

python data/convert_from_hf.py --dataset WillHeld/EmoCF --split train --audio-dir EmoCF --output audio_inputs.jsonl
python src/cava/inference.py --task emotion
python src/cava/inference.py --task tone_aware_reply
