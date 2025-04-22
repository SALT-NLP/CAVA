export CATS_CACHE_SEED=0

python data/convert_from_hf.py --dataset MichaelR207/pronunciation_control --split train --audio-dir ReverseSpellingBee --output audio_inputs.jsonl
python src/cats/inference.py --task pronunciation_oed
python src/cats/inference.py --task pronunciation_audio
