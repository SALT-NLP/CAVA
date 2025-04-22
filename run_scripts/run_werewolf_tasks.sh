export CATS_CACHE_SEED=0

python data/convert_from_hf.py --dataset WillHeld/werewolf --split train --audio-dir Werewolf --output audio_inputs.jsonl
python src/cats/inference.py --task deception_detection
python src/cats/inference.py --task deception_vote_prediction
