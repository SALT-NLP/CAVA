export CATS_CACHE_SEED=5

python data/convert_from_hf.py --dataset rma9248/CATS-ami-speaker-diarization-audio --split train --audio-dir SpeakerDiarization --output audio_inputs.jsonl
python data/convert_from_hf.py --dataset rma9248/CATS-ami-next-speaker-audio --split train --audio-dir NextSpeaker --output audio_inputs.jsonl
python src/cats/inference.py --task speaker_diarization
python src/cats/inference.py --task next_speaker
