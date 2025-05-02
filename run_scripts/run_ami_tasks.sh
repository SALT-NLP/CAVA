export CAVA_CACHE_SEED=5

# Get model names from arguments or use default
MODELS=""
if [ $# -gt 0 ]; then
    MODELS="--models $@"
fi

python data/convert_from_hf.py --dataset rma9248/CATS-ami-speaker-diarization-audio --split train --audio-dir SpeakerDiarization --output audio_inputs.jsonl
python data/convert_from_hf.py --dataset rma9248/CATS-ami-next-speaker-audio --split train --audio-dir NextSpeaker --output audio_inputs.jsonl
python src/cava/inference.py --task speaker_diarization $MODELS
python src/cava/inference.py --task next_speaker $MODELS
