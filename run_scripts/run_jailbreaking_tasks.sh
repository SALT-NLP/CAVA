export CAVA_CACHE_SEED=0

# Get model names from arguments or use default
MODELS=""
if [ $# -gt 0 ]; then
    MODELS="--models $@"
fi

python data/convert_from_hf.py --dataset WillHeld/AudioJailbreakPersuasive --split train --audio-dir jailbreaking/jailbreak_persuasive --output audio_inputs.jsonl
python data/convert_from_hf.py --dataset WillHeld/AudioJailbreakPersuasive --split basic --audio-dir jailbreaking/jailbreak_basic --output audio_inputs.jsonl
python src/cava/inference.py --task jailbreak_base $MODELS
python src/cava/inference.py --task jailbreak $MODELS
