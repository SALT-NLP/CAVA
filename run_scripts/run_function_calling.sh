export CAVA_CACHE_SEED=0

# Get model names from arguments or use default
MODELS=""
if [ $# -gt 0 ]; then
    MODELS="--models $@"
fi

python tests/test_function_data_prep.py
python src/cava/inference.py --task function_calling $MODELS
