export CATS_CACHE_SEED=0

python tests/test_function_data_prep.py
python src/cats/inference.py --task function_calling
