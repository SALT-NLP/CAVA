# Function Call Evaluation Tools

This repository contains scripts for preparing and evaluating function calling capabilities of language models with the STOP (Spoken Task Oriented Semantic Parsing) dataset.

## Scripts

- `eval_function_call.py`: Evaluates language models on function calling tasks using audio inputs and compares against gold standard function calls
- `extract_openai_functions.py`: Converts STOP dataset intents into OpenAI function calling format
- `extract_stop_utterances.py`: Extracts and filters utterances from the STOP dataset
- `fairseq_to_hf.py`: Converts STOP dataset from fairseq format to HuggingFace format

## Workflow

1. Convert dataset: `fairseq_to_hf.py`
2. Extract utterances: `extract_stop_utterances.py`
3. Extract function definitions: `extract_openai_functions.py`
4. Evaluate models: `eval_function_call.py`

## End-to-End Example

```bash
# 1. Download the STOP dataset
wget https://dl.fbaipublicfiles.com/stop/stop.tar.gz
tar -xzf stop.tar.gz

# 2. Convert the dataset from fairseq to HuggingFace format
python fairseq_to_hf.py --stop_root stop --output_dir .

# 3. Extract OpenAI function definitions
python extract_openai_functions.py --dataset_path ./stop_dataset --output_dir .

# 4. Extract utterances from the STOP dataset
python extract_stop_utterances.py --dataset stop_dataset --output stop_utterances.json

# 5. Run evaluation
python eval_function_call.py \
    --model gpt-4o-audio-preview \
```

Note: You may need to adjust parameters based on your specific needs. Check each script's help documentation for available options.

## Requirements

- Python packages: datasets, tqdm, openai, pandas, soundfile
- OpenAI API access
- STOP dataset in fairseq format
