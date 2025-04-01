<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="./img/CATS-white.svg">
    <source media="(prefers-color-scheme: light)" srcset="./img/CATS.svg">
    <img alt="Shows a black logo in light color mode and a white one in dark color mode.">
  </picture>
</p>

# CATS - Comprehensive Assessment for Testing Speech

A framework for evaluating audio models across multiple tasks.

## Setup

### Create environment and install packages

```bash
conda create -n "cats" python=3.12 ipython -y
conda activate cats
pip install -e .
```

### Add .env file

Include the .env file in the following format:

```
OPENAI_API_KEY=[KEY]
GEMINI_API_KEY=[KEY]
```

## Adding a new dataset

To add a new dataset to the evaluation framework, follow these steps:

### 1. Prepare your audio files

- Ensure your audio files are in a compatible format (WAV is recommended)
- Place them in a dedicated directory within the project structure
- For example: `data/YourDatasetName/audio/`

### 2. Create a dataset description file

Create a JSONL file with entries describing each audio file. Each line should be a valid JSON object with the following structure:

```json
{
  "filename": "your_audio_file.wav",
  "field_name": "ground_truth_value",
  "sentence": "optional context or transcript",
  ... # All Additional Relevant Metadata
```

For example, for an emotion classification task:
```json
{
  "filename": "angry_speech_1.wav",
  "emotion": "anger",
  "sentence": "I can't believe you did that!",
  "voice_gender": "female"
}
```

### 3. Using HuggingFace datasets

You can convert audio datasets from HuggingFace to the CATS format using the included conversion script. This allows you to leverage existing audio datasets without manual file preparation.

#### HuggingFace Dataset Converter

The `convert_from_hf.py` script converts any HuggingFace audio dataset to CATS format:

```bash
python convert_from_hf.py \
  --dataset WillHeld/werewolf \
  --split train \
  --audio-dir werewolf_data \
  --output audio_inputs.jsonl \
  --preserve-columns
```

This will:
1. Download the specified dataset from HuggingFace
2. Extract the audio files to `data/werewolf_data/`
3. Create a JSONL file at `data/werewolf_data/audio_inputs.jsonl` with entries like:

```json
{"filename": "0.wav", "werewolf": ["Justin", "Mike"], "PlayerNames": ["Justin", "Caitlynn", "Mitchell", "James", "Mike"], "endRoles": ["Werewolf", "Tanner", "Seer", "Robber", "Werewolf"], "votingOutcome": [3, 0, 3, 0, 0]}
```

You can then use this dataset like any other CATS dataset by configuring a task with:
- `audio_dir: "werewolf_data/"`
- `data_file: "audio_inputs.jsonl"`

For more options and customization:

```bash
python convert_from_hf.py --help
```

### 4. Configure a new task

Add a new task configuration in `src/cats/config.py` by updating the `create_task_configs()` function:

```python
def create_task_configs() -> Dict[str, TaskConfig]:
    return {
        # ... existing tasks ...
        
        "your_task": TaskConfig(
            name="your_task",
            prompt_template="Your task-specific prompt here",
            labels=["label1", "label2", "label3"],  # Optional for classification tasks
            max_new_tokens=10,                      # Adjust based on expected response length
            field_name="your_field_name",           # Field containing ground truth
            audio_dir="your_dataset_directory/",    # Path to audio files
            data_file="your_dataset_inputs.jsonl",  # Formatted data file
        ),
    }
```

### 5. Run evaluation

Run the evaluation using the command:

```python
import cats

# To run evaluation on your new task
from cats.inference import main
main()
```

You can also modify the `main()` function in `src/cats/inference.py` to specify your task:

```python
def main():
    # Reset API counters
    reset_api_counters()
    
    # Get available tasks
    tasks = create_task_configs()
    
    # Specify your task
    task_name = "your_task"  
    task_config = tasks[task_name]
    
    # ... rest of the function
```

## Prompt Templates

Prompt templates are used to guide the model in performing the task. 
Templates can include placeholders for dynamic content using the format `{placeholder_name}`.

For example:
```python
prompt_template="Analyze the audio and determine if the speaker sounds {emotion_type}. Respond with only 'yes' or 'no'."
```

When no placeholders are used, the template is used as a prefix to the audio input.

## Output format

After running evaluation, results will be saved in files named:
`[data_file]_[model_name]_[task_name]`

For example:
`audio_inputs.jsonl_Qwen2-Audio-7B-Instruct_emotion`

The output file will contain the original records with added prediction fields.

## Speech Output Evaluation

For tasks that require evaluating a model's speech output (such as pronunciation or speech synthesis):

1. Set the `speech_output` parameter to `True` in your task configuration
2. Specify an `output_audio_dir` where generated audio will be saved
3. Define an appropriate evaluation metric in the task configuration
