# MEAU - Multitask Evaluation of Audio Models

A framework for evaluating audio models across multiple tasks.

## Setup

### Create environment and install packages

```bash
conda create -n "meau" python=3.12 ipython -y>
conda activate meau
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

### 3. Configure a new task

Add a new task configuration in `src/meau/config.py` by updating the `create_task_configs()` function:

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

### 4. Run evaluation

Run the evaluation using the command:

```python
import meau

# To run evaluation on your new task
from meau.inference import main
main()
```

You can also modify the `main()` function in `src/meau/inference.py` to specify your task:

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

## Output format

After running evaluation, results will be saved in files named:
`[data_file]_[model_name]_[task_name]`

For example:
`audio_inputs.jsonl_Qwen2-Audio-7B-Instruct_emotion`

The output file will contain the original records with added prediction fields.
