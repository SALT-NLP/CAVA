#!/bin/bash

# run_all_tasks.sh - Master script to run all CAVA tasks with specified models

# Default models if none provided
DEFAULT_MODELS=("gemini-2.5-pro-preview-03-25" "models/gemini-2.0-flash-exp" "gpt-4o-audio-preview" "pipeline_gpt-4o_gpt-4o-mini-tts_gpt-4o-mini-transcribe")

# Use provided models or default ones
if [ $# -gt 0 ]; then
    MODELS=("$@")
else
    MODELS=("${DEFAULT_MODELS[@]}")
fi

# Display models being evaluated
echo "Running evaluation on the following models:"
for model in "${MODELS[@]}"; do
    echo "- $model"
done
echo ""

# Convert model array to space-separated string for passing to scripts
MODEL_ARGS="${MODELS[@]}"

# Function to run a script and report completion
run_script() {
    script_path="$1"
    script_name=$(basename "$script_path")
    
    echo "========================================================"
    echo "Starting $script_name with models: $MODEL_ARGS"
    echo "========================================================"
    
    # Run the script with the models as arguments
    bash "$script_path" $MODEL_ARGS
    
    echo "========================================================"
    echo "Completed $script_name"
    echo "========================================================"
    echo ""
}

# Run all scripts in the run_scripts directory
run_script run_scripts/run_ami_tasks.sh
run_script run_scripts/run_counterfactual_tasks.sh
run_script run_scripts/run_function_calling.sh
run_script run_scripts/run_jailbreaking_tasks.sh
run_script run_scripts/run_jeopardy.sh
run_script run_scripts/run_multimodal_instruction_following.sh
run_script run_scripts/run_reverse_spelling_bee.sh
run_script run_scripts/run_werewolf_tasks.sh

echo "All evaluations completed successfully!"
