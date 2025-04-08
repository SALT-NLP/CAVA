"""
accuracy_script_detection.py calculates the accuracy of predictions for the werewolf deception detection task (i.e. predicts which character is the werewolf).
It measures the percentage of predictions which are correct.
"""

import json

# Path to your file
file_path = "werewolf.jsonl_gemini-2.0-flash-exp_deception_detection_1" #'data.txt'  # or 'data.jsonl'
#file_path = "werewolf.jsonl_gpt-4o-audio-preview_deception_detection_1"

total = 0
correct_predictions = 0

with open(file_path, 'r') as f:
    for line in f:
        if line.strip():  # Skip empty lines
            entry = json.loads(line)
            prediction = entry["prediction"].strip('.')
            if prediction in entry["werewolf"]:
                correct_predictions += 1
            if prediction == "None" and entry["werewolf"] == []:
                correct_predictions += 1
            total += 1

# Compute proportion
proportion = ((correct_predictions*100) / total) if total > 0 else 0
print(f"Percentage of correct werewolf predictions: {proportion:.2f}%")
