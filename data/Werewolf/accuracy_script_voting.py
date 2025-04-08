"""
accuracy_script_voting.py calculates the accuracy of predictions for the werewolf voting task.
It measures the percentage of predictions which are correct on a per-vote basis (i.e. each vote is compared to the prediction for that vote)
Note that if there are more predictions than votes cast, it only looks at the first few predictions 
(i.e. if there are 5 predictions but only 3 votes cast, only look at first 3 predictions).
"""

import json
from typing import List


def clean_prediction(prediction_str: str) -> List[str]:
    """
    Cleans and parses the prediction field which is a multiline string of predicted vote names.
    """
    lines = prediction_str.strip().splitlines()
    votes = []
    for line in lines:
        clean_line = line.strip()
        # If the line contains keywords that indicate it's header text, skip it.
        if any(keyword in clean_line.lower() for keyword in ["predicted", "votes", "based on the provided audio"]):
            continue
        # Remove markdown asterisks if present.
        clean_line = clean_line.replace("*", "").strip()
        votes.append(clean_line)
    return votes


def check_accuracy(file_path: str) -> None:
    """
    Calculates the accuracy of the predictions in the given file.
    Note that if there are more predictions than votes cast, it only looks at the first few predictions 
    (i.e. if there are 5 predictions but only 3 votes cast, only look at first 3 predictions). 
    """
    total_predictions = 0
    correct_predictions = 0

    with open(file_path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            player_names = data.get("PlayerNames", [])
            voting_outcome = data.get("votingOutcome", [])
            prediction_str = data.get("prediction", "")
            filename = data.get("filename", "unknown")

            predictions = clean_prediction(prediction_str)

            for i, vote_index in enumerate(voting_outcome):
                if isinstance(vote_index, str) and vote_index.strip().upper() == "N/A":
                    print(f"[{filename}] Skipping N/A vote for player {i}")
                    continue
                try:
                    vote_index = int(vote_index)
                    true_vote_name = player_names[vote_index]
                except (ValueError, IndexError):
                    print(f"\n[{filename}] Invalid vote index: {vote_index}")
                    print(f"  -> PlayerNames: {player_names}")
                    print(f"  -> VotingOutcome: {voting_outcome}")
                    print(f"  -> Full line: {data}")
                    continue

                model_vote_name = predictions[i] if i < len(predictions) else None
                total_predictions += 1

                if model_vote_name and model_vote_name.lower() == true_vote_name.lower():
                    print(predictions)
                    print(filename)
                    correct_predictions += 1

    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    print(f"Accuracy: {accuracy:.2%} ({correct_predictions}/{total_predictions} correct)")


#check_accuracy("werewolf.jsonl_gemini-2.0-flash-exp_deception_character_prediction")
check_accuracy("werewolf.jsonl_gpt-4o-audio-preview_deception_character_prediction")