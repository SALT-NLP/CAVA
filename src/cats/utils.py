
def parse_next_speaker_response(response_text):
    '''
    This function will help to parse the output from LM which contains both reasoning and speaker label to the speaker label like "A" or "B"'''
    try:
        parts = response_text.split('\n')
        
        reasoning = ""
        speaker = ""
        
        # Parse each part
        for part in parts:
            if part.startswith("Reasoning:"):
                reasoning = part.replace("Reasoning:", "").strip()
            elif part.startswith("Speaker:"):
                speaker = part.replace("Speaker:", "").strip()
                speaker=speaker.replace(".", "")
        if not speaker or not reasoning:
            raise ValueError("Invalid response, can't parse speaker or reasoning")
        elif "A" not in speaker and "B" not in speaker and "C" not in speaker and "D" not in speaker and "E" not in speaker:
            raise ValueError(f"Invalid speaker: {speaker}") 
        print(speaker)       
        return speaker
import re

import numpy as np
from scipy.optimize import linear_sum_assignment


def get_jer_score(expected_value, predicted_value):
    """
    This is a function to calculate the Jaccard Error Rate for Speaker Diarization Task
    Given the ground truth speaker order(expected_value) and the parsed speaker order dictionary answered by the audio model, this function will return a Jaccard Error Rate from 0 to 1
    """
    ref_dict = {i + 1: speaker for i, speaker in enumerate(expected_value)}
    sys_dict = predicted_value
    random_dict = {i + 1: 1 for i, speaker in enumerate(expected_value)}
    try:
        if sys_dict[1] == "fail":
            sys_dict = random_dict
            print("use a random prediction")
    except:
        pass
    return calculate_jer(ref_dict, sys_dict)


def get_der_score(expected_value, predicted_value):
    """
    This is a function to calculate the simple version of Diarization Error Rate for Speaker Diarization Task (#miss-classified sentence/#total sentences)
    Given the ground truth speaker order(expected_value) and the parsed speaker order dictionary answered by the audio model, this function will return a DER from 0 to 1
    """
    ref_dict = {i + 1: speaker for i, speaker in enumerate(expected_value)}
    sys_dict = predicted_value
    random_dict = {i + 1: 1 for i, speaker in enumerate(expected_value)}
    try:
        if sys_dict[1] == "fail":
            sys_dict = random_dict
            print("use a random prediction")
    except:
        pass
    return calculate_der(ref_dict, sys_dict)


def calculate_jer(ref_dict, sys_dict):
    # Get unique speakers
    ref_speakers = sorted(set(ref_dict.values()))
    sys_speakers = sorted(set(sys_dict.values()))

    n_ref = len(ref_speakers)
    n_sys = len(sys_speakers)

    # Handle edge cases
    if n_ref == 0 and n_sys > 0:
        return 100.0
    elif n_ref > 0 and n_sys == 0:
        return 100.0
    elif n_ref == n_sys == 0:
        return 0.0

    # Create contingency matrix (intersection)
    cm = np.zeros((n_ref, n_sys))
    ref_counts = np.zeros(n_ref)
    sys_counts = np.zeros(n_sys)

    # Fill matrices
    for sent in ref_dict:
        ref_spk = ref_dict.get(sent)
        sys_spk = sys_dict.get(sent)
        if sys_spk == None:
            sys_spk = sys_speakers[0]  # randomly assign to the first speaker
        ref_idx = ref_speakers.index(ref_spk)
        sys_idx = sys_speakers.index(sys_spk)
        cm[ref_idx, sys_idx] += 1
        ref_counts[ref_idx] += 1
        sys_counts[sys_idx] += 1

    # Calculate JER for each possible pairing
    ref_durs = np.tile(ref_counts, [n_sys, 1]).T
    sys_durs = np.tile(sys_counts, [n_ref, 1])
    intersect = cm
    union = ref_durs + sys_durs - intersect

    # Avoid division by zero
    # union == 0  => intersection == 0
    # wont happen because each speaker has at least one sentence though
    union[union == 0] = 1
    jer_speaker = 1 - (intersect / union)

    # Find optimal mapping using Hungarian algorithm
    ref_speaker_inds, sys_speaker_inds = linear_sum_assignment(jer_speaker)  # O(n^3)
    # print(ref_speaker_inds, sys_speaker_inds)

    # Calculate JER for each reference speaker
    jers = np.ones(n_ref, dtype="float64")
    for ref_idx, sys_idx in zip(ref_speaker_inds, sys_speaker_inds):
        jers[ref_idx] = jer_speaker[ref_idx, sys_idx]
        # print(jers[ref_idx])
        # print(cm[ref_idx, sys_idx], ref_counts[ref_idx], sys_counts[sys_idx])
        # print(union[ref_idx, sys_idx])

    return float(np.mean(jers))


def calculate_der(ref_dict, sys_dict):
    """
    Calculate the Diarization Error Rate (DER) between reference and system speaker diarization.

    DER is defined as the fraction of time that is not attributed correctly to a speaker.
    It's calculated as: (false_alarm + missed_detection + speaker_confusion) / total_time

    For sentence-level diarization, we can simplify this to be the proportion of sentences
    that are not correctly attributed after finding the optimal mapping.

    Args:
        ref_dict: Dictionary mapping sentence IDs to speaker IDs in the reference
        sys_dict: Dictionary mapping sentence IDs to speaker IDs in the system output

    Returns:
        float: The Diarization Error Rate (0.0 to 1.0, lower is better)
    """
    # Get unique speakers
    ref_speakers = sorted(set(ref_dict.values()))
    sys_speakers = sorted(set(sys_dict.values()))

    n_ref = len(ref_speakers)
    n_sys = len(sys_speakers)

    # Handle edge cases
    if n_ref == 0 and n_sys > 0:
        return 1.0  # All false alarms
    elif n_ref > 0 and n_sys == 0:
        return 1.0  # All missed detections
    elif n_ref == n_sys == 0:
        return 0.0  # Both empty, perfect match

    # Create confusion matrix
    cm = np.zeros((n_ref, n_sys))

    # Fill the confusion matrix
    for sent in ref_dict:
        ref_spk = ref_dict.get(sent)
        sys_spk = sys_dict.get(sent)
        if sys_spk == None:
            sys_spk = sys_speakers[0]  # randomly assign to the first speaker
        ref_idx = ref_speakers.index(ref_spk)
        sys_idx = sys_speakers.index(sys_spk)
        cm[ref_idx, sys_idx] += 1

    neg_cm = -cm
    total_sentences = len(ref_dict)

    ref_inds, sys_inds = linear_sum_assignment(neg_cm)

    # Calculate the number of correctly classified sentences with optimal mapping
    correct_sentences = sum(cm[i, j] for i, j in zip(ref_inds, sys_inds))

    # Calculate DER
    der = 1.0 - (correct_sentences / total_sentences)

    return der


def parse_speaker_label_response(response_text):
    # This function will help to parse the output from LM to a dictionary that maps from the sentence index to the speaker label
    try:
        speaker_labels = {}

        lines = [line.strip().lower() for line in response_text.strip().split("\n") if line.strip()]

        pattern = re.compile(r"^sentence\s*(\d+)\s*:\s*speaker\s*(\d+)[\.\s]*$", re.IGNORECASE)

        for line in lines:
            match = pattern.match(line)
            if match:
                sentence_num, speaker_num = match.groups()
                speaker_labels[int(sentence_num)] = int(speaker_num)  # Convert to integers
            else:
                pass
                # print(f"Warning: Could not parse line -> '{line}'")

        if not speaker_labels or len(speaker_labels) <= 10:  # if parsing failed, use more robust method
            speaker_labels = {}
            current_sentence = None
            for i, line in enumerate(lines):
                if "sentence" in line.lower() and any(char.isdigit() for char in line):
                    # Extract the sentence number
                    parts = line.lower().split("sentence")
                    if len(parts) > 1:
                        digits = "".join(char for char in parts[1] if char.isdigit())
                        if digits:
                            sentence_num = int(digits)
                            current_sentence = sentence_num

                            # Check if this line also contains a speaker label
                            if "speaker" in line.lower() and any(char.isdigit() for char in line.split("speaker")[1]):
                                speaker_parts = line.lower().split("speaker")
                                speaker_digits = "".join(char for char in speaker_parts[1] if char.isdigit())
                                if speaker_digits:
                                    speaker_num = int(speaker_digits)
                                    speaker_labels[current_sentence] = speaker_num
                                    current_sentence = None  # Reset current sentence

                # If the previous sentence had no speaker label and the current line has no sentence number, check this line for a speaker label
                elif current_sentence is not None and "speaker" in line.lower():
                    speaker_parts = line.lower().split("speaker")
                    if len(speaker_parts) > 1:
                        speaker_digits = "".join(char for char in speaker_parts[1] if char.isdigit())
                        if speaker_digits:
                            speaker_num = int(speaker_digits)
                            speaker_labels[current_sentence] = speaker_num
                            current_sentence = None  # Reset current sentence

        if not speaker_labels:
            print("Warning: No speaker labels found in response")
            print(response_text)
        return (
            speaker_labels if speaker_labels else {1: "fail"}
        )  # dont want to return empty dictionary, but a dictionary that indicate failure

    except Exception as e:
        print(f"Error parsing response: {e}")
        return None
