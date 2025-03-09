import argparse
import os
import json
from typing import List
import pandas as pd
from pydub import AudioSegment
from pathlib import Path

root_dir = "../../data/SpeakerDiarization/"
absolutepath = "data/SpeakerDiarization/"

def get_meeting_ids() -> List[str]:
    corpus_path = os.path.join(root_dir, "AMI_meetings_Information")
    meeting_ids = set()
    
    # Ensure the directory exists
    if not os.path.exists(corpus_path):
        raise FileNotFoundError(f"Directory not found: {corpus_path}")
    
    # Iterate over CSV files in the directory
    for csv_file in os.listdir(corpus_path):
        if csv_file.endswith("_turns.csv"):  # Ensure we are processing the right files
            meeting_id = csv_file.split("_")[0]  # Extract the meeting ID
            meeting_ids.add(meeting_id)
    return sorted(meeting_ids)

def convert_to_unlabeled_transcribed_text(df) -> str:
    transcriptions = []
    label = 1
    for index, row in df.iterrows():
        speaker = row['speaker']
        start = row['start_time']
        end = row['end_time']
        words = row['words_ref']
        
        transcriptions.append(f"Sentence {label}: {words.strip()}")
        label += 1
    formatted_transcription = "\n".join(transcriptions)
    #print(formatted_transcription)
    return formatted_transcription


def process_meeting(audio_id, sentences = 20):
    """Process a single meeting's audio and CSV files"""
    
    audio_path = os.path.join(root_dir,"amicorpus", audio_id,"audio",f"{audio_id}.Mix-Headset.wav")
    csv_path = os.path.join(root_dir, "AMI_meetings_Information",f"{audio_id}_turns.csv")
    output_dir = os.path.join(root_dir, "audio",audio_id,f"#sentences={sentences}")
    #output_audio_path = os.path.join(output_dir, "meeting_cut.wav")
    #output_config_path = os.path.join(output_dir, "transcription_config.json")
    
    # Create output directory
    #os.makedirs(output_dir, exist_ok=True)
    
    try:
        audio = AudioSegment.from_wav(audio_path)
        df = pd.read_csv(csv_path)
        config_data = []

        filtered_df = df[(df['class'] != 'minor') & (df['duration'] > 0.5) & (df['word_count'] > 2)]
        label = 1
        for i in range(0, len(filtered_df), sentences):
            if i + sentences > len(filtered_df):
                break
            first_non_minor_row = filtered_df.iloc[i]
            start_time = first_non_minor_row['start_time']
            start_time = max(start_time - 3, 0)
            end_time = filtered_df.iloc[i + sentences -1]['end_time'] + 1
            output_audio_path = os.path.join(output_dir, str(label), "meeting_cut.wav")
            os.makedirs(os.path.dirname(output_audio_path), exist_ok=True)
            #transcript_with_speaker = convert_to_transcribed_text(filtered_df.iloc[i:i+sentences])
            valid_speakers = valid_speakers = set(filtered_df.iloc[i:i+sentences]['speaker'].unique())
            speaker_order = filtered_df.iloc[i:i+sentences]['speaker'].tolist()
            test_transcript = convert_to_unlabeled_transcribed_text(filtered_df.iloc[i:i+sentences])
            temp_output_dir = os.path.join("audio",audio_id,f"#sentences={sentences}")
            audio_segment = audio[start_time * 1000:end_time * 1000]
            audio_segment.export(output_audio_path, format="wav")
            output_audio_path = os.path.join(temp_output_dir, str(label), "meeting_cut.wav")
            # Create configuration data
            config_data.append({
                "meeting_id": audio_id,
                "label": label,
                "start_time": start_time,
                "end_time": end_time,
                #"transcript_with_speaker": transcript_with_speaker,
                "transcript_without_speaker": test_transcript,
                "filename": output_audio_path,
                "valid_speakers": list(valid_speakers),
                "speaker_order": speaker_order,
                "num_speakers": len(valid_speakers)
            })


            print(f"Processed {audio_id}")
            print(f"Audio saved at: {output_audio_path}")
            label += 1
            
        return config_data
            
    except Exception as e:
        print(f"Error processing {audio_id}: {str(e)}")
        return None

def process_all_meetings(sentences = 20):

    meetings = get_meeting_ids()
    # Process each meeting
    successful = 0
    failed = 0
    config_data = []
    for meeting in meetings:
        result = process_meeting(meeting, sentences)
        if result:
            config_data.extend(result)
            successful += 1
        else:
            failed += 1
    
    result_path = os.path.join(root_dir, "audio_inputs.jsonl")
    with open(result_path, "w") as json_file:
        for data in config_data:
            json.dump(data, json_file)
            json_file.write("\n")
    
    print(f"\nProcessing complete:")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Length of the dataset: {len(config_data)}")
    print(f"Results saved to: {result_path}")

def main(args):

    process_all_meetings(sentences=args.sentences)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process meetings with specified parameters')
    
    parser.add_argument('--sentences', type=int, default=20,
                        help='Number of sentences per datapoint')
    args = parser.parse_args()
    main(args)