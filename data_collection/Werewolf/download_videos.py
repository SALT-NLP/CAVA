"""
download_videos.py contains the code which downloads and splits the Werewolf videos, before creating an audio dataset 
which can be uploaded to huggingface.
"""
import os
import json
import ffmpeg
import yt_dlp
import librosa
import soundfile as sf
import pandas as pd
from datasets import load_dataset

JSON_folder = "vote_outcome_youtube_released"
output_parent_folder = "output_videos"
url_mapping_file = "youtube_urls_released.json"

HF_dataset_folder = "hf_audio_dataset"
audio_folder = os.path.join(HF_dataset_folder, "data")
metadata_file = os.path.join(HF_dataset_folder, "metadata.csv")

os.makedirs(audio_folder, exist_ok=True)


def load_urls():
    with open(url_mapping_file, "r") as file:
        return json.load(file)


def download_videos(url, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    output_filename = os.path.join(output_folder, "original_video.mp4")

    ydl_opts = {
        "format": "best[ext=mp4]",
        "outtmpl": output_filename
    }
    video_url = f"https://www.youtube.com/watch?v={url}"
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([video_url])
        print(f"Downloaded video: {output_filename}")
        return output_filename
    except Exception as e:
        print(f"Error downloading video: {e}")
        return None  # Return None if download fails


def trim_video(input_file, start_time, end_time, output_file):
    
    ffmpeg.input(input_file, ss=start_time, to=end_time).output(
        output_file, codec="copy").run(overwrite_output=True)
    print(f"Video trimmed and saved to {output_file}")


def process_from_json(json_file, url):
    with open(json_file, "r") as file:
        games_data = json.load(file)
    
    json_filename = os.path.basename(json_file).replace(".json", "")
    output_folder = os.path.join(output_parent_folder, json_filename)

    os.makedirs(output_folder, exist_ok=True)

    video_file = download_videos(url, output_folder)
    if video_file is None:
        print(f"Error downloading Youtube link for JSON file: {json_file}")
        return
    
    game_entries = {k: v for k, v in games_data.items() if k.startswith("Game") and "startTime" in v and "endTime" in v}
    sorted_games = sorted(game_entries.items(), key=lambda x: int(x[0].replace("Game", "")))
    
    iteration = 1
    for game_name, game_info in sorted_games:
        print(game_info.keys())
        start_time = game_info["startTime"]
        end_time = game_info["endTime"]
        output_file_name = f"game_{iteration}.mp4"
        output_file = os.path.join(output_folder, output_file_name)
        trim_video(video_file, start_time, end_time, output_file)
        iteration += 1


def process_json_files():
    video_urls = load_urls()
    json_files = [os.path.join(JSON_folder, file) for file in os.listdir(JSON_folder)]
    for file in json_files:
        json_name = os.path.basename(file).replace(".json", "")
        print(json_name)
        video_url = video_urls.get(json_name)
        process_from_json(file, video_url)
    print("JSON files processed")


def upload_hugging_face():
    metadata = []

    for json_file in os.listdir(JSON_folder):
        if not json_file.endswith(".json"):
            continue

        json_path = os.path.join(JSON_folder, json_file)

        if not os.path.exists(json_path):
            print(f"Skipping {json_path} as not found")
            continue


        with open(json_path, "r") as file:
            games_data = json.load(file)

        json_filename = os.path.basename(json_file).replace(".json", "")

        output_folder = os.path.join(output_parent_folder, json_filename)
        if not os.path.exists(output_folder):
            print(f"no matching audio folder: {json_filename}")
            continue

        sorted_games = sorted(
    ((k, v) for k, v in games_data.items() if k.startswith("Game") and "startTime" in v and "endTime" in v),
    key=lambda x: int(x[0].replace("Game", ""))
)

        iteration = 1
        for game_name, game_info in sorted_games:
            if "startTime" not in game_info or "endTime" not in game_info:
                print(f"Skipping {game_name} in {json_file} due to missing startTime or endTime.")
                continue

            start_time = game_info["startTime"]
            end_time = game_info["endTime"]
            player_names = json.dumps(game_info["playerNames"])
            voting_outcome = json.dumps(game_info["votingOutcome"])
            start_roles = json.dumps(game_info["startRoles"])
            end_roles = json.dumps(game_info["endRoles"])
            warning = json.dumps(game_info["warning"])

            audio_filename = os.path.join(output_folder, f"game_{iteration}.mp4")
            if not os.path.exists(audio_filename):
                print(f"{audio_filename} not found")
                continue
            
            wav_file = os.path.join(audio_folder, f"{json_filename}_game_{iteration}.wav")
            y, sr = librosa.load(audio_filename, sr=None)
            sf.write(wav_file, y, sr)

            metadata.append([f"data/{json_filename}_game_{iteration}.wav", 
                             start_time,
                             end_time,
                             player_names,
                             voting_outcome,
                             start_roles,
                             end_roles,
                             warning
                             ])
            iteration += 1
            
    df = pd.DataFrame(metadata, columns=["file_name", "startTime", "endTime", "playerName", "votingOutcome", "startRoles", 
                                         "endRoles", "warning"])
    df.to_csv(metadata_file, index=False)
    print("Hugging Face Dataframe prepared")


process_json_files()
upload_hugging_face()

dataset = load_dataset("audiofolder", data_dir=HF_dataset_folder)
print(dataset)



    