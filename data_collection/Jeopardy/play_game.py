import asyncio
import base64
import os
import time
import random
import json
from queue import PriorityQueue
from dataclasses import dataclass, field
from threading import Thread
import numpy as np
import openai
from google import genai
from google.genai import types
from dotenv import load_dotenv
from pydub import AudioSegment
from gtts import gTTS
import librosa
from tqdm import tqdm
import pandas as pd
from models import *
import os, json, time, asyncio
import pandas as pd
from tqdm import tqdm

load_dotenv()


def read_jsonl(file_path):
    """
    Read a JSONL file and return its contents as a list of Python objects.
    """
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                if line.strip():  # Skip empty lines
                    item = json.loads(line)
                    data.append(item)
        print(f"Successfully read {len(data)} items from {file_path}")
        return data
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return []
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        return []
    except Exception as e:
        print(f"Unexpected error: {e}")
        return []

def run_competition(openai_key: str, gemini_key: str):
    dataset_path = "/nlp/scr/askhan1/jeopardy/CATS/data/jeopardy/audio_inputs.jsonl"

    jeopardy_questions = read_jsonl(dataset_path)

    model_handlers = {
        # "GPT-4o_realtime": OpenAIHandler(api_key=openai_key)
        "Gemini-2.0_multimodal": GeminiLiveHandler(api_key=gemini_key)
    }

    for name, model in model_handlers.items():
        results_file = f"results/{name}.json"

        # Load existing results if the file exists. this helps if your system stalls and you want to resume experiment from where you left off
        if os.path.exists(results_file):
            with open(results_file, "r") as f:
                all_rounds = json.load(f)
            # Set the round_index to the next new round.
            round_index = len(all_rounds)
            print(f"Loaded existing results with {round_index} rounds.")
        else:
            all_rounds = {}
            round_index = 0
            print("No existing results found; starting fresh.")

        c = 0
        for row in tqdm(jeopardy_questions, total=len(jeopardy_questions)):
            if c < round_index: # makes sure model 
                c += 1
                continue

            print("Global Round:", round_index, 'C:', c)
            questionpath = "/nlp/scr/askhan1/jeopardy/CATS/data/jeopardy/data/" + row['filename']
            question = row["question"]
            correct_answer = row["answer"]
            category = row['category']
            print(f"\nQuestion: {question} | Answer: {correct_answer}")


            round_result = model.process_file(questionpath, correct_answer, question)
            
            # Prepare round data with results for each model.
            round_data = {
                "question": question,
                "correct_answer": correct_answer,
                "category": category,
                "results": {f"{name}": round_result}
            }
            
            # If the round already exists, merge the new results with the existing ones.
            if str(round_index) in all_rounds:
                existing_round = all_rounds[str(round_index)]
                existing_round["results"].update(round_data["results"])
                all_rounds[str(round_index)] = existing_round
            else:
                all_rounds[str(round_index)] = round_data

            # Save the updated rounds to the JSON file.
            with open(results_file, "w") as f:
                json.dump(all_rounds, f, indent=4)
            print(f"Saved results for round {round_index} to {results_file}.")
            
            c += 1
            round_index += 1
            # Pause between rounds if necessary.
            time.sleep(5)

    print("processing finished and results saved.")



# --------------------------
# Main Execution
# --------------------------
if __name__ == "__main__":
    run_competition(os.getenv("OPENAI_API_KEY"), os.getenv("GOOGLE_API_KEY"))
