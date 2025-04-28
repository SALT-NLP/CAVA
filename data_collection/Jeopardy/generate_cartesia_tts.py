import json
import os
import requests
import random
import time
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv
from datasets import Dataset, Audio, DatasetDict
import openai
import argparse

# Load environment variables
load_dotenv()

# Use the UUIDs provided as voice IDs
CARTESIA_VOICES = [
    "c99d36f3-5ffd-4253-803a-535c1bc9c306",
    "bf0a246a-8642-498a-9950-80c35e9276b5",
    "78ab82d5-25be-4f7d-82b3-7ad64e5b85b2",
    "6f84f4b8-58a2-430c-8c79-688dad597532",
    "c8605446-247c-4d39-acd4-8f4c28aa363c",
    "00967b2f-88a6-4a31-8153-110a92134b9f",
    "0c8ed86e-6c64-40f0-b252-b773911de6bb",
    "146485fd-8736-41c7-88a8-7cdd0da34d84",
    "9fa83ce3-c3a8-4523-accc-173904582ced",
    "17ab4eb9-ef77-4a31-85c5-0603e9fce546",
    "c378e743-e7dc-49da-b9ce-8377b543acdd",
    "5abd2130-146a-41b1-bcdb-974ea8e19f56",
    "f4e8781b-a420-4080-81cf-576331238efa",
    "23e9e50a-4ea2-447b-b589-df90dbb848a2",
    "58db94c7-8a77-46a7-9107-b8b957f164a0",
    "ab109683-f31f-40d7-b264-9ec3e26fb85e",
    "6adbb439-0865-468c-9e68-adbb0eb2e71c",
    "81db94f2-ea76-4e5a-94bf-c92be997270d",
    "607167f6-9bf2-473c-accc-ac7b3b66b30b",
    "87bc56aa-ab01-4baa-9071-77d497064686",
    "97f4b8fb-f2fe-444b-bb9a-c109783a857a",
    "729651dc-c6c3-4ee5-97fa-350da1f88600",
    "f9836c6e-a0bd-460e-9d3c-f7299fa60f94",
    "7fe6faca-172f-4fd9-a193-25642b8fdb07",
    "2a4d065a-ac91-4203-a015-eb3fc3ee3365",
    "3cbf8fed-74d5-4690-b715-711fcf8d825f",
    "87177869-f798-48ae-870f-e07d0c960a1e",
    "34bde396-9fde-4ebf-ad03-e3a1d1155205",
    "daf747c6-6bc2-4083-bd59-aa94dce23f5d",
    "996a8b96-4804-46f0-8e05-3fd4ef1a87cd",
    "13524ffb-a918-499a-ae97-c98c7c4408c4",
    "41f3c367-e0a8-4a85-89e0-c27bae9c9b6d",
    "63ff761f-c1e8-414b-b969-d1833d1c870c",
]


def generate_prefixed_answer(question, answer):
    """
    Use OpenAI API to generate a proper answer with the correct prefix (Who is, What is, etc.)
    """
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        print("Warning: OpenAI API key not provided. Using fallback prefix.")
        return f"What is {answer}"

    client = openai.OpenAI(api_key=openai_api_key)
    tries = 0
    answer = answer.replace('"', "")
    while tries < 10:
        tries += 1
        response = client.chat.completions.create(
            model="gpt-4o-mini-2024-07-18",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an assistant that reformats trivia answers to include appropriate question prefixes according to Jeopardy standards. "
                        "Return ONLY the answer with the correct prefix. The original answer MUST be in the reformatted version. "
                        "For example if the answer is 'Answer: LLM', just write 'What is an LLM?'."
                    ),
                },
                {"role": "user", "content": f"Answer: {answer}"},
            ],
            temperature=1,
            max_tokens=50,
        )
        rewrite = response.choices[0].message.content.strip()
        print(answer, rewrite)
        if answer.lower() in rewrite.lower():
            break
    return rewrite


def generate_speech(text, output_file, voice_id):
    """Generate speech from text and save to file."""
    cartesia_api_key = os.getenv("CARTESIA_API_KEY")
    if not cartesia_api_key:
        raise ValueError("Cartesia API key is required. Set CARTESIA_API_KEY environment variable.")

    # Set up headers and payload
    headers = {"Cartesia-Version": "2024-11-13", "X-API-Key": cartesia_api_key, "Content-Type": "application/json"}

    payload = {
        "model_id": "sonic-2",
        "transcript": text,
        "voice": {"mode": "id", "id": voice_id},
        "language": "en",
        "output_format": {"container": "wav", "sample_rate": 44100, "encoding": "pcm_s16le"},
    }

    # Make API request
    response = requests.post("https://api.cartesia.ai/tts/bytes", json=payload, headers=headers)

    # Check for errors
    if response.status_code != 200:
        error_msg = f"Error: {response.status_code} - {response.text}"
        print(error_msg)
        return False

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Save audio file
    with open(output_file, "wb") as f:
        f.write(response.content)

    return True


def process_jsonl(input_file, output_dir):
    """Process JSONL file and generate audio for answers with proper prefixes."""
    os.makedirs(output_dir, exist_ok=True)

    # Read all lines from the JSONL file
    with open(input_file, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    # Process each line
    dataset_items = []
    for line_idx, line in enumerate(tqdm(lines, desc="Generating audio")):
        data = json.loads(line)

        # Get question and answer
        if "question" not in data or "answer" not in data:
            print(f"Warning: Required fields not found in line {line_idx+1}. Skipping.")
            continue

        question = data["question"]
        answer = data["answer"]

        # Generate prefixed answer using OpenAI
        prefixed_answer = data["prefixed_answer"]  # generate_prefixed_answer(question, answer)

        # Create output filename
        filename = data.get("filename", f"{line_idx}.wav")
        output_path = os.path.join(output_dir, filename)

        # Randomly select voice
        voice_id = random.choice(CARTESIA_VOICES)

        # Generate speech
        success = generate_speech(question, output_path, voice_id)

        # Only add to dataset if speech generation was successful
        if success:
            # Store item for dataset creation
            item = {k: v for k, v in data.items()}
            item["audio"] = output_path
            item["voice_id"] = voice_id
            item["prefixed_answer"] = prefixed_answer
            dataset_items.append(item)

        # Add a small delay to avoid hitting rate limits
        time.sleep(0.1)

    return dataset_items


def create_and_push_dataset(dataset_items, repo_id):
    """Create a Hugging Face dataset and push it to the Hub."""
    # Create the dataset
    dataset = Dataset.from_list(dataset_items)

    # Cast the audio column to Audio feature
    dataset = dataset.cast_column("audio", Audio())

    dataset_dict = DatasetDict({"combined": dataset})
    # Push to the Hub
    dataset_dict.push_to_hub(
        repo_id, commit_message="Upload dataset with Cartesia TTS audio samples and prefixed answers"
    )

    print(f"Dataset pushed to https://huggingface.co/datasets/{repo_id}")
    return dataset_dict


def main():
    parser = argparse.ArgumentParser(
        description="Generate TTS audio files from a JSONL file and upload to Hugging Face"
    )
    parser.add_argument("input_file", help="Input JSONL file")
    parser.add_argument("repo_id", help="Hugging Face dataset repository ID (username/dataset-name)")
    parser.add_argument("--output-dir", default="outputs", help="Output directory for audio files")

    args = parser.parse_args()

    # Process JSONL and create audio files
    dataset_items = process_jsonl(
        args.input_file,
        args.output_dir,
    )

    # Create and push dataset to HF Hub
    create_and_push_dataset(dataset_items, args.repo_id)


if __name__ == "__main__":
    main()
