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
import wave
import whisper 

# --------------------------
# Constants and Utilities
# --------------------------
SAMPLE_RATE = 24000
FRAME_SIZE = 240

def get_frames_from_file(file_path, frame_size=FRAME_SIZE):
    """Convert an audio file to frames."""
    audio = AudioSegment.from_file(file_path).set_frame_rate(SAMPLE_RATE).set_channels(1)
    samples = np.array(audio.get_array_of_samples(), dtype=np.int16)
    frames = []
    num_frames = len(samples) // frame_size
    for i in range(num_frames):
        start = i * frame_size
        end = start + frame_size
        frames.append((SAMPLE_RATE, samples[start:end].reshape(1, -1)))
    remainder = len(samples) % frame_size
    if remainder:
        leftover = samples[-remainder:]
        padded = np.pad(leftover, (0, frame_size - remainder))
        frames.append((SAMPLE_RATE, padded.reshape(1, -1)))
    return frames

def encode_audio(sample_rate, data):
    """Encode a single frame of audio data into base64 PCM."""
    segment = AudioSegment(
        data.tobytes(),
        frame_rate=sample_rate,
        sample_width=data.dtype.itemsize,
        channels=1
    )
    pcm_audio = segment.set_frame_rate(SAMPLE_RATE).set_channels(1).set_sample_width(2).raw_data
    return base64.b64encode(pcm_audio).decode("utf-8")

def check_answer(transcribed_answer, correct_answer):
    """Use GPT-4 to evaluate if the transcribed answer is correct."""
    print(f"CHECKING ANSWER: {transcribed_answer} | {correct_answer}")

    prompt = f"""
        You are an expert Jeopardy evaluator.
        Determine if the contestant's answer is correct. The answer must be worded in the form of a question, beginning with “What is” or “Who are,” for example."
        Correct Answer: "{correct_answer}"
        Contestant Answer: "{transcribed_answer}"
        Provide only "Correct" or "Incorrect" as output.
    """
    try:
        client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = client.chat.completions.create(
            model="o3-mini",
            messages=[{"role": "user", "content": prompt}],
            reasoning_effort="high"
        )
        llm_response = response.choices[0].message.content.strip()
        is_correct = llm_response.lower() == "correct"
        return is_correct, 0.0
    except Exception as e:
        print(f"Error in check_answer: {e}")
        return False, "Error"

# Model Handler Base Class
class BaseModelHandler:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.correct_answer = None

    def process_file(self, input_data: str, correct_answer: str, text_input: str = None) -> dict:
        """
        input_data: for audio models, the audio file path; for text models, this can be ignored.
        correct_answer: the expected answer.
        text_input: the question text for text-text model implementation. can be ignored for now.
        """
        raise NotImplementedError("Subclasses must implement process_file.")


# gpt40 realtime handler
class OpenAIHandler(BaseModelHandler):
    def __init__(self, api_key: str):
        super().__init__(api_key)
        self.client = openai.Client(api_key=self.api_key)
        print("OpenAI (audio) handler initialized.")

    async def _process_file_async(self, input_data: str) -> dict:
        """
        async helper that wraps the synchronous processing code.
        This method does NOT perform multiple attempts; it is expected that
        `process_file` handles timeouts/retries.
        """

        with self.client.beta.realtime.connect(model="gpt-4o-mini-realtime-preview-2024-12-17") as conn:
            print("OpenAIHandler: Connected")
            # Update the session with turn detection and instructions.
            conn.session.update(
                session={
                    "turn_detection": {"type": "server_vad"},
                    "instructions": (
                        "You are a contestant on Jeopardy. the answer must be worded in the form of a question, beginning with “What is” or “Who are,” for example."
                    ),
                }
            )

            # Extract frames from the input file.
            frames = get_frames_from_file(input_data, 320)
            i = 0
            for sample_rate, array in frames:
                audio_message = encode_audio(sample_rate, array.squeeze())
                conn.input_audio_buffer.append(audio=audio_message)
                await asyncio.sleep(0.01)

                if i == len(frames) - 1:
                    start_time = time.time()


                i += 1

            print("OpenAIHandler: Sent frames")

            final_response = ""
            processing_time = None

            # Process events from the realtime connection.
            for event in conn:
                print(event.type)
                if event.type == "response.created" and processing_time is None:
                    processing_time = time.time() - start_time
                if event.type == "response.audio_transcript.done":
                    final_response = event.transcript
                elif event.type == "response.done":
                    break

            final_response = final_response.strip()
            print("OpenAIHandler: Audio transcription done.")

            return {
                "transcript": final_response,
                "audio_path": "gpt_assistant_output.wav",
                "processing_time": f"{processing_time:.6f}" if processing_time else "N/A",
                "success": True
            }


    def process_file(self, input_data: str, correct_answer: str, text_input: str = None) -> dict:
        """
        Synchronous wrapper for the asynchronous live-processing and transcription.
        'input_data' is the path to the input WAV file.
        'correct_answer' is used to check the response (if applicable).
        Implements a retry mechanism (up to 3 attempts) with a timeout.
        """
        attempts = 0
        last_exception = None
        while attempts < 3:
            try:
                # Set an overall timeout for the asynchronous processing.
                result = asyncio.run(asyncio.wait_for(self._process_file_async(input_data), timeout=30))
                if not result.get("transcript"):
                    raise Exception("Empty transcript received.")
                is_correct = False
                if correct_answer:
                    is_correct, _ = check_answer(result["transcript"], correct_answer)
                result["correct"] = is_correct
                return result
            except asyncio.TimeoutError:
                print(f"OpenAIHandler: Timeout error. Attempt {attempts + 1} of 3.")
                attempts += 1
            except Exception as e:
                print(f"OpenAIHandler: Exception encountered: {e}. Attempt {attempts + 1} of 3.")
                last_exception = e
                attempts += 1
        return {
            "transcript": "",
            "audio_path": "",
            "processing_time": "Error",
            "success": False,
            "error": f"Failed after 3 attempts: {last_exception}"
        }




# Model handler for gemini 2.0 flash-exp
class GeminiLiveHandler(BaseModelHandler):
    def __init__(self, api_key: str):
        super().__init__(api_key)
        self.client = genai.Client(api_key=api_key, http_options={"api_version": "v1alpha"})
        self.model_id = "gemini-2.0-flash-exp"
        preloaded_instruction = (
            "You are a contestant on Jeopardy. the answer must be worded in the form of a question, beginning with “What is” or “Who are,” for example."
        )
        instruction_content = types.Content(parts=[types.Part(text=preloaded_instruction)])
        self.config = {
            "response_modalities": ["AUDIO"],
            "system_instruction": instruction_content,
        }

        self.pcm_file = "output.pcm"
        self.wav_file = "gemini_multimodel_output.wav"
        self.whisper_model = whisper.load_model("large-v3")
        print("Gemini Live audio handler initialized.")

    @staticmethod
    def convert_pcm_to_wav(pcm_file, wav_file, sample_rate=24000, num_channels=1, sample_width=2):
        """Convert accumulated PCM data into a WAV file."""
        with open(pcm_file, "rb") as pcm:
            pcm_data = pcm.read()
        with wave.open(wav_file, "wb") as wav:
            wav.setnchannels(num_channels)
            wav.setsampwidth(sample_width)
            wav.setframerate(sample_rate)
            wav.writeframes(pcm_data)
        print(f"Converted {pcm_file} to {wav_file}")

    async def _process_file_async(self, input_data: str) -> dict:
        print("GeminiLiveHandler: Starting asynchronous processing.")
        # Remove any existing PCM file.
        if os.path.exists(self.pcm_file):
            os.remove(self.pcm_file)

        try:
            frames = get_frames_from_file(input_data, frame_size=320)
            print(f"GeminiLiveHandler: Total frames extracted: {len(frames)}")
        except Exception as e:
            return {
                "transcript": "",
                "audio_path": None,
                "processing_time": "Error",
                "success": False,
                "error": f"Error reading input WAV file: {e}"
            }

        print("GeminiLiveHandler: Connecting to Gemini (live mode)...")
        processing_time = None
        async with self.client.aio.live.connect(model=self.model_id, config=self.config) as session:
            # Send each frame to the live session.
            for i, (frame_rate, frame_array) in enumerate(frames):
                frame_bytes = frame_array.tobytes()
                is_last = (i == len(frames) - 1)
                audio_input = types.LiveClientRealtimeInput(
                    media_chunks=[types.Blob(data=frame_bytes, mime_type="audio/pcm")]
                )
                await session.send(input=audio_input, end_of_turn=is_last)
                await asyncio.sleep(0.01)

                if is_last:
                    start_time = time.time()
                

            print("GeminiLiveHandler: Done sending frames.")
            is_response_complete = False
            received_response = False
            got_response = False
            while not is_response_complete:
                async for response in session.receive():
                    print(response.server_content)
                    if response.server_content and response.server_content.model_turn:
                        if not received_response:
                            processing_time = time.time() - start_time
                            received_response = True
                        for part in response.server_content.model_turn.parts:
                            if part.inline_data and part.inline_data.data:
                                audio_data = part.inline_data.data
                                with open(self.pcm_file, "ab") as f:
                                    f.write(audio_data)
                    if received_response and response.server_content and response.server_content.turn_complete:
                        is_response_complete = True
                        if response.text:
                            print("GeminiLiveHandler >", response.text)
                        break

            self.convert_pcm_to_wav(self.pcm_file, self.wav_file, sample_rate=SAMPLE_RATE)
            print("GeminiLiveHandler: Audio output received and converted.")
            print(f"GeminiLiveHandler: Processing time: {processing_time} seconds")

        print("GeminiLiveHandler: Starting transcription of the generated audio using Whisper...")
        try:
            result = self.whisper_model.transcribe(self.wav_file)
            transcript = result["text"]
            print("GeminiLiveHandler: Transcription result:", transcript)
        except Exception as e:
            print("GeminiLiveHandler: Error during transcription:", e)
            transcript = ""

        return {
            "transcript": transcript.strip(),
            "audio_path": self.wav_file,
            "processing_time": f"{processing_time:.6f}" if processing_time else "N/A",
            "success": True
        }

    def process_file(self, input_data: str, correct_answer: str, text_input: str = None) -> dict:
        """
        Synchronous wrapper for the asynchronous live-processing and transcription.
        'input_data' is the path to the input WAV file.
        'correct_answer' is used to check the response (if applicable).
        Implements a retry mechanism (up to 3 attempts) with a timeout.
        """
        attempts = 0
        last_exception = None
        while attempts < 3:
            try:
                # Set an overall timeout for the asynchronous processing.
                result = asyncio.run(asyncio.wait_for(self._process_file_async(input_data), timeout=30))
                if not result.get("transcript"):
                    raise Exception("Empty transcript received.")
                is_correct = False
                if correct_answer:
                    is_correct, _ = check_answer(result["transcript"], correct_answer)
                result["correct"] = is_correct
                return result
            except asyncio.TimeoutError:
                print(f"GeminiLiveHandler: Timeout error. Attempt {attempts + 1} of 3.")
                attempts += 1
            except Exception as e:
                print(f"GeminiLiveHandler: Exception encountered: {e}. Attempt {attempts + 1} of 3.")
                last_exception = e
                attempts += 1
        return {
            "transcript": "",
            "audio_path": None,
            "processing_time": "Error",
            "success": False,
            "error": f"Failed after 3 attempts: {last_exception}"
        }


if __name__ == "__main__":
    load_dotenv()
    model = GeminiLiveHandler(api_key=os.getenv("GOOGLE_API_KEY"))
    result = model.process_file("input.wav", "apple")
    print("Final result:", result)
