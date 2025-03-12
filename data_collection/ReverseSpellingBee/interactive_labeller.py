#!/usr/bin/env python
import argparse
import csv
import os
import sys
import time
import random
import threading
import numpy as np
import pandas as pd
import signal
import shutil
import sounddevice as sd
from dotenv import load_dotenv
from datasets import load_dataset
import curses
import io  # For handling bytes in audio data

# Global cache for audio files loaded from disk
audio_cache = {}
DEBUG = False  # Set via --debug flag
DEBUG_LOG_FILE = "debug.log"

# Global locks
audio_lock = threading.Lock()   # Ensure sequential audio calls
stop_lock = threading.Lock()    # Ensure only one thread stops audio at a time

def debug_print(msg):
    """Append debug messages to DEBUG_LOG_FILE with a timestamp, if debugging is enabled."""
    if DEBUG:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(DEBUG_LOG_FILE, "a") as f:
            f.write(f"[{timestamp}] {msg}\n")

# --- Debug and Audio Functions ---
def debug_audio_full(audio_item, label):
    debug_print(f"{label} FULL INFO: {audio_item}")

def debug_audio_info(audio_item, name):
    debug_print(f"{name} info: {audio_item}")

def load_audio_from_path(path):
    """Load an audio file from disk, caching the result."""
    debug_print(f"Loading audio from: {path}")
    if path in audio_cache:
        debug_print(f"Using cached audio for: {path}")
        return audio_cache[path]
    try:
        import soundfile as sf
        array, samplerate = sf.read(path, dtype='float32')
        data = {"array": array, "sampling_rate": samplerate, "path": path}
        audio_cache[path] = data
        debug_print(f"Loaded audio; samplerate={samplerate}, shape={np.array(array).shape}")
        return data
    except Exception as e:
        debug_print(f"Error loading audio from {path}: {e}")
        return None

class PlaybackHandle:
    """Wrapper for a playback thread using sounddevice."""
    def __init__(self, thread):
        self.thread = thread
    def is_playing(self):
        return self.thread.is_alive()
    def stop(self):
        debug_print("PlaybackHandle.stop() called.")
        with stop_lock:
            try:
                sd.stop()
                time.sleep(0.05)  # Allow a short time for buffers to clear
            except Exception as e:
                debug_print(f"sd.stop() error: {e}")

def play_audio_clip(audio_item):
    """
    Play an audio clip given an audio item.
    If audio_item is not a dict, treat it as a file path.
    If no "array" is present, attempt to load from "bytes" (if available),
    then fallback to loading from "path".
    """
    debug_print("play_audio_clip: Received audio_item: " + repr(str(audio_item))[:100])
    if not isinstance(audio_item, dict):
        audio_item = load_audio_from_path(audio_item)
        if audio_item is None:
            debug_print("play_audio_clip: Failed to load audio from file path.")
            return None

    if "array" not in audio_item or audio_item["array"] is None:
        found_bytes = False
        for k, v in audio_item.items():
            if isinstance(v, (bytes, bytearray)):
                try:
                    import soundfile as sf
                    with io.BytesIO(v) as f:
                        array, samplerate = sf.read(f, dtype='float32')
                    audio_item["array"] = array
                    audio_item["sampling_rate"] = samplerate
                    debug_print("play_audio_clip: Loaded audio from bytes under key " + k +
                                f"; samplerate={samplerate}, shape={np.array(array).shape}")
                    found_bytes = True
                    break
                except Exception as e:
                    debug_print("play_audio_clip: Failed to load from bytes under key " + k + f": {e}")
        if not found_bytes:
            if "path" in audio_item and audio_item["path"]:
                loaded = load_audio_from_path(audio_item["path"])
                if loaded:
                    audio_item.update(loaded)
        if "array" not in audio_item or audio_item["array"] is None:
            debug_print("play_audio_clip: No valid audio data in audio_item.")
            return None

    try:
        array = np.array(audio_item["array"])
        if np.any(np.isnan(array)):
            debug_print("play_audio_clip: Array contains NaN, replacing with zeros.")
            array = np.nan_to_num(array)
        samplerate = int(audio_item["sampling_rate"])
        debug_print(f"play_audio_clip: Starting playback with samplerate={samplerate}")
        def playback():
            with audio_lock:
                try:
                    sd.play(array, samplerate)
                    debug_print("play_audio_clip: sd.play() called; waiting for playback to finish.")
                    sd.wait()
                    debug_print("play_audio_clip: Playback finished.")
                except sd.PortAudioError as e:
                    if "-50" in str(e):
                        debug_print("Ignoring AUHAL error -50 (Mac CoreAudio conflict).")
                    else:
                        debug_print("Unexpected PortAudioError: " + str(e))
                except Exception as e:
                    debug_print("play_audio_clip: Playback exception: " + str(e))
        thread = threading.Thread(target=playback, daemon=True)
        thread.start()
        debug_print("play_audio_clip: Playback thread started.")
        return PlaybackHandle(thread)
    except Exception as e:
        debug_print("play_audio_clip: Exception processing audio_item: " + str(e))
        return None

def play_row_audio(audio1, audio2, stop_event):
    """
    Play two audio clips in sequence with a short pause between them.
    Before starting the second clip, check if the stop event is set.
    """
    play_handles = []
    debug_print("play_row_audio: Starting first audio clip.")
    handle1 = play_audio_clip(audio1)
    if handle1 is None:
        debug_print("play_row_audio: First audio clip failed.")
        return play_handles
    play_handles.append(handle1)
    while handle1.is_playing():
        if stop_event.is_set():
            debug_print("play_row_audio: Stop event set during first clip.")
            handle1.stop()
            return play_handles
        time.sleep(0.05)
    # Check stop event before starting second clip.
    if stop_event.is_set():
        debug_print("play_row_audio: Stop event set after first clip; not starting second clip.")
        return play_handles
    time.sleep(0.1)  # 100ms pause before second clip.
    debug_print("play_row_audio: Starting second audio clip.")
    if audio2 is None:
        return play_handles
    handle2 = play_audio_clip(audio2)
    if handle2 is not None:
        play_handles.append(handle2)
        while handle2.is_playing():
            if stop_event.is_set():
                debug_print("play_row_audio: Stop event set during second clip.")
                handle2.stop()
                return play_handles
            time.sleep(0.05)
    return play_handles

def clear_audio_queue(play_handles):
    """
    Stop and join all playback threads, then clear the queue.
    """
    debug_print("Clearing audio queue...")
    for handle in play_handles:
        if handle is not None:
            handle.stop()
            try:
                handle.thread.join(timeout=0.1)
            except Exception as e:
                debug_print(f"Error joining thread: {e}")
    play_handles.clear()
    debug_print("Audio queue cleared.")

# --- CSV Handling ---
def save_results_csv(results, results_file):
    fieldnames = ["word", "region", "OED", "label"]
    try:
        with open(results_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for res in results:
                writer.writerow(res)
        debug_print("CSV saved successfully.")
    except Exception as e:
        debug_print("Error saving CSV: " + str(e))

def load_results_csv(results_file):
    if not os.path.exists(results_file):
        debug_print("CSV file not found; starting with empty results.")
        return []
    results = []
    try:
        with open(results_file, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                results.append(row)
        debug_print(f"Loaded {len(results)} results from CSV.")
    except Exception as e:
        debug_print("Error loading CSV: " + str(e))
    return results

def print_row_curses(stdscr, row, current_index, total_rows):
    """
    Display a row's data (excluding audio) on the curses screen.
    """
    stdscr.clear()
    y = 0
    stdscr.addstr(y, 0, f"Labeling row {current_index+1} / {total_rows}")
    y += 2
    for key, value in row.items():
        if key not in {"audio", "GPT4o_pronunciation"}:
            stdscr.addstr(y, 0, f"{key}: {str(value).strip()}")
            y += 1
    y += 1
    stdscr.addstr(y, 0, "=" * 80)
    y += 1
    stdscr.addstr(y, 0, "Instructions:")
    y += 1
    stdscr.addstr(y, 0, "Left Arrow = previous, Right Arrow = next")
    y += 1
    stdscr.addstr(y, 0, "R = repeat, S = same, D = different, B = bad, Q = quit")
    stdscr.refresh()

def signal_handler(sig, frame):
    raise KeyboardInterrupt

# --- Main Interactive Loop with Curses ---
def main_curses(stdscr):
    # Initialize curses.
    curses.curs_set(0)  # Hide cursor.
    stdscr.keypad(True)
    debug_print("Starting main_curses.")

    # Load environment and parse arguments.
    load_dotenv()
    parser = argparse.ArgumentParser(
        description="Interactive labeling tool for annotating hard cases from a Hugging Face dataset."
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic behavior.")
    parser.add_argument("--huggingface_repo", type=str, default="MichaelR207/wiktionary_pronunciations-final",
                        help="Hugging Face repo id (default: '-final' dataset).")
    parser.add_argument("--results_file", type=str, default="michael_labels.csv",
                        help="CSV file to store labeling results.")
    parser.add_argument("--start_index", type=int, default=0,
                        help="Index to start labeling from (skipping previous ones).")
    parser.add_argument("--pos", type=str, default="Proper Noun,Noun",
                        help="Comma-separated list of prioritized parts of speech (default: 'Proper Noun,Noun').")
    parser.add_argument("--judge_match", action="store_true",
                        help="If set, filter for rows where gpt4o_correct is True (default: False).")
    parser.add_argument("--target", type=int, default=1000,
                        help="Target number of rows to mark as 'Different' (0 means no target limit).")
    parser.add_argument("--row_cap", type=int, default=0,
                        help="Maximum number of rows to process (0 means no cap).")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging.")
    args = parser.parse_args()
    global DEBUG
    DEBUG = args.debug
    debug_print("Arguments parsed.")

    signal.signal(signal.SIGINT, signal_handler)
    signal.siginterrupt(signal.SIGINT, True)
    random.seed(args.seed)
    debug_print("Random seed set.")

    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        stdscr.addstr("Error: HF_TOKEN not found in environment variables. Set it in your .env file.\n")
        stdscr.refresh()
        time.sleep(2)
        sys.exit(1)
    debug_print("HF_TOKEN found.")

    stdscr.addstr(f"Loading dataset from '{args.huggingface_repo}'...\n")
    stdscr.refresh()
    ds = load_dataset(args.huggingface_repo, split="train", token=hf_token)
    stdscr.addstr(f"Dataset loaded. Total rows: {ds.num_rows}\n")
    stdscr.refresh()
    debug_print(f"Dataset loaded with {ds.num_rows} rows.")

    df = ds.to_pandas()
    target_value = True if args.judge_match else False
    df = df[df["gpt4o_correct"] == target_value]
    debug_print(f"Filtered dataset on gpt4o_correct == {target_value}")

    pos_priority_list = [p.strip() for p in args.pos.split(",")]
    def get_priority(pos_list):
        pos_list_lower = [p.lower() for p in pos_list]
        for i, priority in enumerate(pos_priority_list):
            if priority.lower() in pos_list_lower:
                return i
        return len(pos_priority_list)
    df["priority"] = df["pos"].apply(get_priority)
    df_noun = df[df["priority"] < len(pos_priority_list)]
    df_non_noun = df[df["priority"] >= len(pos_priority_list)]
    df_noun = df_noun.sort_values(by=["priority", "word"], ascending=True)
    df_non_noun = df_non_noun.sort_values(by=["word"], ascending=True)
    df_filtered = pd.concat([df_noun, df_non_noun])
    debug_print(f"Filtered dataset contains {df_filtered.shape[0]} rows.")

    total_filtered = df_filtered.shape[0]
    stdscr.addstr(f"Filtered dataset contains {total_filtered} rows.\n")
    stdscr.addstr("Press any key to continue...\n")
    stdscr.refresh()
    stdscr.getch()

    rows = df_filtered.to_dict(orient="records")
    total_rows = len(rows)
    results = load_results_csv(args.results_file)
    current_index = max(args.start_index, len(results))
    different_count = sum(1 for r in results if r.get("label", "").lower() == "different")
    debug_print(f"Starting labeling loop at row {current_index} with {different_count} 'Different' labels.")

    stop_event = threading.Event()

    def play_current_audio(row):
        stop_event.clear()
        play_handles = []
        def playback():
            nonlocal play_handles
            debug_print("Starting play_row_audio for current row.")
            play_handles = play_row_audio(row["audio"], row["GPT4o_pronunciation"], stop_event)
            debug_print("play_row_audio finished; play_handles: " + repr(play_handles))
        thread = threading.Thread(target=playback, daemon=True)
        thread.start()
        # Wait briefly (up to 1 second) for play_handles to be populated.
        timeout = time.time() + 1.0
        while time.time() < timeout and not play_handles:
            time.sleep(0.01)
        debug_print("play_current_audio: Returning handles: " + repr(play_handles))
        return thread, play_handles

    # --- Main interactive labeling loop ---
    while (current_index < total_rows and
           (args.row_cap == 0 or current_index < args.row_cap) and
           (args.target == 0 or different_count < args.target)):
        row = rows[current_index]
        debug_print(f"Processing row {current_index+1}")
        print_row_curses(stdscr, row, current_index, total_rows)
        
        # Start audio playback.
        stop_event.clear()
        audio_thread, play_handles = play_current_audio(row)
        # Block indefinitely for key input.
        key = stdscr.getch()
        debug_print(f"Key received: {key}")
        # Flush any extra keys.
        curses.flushinp()
        
        # If too many audio files are queued, reset and replay the current row.
        if len(play_handles) > 2:
            debug_print("Too many audio files queued; repeating current row.")
            clear_audio_queue(play_handles)
            stdscr.addstr(0, 0, "Too many queued audio, repeating row. Please wait...")
            stdscr.refresh()
            time.sleep(1)
            continue
        
        debug_print("Key pressed; stopping audio playback.")
        stop_event.set()
        clear_audio_queue(play_handles)

        chosen_label = None
        # Process key input.
        if key == curses.KEY_LEFT:
            if current_index > 0:
                debug_print("Left key: moving to previous row.")
                current_index -= 1
            else:
                stdscr.addstr(0, 0, "Already at the first example.")
                stdscr.refresh()
                time.sleep(1)
            continue
        elif key == curses.KEY_RIGHT:
            chosen_label = "Skip"
            debug_print("Right key: skipping row.")
            current_index += 1
        elif key in (ord('r'), ord('R')):
            debug_print("R key: repeating row (no advancement).")
            chosen_label = None
        elif key in (ord('s'), ord('S')):
            chosen_label = "Same"
            debug_print("S key: marking row as 'Same'.")
            current_index += 1
        elif key in (ord('d'), ord('D')):
            chosen_label = "Different"
            debug_print("D key: marking row as 'Different'.")
            if row.get("gpt4o_correct") == False:
                different_count += 1
            current_index += 1
        elif key in (ord('b'), ord('B')):
            chosen_label = "Bad"
            debug_print("B key: marking row as 'Bad'.")
            current_index += 1
        elif key in (ord('q'), ord('Q')):
            debug_print("Q key: quitting.")
            break
        else:
            chosen_label = "Skip"
            debug_print("Unrecognized key; defaulting to 'Skip'.")
            current_index += 1

        if chosen_label is not None:
            res_entry = {
                "word": str(row.get("word", "")).strip(),
                "region": str(row.get("region", "")).strip(),
                "OED": str(row.get("OED", "")).strip(),
                "label": chosen_label
            }
            debug_print(f"Saving result for row {current_index}: {res_entry}")
            if current_index - 1 < len(results):
                results[current_index - 1] = res_entry
            else:
                results.append(res_entry)
            save_results_csv(results, args.results_file)

    stdscr.clear()
    stdscr.addstr("Labeling complete!\n")
    stdscr.addstr(f"Final 'Different' count: {different_count}\n")
    stdscr.addstr("Press any key to exit.")
    stdscr.refresh()
    stdscr.getch()
    debug_print("Exiting main loop.")

if __name__ == "__main__":
    curses.wrapper(main_curses)