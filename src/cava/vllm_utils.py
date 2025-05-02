from pathlib import Path
import subprocess
import os
import time
import atexit
import signal
import sys
import torch
import threading
from math import log2, floor

# List to keep track of subprocesses
processes = []


def get_optimal_tp_size():
    """
    Determine the optimal tensor parallel size based on available GPUs.
    Returns the largest power of 2 that doesn't exceed the GPU count.
    """
    gpu_count = torch.cuda.device_count()
    if gpu_count <= 1:
        return 1

    # Find the largest power of 2 that doesn't exceed gpu_count
    return 2 ** floor(log2(gpu_count))


def start_vllm_server(model_name, tool_call_parser="hermes", LOG_PATH="VLLM_LOGS"):
    """
    Start a vLLM server with automatic GPU allocation and fixed port 8001.
    The function waits until the server outputs "Application startup complete."

    Args:
        model_name: The name or path of the model to load
        torch_dtype: Data type for torch operations

    Returns:
        process: The server process
    """
    # Set up environment
    env = os.environ.copy()
    tp_size = get_optimal_tp_size()

    # Build the command
    command = [
        "vllm",
        "serve",
        model_name,
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        tool_call_parser,
        "--api-key",
        "cava",
    ]

    # Add tensor parallelism if using multiple GPUs
    if tp_size > 1:
        command.extend(["--tensor-parallel-size", str(tp_size)])

    # Prepare for logging and process monitoring
    model_save_name = model_name.split("/")[-1]
    Path(LOG_PATH).mkdir(parents=True, exist_ok=True)
    log_file = f"{LOG_PATH}/vllm_server_{model_save_name}.log"

    # Event to signal when the server is ready
    server_ready = threading.Event()

    # Function to monitor the log file for the startup completion message
    def monitor_log():
        with open(log_file, "r") as log:
            while True:
                line = log.readline()
                if not line:
                    time.sleep(0.1)  # Short sleep to avoid busy waiting
                    continue

                if "Application startup complete" in line:
                    server_ready.set()
                    print(f"Server for model {model_name} is ready!")
                    break

    # Start the subprocess with output redirection to log file
    with open(log_file, "w") as log:
        process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)

    processes.append(process)

    # Start the log monitoring thread
    monitor_thread = threading.Thread(target=monitor_log)
    monitor_thread.daemon = True  # Thread will exit when main program exits
    monitor_thread.start()

    print(
        f"Starting vLLM server with model {model_name} using all available GPUs with tensor parallelism size {tp_size}"
    )
    print(f"Process ID: {process.pid}")
    print("Waiting for server initialization...")

    # Wait for the server to be ready
    server_ready.wait()  # This will block until the server_ready event is set

    return process
