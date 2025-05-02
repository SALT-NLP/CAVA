#!/usr/bin/env python
import sys
import os

from cava.data import prep_data_function_calling, filter_top_calls

if __name__ == "__main__":
    print("Starting function calling data preparation...")
    prep_data_function_calling()
    print("Function calling data preparation completed.")
    filter_top_calls("./data/function_calling_test/audio_inputs.jsonl")
