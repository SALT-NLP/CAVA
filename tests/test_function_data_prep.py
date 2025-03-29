#!/usr/bin/env python
import sys
import os

# Add the parent directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.cats.data import prep_function_calling_data

if __name__ == "__main__":
    print("Starting function calling data preparation...")
    prep_function_calling_data()
    print("Function calling data preparation completed.")