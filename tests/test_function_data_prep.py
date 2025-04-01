#!/usr/bin/env python
import sys
import os

from cats.data import prep_data_function_calling

if __name__ == "__main__":
    print("Starting function calling data preparation...")
    prep_data_function_calling()
    print("Function calling data preparation completed.")
