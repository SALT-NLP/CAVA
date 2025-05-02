#!/usr/bin/env python3
"""
CAVA Model Evaluation Script - Compares multiple models on Jeopardy tasks

This script evaluates the performance of different models on Jeopardy question answering
by analyzing the JSONL output files. It compares models based on:
1. Correctness (using the PEDANT score)
2. Response latency

Author: (Your name)
Date: April 15, 2025
"""

import argparse
import glob
import json
import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import re

# Import the PEDANT score function from the existing codebase
from cava.utils import get_pedant_score


def load_jsonl(filepath: str) -> List[Dict[str, Any]]:
    """
    Load data from a JSONL file

    Args:
        filepath: Path to the JSONL file

    Returns:
        List of records from the file
    """
    records = []
    try:
        with open(filepath, "r") as f:
            for line in f:
                if line.strip():  # Skip empty lines
                    records.append(json.loads(line))
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
    return records


def find_model_files(data_dir: str, base_filename: str) -> Dict[str, str]:
    """
    Find all model result files for a given dataset

    Args:
        data_dir: Directory containing the data files
        base_filename: Base filename of the dataset

    Returns:
        Dictionary mapping model names to their result files
    """
    # Pattern matches the format: base_filename_modelname_taskname
    pattern = f"{base_filename}_*_jeopardy"
    model_files = {}

    for filepath in glob.glob(os.path.join(data_dir, pattern)):
        # Extract model name from the filename
        parts = os.path.basename(filepath).split("_")
        if len(parts) >= 3:
            # The model name is everything between the base_filename and _jeopardy
            model_name_parts = os.path.basename(filepath).replace(base_filename + "_", "").split("_jeopardy")[0]
            model_files[model_name_parts] = filepath

    return model_files


def extract_model_name(filename: str) -> str:
    """
    Extract a clean model name from the filename for display

    Args:
        filename: The filename to parse

    Returns:
        A clean, display-friendly model name
    """
    if "gemini" in filename.lower():
        return "Gemini 2.0 Flash"
    elif "gpt-4o-audio" in filename.lower():
        return "GPT-4o Audio"
    elif "pipeline" in filename.lower():
        return "Pipeline (GPT-4o)"
    else:
        return filename


def compare_models(model_data: Dict[str, List[Dict[str, Any]]]) -> pd.DataFrame:
    """
    Compare models based on correctness and latency

    Args:
        model_data: Dictionary mapping model names to their results

    Returns:
        DataFrame with model comparison results
    """
    results = []

    # Find all unique question IDs (filenames) across all models
    all_questions = set()
    for model_results in model_data.values():
        for record in model_results:
            all_questions.add(record.get("filename"))

    # For each question, compare all models
    for question_id in sorted(all_questions):
        question_results = {
            "filename": question_id,
            "question": None,
            "answer": None,
        }

        model_scores = {}

        for model_name, model_results in model_data.items():
            # Find the result for this question
            result = next((r for r in model_results if r.get("filename") == question_id), None)

            if result:
                # Store question and answer information (should be the same for all models)
                if not question_results["question"]:
                    question_results["question"] = result.get("question")
                    question_results["answer"] = result.get("answer")
                    question_results["category"] = result.get("category")

                # Calculate PEDANT score if not already in the result
                prediction = result.get("prediction", "")
                answer = result.get("answer", "")
                question = result.get("question", "")

                # Get the PEDANT score (correctness)
                score = get_pedant_score(answer, prediction, question)

                # Store model results
                model_scores[model_name] = {
                    "prediction": prediction,
                    "latency": result.get("latency", float("inf")),
                    "score": score,
                }

        # Determine the winning model for this question
        winning_model = None
        min_latency = float("inf")

        # First, find models with correct answers
        correct_models = [model for model, data in model_scores.items() if data["score"] == 1]

        if correct_models:
            # Among correct models, find the one with the lowest latency
            for model in correct_models:
                if model_scores[model]["latency"] < min_latency:
                    min_latency = model_scores[model]["latency"]
                    winning_model = model

        # Store the winner
        question_results["winning_model"] = winning_model

        # Add all model scores
        for model_name, data in model_scores.items():
            question_results[f"{model_name}_prediction"] = data["prediction"]
            question_results[f"{model_name}_score"] = data["score"]
            question_results[f"{model_name}_latency"] = data["latency"]
            question_results[f"{model_name}_winner"] = model_name == winning_model

        results.append(question_results)

    # Convert to DataFrame
    df = pd.DataFrame(results)
    return df


def analyze_results(df: pd.DataFrame, model_names: List[str]) -> Dict[str, Any]:
    """
    Analyze the results and compute statistics

    Args:
        df: DataFrame with model comparison results
        model_names: List of model names to analyze

    Returns:
        Dictionary with analysis results
    """
    analysis = {}

    # Overall statistics
    total_questions = len(df)
    analysis["total_questions"] = total_questions

    # Per-model statistics
    for model in model_names:
        model_stats = {}

        # Accuracy
        score_col = f"{model}_score"
        if score_col in df.columns:
            model_stats["accuracy"] = df[score_col].mean()
        else:
            model_stats["accuracy"] = float("nan")

        # Average latency
        latency_col = f"{model}_latency"
        if latency_col in df.columns:
            model_stats["avg_latency"] = df[latency_col].mean()
        else:
            model_stats["avg_latency"] = float("nan")

        # Win rate
        winner_col = f"{model}_winner"
        if winner_col in df.columns:
            model_stats["win_count"] = df[winner_col].sum()
            model_stats["win_rate"] = df[winner_col].mean()
        else:
            model_stats["win_count"] = 0
            model_stats["win_rate"] = 0.0

        analysis[model] = model_stats

    # Category-based analysis
    if "category" in df.columns:
        category_stats = {}
        categories = df["category"].unique()

        for category in categories:
            category_df = df[df["category"] == category]
            cat_stat = {"total": len(category_df)}

            for model in model_names:
                score_col = f"{model}_score"
                if score_col in category_df.columns:
                    cat_stat[f"{model}_accuracy"] = category_df[score_col].mean()

                winner_col = f"{model}_winner"
                if winner_col in category_df.columns:
                    cat_stat[f"{model}_win_rate"] = category_df[winner_col].mean()

            category_stats[category] = cat_stat

        analysis["categories"] = category_stats

    return analysis


def print_results(analysis: Dict[str, Any], model_names: List[str]):
    """
    Print the win rates for each model

    Args:
        analysis: Analysis results
        model_names: List of model names
    """
    print("\n" + "=" * 40)
    print(f"CAVA MODEL WIN RATES")
    print("=" * 40)

    # Format the model names for display
    display_names = {model: extract_model_name(model) for model in model_names}

    # Print total questions
    total_questions = analysis["total_questions"]
    print(f"\nTotal questions evaluated: {total_questions}")

    # Calculate questions where no model got a correct answer
    total_wins = sum(analysis[model]["win_count"] for model in model_names)
    no_correct_answers = total_questions - total_wins
    print(
        f"Questions with no correct answers: {no_correct_answers}/{total_questions} ({no_correct_answers/total_questions*100:.2f}%)"
    )

    # Print win rate for each model
    print("\nWIN RATES:")
    print("-" * 40)

    # Sort models by win rate (descending)
    sorted_models = sorted(model_names, key=lambda m: analysis[m]["win_rate"], reverse=True)

    for model in sorted_models:
        stats = analysis[model]
        display_name = display_names[model]
        wins = int(stats["win_count"])
        win_rate = stats["win_rate"] * 100

        print(f"{display_name:<20}: {wins}/{total_questions} questions ({win_rate:.2f}%)")


def generate_detailed_report(df: pd.DataFrame, model_names: List[str], output_file: Optional[str] = None):
    """
    Generate a detailed report of wins and losses

    Args:
        df: DataFrame with model comparison results
        model_names: List of model names
        output_file: Optional file to save the report to
    """
    report = []

    # Format the model names for display
    display_names = {model: extract_model_name(model) for model in model_names}

    # Add header
    report.append("DETAILED QUESTION ANALYSIS")
    report.append("=" * 80)
    report.append("")

    # Sort by winning model to group results
    if "winning_model" in df.columns:
        sorted_df = df.sort_values(["winning_model", "category", "filename"])
    else:
        sorted_df = df.sort_values(["category", "filename"])

    # Process each question
    for _, row in sorted_df.iterrows():
        question_id = row.get("filename", "Unknown")
        category = row.get("category", "Unknown")
        question = row.get("question", "Unknown")
        answer = row.get("answer", "Unknown")
        winning_model = row.get("winning_model", None)

        report.append(f"Question ID: {question_id}")
        report.append(f"Category: {category}")
        report.append(f"Question: {question}")
        report.append(f"Correct Answer: {answer}")

        if winning_model:
            report.append(f"Winner: {display_names.get(winning_model, winning_model)}")
        else:
            report.append("Winner: None (No model answered correctly)")

        report.append("\nModel Performance:")
        report.append("-" * 40)

        for model in model_names:
            prediction = row.get(f"{model}_prediction", "N/A")
            score = row.get(f"{model}_score", "N/A")
            latency = row.get(f"{model}_latency", "N/A")
            is_winner = row.get(f"{model}_winner", False)

            display_name = display_names.get(model, model)
            score_text = f"{score:.2f}" if isinstance(score, float) else score
            latency_text = f"{latency:.2f}s" if isinstance(latency, float) else latency
            winner_mark = " ✓" if is_winner else ""

            report.append(f"{display_name}{winner_mark}:")
            report.append(f"  Prediction: {prediction}")
            report.append(f"  Score: {score_text}")
            report.append(f"  Latency: {latency_text}")

        report.append("\n" + "=" * 80 + "\n")

    # Join the report
    report_text = "\n".join(report)

    # Save to file if requested
    if output_file:
        with open(output_file, "w") as f:
            f.write(report_text)

    return report_text


def main():
    parser = argparse.ArgumentParser(description="CAVA Model Evaluation Tool")
    parser.add_argument("--data-dir", type=str, default="data/jeopardy", help="Directory containing the data files")
    parser.add_argument("--base-file", type=str, default="audio_inputs.jsonl", help="Base filename of the dataset")

    args = parser.parse_args()

    # Find all model files
    print(f"Looking for model result files in {args.data_dir} with base filename {args.base_file}")
    model_files = find_model_files(args.data_dir, args.base_file)

    if not model_files:
        print("No model result files found!")
        return

    print(f"Found {len(model_files)} model result files:")
    for model_name, filepath in model_files.items():
        print(f"  {extract_model_name(model_name)}: {os.path.basename(filepath)}")

    # Load data from all model files
    model_data = {}
    for model_name, filepath in model_files.items():
        print(f"Loading data for {extract_model_name(model_name)}...")
        model_data[model_name] = load_jsonl(filepath)
        print(f"  Loaded {len(model_data[model_name])} records")

    # Compare models
    print("Comparing models...")
    df = compare_models(model_data)

    # Analyze results
    print("Analyzing results...")
    analysis = analyze_results(df, list(model_data.keys()))

    # Print results (only win rates)
    print_results(analysis, list(model_data.keys()))

    # print(generate_detailed_report(df, list(model_data.keys())))

    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()
