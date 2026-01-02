#!/usr/bin/env python3

import os
import csv
import pandas as pd
import numpy as np
import sys
from pathlib import Path


def safe_int(x):
    try:
        return int(x)
    except (ValueError, TypeError):
        return np.nan


def export_result_to_csv(result, filename="attack_results.csv"):
    """
    Appends a single attack result (dictionary) to a CSV file.
    
    If the file does not exist, it creates the file and writes the header row 
    based on the keys in the result dictionary.
    
    Args:
        result (dict): A dictionary containing the attack result.
        filename (str): The name of the CSV file to write to.
    """
    # 1. Define the field names (header) based on the expected format
    fieldnames = list(result.keys())
    
    # Check if the file already exists
    file_exists = os.path.exists(filename)
    
    # Open the file in append mode ('a'). 
    # 'newline=""' is crucial for consistent CSV writing across platforms.
    try:
        with open(filename, 'a', newline='', encoding='utf-8') as csvfile:
            # Create a DictWriter object
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            # 2. Write the header only if the file is being created
            if not file_exists:
                print(f"Creating new file: {filename} and writing header.")
                writer.writeheader()
                
            # 3. Append the data row
            writer.writerow(result)
            
    except Exception as e:
        print(f"An error occurred while writing to CSV: {e}")



def analyze(csv_path, outfile):
    csv_path = Path(csv_path)
    df = pd.read_csv(str(csv_path))

    # 0. raw counts
    counts = df["success"].astype(str).str.strip().str.lower().value_counts()
    n_true, n_false, n_skipped = counts.get("true", 0), counts.get("false", 0), counts.get("skipped", 0)

    # 1. discard skipped
    df = df[df["success"].astype(str).str.strip().str.lower() != "skipped"].copy()
    if df.empty:
        print("❌  No attacked samples found (every row is 'Skipped').")
        sys.exit(0)

    # 2. boolean success
    df["is_success"] = df["success"].astype(str).str.strip().str.lower() == "true"

    # 3. queries
    df["queries"] = df["queries"].apply(safe_int)
    df = df.dropna(subset=["queries"])
    if df.empty:
        print("❌  No rows with valid query counts.")
        sys.exit(0)

    # 4. metrics
    attacked_total = len(df)
    successful_total = df["is_success"].sum()
    success_rate = successful_total / attacked_total
    avg_queries_attacked = df["queries"].mean()
    avg_queries_success = df[df["is_success"]]["queries"].mean()

    # 5. build report block
    report_lines = [
        "Hard-label attack summary",
        "-" * 40,
        f"True (successful)      : {n_true:>8}",
        f"False (failed)         : {n_false:>8}",
        f"Skipped                : {n_skipped:>8}",
        "-" * 40,
        f"Attacked (non-skipped) : {attacked_total:>8}",
        f"Successful attacks     : {successful_total:>8}",
        f"Attack success rate    : {success_rate:>8.2%}",
        f"Avg. queries (attacked): {avg_queries_attacked:>8.1f}",
        f"Avg. queries (success) : {avg_queries_success:>8.1f}",
        "-" * 40,
    ]
    report_txt = "\n".join(report_lines)

    # 6. stdout
    print(report_txt)

    # 7. save to file
    out_file = Path(outfile)
    # Force .txt extension for the human-readable report
    if out_file.suffix.lower() != ".txt":
        out_file = out_file.with_suffix(".txt")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(report_txt, encoding="utf-8")
    print(f"\nReport saved to {out_file}")


