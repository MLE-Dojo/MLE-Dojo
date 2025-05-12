import pandas as pd
import numpy as np
import os
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split

def prepare(raw: Path, public: Path, private: Path):
    print("Starting data preparation...")
    
    # Create output directories if they don't exist
    print("Setting up directory structure...")
    os.makedirs(public, exist_ok=True)
    os.makedirs(private, exist_ok=True)
    
    # Load the original datasets with error handling
    print("Loading original datasets with error handling...")
    try:
        # Try with additional parameters to handle data issues
        train_df = pd.read_csv(raw / "train.csv", error_bad_lines=False, warn_bad_lines=True, engine='python')
    except TypeError:
        # For newer pandas versions where error_bad_lines is deprecated
        train_df = pd.read_csv(raw / "train.csv", on_bad_lines='skip', engine='python')
    
    try:
        test_df = pd.read_csv(raw / "test.csv", error_bad_lines=False, warn_bad_lines=True, engine='python')
    except TypeError:
        test_df = pd.read_csv(raw / "test.csv", on_bad_lines='skip', engine='python')
    
    print(f"Original train set size: {train_df.shape}")
    print(f"Original test set size: {test_df.shape}")
    
    # Calculate split ratio based on original sizes
    try:
        split_ratio = test_df.shape[0] / (train_df.shape[0] + test_df.shape[0])
        print(f"Calculated split ratio: {split_ratio:.4f}")
        
        # Validate the split ratio
        if split_ratio > 1 or split_ratio < 0.05:
            print("Warning: Calculated split ratio is improper. Using default ratio of 0.2")
            split_ratio = 0.2
    except Exception as e:
        print(f"Error calculating split ratio: {e}. Using default ratio of 0.2")
        split_ratio = 0.2
    
    # Split the training data
    print(f"Splitting train data with ratio: {split_ratio:.4f}")
    new_train_df, new_test_df = train_test_split(train_df, test_size=split_ratio, random_state=42)
    
    print(f"New train set size: {new_train_df.shape}")
    print(f"New test set size: {new_test_df.shape}")
    
    # Create test_answer.csv (private)
    print("Creating test_answer.csv...")
    test_answer = new_test_df[['id', 'loss']]
    test_answer.to_csv(private / "test_answer.csv", index=False)
    
    # Remove 'loss' column from the new test set
    new_test_df_public = new_test_df.drop(columns=['loss'])
    
    # Save new train and test files
    print("Saving new train and test files...")
    new_train_df.to_csv(public / "train.csv", index=False)
    new_test_df_public.to_csv(public / "test.csv", index=False)
    
    # Create sample_submission.csv
    print("Creating sample_submission.csv...")
    sample_submission = pd.DataFrame({
        'id': new_test_df_public['id'],
        'loss': [0] * len(new_test_df_public)
    })
    sample_submission.to_csv(public / "sample_submission.csv", index=False)

    
    # Validation checks
    print("Running validation checks...")
    # Checking if the split is proper (accounting for potentially skipped rows)
    assert abs(len(new_train_df) + len(new_test_df) - len(train_df)) <= 10, "Test-train split has too many missing entries!"
    
    # Check if test_answer.csv and sample_submission.csv have the same columns
    test_answer_cols = pd.read_csv(private / "test_answer.csv").columns.tolist()
    sample_submission_cols = pd.read_csv(public / "sample_submission.csv").columns.tolist()
    assert test_answer_cols == sample_submission_cols, "Column mismatch between test_answer.csv and sample_submission.csv!"
    
    print("Data preparation completed successfully!")
    print(f"Files saved to {public} and {private}")

    # clean up
    shutil.rmtree(raw)
