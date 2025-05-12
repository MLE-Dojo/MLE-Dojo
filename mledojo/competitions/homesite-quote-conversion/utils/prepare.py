#!/usr/bin/env python3

import os
import pandas as pd
import numpy as np
import shutil
import subprocess
import random
import zipfile
from pathlib import Path
from sklearn.model_selection import train_test_split

def prepare(raw: Path, public: Path, private: Path):
    """
    Prepare the data for the Homesite Quote Conversion competition by:
    1. Organizing the original data
    2. Creating new train/test splits
    3. Preparing submission files
    4. Validating the results
    """
    print("Starting data preparation for Homesite Quote Conversion competition")
    
    # rm homesite-quote-conversion.zip
    os.system("rm homesite-quote-conversion.zip")

    # unzip other zip files
    for file in raw.glob("*.zip"):
        os.system(f"unzip {file} -d {raw}")

    # Define original directory
    original_dir = raw / 'original'
    
    # Create directories if they don't exist
    for directory in [public, private, original_dir]:
        if not directory.exists():
            directory.mkdir(parents=True, exist_ok=True)
            print(f"Created directory: {directory}")
    
    # Check if original folder already has files
    original_has_files = len(list(original_dir.iterdir())) > 0
    
    if not original_has_files:
        print("Moving original files to 'original' folder...")
        for file in ['train.csv', 'test.csv', 'sample_submission.csv']:
            if (raw / file).exists():
                shutil.copy(raw / file, original_dir / file)
                print(f"Copied {file} to original folder")
    else:
        print("Original folder already has files, using those as reference")
    
    # Load original data
    print("Loading original data...")
    train_file = original_dir / 'train.csv'
    test_file = original_dir / 'test.csv'
    
    if not train_file.exists() or not test_file.exists():
        print("Error: Original train or test file not found!")
        return
    
    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)
    
    print(f"Original train shape: {train_df.shape}")
    print(f"Original test shape: {test_df.shape}")
    
    # Identify target and ID columns
    id_column = "QuoteNumber"
    target_column = "QuoteConversion_Flag"
    
    print(f"Identified ID column: {id_column}")
    print(f"Identified target column: {target_column}")
    
    # Calculate split ratio
    try:
        original_test_size = len(test_df)
        original_train_size = len(train_df)
        test_ratio = original_test_size / (original_test_size + original_train_size)
        
        print(f"Original split ratio: test_size={test_ratio:.4f} (test: {original_test_size}, train: {original_train_size})")
        
        if test_ratio <= 0 or test_ratio >= 1:
            print("Invalid split ratio, using default value of 0.2")
            test_ratio = 0.2
    except Exception as e:
        print(f"Error calculating split ratio: {e}")
        print("Using default test_ratio of 0.2")
        test_ratio = 0.2
    
    # Create new train-test split from original train data
    print(f"Creating new train-test split with test_ratio={test_ratio}...")
    train_new, test_new = train_test_split(train_df, test_size=test_ratio, random_state=42)
    
    print(f"New train shape: {train_new.shape}")
    print(f"New test shape: {test_new.shape}")
    
    # Create test_answer.csv (private)
    print("Creating test_answer.csv...")
    test_answer = test_new[[id_column, target_column]].copy()
    test_answer.to_csv(private / 'test_answer.csv', index=False)
    
    # Create sample_submission.csv (public)
    print("Creating sample_submission.csv...")
    sample_submission = test_new[[id_column]].copy()
    sample_submission[target_column] = 0  # Fill with zeros as default values
    sample_submission.to_csv(public / 'sample_submission.csv', index=False)
    
    # Remove target column from test data
    print("Removing target column from test data...")
    test_new_public = test_new.drop(target_column, axis=1)
    
    # Save new train and test files
    print("Saving new train and test files...")
    train_new_path = public / 'train.csv'
    test_new_path = public / 'test.csv'
    
    train_new.to_csv(train_new_path, index=False)
    test_new_public.to_csv(test_new_path, index=False)
    
    # Validation
    print("Validating train-test split...")
    # Check if test file doesn't have target column
    assert target_column not in pd.read_csv(test_new_path).columns, f"{target_column} should not be in test.csv"
    
    # Check if train and test have the same number of columns (except target column)
    train_cols = set(pd.read_csv(train_new_path).columns)
    test_cols = set(pd.read_csv(test_new_path).columns)
    assert len(train_cols) == len(test_cols) + 1, "Train should have one more column than test"
    assert train_cols - test_cols == {target_column}, f"The only different column should be {target_column}"
    
    # Check if test_answer and sample_submission have the same ID values
    test_answer_ids = set(pd.read_csv(private / 'test_answer.csv')[id_column])
    sample_submission_ids = set(pd.read_csv(public / 'sample_submission.csv')[id_column])
    assert test_answer_ids == sample_submission_ids, "test_answer.csv and sample_submission.csv should have the same IDs"
    

    # clean up
    shutil.rmtree(raw)