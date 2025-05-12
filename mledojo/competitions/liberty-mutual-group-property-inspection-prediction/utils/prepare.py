#!/usr/bin/env python3

import os
import pandas as pd
import numpy as np
import shutil
import subprocess
import zipfile
from pathlib import Path
from sklearn.model_selection import train_test_split

def prepare(raw: Path, public: Path, private: Path):
    """
    Prepare data for the Liberty Mutual competition by creating new train/test splits.
    
    Args:
        raw: Path to the raw data directory
        public: Path to the public data directory
        private: Path to the private data directory
    """
    print("Starting data preparation process...")

    # unzip the data.zip file
    os.system(f"unzip {raw / 'train.csv.zip'} -d {raw}")
    os.system(f"unzip {raw / 'test.csv.zip'} -d {raw}")
    os.system(f"unzip {raw / 'sample_submission.csv.zip'} -d {raw}")

    # Create directories if they don't exist
    original_path = raw / 'original'
    for path in [public, private, original_path]:
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            print(f"Created directory: {path}")

    # Check if files already exist in original folder
    train_path = original_path / 'train.csv'
    test_path = original_path / 'test.csv'
    sample_submission_path = original_path / 'sample_submission.csv'

    # Move files to original folder if they're not already there
    if not train_path.exists():
        raw_train_path = raw / 'train.csv'
        if raw_train_path.exists():
            shutil.copy(raw_train_path, train_path)
            print(f"Copied train.csv to original folder")
        else:
            print(f"Error: {raw_train_path} not found")
            return

    if not test_path.exists():
        raw_test_path = raw / 'test.csv'
        if raw_test_path.exists():
            shutil.copy(raw_test_path, test_path)
            print(f"Copied test.csv to original folder")
        else:
            print(f"Error: {raw_test_path} not found")
            return

    if not sample_submission_path.exists():
        raw_sample_path = raw / 'sample_submission.csv'
        if raw_sample_path.exists():
            shutil.copy(raw_sample_path, sample_submission_path)
            print(f"Copied sample_submission.csv to original folder")
        else:
            print(f"Error: {raw_sample_path} not found")

    # Load data
    print("Loading data files...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    # Identify target and ID columns
    id_col = "Id"
    target_col = "Hazard"
    print(f"Identified target column: {target_col}, ID column: {id_col}")

    # Calculate split ratio based on original sizes
    try:
        train_size = len(train_df)
        test_size = len(test_df)
        test_ratio = test_size / (train_size + test_size)
        
        if test_ratio <= 0 or test_ratio >= 1:
            print(f"Calculated test ratio {test_ratio} is improper, using default ratio of 0.2")
            test_ratio = 0.2
        else:
            print(f"Original data sizes - Train: {train_size}, Test: {test_size}")
            print(f"Calculated test ratio: {test_ratio}")
    except Exception as e:
        print(f"Error calculating split ratio: {e}, using default ratio of 0.2")
        test_ratio = 0.2

    # Create new train/test split from original train data
    print("Creating new train/test split...")
    X = train_df.drop(columns=[target_col])
    y = train_df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, random_state=42)
    
    # Create new dataframes
    new_train_df = pd.concat([X_train, y_train], axis=1)
    new_test_df = X_test.copy()
    
    # Create test answers
    test_answers_df = pd.DataFrame({
        id_col: X_test[id_col],
        target_col: y_test
    })
    
    # Create sample submission with different values (zeros)
    sample_submission_df = pd.DataFrame({
        id_col: X_test[id_col],
        target_col: np.zeros(len(X_test))
    })
    
    # Validate the split
    assert len(new_train_df) + len(new_test_df) == len(train_df), "Split sizes don't match original size"
    assert len(new_test_df) > 0, "Test set is empty"
    assert len(new_train_df) > 0, "Train set is empty"
    assert set(new_test_df[id_col].values) == set(test_answers_df[id_col].values), "Test IDs don't match test answers"
    print("Split validation passed")

    # Save new files
    print("Saving new files...")
    new_train_path = public / 'train.csv'
    new_test_path = public / 'test.csv'
    test_answers_path = private / 'test_answer.csv'
    new_sample_path = public / 'sample_submission.csv'
    
    new_train_df.to_csv(new_train_path, index=False)
    new_test_df.to_csv(new_test_path, index=False)
    test_answers_df.to_csv(test_answers_path, index=False)
    sample_submission_df.to_csv(new_sample_path, index=False)


    print("Data preparation complete!")

    # clean up
    shutil.rmtree(raw)