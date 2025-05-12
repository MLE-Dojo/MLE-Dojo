#!/usr/bin/env python3

import os
import pandas as pd
import numpy as np
import shutil
import subprocess
from sklearn.model_selection import train_test_split
import zipfile
from pathlib import Path

def prepare(raw: Path, public: Path, private: Path):
    """
    Prepare data for Mercedes-Benz Greener Manufacturing by creating new train/test splits.
    
    Args:
        raw: Path to the raw data directory
        public: Path to the public data directory
        private: Path to the private data directory
    """
    print("Starting data preparation...")
    
    # unzip the data.zip file
    os.system(f"unzip {raw / 'test.csv.zip'} -d {raw}")
    os.system(f"unzip {raw / 'train.csv.zip'} -d {raw}")
    os.system(f"unzip {raw / 'sample_submission.csv.zip'} -d {raw}")
    
    # Create necessary directories if they don't exist
    public.mkdir(parents=True, exist_ok=True)
    private.mkdir(parents=True, exist_ok=True)
    original_dir = raw / "original"
    
    # Check if 'original' folder exists, if not create it and move files
    if not original_dir.exists():
        print("Original folder doesn't exist. Creating and moving files...")
        original_dir.mkdir(parents=True, exist_ok=True)
        
        # Check and move train.csv, test.csv and sample_submission.csv to original folder
        for file in ['train.csv', 'test.csv', 'sample_submission.csv']:
            if (raw / file).exists():
                shutil.copy2(raw / file, original_dir / file)
                print(f"Copied {file} to original folder")
    
    # Load original train and test data
    print("Loading original data...")

    train_df = pd.read_csv(original_dir / 'train.csv')
    test_df = pd.read_csv(original_dir / 'test.csv')
    
    print(f"Original train data shape: {train_df.shape}")
    print(f"Original test data shape: {test_df.shape}")
    
    # Identify target column ('y') and ID column ('ID')
    id_column = 'ID'
    target_column = 'y'
    
    print(f"Identified ID column: {id_column}")
    print(f"Identified target column: {target_column}")
    
    # Calculate split ratio based on original test/train sizes
    try:
        split_ratio = len(test_df) / len(train_df)
        if split_ratio <= 0 or split_ratio > 1:
            print("Invalid split ratio calculated. Using default split ratio of 0.2")
            split_ratio = 0.2
    except Exception as e:
        print(f"Error calculating split ratio: {str(e)}. Using default split ratio of 0.2")
        split_ratio = 0.2
    split_ratio = 0.2 #hard code it 
    print(f"Using test/train split ratio: {split_ratio:.4f}")
    
    # Create new train and test split from original train data
    print("Creating new train/test split...")
    X = train_df.drop(columns=[target_column])
    y = train_df[target_column]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split_ratio, random_state=42)
    
    print(f"New train data size: {len(X_train)}")
    print(f"New test data size: {len(X_test)}")
    
    # Create new train DataFrame
    new_train_df = X_train.copy()
    new_train_df[target_column] = y_train
    
    # Create new test DataFrame (without target column)
    new_test_df = X_test.copy()
    
    # Create test answers DataFrame
    test_answer_df = pd.DataFrame({
        id_column: X_test[id_column],
        target_column: y_test
    })
    
    # Create sample submission DataFrame
    sample_submission_df = pd.DataFrame({
        id_column: X_test[id_column],
        target_column: np.random.normal(100, 10, size=len(X_test))
    })
    
    # Assertions for validation
    print("Validating data splits...")
    assert len(new_train_df) + len(new_test_df) == len(train_df), "Total size mismatch"
    assert len(new_test_df) == len(test_answer_df), "Test size mismatch"
    assert len(sample_submission_df) == len(test_answer_df), "Sample submission size mismatch"
    assert all(col in new_test_df.columns for col in X.columns if col != target_column), "Column mismatch"
    assert target_column not in new_test_df.columns, "Test data should not contain target column"
    print("Validation passed!")
    
    # Save files
    print("Saving files...")
    
    # Save test_answer.csv to private directory
    test_answer_df.to_csv(private / 'test_answer.csv', index=False)
    print(f"Saved test_answer.csv to {private}")
    
    # Save train.csv, test.csv, and sample_submission.csv to public directory
    new_train_df.to_csv(public / 'train.csv', index=False)
    new_test_df.to_csv(public / 'test.csv', index=False)
    sample_submission_df.to_csv(public / 'sample_submission.csv', index=False)
    

    # clean up
    shutil.rmtree(raw)

