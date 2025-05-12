#!/usr/bin/env python
# filepath: prepare.py
import os
import pandas as pd
import numpy as np
import shutil
import subprocess
import zipfile
from sklearn.model_selection import train_test_split
from pathlib import Path

def prepare(raw: Path, public: Path, private: Path):
    """
    Prepares data for the NYC Taxi Trip Duration competition.
    Creates new train-test splits from the original training data.
    
    Args:
        raw: Path to the raw data directory
        public: Path to the public data directory
        private: Path to the private data directory
    """
    print("Starting data preparation process...")
    
    # unzip the data.zip file
    os.system(f"unzip {raw / 'test.zip'} -d {raw}")
    os.system(f"unzip {raw / 'train.zip'} -d {raw}")
    os.system(f"unzip {raw / 'sample_submission.zip'} -d {raw}")

    # Create necessary directories
    print("Creating necessary directories...")
    public.mkdir(parents=True, exist_ok=True)
    private.mkdir(parents=True, exist_ok=True)
    
    # Check if 'original' folder exists
    original_dir = raw / 'original'
    if original_dir.exists():
        print("Original folder exists, using files from there...")
    else:
        print("Original folder doesn't exist, creating and moving files...")
        original_dir.mkdir(parents=True, exist_ok=True)
        
        # Move train and test files to original folder
        for file in ['train.csv', 'test.csv', 'sample_submission.csv']:
            file_path = raw / file
            if file_path.exists():
                shutil.copy(file_path, original_dir / file)
                print(f"Copied {file} to original folder")
    
    # Load the data
    print("Loading data files...")
    try:
        train_df = pd.read_csv(original_dir / 'train.csv')
        test_df = pd.read_csv(original_dir / 'test.csv')
        print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")
        
        # Identify target column by comparing train and test columns
        train_cols = set(train_df.columns)
        test_cols = set(test_df.columns)
        
        # The target column would be present in train but not in test
        target_cols = train_cols - test_cols
        
        if len(target_cols) == 1:
            target_col = list(target_cols)[0]
            print(f"Identified target column: {target_col}")
        else:
            target_col = 'trip_duration'
            print(f"Could not uniquely identify target column, using default: {target_col}")
        
        # Identify ID column
        id_col = None
        for col in train_df.columns:
            if 'id' in col.lower() and train_df[col].nunique() == train_df.shape[0]:
                id_col = col
                break
        
        if id_col is None:
            id_col = 'id'  # Default
        
        print(f"Identified ID column: {id_col}")
        
        # Calculate split ratio based on original sizes
        try:
            original_ratio = test_df.shape[0] / train_df.shape[0]
            print(f"Original test/train ratio: {original_ratio:.4f}")
            
            if original_ratio <= 0 or original_ratio >= 1:
                print("Ratio is improper, using default ratio of 0.25")
                original_ratio = 0.25
        except:
            print("Error calculating ratio, using default ratio of 0.25")
            original_ratio = 0.25
        
        # Create new train-test split from original train data
        print("Creating new train-test split...")
        X_train, X_test, y_train, y_test = train_test_split(
            train_df.drop(target_col, axis=1),
            train_df[target_col],
            test_size=original_ratio,
            random_state=42
        )
        
        # Combine features and target for new train
        new_train = X_train.copy()
        new_train[target_col] = y_train.values
        
        # Create new test without target
        new_test = X_test.copy()
        
        print(f"New train shape: {new_train.shape}, New test shape: {new_test.shape}")
        
        # Create test_answer file with id and target
        test_answer = pd.DataFrame({
            id_col: X_test[id_col],
            target_col: y_test.values
        })
        
        # Save test_answer to private folder
        test_answer_path = private / 'test_answer.csv'
        test_answer.to_csv(test_answer_path, index=False)
        print(f"Saved test answers to {test_answer_path}")
        
        # Create sample_submission with same format but different values
        sample_submission = pd.DataFrame({
            id_col: X_test[id_col],
            target_col: np.random.randint(100, 1000, size=X_test.shape[0])
        })
        
        # Save new train and test files
        new_train_path = public / 'train.csv'
        new_test_path = public / 'test.csv'
        sample_submission_path = public / 'sample_submission.csv'
        
        new_train.to_csv(new_train_path, index=False)
        new_test.to_csv(new_test_path, index=False)
        sample_submission.to_csv(sample_submission_path, index=False)
        
        print(f"Saved new train, test, and sample submission files")
        
        # Validating the train test split with assertions
        print("Validating train-test split...")
        assert new_train.shape[0] + new_test.shape[0] == train_df.shape[0], "Train-test split sizes don't match original"
        assert target_col in new_train.columns, f"Target column {target_col} not found in new train data"
        assert target_col not in new_test.columns, f"Target column {target_col} should not be in new test data"
        assert test_answer.shape[0] == new_test.shape[0], "Test answer and test data don't match in size"
        assert sample_submission.shape[0] == new_test.shape[0], "Sample submission and test data don't match in size"
        
            
    except Exception as e:
        print(f"Error in data preparation: {e}")
        raise

    # clean up
    shutil.rmtree(raw)
