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
    Prepare the data for the ICR Identify Age-Related Conditions competition by:
    1. Organizing the original data
    2. Creating new train/test splits
    3. Preparing submission files
    4. Validating the results
    """
    print("Starting data preparation...")
    
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
        for file_name in ['train.csv', 'test.csv', 'greeks.csv', 'sample_submission.csv']:
            if (raw / file_name).exists():
                shutil.copy(raw / file_name, original_dir / file_name)
                print(f"Copied {file_name} to original folder")
    else:
        print("Original folder already has files, using those as reference")
    
    # Load data
    print("Loading data files...")
    train_path = original_dir / 'train.csv'
    test_path = original_dir / 'test.csv'
    
    try:
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")
    except Exception as e:
        print(f"Error loading data files: {e}")
        train_df = None
        test_df = None
    
    # Identify target and ID columns
    id_column = 'Id'
    target_column = 'Class'
    
    print(f"Identified columns - ID: {id_column}, Target: {target_column}")
    
    # Calculate split ratio
    try:
        if train_df is not None and test_df is not None:
            test_train_ratio = test_df.shape[0] / train_df.shape[0]
            if test_train_ratio > 1 or test_train_ratio <= 0:
                print(f"Invalid calculated ratio: {test_train_ratio}, using default ratio of 0.2")
                test_train_ratio = 0.2
            else:
                print(f"Calculated test/train ratio: {test_train_ratio}")
        else:
            print("Using default test/train ratio: 0.2")
            test_train_ratio = 0.2
    except Exception as e:
        print(f"Error calculating ratio: {e}, using default ratio of 0.2")
        test_train_ratio = 0.2

    test_train_ratio = 0.2 #hardcode it, otherwise too small
    
    if train_df is not None:
        print("Creating new train/test split...")
        X = train_df.drop(columns=[target_column])
        y = train_df[target_column]
        
        # Creating new train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_train_ratio, random_state=42, stratify=y
        )
        
        # Recreate dataframes
        new_train_df = pd.concat([X_train, y_train], axis=1)
        new_test_df = X_test.copy()
        
        # Create test_answer.csv
        test_answer_df = pd.DataFrame({
            id_column: new_test_df[id_column],
            target_column: y_test.values
        })
        
        print(f"New train shape: {new_train_df.shape}, New test shape: {new_test_df.shape}")
        
        # Validate split
        assert new_train_df.shape[0] + new_test_df.shape[0] == train_df.shape[0], "Split size doesn't match original size"
        assert len(new_test_df) > 0, "Test set is empty"
        assert len(new_train_df) > 0, "Train set is empty"
        print("Split validation passed!")
        
        # Create sample_submission
        sample_submission_df = pd.DataFrame({
            id_column: new_test_df[id_column],
            'class_0': [0.5] * len(new_test_df),
            'class_1': [0.5] * len(new_test_df)
        })
        
        # Save private test answers
        private_test_answer_path = private / 'test_answer.csv'
        test_answer_df.to_csv(private_test_answer_path, index=False)
        print(f"Saved test answers to {private_test_answer_path}")
        
        # Save files to public directory
        new_train_path = public / 'train.csv'
        new_test_path = public / 'test.csv'
        sample_submission_path = public / 'sample_submission.csv'
        
        new_train_df.to_csv(new_train_path, index=False)
        new_test_df.to_csv(new_test_path, index=False)
        sample_submission_df.to_csv(sample_submission_path, index=False)
        print(f"Saved new train/test files to {public}")

    
    print("Data preparation complete!")

    # clean up
    shutil.rmtree(raw)
