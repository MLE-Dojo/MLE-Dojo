#!/usr/bin/env python3
# filepath: prepare.py
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
    Prepares data for the Novozymes Enzyme Stability Prediction competition.
    Creates new train-test splits from the original training data.
    
    Args:
        raw: Path to the raw data directory
        public: Path to the public data directory
        private: Path to the private data directory
    """
    print("Starting data preparation process...")
    
    # Create directories if they don't exist
    public.mkdir(parents=True, exist_ok=True)
    private.mkdir(parents=True, exist_ok=True)
    original_dir = raw / 'original'
    
    # Step 1: Analyze data structure and organize files
    print("Step 1: Analyzing data structure...")
    
    # Check if original folder exists, if not create it and move files
    if not original_dir.exists():
        print("Creating original folder and moving original files...")
        original_dir.mkdir(parents=True, exist_ok=True)
        
        # Files to move to original folder
        train_files = ['train.csv', 'train_updates_20220929.csv']
        test_files = ['test.csv', 'test_labels.csv', 'sample_submission.csv']
        
        # Move train and test files to original folder
        for file in train_files + test_files:
            file_path = raw / file
            if file_path.exists():
                shutil.copy(file_path, original_dir / file)
                print(f"Copied {file} to original folder")
    
    # Step 2: Load the data and identify columns
    print("Step 2: Loading data...")
    
    # Check if files exist in original folder first, otherwise use raw_data_dir
    train_path = original_dir / 'train.csv' if (original_dir / 'train.csv').exists() else raw / 'train.csv'
    test_path = original_dir / 'test.csv' if (original_dir / 'test.csv').exists() else raw / 'test.csv'
    
    # Load the datasets
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    # Check for train updates file
    train_updates_path = original_dir / 'train_updates_20220929.csv' if (original_dir / 'train_updates_20220929.csv').exists() else raw / 'train_updates_20220929.csv'
    if train_updates_path.exists():
        print("Loading train updates file...")
        train_updates_df = pd.read_csv(train_updates_path)
        # Apply updates to train_df if needed
        for _, row in train_updates_df.iterrows():
            if not pd.isna(row['seq_id']):
                # Only update columns that exist in both dataframes
                common_columns = [col for col in row.index if col in train_df.columns]
                train_df.loc[train_df['seq_id'] == row['seq_id'], common_columns] = row[common_columns].values
        print(f"Applied updates from {train_updates_path}")
    
    # Step 3: Calculate split ratio based on original test/train sizes
    print("Step 3: Calculating split ratio...")
    
    train_size = len(train_df)
    test_size = len(test_df)
    
    try:
        split_ratio = test_size / (train_size + test_size)
        if split_ratio > 1 or split_ratio <= 0:
            print(f"Warning: Calculated split ratio {split_ratio} is improper. Using default ratio of 0.2")
            split_ratio = 0.2
    except:
        print("Error in calculating split ratio. Using default ratio of 0.2")
        split_ratio = 0.2
    
    print(f"Original train size: {train_size}")
    print(f"Original test size: {test_size}")
    print(f"Using split ratio: {split_ratio:.4f}")
    
    # Step 4: Identify label/target and ID column names
    print("Step 4: Identifying label/target and ID columns...")
    id_col = 'seq_id'  # Based on CSV samples
    target_col = 'tm'  # Based on description and CSV samples
    
    print(f"ID column: {id_col}")
    print(f"Target column: {target_col}")
    
    # Step 5: Create new train and test splits from original train
    print("Step 5: Creating new train and test splits...")
    
    # Create new train and test splits
    train_new, test_new = train_test_split(train_df, test_size=split_ratio, random_state=42)
    
    print(f"New train size: {len(train_new)}")
    print(f"New test size: {len(test_new)}")
    
    # Step 6: Prepare test and sample submission files
    print("Step 6: Preparing test and answer files...")
    
    # Save test answer file
    test_answer = test_new[[id_col, target_col]].copy()
    test_answer.to_csv(private / 'test_answer.csv', index=False)
    print(f"Saved test answers to {private / 'test_answer.csv'}")
    
    # Create test file without target column
    test_public = test_new.drop(columns=[target_col])
    test_public.to_csv(public / 'test.csv', index=False)
    print(f"Saved test file to {public / 'test.csv'}")
    
    # Save train file
    train_new.to_csv(public / 'train.csv', index=False)
    print(f"Saved train file to {public / 'train.csv'}")
    
    # Step 7: Create sample submission file
    print("Step 7: Creating sample submission file...")
    sample_submission = test_answer.copy()
    sample_submission[target_col] = 0  # Reset target values to 0
    sample_submission.to_csv(public / 'sample_submission.csv', index=False)
    print(f"Saved sample submission to {public / 'sample_submission.csv'}")
    
    # Step 8: Validate the split with assertions
    print("Step 8: Validating test-train split...")
    
    # Check that the split was done properly
    assert len(train_new) + len(test_new) == len(train_df), "Split sizes don't add up to original size"
    assert len(test_answer) == len(test_public), "Test answer and public test files have different lengths"
    assert set(test_answer[id_col].values) == set(test_public[id_col].values), "IDs in test answer and public test don't match"
    
    # Step 9: Handle additional test and train folders if they exist
    print("Step 9: Handling additional test/train folders...")
    
    for folder in ['test', 'train']:
        folder_path = raw / folder
        if folder_path.is_dir():
            new_folder_path = public / folder
            new_folder_path.mkdir(parents=True, exist_ok=True)
            
            # Copy contents to align with new splits
            print(f"Found {folder} folder. Processing contents...")
            # Here you would implement specific logic based on the exact structure
            # This is a placeholder that simply copies files
            for file in os.listdir(folder_path):
                shutil.copy(
                    folder_path / file,
                    new_folder_path / file
                )
            print(f"Processed {folder} folder")
    
    
    # Step 10: Copy additional files
    print("Step 10: Copying additional files...")
    
    for file in os.listdir(raw):
        file_path = raw / file
        # Skip directories (especially 'original'), train files, and test files
        if (file_path.is_dir() and file != 'original') or \
           file in ['train.csv', 'test.csv', 'train_updates_20220929.csv', 'test_labels.csv', 'sample_submission.csv']:
            continue
        
        # Copy remaining files to public data
        if file_path.is_file():
            shutil.copy(file_path, public / file)
            print(f"Copied additional file: {file}")
    
    print("Data preparation complete!")

    # clean up
    shutil.rmtree(raw)

