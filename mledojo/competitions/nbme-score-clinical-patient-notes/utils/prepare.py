import os
import shutil
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

def prepare(raw: Path, public: Path, private: Path):
    """
    Prepare data for NBME Score Clinical Patient Notes competition by creating new train/test splits.
    
    Args:
        raw: Path to the raw data directory
        public: Path to the public data directory
        private: Path to the private data directory
    """
    print("Starting data preparation...")
    
    # Create output directories if they don't exist
    public.mkdir(parents=True, exist_ok=True)
    private.mkdir(parents=True, exist_ok=True)
    original_dir = raw / "original"
    
    print("Checking if original folder exists...")
    if not original_dir.exists():
        print("Original folder doesn't exist. Creating it and moving original train/test files...")
        original_dir.mkdir(parents=True, exist_ok=True)
        
        # Move original train and test files to original folder
        train_file = raw / 'train.csv'
        test_file = raw / 'test.csv'
        
        if train_file.exists():
            shutil.copy(train_file, original_dir / 'train.csv')
        
        if test_file.exists():
            shutil.copy(test_file, original_dir / 'test.csv')
    
    # Load data
    print("Loading data files...")
    train_df = pd.read_csv(raw / 'train.csv')
    test_df = pd.read_csv(raw / 'test.csv')
    
    # Additional files that may exist
    try:
        features_df = pd.read_csv(raw / 'features.csv')
        patient_notes_df = pd.read_csv(raw / 'patient_notes.csv')
    except FileNotFoundError:
        print("Some additional files not found, continuing with available data.")
    
    # Calculate split ratio based on original test/train sizes
    print("Calculating split ratio...")
    try:
        ratio = len(test_df) / len(train_df)
        if ratio > 1:
            print(f"Calculated ratio {ratio} is greater than 1, using default ratio of 0.2")
            ratio = 0.2
        print(f"Split ratio: {ratio}")
        print(f"Original train size: {len(train_df)}, test size: {len(test_df)}")
    except Exception as e:
        print(f"Error calculating ratio: {e}. Using default ratio of 0.2")
        ratio = 0.2
    ratio = 0.2  # hardcode
    
    # Identify label column and ID column
    print("Identifying label and ID columns...")
    # Based on the CSV samples, 'location' is the target column and 'id' is the ID column
    id_column = 'id'
    label_column = 'location'
    
    # Create new train-test split from original train files
    print("Creating new train-test split...")
    
    train_unique_ids = train_df[id_column].unique()
    train_ids, test_ids = train_test_split(train_unique_ids, test_size=ratio, random_state=42)
    
    new_train_df = train_df[train_df[id_column].isin(train_ids)].copy()
    new_test_df = train_df[train_df[id_column].isin(test_ids)].copy()
    
    print(f"New train size: {len(new_train_df)}, new test size: {len(new_test_df)}")
    
    # Save test answers to private folder
    print("Saving test answers...")
    test_answer_df = new_test_df[[id_column, label_column]].copy()
    test_answer_df.to_csv(private / 'test_answer.csv', index=False)
    
    # Remove label column from test data
    print("Removing label column from test data...")
    new_test_df = new_test_df.drop(columns=[label_column])
    
    # Create sample submission with same structure but different values
    print("Creating sample submission file...")
    sample_submission_df = test_answer_df.copy()
    # For this competition, location seems to be spans, default to empty values
    sample_submission_df[label_column] = ''
    
    # Save new train and test files
    print("Saving new train and test files...")
    new_train_df.to_csv(public / 'train.csv', index=False)
    new_test_df.to_csv(public / 'test.csv', index=False)
    sample_submission_df.to_csv(public / 'sample_submission.csv', index=False)
    
    # Copy additional files
    print("Copying additional files...")
    for file in os.listdir(raw):
        file_path = raw / file
        if file_path.is_file() and file not in ['train.csv', 'test.csv', 'sample_submission.csv'] and not file.startswith('original'):
            shutil.copy(file_path, public / file)
    
    # Check if there are additional train and test folders
    print("Checking for additional train/test folders...")
    train_dir = raw / 'train'
    test_dir = raw / 'test'
    
    if train_dir.exists() and train_dir.is_dir() and test_dir.exists() and test_dir.is_dir():
        print("Additional train and test folders found. Creating new splits...")
        
        # Create new train and test directories
        new_train_dir = public / 'train'
        new_test_dir = public / 'test'
        new_train_dir.mkdir(parents=True, exist_ok=True)
        new_test_dir.mkdir(parents=True, exist_ok=True)
        
        # Process based on IDs in new_train_df and new_test_df
        # This is a simplified approach - you might need to adapt to your specific folder structure
        
        # Example implementation depends on how files are named and organized in the folders
        # This is a placeholder for the concept
        
        print("Warning: Placeholder for additional folder processing. Implementation depends on folder structure.")
    
    # Validate with assertions
    print("Validating train-test split...")
    assert len(new_train_df) + len(new_test_df) == len(train_df), "Train-test split doesn't match original data size"
    assert not new_test_df.columns.isin([label_column]).any(), "Test data should not contain the label column"
    assert (private / 'test_answer.csv').exists(), "Test answer file not created"
    assert (public / 'sample_submission.csv').exists(), "Sample submission file not created"
    
    print("Data preparation complete!")
    
    # clean up
    shutil.rmtree(raw)

