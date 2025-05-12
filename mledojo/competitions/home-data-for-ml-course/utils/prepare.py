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
    Prepare data for machine learning by creating train/test splits and necessary files.
    
    Args:
        raw: Path to raw data directory
        public: Path to public output directory
        private: Path to private output directory
    """
    print("Starting data preparation...")
    
    # Define paths
    original_dir = raw / 'original'
    
    # Create directories if they don't exist
    for dir_path in [public, private, original_dir]:
        if not dir_path.exists():
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"Created directory: {dir_path}")
    
    # Step 1: Analyze data structure and identify original train/test splits
    print("Step 1: Analyzing data structure...")
    
    # Check if original folder exists with train/test files
    if (original_dir / 'train.csv').exists() and (original_dir / 'test.csv').exists():
        print("Using train/test files from original folder")
        train_path = original_dir / 'train.csv'
        test_path = original_dir / 'test.csv'
    else:
        # If not, move the train/test files to original folder
        train_path = raw / 'train.csv'
        test_path = raw / 'test.csv'
        
        if train_path.exists() and test_path.exists():
            # Copy files to original directory
            shutil.copy2(train_path, original_dir)
            shutil.copy2(test_path, original_dir)
            print("Copied train/test files to original folder")
        else:
            print("Error: train.csv or test.csv not found in data_raw directory")
            return
    
    # Read the data
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    print(f"Train data shape: {train_df.shape}")
    print(f"Test data shape: {test_df.shape}")
    
    # Step 2: Calculate split ratio based on original test/train sizes
    print("\nStep 2: Calculating split ratio...")
    
    try:
        train_size = train_df.shape[0]
        test_size = test_df.shape[0]
        total_size = train_size + test_size
        split_ratio = test_size / total_size
        
        print(f"Original train size: {train_size}")
        print(f"Original test size: {test_size}")
        print(f"Original total size: {total_size}")
        print(f"Calculated split ratio: {split_ratio:.4f}")
        
        # Check if split ratio is between 0 and 1
        if split_ratio <= 0 or split_ratio >= 1:
            print("Warning: Invalid split ratio calculated. Using default ratio of 0.2")
            split_ratio = 0.2
    except Exception as e:
        print(f"Error calculating split ratio: {e}")
        print("Using default ratio of 0.2")
        split_ratio = 0.2
    
    # Step 3 & 4: Identify target/label column and create new train/test splits
    print("\nStep 3 & 4: Creating new train/test splits...")
    
    # Identify target column: SalePrice for this dataset
    target_col = 'SalePrice'
    id_col = 'Id'
    
    print(f"Target column identified: {target_col}")
    print(f"ID column identified: {id_col}")
    
    # Create new train/test split from original train data
    train_new, test_new = train_test_split(train_df, test_size=split_ratio, random_state=42)
    
    print(f"New train data shape: {train_new.shape}")
    print(f"New test data shape: {test_new.shape}")
    
    # Step 5 & 6: Create test files without target column and save test answers
    print("\nStep 5 & 6: Creating test files and saving test answers...")
    
    # Create test_answer.csv
    test_answer = test_new[[id_col, target_col]].copy()
    test_answer_path = private / 'test_answer.csv'
    test_answer.to_csv(test_answer_path, index=False)
    print(f"Saved test answers to {test_answer_path}")
    
    # Remove target column from test data
    test_new_public = test_new.drop(columns=[target_col])
    
    # Step 7: Create sample_submission.csv
    print("\nStep 7: Creating sample_submission.csv...")
    
    # Create sample submission with dummy prices
    # Debug info
    print(f"test_new_public shape: {test_new_public.shape}")
    print(f"test_new_public['{id_col}'] count: {test_new_public[id_col].count()}")
    
    # Create sample submission using the ids from test_new
    sample_submission = pd.DataFrame({
        id_col: test_new_public[id_col],
        target_col: np.random.randint(100000, 300000, size=len(test_new_public))
    })
    
    # Check the sample submission shape before saving
    print(f"Sample submission shape: {sample_submission.shape}")
    
    sample_submission_path = public / 'sample_submission.csv'
    sample_submission.to_csv(sample_submission_path, index=False)
    print(f"Saved sample submission to {sample_submission_path}")
    
    # Step 8: Not applicable for this dataset as there are no additional folders
    
    # Step 9: Validate with assertions
    print("\nStep 9: Validating train/test split...")
    
    # Assert that the shapes are correct
    assert train_new.shape[0] > 0, "Train set is empty"
    assert test_new.shape[0] > 0, "Test set is empty"
    assert test_new_public.shape[0] == test_answer.shape[0], "Test and test_answer sizes don't match"
    assert sample_submission.shape[0] == test_new_public.shape[0], "Sample submission and test sizes don't match"
    
    print("Validation successful")
    
    # Save new train and test files
    train_new_path = public / 'train.csv'
    test_new_path = public / 'test.csv'
    
    train_new.to_csv(train_new_path, index=False)
    test_new_public.to_csv(test_new_path, index=False)
    
    print(f"Saved new train data to {train_new_path}")
    print(f"Saved new test data to {test_new_path}")
    
    
    print("\nData preparation completed successfully!")

    # clean up
    shutil.rmtree(raw)