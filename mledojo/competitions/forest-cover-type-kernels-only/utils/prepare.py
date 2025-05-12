import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import shutil
from pathlib import Path

def prepare(raw: Path, public: Path, private: Path):
    """
    Prepare data for evaluating an agent in a sandbox environment.
    Split the training data into new train and test sets, and create necessary submission files.
    
    Args:
        raw: Path to the raw data directory
        public: Path to the public data directory
        private: Path to the private data directory
    """
    print("Starting data preparation...")
    
    # Create output directories if they don't exist
    os.makedirs(public, exist_ok=True)
    os.makedirs(private, exist_ok=True)
    
    # unzip the zip files
    os.system(f"unzip {raw / 'train.csv.zip'} -d {raw}")
    os.system(f"unzip {raw / 'test.csv.zip'} -d {raw}")
    os.system(f"unzip {raw / 'sample_submission.csv.zip'} -d {raw}")
    os.system(f"unzip {raw / 'sampleSubmission.csv.zip'} -d {raw}")
    # Load the original data
    print("Loading original data...")
    try:
        train_df = pd.read_csv(raw / "train.csv")
        test_df = pd.read_csv(raw / "test.csv")
        
        # Check for sample_submission.csv or sampleSubmission.csv
        sample_submission_path = raw / "sample_submission.csv"
        if not os.path.exists(sample_submission_path):
            sample_submission_path = raw / "sampleSubmission.csv"
            
        sample_submission = pd.read_csv(sample_submission_path)
        
        print(f"Loaded train.csv with {len(train_df)} rows and {len(train_df.columns)} columns")
        print(f"Loaded test.csv with {len(test_df)} rows and {len(test_df.columns)} columns")
        print(f"Loaded sample submission with {len(sample_submission)} rows and {len(sample_submission.columns)} columns")
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Calculate split ratio based on original test/train sizes
    try:
        original_train_size = len(train_df)
        original_test_size = len(test_df)
        split_ratio = original_test_size / (original_train_size + original_test_size)
        
        # Validate split ratio
        if split_ratio <= 0.05 or split_ratio >= 1:
            print(f"Calculated split ratio {split_ratio} is improper. Using default ratio of 0.2")
            split_ratio = 0.2
        else:
            print(f"Using calculated split ratio: {split_ratio}")
        
        print(f"Original train size: {original_train_size}, Original test size: {original_test_size}")
    except Exception as e:
        print(f"Error calculating split ratio: {e}")
        split_ratio = 0.2
        print(f"Using default split ratio: {split_ratio}")
    
    # Split the training data
    print(f"Splitting training data with ratio {split_ratio}...")
    try:
        # For this competition, we need to ensure balance across cover types
        # Split the data stratified by Cover_Type
        new_train_df, new_test_df = train_test_split(
            train_df, test_size=split_ratio, random_state=42, stratify=train_df['Cover_Type']
        )
        
        print(f"New train size: {len(new_train_df)}, New test size: {len(new_test_df)}")
        
        # Validate the split
        assert len(new_train_df) + len(new_test_df) == len(train_df), "Split sizes don't match original size"
        assert len(new_test_df) > 0, "Test set is empty"
        assert len(new_train_df) > 0, "Train set is empty"
        
        # Create test data without the target column (Cover_Type)
        print("Creating test.csv for public folder...")
        test_cols = test_df.columns  # Use the same columns as original test
        new_test_public = new_test_df[test_cols].copy() if set(test_cols).issubset(set(new_test_df.columns)) else new_test_df.drop(columns=['Cover_Type'])
        
        # Create test_answer.csv with Id and Cover_Type
        print("Creating test_answer.csv...")
        test_answer = new_test_df[['Id', 'Cover_Type']].copy()
        
        # Create sample_submission.csv with Id and Cover_Type (default value 1)
        print("Creating sample_submission.csv...")
        sample_submission_new = pd.DataFrame({'Id': new_test_df['Id']})
        sample_submission_new['Cover_Type'] = 1  # Default prediction
        
        # Validate that test_answer and sample_submission have the same columns
        assert set(test_answer.columns) == set(sample_submission_new.columns), "Columns in test_answer and sample_submission don't match"
        
        # Save the files
        print("Saving new files...")
        new_train_df.to_csv(public / "train.csv", index=False)
        new_test_public.to_csv(public / "test.csv", index=False)
        sample_submission_new.to_csv(public / "sample_submission.csv", index=False)
        test_answer.to_csv(private / "test_answer.csv", index=False)
        
        print("Data preparation completed successfully!")
    except Exception as e:
        print(f"Error during data preparation: {e}")
        import traceback
        traceback.print_exc()
    
    # clean up
    shutil.rmtree(raw)