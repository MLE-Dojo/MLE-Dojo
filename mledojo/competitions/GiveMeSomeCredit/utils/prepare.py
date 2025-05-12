import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import shutil
from pathlib import Path

def prepare(raw: Path, public: Path, private: Path):
    """
    Prepares data for the 'Give Me Some Credit' competition by:
    1. Creating new train/test splits from the original training data
    2. Creating sample_submission.csv and test_answer.csv files
    3. Organizing data into appropriate directories
    
    Args:
        raw: Path to the raw data directory
        public: Path to the public data directory
        private: Path to the private data directory
    """
    print("Starting data preparation process...")
    
    # Create output directories if they don't exist
    public.mkdir(exist_ok=True, parents=True)
    private.mkdir(exist_ok=True, parents=True)
    print("Created output directories")
    
    # Read the original train and test files
    train_df = pd.read_csv(raw / "cs-training.csv")
    test_df = pd.read_csv(raw / "cs-test.csv")
    
    print(f"Original dataset: {len(train_df)} training samples, {len(test_df)} test samples")
    
    # Clean up the index column if it exists as an unnamed column
    if train_df.columns[0] == 'Unnamed: 0' or train_df.columns[0] == '':
        train_df = train_df.rename(columns={train_df.columns[0]: 'Id'})
    if test_df.columns[0] == 'Unnamed: 0' or test_df.columns[0] == '':
        test_df = test_df.rename(columns={test_df.columns[0]: 'Id'})
    
    # Calculate the split ratio based on original data
    try:
        test_ratio = len(test_df) / (len(train_df) + len(test_df))
        print(f"Calculated test ratio: {test_ratio}")
        if test_ratio > 1 or test_ratio < 0.05:
            print(f"Calculated test ratio {test_ratio} is improper, using default value of 0.2")
            test_ratio = 0.2
    except Exception as e:
        print(f"Error calculating test ratio: {e}, using default value of 0.2")
        test_ratio = 0.2
    
    print(f"Using test ratio: {test_ratio}")
    
    # Split the original training data into new train and test sets
    train_features = train_df.drop(columns=['SeriousDlqin2yrs'])
    train_target = train_df['SeriousDlqin2yrs']
    
    X_train, X_test, y_train, y_test = train_test_split(
        train_features, 
        train_target, 
        test_size=test_ratio,
        random_state=42
    )
    
    print(f"Split data: {len(X_train)} new training samples, {len(X_test)} new test samples")
    
    # Verify the split is correct
    assert len(X_train) + len(X_test) == len(train_df), "Split sizes don't add up to original size"
    assert abs(len(X_test) / len(train_df) - test_ratio) < 0.01, "Test ratio is not as expected"
    
    # Create the new training file
    new_train_df = pd.concat([pd.Series(y_train).reset_index(drop=True), X_train.reset_index(drop=True)], axis=1)
    new_train_df.to_csv(public / "cs-training.csv", index=False)
    print("Created new training file")
    
    # Create the new test file (without the target column)
    new_test_df = X_test.reset_index(drop=True)
    new_test_df.to_csv(public / "cs-test.csv", index=False)
    print("Created new test file")
    
    # Create sample_submission.csv
    sample_entry_df = pd.read_csv(raw / "sampleEntry.csv")
    sample_submission_df = pd.DataFrame({
        'Id': new_test_df['Id'],
        'Probability': [0.0] * len(new_test_df)
    })
    sample_submission_df.to_csv(public / "sample_submission.csv", index=False)
    print("Created sample_submission.csv (sample_submission.csv)")
    
    # Create test_answer.csv (the ground truth for the test set)
    test_answer_df = pd.DataFrame({
        'Id': new_test_df['Id'],
        'Probability': y_test.reset_index(drop=True)
    })
    test_answer_df.to_csv(private / "test_answer.csv", index=False)
    print("Created test_answer.csv")
    
    # Verify test_answer.csv and sample_submission.csv have the same columns
    sample_submission_cols = list(pd.read_csv(public / "sample_submission.csv").columns)
    test_answer_cols = list(pd.read_csv(private / "test_answer.csv").columns)
    assert sample_submission_cols == test_answer_cols, "Column mismatch between test_answer.csv and sample_submission.csv"
    print("Verified test_answer.csv and sample_submission.csv have the same columns")
    
    # Copy additional files if they exist
    data_dict_path = raw / "Data Dictionary.xls"
    if data_dict_path.exists():
        shutil.copy(data_dict_path, public)
        print("Copied Data Dictionary.xls")
    
    print("Data preparation complete!")

    # clean up
    shutil.rmtree(raw)