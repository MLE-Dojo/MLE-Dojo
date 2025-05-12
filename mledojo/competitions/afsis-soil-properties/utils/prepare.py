import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import shutil
from pathlib import Path

def prepare(raw: Path, public: Path, private: Path):
    """
    Prepare data for agent evaluation by creating a new train/test split from the existing training data.
    Also creates test_answer.csv and sample_submission.csv files.
    
    Args:
        raw: Path to the raw data directory
        public: Path to the public output directory
        private: Path to the private output directory
    """
    print("Starting data preparation process...")
    
    # Create output directories if they don't exist
    for directory in [public, private]:
        if not directory.exists():
            directory.mkdir(parents=True)
            print(f"Created directory: {directory}")
    
    # extract the zip files
    os.system(f"unzip {raw / 'train.zip'}")
    os.system(f"unzip {raw / 'test.zip'}")
    
    # Load the training data
    print("Loading training data...")
    train_data = pd.read_csv(raw / 'training.csv')
    
    # Load the test data to calculate the original split ratio
    print("Loading test data...")
    test_data = pd.read_csv(raw / 'sorted_test.csv')
    
    # Calculate the split ratio based on original test/train sizes
    try:
        print("Calculating split ratio...")
        original_train_size = len(train_data)
        original_test_size = len(test_data)
        total_size = original_train_size + original_test_size
        test_ratio = original_test_size / total_size
        
        # Check if the ratio is reasonable
        if test_ratio <= 0.05 or test_ratio >= 1.0:
            print(f"Calculated test ratio {test_ratio} is out of reasonable range. Using default ratio of 0.3")
            test_ratio = 0.3
        else:
            print(f"Using calculated test ratio: {test_ratio}")
    except Exception as e:
        print(f"Error calculating split ratio: {e}. Using default ratio of 0.3")
        test_ratio = 0.3
    
    print(f"Train size: {original_train_size}, Test size: {original_test_size}, Test ratio: {test_ratio}")
    
    # Split the train data into new train and test sets
    print("Splitting data into new train and test sets...")
    new_train, new_test = train_test_split(train_data, test_size=test_ratio, random_state=42)
    
    print(f"New train size: {len(new_train)}, New test size: {len(new_test)}")
    
    # Create test_answer.csv (copy of the new test data with target columns only)
    print("Creating test_answer.csv...")
    test_answer = new_test[['PIDN', 'Ca', 'P', 'pH', 'SOC', 'Sand']].copy()
    
    # Create sample_submission.csv (structure from test data but with zeros for target values)
    print("Creating sample_submission.csv...")
    sample_submission = new_test[['PIDN']].copy()
    for col in ['Ca', 'P', 'pH', 'SOC', 'Sand']:
        sample_submission[col] = 0
    
    # Save the new files
    print("Saving new train data...")
    new_train.to_csv(public / 'train.csv', index=False)
    
    print("Saving new test data...")
    # For test data, we should remove the target columns
    new_test_public = new_test.drop(columns=['Ca', 'P', 'pH', 'SOC', 'Sand'])
    new_test_public.to_csv(public / 'test.csv', index=False)
    
    print("Saving sample_submission.csv...")
    sample_submission.to_csv(public / 'sample_submission.csv', index=False)
    
    print("Saving test_answer.csv...")
    test_answer.to_csv(private / 'test_answer.csv', index=False)
    
    # Validation checks
    print("Performing validation checks...")
    
    # Check if the test train split is proper
    assert len(new_train) + len(new_test) == len(train_data), "Split validation failed: total count mismatch"
    assert abs(len(new_test) / len(train_data) - test_ratio) < 0.01, "Split validation failed: ratio mismatch"
    print("Split validation passed!")
    
    # Check if the test_answer.csv and sample_submission.csv have the same columns
    test_answer_cols = list(test_answer.columns)
    sample_submission_cols = list(sample_submission.columns)
    assert test_answer_cols == sample_submission_cols, "Column validation failed: columns don't match"
    print("Column validation passed!")
    
    # Final confirmation
    print("Data preparation completed successfully!")
    print(f"New files created in: {public} and {private}")

    # clean up
    shutil.rmtree(raw)
