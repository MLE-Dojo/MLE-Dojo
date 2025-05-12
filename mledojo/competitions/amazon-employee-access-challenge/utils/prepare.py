import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path
import shutil

def prepare(raw: Path, public: Path, private: Path):
    """
    Prepares data for evaluation by:
    1. Loading the original data
    2. Creating a new train/test split from the training data
    3. Creating test_answer.csv and sample_submission.csv
    4. Saving all files to the appropriate directories
    
    Args:
        raw: Path to the raw data directory
        public: Path to the public output directory
        private: Path to the private output directory
    """
    print("Starting data preparation...")
    
    # Create output directories if they don't exist
    os.makedirs(public, exist_ok=True)
    os.makedirs(private, exist_ok=True)
    print("Created output directories")
    
    # Load original data
    try:
        train_df = pd.read_csv(raw / "train.csv")
        test_df = pd.read_csv(raw / "test.csv")
        sample_submission = pd.read_csv(raw / "sampleSubmission.csv")
        print(f"Loaded original data - train: {train_df.shape}, test: {test_df.shape}")
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Calculate split ratio based on original sizes
    try:
        original_train_size = train_df.shape[0]
        original_test_size = test_df.shape[0]
        test_ratio = original_test_size / (original_train_size + original_test_size)
        
        # Validate the ratio
        if test_ratio <= 0.05 or test_ratio >= 1:
            print(f"Calculated ratio {test_ratio} is invalid, using default 0.2")
            test_ratio = 0.2
        else:
            print(f"Using calculated test ratio: {test_ratio}")
    except Exception as e:
        print(f"Error calculating split ratio: {e}. Using default 0.2")
        test_ratio = 0.2
    
    # Create new train and test splits
    try:
        new_train, new_test = train_test_split(train_df, test_size=test_ratio, random_state=42)
        print(f"Created new split - new train: {new_train.shape}, new test: {new_test.shape}")
        
        # Assertion to verify the split is proper
        assert len(new_train) + len(new_test) == len(train_df), "Split size mismatch!"
        assert 0.05 <= len(new_test) / len(train_df) <= 0.95, "Split ratio is not within reasonable bounds!"
        print("Split validation passed")
    except Exception as e:
        print(f"Error creating split: {e}")
        return
    
    # Create test_answer.csv (private)
    try:
        # Add 'id' column to new test data to match test.csv format
        new_test_copy = new_test.copy()
        new_test_copy.reset_index(inplace=True)
        new_test_copy.rename(columns={'index': 'id'}, inplace=True)
        
        # Save the ACTION column as the answers/target
        test_answer = new_test_copy[['id', 'ACTION']]
        test_answer.to_csv(private / "test_answer.csv", index=False)
        print("Created test_answer.csv in private folder")
    except Exception as e:
        print(f"Error creating test_answer.csv: {e}")
        return
    
    # Create new test.csv for public folder (without ACTION column)
    try:
        public_test = new_test_copy.drop(columns=['ACTION'])
        public_test.to_csv(public / "test.csv", index=False)
        print("Created new test.csv in public folder")
    except Exception as e:
        print(f"Error creating public test.csv: {e}")
        return
    
    # Create sample_submission.csv for public folder
    try:
        sample_sub = pd.DataFrame({
            'id': public_test['id'],
            'ACTION': [0] * len(public_test)
        })
        sample_sub.to_csv(public / "sample_submission.csv", index=False)
        print("Created sample_submission.csv in public folder")
    except Exception as e:
        print(f"Error creating sample_submission.csv: {e}")
        return
    
    # Save new train data to public folder
    try:
        new_train.to_csv(public / "train.csv", index=False)
        print("Created new train.csv in public folder")
    except Exception as e:
        print(f"Error creating train.csv: {e}")
        return
    
    # Check that test_answer.csv and sample_submission.csv have the same ids in the same order
    try:
        test_answer = pd.read_csv(private / "test_answer.csv")
        sample_sub = pd.read_csv(public / "sample_submission.csv")
        
        # Assert columns match
        assert set(test_answer.columns) == set(['id', 'ACTION']), "test_answer.csv columns don't match expected format"
        assert set(sample_sub.columns) == set(['id', 'ACTION']), "sample_submission.csv columns don't match expected format"
        
        # Assert ids match
        assert all(test_answer['id'] == sample_sub['id']), "IDs in test_answer.csv and sample_submission.csv don't match"
        print("Validation passed: test_answer.csv and sample_submission.csv formats match")
    except Exception as e:
        print(f"Error in validation: {e}")
        return
    
    print("Data preparation completed successfully!")

    # clean up
    shutil.rmtree(raw)