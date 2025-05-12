import os
import pandas as pd
import numpy as np
import shutil
import subprocess
import zipfile
from pathlib import Path
from sklearn.model_selection import train_test_split

def prepare(raw: Path, public: Path, private: Path):
    """
    Prepare data for Kobe Bryant Shot Selection competition.
    
    Args:
        raw: Path to raw data directory
        public: Path to public data directory
        private: Path to private data directory
    """
    print("Starting data preparation process...")

    # unzip the data.zip file
    os.system(f"unzip {raw / 'data.csv.zip'} -d {raw}")
    os.system(f"unzip {raw / 'sample_submission.csv.zip'} -d {raw}")
    
    # Create necessary directories
    os.makedirs(public, exist_ok=True)
    os.makedirs(private, exist_ok=True)
    os.makedirs(raw / "original", exist_ok=True)
    
    print("Created necessary directories.")
    
    # Step 1: Check if original folder exists and move files if needed
    if not (raw / "original" / "data.csv").exists():
        print("Moving original files to original folder...")
        shutil.copy(raw / "data.csv", raw / "original" / "data.csv")
        shutil.copy(raw / "sample_submission.csv", raw / "original" / "sample_submission.csv")
    
    # Step 2: Calculate the split ratio based on original data
    print("Calculating split ratio based on original data...")
    
    # Load the data
    data = pd.read_csv(raw / "data.csv")
    sample_submission = pd.read_csv(raw / "sample_submission.csv")
    
    # Identify test samples (where shot_made_flag is null) and train samples
    train_data = data[data['shot_made_flag'].notna()]
    test_data_original = data[data['shot_made_flag'].isna()]
    
    train_size = len(train_data)
    test_size = len(test_data_original)
    
    # Calculate the ratio
    try:
        ratio = test_size / train_size
        print(f"Original train size: {train_size}, test size: {test_size}")
        print(f"Calculated split ratio: {ratio:.4f}")
        if ratio > 1:
            print("Warning: Calculated ratio > 1, using default ratio of 0.2")
            ratio = 0.2
    except Exception as e:
        print(f"Error calculating ratio: {e}, using default ratio of 0.2")
        ratio = 0.2
    
    # Step 3 & 4: Identify target and ID columns and create new train-test split
    print("Creating new train-test split...")
    
    target_column = 'shot_made_flag'  # As identified from the description
    id_column = 'shot_id'  # As identified from the description
    
    # Split the train_data into new train and test sets
    train_new, test_new = train_test_split(train_data, test_size=ratio, random_state=42)
    
    print(f"New train size: {len(train_new)}, new test size: {len(test_new)}")
    
    # Step 5 & 6: Create test data without target column and save test answers
    print("Creating test data and saving test answers...")
    
    # Save test answers
    test_answers = test_new[[id_column, target_column]].copy()
    test_answers.to_csv(private / "test_answer.csv", index=False)
    
    # Create test data without target column
    test_new_public = test_new.drop(columns=[target_column])
    
    # Step 7: Create sample submission file
    print("Creating sample submission file...")
    sample_submission_new = pd.DataFrame({
        id_column: test_new_public[id_column],
        target_column: 0.5  # Default prediction value
    })
    sample_submission_new.to_csv(public / "sample_submission.csv", index=False)
    
    # Step 8: Handle additional folders if any (not applicable in this case)
    
    # Step 9: Validate with assertions
    print("Validating the split...")
    assert len(train_new) + len(test_new) == len(train_data), "Train-test split validation failed!"
    assert target_column not in test_new_public.columns, "Test data should not contain target column!"
    assert len(test_answers) == len(test_new_public), "Test answers and test data size mismatch!"
    
    # Step 10: Check if we need to zip files based on size
    print("Checking file sizes...")
    
    # Save train and test data
    train_path = public / "train.csv"
    test_path = public / "test.csv"
    
    train_new.to_csv(train_path, index=False)
    test_new_public.to_csv(test_path, index=False)
    

    # clean up
    shutil.rmtree(raw)
