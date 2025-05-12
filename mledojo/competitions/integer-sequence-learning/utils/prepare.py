import os
import pandas as pd
import numpy as np
import shutil
import zipfile
import subprocess
import random
from pathlib import Path
from sklearn.model_selection import train_test_split

def prepare(raw: Path, public: Path, private: Path):
    """
    Prepares data for the Integer Sequence Learning competition by creating
    new train/test splits from the original train data.
    
    Args:
        raw: Path to the raw data directory
        public: Path to the public data directory
        private: Path to the private data directory
    """
    print("Starting data preparation...")
    
    # Create necessary directories
    public.mkdir(parents=True, exist_ok=True)
    private.mkdir(parents=True, exist_ok=True)
    
    # rm integer-sequence-learning.zip
    os.system("rm integer-sequence-learning.zip")

    # unzip other zip files
    for file in raw.glob("*.zip"):
        os.system(f"unzip {file} -d {raw}")
        os.system(f"rm {file}")

    # Define original directory
    original_dir = raw / "original"
    
    # Step 1: Analyze data structure and identify original train/test splits
    print("Analyzing data structure...")
    
    if original_dir.exists():
        print("Original folder exists, using files from there.")
    else:
        print("Creating original folder and moving files there...")
        original_dir.mkdir(parents=True, exist_ok=True)
        
        # Move original files to original directory
        for file in ["train.csv", "test.csv", "sample_submission.csv"]:
            src_path = raw / file
            if src_path.exists():
                shutil.copy(src_path, original_dir / file)
                print(f"Copied {file} to original folder")
    
    # Load data
    train_path = original_dir / "train.csv" if original_dir.exists() else raw / "train.csv"
    test_path = original_dir / "test.csv" if original_dir.exists() else raw / "test.csv"
    
    print(f"Loading train data from {train_path}")
    train_df = pd.read_csv(train_path)
    print(f"Loading test data from {test_path}")
    test_df = pd.read_csv(test_path)
    
    # Step 2: Calculate the split ratio
    train_size = len(train_df)
    test_size = len(test_df)
    
    try:
        test_train_ratio = test_size / train_size
        if test_train_ratio >= 1 or test_train_ratio <= 0:
            print(f"Calculated ratio {test_train_ratio} is improper, using default 0.2")
            test_train_ratio = 0.2
        else:
            print(f"Using calculated test/train ratio: {test_train_ratio}")
    except:
        print("Error in calculating ratio, using default 0.2")
        test_train_ratio = 0.2
    
    print(f"Original train size: {train_size}, test size: {test_size}")
    test_train_ratio = 0.2  # hardcode it to 0.2
    
    # Step 3 & 4: Create new train and test splits from original train files
    print("Creating new train-test split...")
    
    # Identify Id and target columns
    id_col = 'Id'  # Based on CSV samples
    target_col = 'Last'  # Based on submission format
    sequence_col = 'Sequence'  # Based on CSV samples
    
    # Split the original train data
    train_df_shuffled = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
    new_test_size = int(len(train_df_shuffled) * test_train_ratio)
    new_train_df = train_df_shuffled[new_test_size:].reset_index(drop=True)
    new_test_df = train_df_shuffled[:new_test_size].reset_index(drop=True)
    
    print(f"New train size: {len(new_train_df)}, new test size: {len(new_test_df)}")
    
    # Step 5 & 6: Prepare test data without target and save test answers
    print("Preparing test data and test answers...")
    
    # Extract the last number from each sequence for test data
    new_test_answers = pd.DataFrame(columns=[id_col, target_col])
    new_test_without_last = pd.DataFrame(columns=[id_col, sequence_col])
    
    for idx, row in new_test_df.iterrows():
        sequence = row[sequence_col]
        seq_numbers = [int(num.strip()) for num in sequence.split(',')]
        last_number = seq_numbers[-1]
        remaining_sequence = ','.join(str(num) for num in seq_numbers[:-1])
        
        # Add to test answers
        new_test_answers = pd.concat([
            new_test_answers,
            pd.DataFrame({
                id_col: [row[id_col]],
                target_col: [last_number]
            })
        ], ignore_index=True)
        
        # Add to test without last element
        new_test_without_last = pd.concat([
            new_test_without_last,
            pd.DataFrame({
                id_col: [row[id_col]],
                sequence_col: [remaining_sequence]
            })
        ], ignore_index=True)
    
    # Step 7: Create sample submission file
    print("Creating sample submission file...")
    sample_submission = new_test_answers.copy()
    # Replace actual answers with zeros or random values
    sample_submission[target_col] = 0
    
    # Step 8: Save all new files
    print("Saving new files...")
    
    # Save train and test files
    new_train_df.to_csv(public / "train.csv", index=False)
    new_test_without_last.to_csv(public / "test.csv", index=False)
    
    # Save test answers (private)
    new_test_answers.to_csv(private / "test_answer.csv", index=False)
    
    # Save sample submission
    sample_submission.to_csv(public / "sample_submission.csv", index=False)
    
    # Step 9: Validate with assertions
    print("Validating splits...")
    assert len(new_train_df) + len(new_test_df) == len(train_df), "Training and test split sizes don't match original"
    assert len(new_test_without_last) == len(new_test_answers), "Test and answer files have different sizes"
    assert set(new_test_without_last[id_col]) == set(new_test_answers[id_col]), "Test and answer files have different IDs"
    
    # Step 11: Copy any additional files from raw to public directory
    print("Copying additional files...")
    
    for item_path in raw.iterdir():
        if item_path.is_file() and item_path.name not in ["train.csv", "test.csv", "sample_submission.csv"]:
            print(f"Copying additional file: {item_path.name}")
            shutil.copy(item_path, public / item_path.name)
    
    print("Data preparation complete!")
    
    # clean up
    shutil.rmtree(raw)

