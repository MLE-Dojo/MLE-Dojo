#!/usr/bin/env python3

import os
import sys
import pandas as pd
import numpy as np
import shutil
import subprocess
import zipfile
import random
from pathlib import Path
from sklearn.model_selection import train_test_split

def prepare(raw: Path, public: Path, private: Path):
    """
    Prepare data for loan default prediction by creating new train/test splits.
    
    Args:
        raw: Path to the raw data directory
        public: Path to the public data directory
        private: Path to the private data directory
    """
    print("Starting data preparation process...")
    
    # Create directories if they don't exist
    public.mkdir(parents=True, exist_ok=True)
    private.mkdir(parents=True, exist_ok=True)
    original_folder = raw / "original"
    original_folder.mkdir(parents=True, exist_ok=True)
    
    print("Directories created or verified.")
    
    # Step 1: Analyze data structure and identify original train/test splits
    print("Analyzing data structure...")
    
    files = os.listdir(raw)
    
    # Move files to original folder if they're not already there
    train_file_path = ""
    test_file_path = ""
    
    if "original" in files and len(os.listdir(original_folder)) > 0:
        print("Using files from existing 'original' folder")
        orig_files = os.listdir(original_folder)
        train_file = [f for f in orig_files if "train" in f]
        test_file = [f for f in orig_files if "test" in f and "sample" not in f]
        
        if train_file and test_file:
            train_file_path = original_folder / train_file[0]
            test_file_path = original_folder / test_file[0]
    else:
        print("Moving files to 'original' folder")
        train_file = [f for f in files if "train" in f]
        test_file = [f for f in files if "test" in f and "sample" not in f]
        sample_submission = [f for f in files if "sample" in f.lower()]
        
        if train_file:
            train_file_path = raw / train_file[0]
            orig_train_path = original_folder / train_file[0]
            shutil.copy2(train_file_path, orig_train_path)
            print(f"Copied {train_file[0]} to original folder")
            
        if test_file:
            test_file_path = raw / test_file[0]
            orig_test_path = original_folder / test_file[0]
            shutil.copy2(test_file_path, orig_test_path)
            print(f"Copied {test_file[0]} to original folder")
            
        if sample_submission:
            sample_sub_path = raw / sample_submission[0]
            orig_sample_path = original_folder / sample_submission[0]
            shutil.copy2(sample_sub_path, orig_sample_path)
            print(f"Copied {sample_submission[0]} to original folder")
    
    # Step 2: Calculate the split ratio based on original test/train sizes
    print("Calculating split ratio...")
    
    default_test_ratio = 0.2
    split_ratio = default_test_ratio
    
    try:
        if train_file_path.exists() and test_file_path.exists():
            train_df = pd.read_csv(train_file_path)
            test_df = pd.read_csv(test_file_path)
            
            train_size = len(train_df)
            test_size = len(test_df)
            
            total_size = train_size + test_size
            calculated_ratio = test_size / total_size
            
            if 0 < calculated_ratio < 1:
                split_ratio = calculated_ratio
                print(f"Calculated split ratio: {split_ratio:.4f}")
                print(f"Original train size: {train_size}, test size: {test_size}")
            else:
                print(f"Invalid calculated ratio: {calculated_ratio}. Using default ratio: {default_test_ratio}")
        else:
            print(f"Train or test file not found. Using default ratio: {default_test_ratio}")
    except Exception as e:
        print(f"Error calculating split ratio: {e}")
        print(f"Using default ratio: {default_test_ratio}")
    
    # Step 3: Create new train and test files from the original train files
    print("Creating new train and test splits...")
    
    if train_file_path.exists():
        train_df = pd.read_csv(train_file_path)
        
        # Step 4: Identify the label/target column
        print("Identifying target/label column...")
        
        # Based on the description, 'loss' is the target column
        target_col = 'loss'
        id_col = 'id'
        
        if target_col in train_df.columns:
            print(f"Target column identified: {target_col}")
            
            # Split the data
            features = train_df.drop(columns=[target_col])
            target = train_df[target_col]
            
            X_train, X_test, y_train, y_test = train_test_split(
                features, target, test_size=split_ratio, random_state=42
            )
            
            # Recombine for the new train set
            new_train = X_train.copy()
            new_train[target_col] = y_train
            
            # Test set shouldn't have the target column
            new_test = X_test.copy()
            
            print(f"New train size: {len(new_train)}, new test size: {len(new_test)}")
            
            # Step 5 & 6: Save test labels separately
            print("Saving test answers...")
            test_answers = pd.DataFrame({
                id_col: X_test[id_col],
                target_col: y_test
            })
            test_answers.to_csv(private / "test_answer.csv", index=False)
            
            # Step 7: Create sample submission file
            print("Creating sample submission file...")
            sample_submission = pd.DataFrame({
                id_col: X_test[id_col],
                target_col: np.zeros(len(X_test))  # Using zeros as placeholder values
            })
            sample_submission.to_csv(public / "sample_submission.csv", index=False)
            
            # Save the new train and test files
            print("Saving new train and test files...")
            
            new_train.to_csv(public / "train.csv", index=False)
            new_test.to_csv(public / "test.csv", index=False)
            
            # Step 8: Handle additional files/folders (not applicable for current structure)
            
            # Step 9: Validate the split
            print("Validating the split...")
            
            # Simple validations
            assert len(new_train) + len(new_test) == len(train_df), "Split sizes don't add up to original"
            assert set(new_test[id_col].values) == set(test_answers[id_col].values), "Test IDs don't match test answers"
            assert target_col not in new_test.columns, "Target column should not be in test set"
            
            print("Validation passed!")
            
            
            # Step 12: Copy additional files
            print("Copying additional files...")
            
            for file in files:
                file_path = raw / file
                if file_path.is_file() and file.lower() != "train_v2.csv" and file.lower() != "test_v2.csv" and "sample" not in file.lower() and file != "original":
                    dest_path = public / file
                    shutil.copy2(file_path, dest_path)
                    print(f"Copied additional file: {file}")
            
            print("Data preparation completed successfully!")
        else:
            print(f"Error: Target column '{target_col}' not found in train file")
    else:
        print("Error: Train file not found")

    # clean up
    shutil.rmtree(raw)
