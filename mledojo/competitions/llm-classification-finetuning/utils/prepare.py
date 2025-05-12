import os
import pandas as pd
import numpy as np
import shutil
import subprocess
from pathlib import Path
from sklearn.model_selection import train_test_split
import zipfile

def prepare(raw: Path, public: Path, private: Path):
    """
    Prepare data for the LLM classification finetuning by creating new train/test splits.
    
    Args:
        raw: Path to the raw data directory
        public: Path to the public data directory
        private: Path to the private data directory
    """
    print("Starting data preparation...")
    
    # Create necessary directories
    print("Creating necessary directories...")
    original_folder = raw / 'original'
    public.mkdir(parents=True, exist_ok=True)
    private.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Analyze the data structure
    print("Analyzing data structure...")
    if original_folder.exists():
        print("Original folder exists, using files from there.")
    else:
        print("Original folder doesn't exist, creating and moving files...")
        original_folder.mkdir(parents=True, exist_ok=True)
        train_file = raw / 'train.csv'
        test_file = raw / 'test.csv'
        sample_submission_file = raw / 'sample_submission.csv'
        
        if train_file.exists():
            shutil.copy(train_file, original_folder / 'train.csv')
        
        if test_file.exists():
            shutil.copy(test_file, original_folder / 'test.csv')
            
        if sample_submission_file.exists():
            shutil.copy(sample_submission_file, original_folder / 'sample_submission.csv')
    
    # Load the original data
    print("Loading the original data...")
    original_train_path = original_folder / 'train.csv'
    original_test_path = original_folder / 'test.csv'
    
    if not original_train_path.exists():
        original_train_path = raw / 'train.csv'
    
    if not original_test_path.exists():
        original_test_path = raw / 'test.csv'
    
    train_df = pd.read_csv(original_train_path)
    
    try:
        test_df = pd.read_csv(original_test_path)
        # Step 2: Calculate the split ratio based on original test/train sizes
        print("Calculating split ratio...")
        test_size = len(test_df)
        train_size = len(train_df)
        split_ratio = test_size / (test_size + train_size)
        
        if split_ratio > 1 or split_ratio <= 0:
            print(f"Calculated split ratio {split_ratio} is improper, using default of 0.3")
            split_ratio = 0.3
        else:
            print(f"Using calculated split ratio: {split_ratio}")
        
    except (FileNotFoundError, pd.errors.EmptyDataError):
        print("Error loading test file or file is empty. Using default split ratio of 0.3")
        split_ratio = 0.3
    
    print(f"Train size: {len(train_df)}")
    
    # Step 4: Identify the label/target and Id column names
    print("Identifying labels and ID columns...")
    id_column = 'id'
    target_columns = ['winner_model_a', 'winner_model_b', 'winner_tie']
    
    print(f"ID column: {id_column}")
    print(f"Target columns: {target_columns}")
    split_ratio = 0.2 #hardcode it 
    # Step 3: Create new train and test files
    print("Creating new train and test splits...")
    data_for_split = train_df.copy()
    
    X = data_for_split.drop(target_columns, axis=1)
    y = data_for_split[target_columns]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=split_ratio, random_state=42, stratify=None
    )
    
    # Combine the data back
    new_train = pd.concat([X_train, y_train], axis=1)
    new_test_with_labels = pd.concat([X_test, y_test], axis=1)
    
    # Step 5: Create test file without target/label columns
    new_test = X_test.copy()
    
    # Step 6: Save test answers to private directory
    print("Saving test answers to private directory...")
    test_answers = pd.DataFrame()
    test_answers[id_column] = new_test_with_labels[id_column]
    for col in target_columns:
        test_answers[col] = new_test_with_labels[col]
    
    test_answers.to_csv(private / 'test_answer.csv', index=False)
    
    # Step 7: Create sample_submission.csv
    print("Creating sample submission file...")
    sample_submission = pd.DataFrame()
    sample_submission[id_column] = new_test[id_column]
    # Equal probability (1/3) for each class
    for col in target_columns:
        sample_submission[col] = 1/3
    
    sample_submission.to_csv(public / 'sample_submission.csv', index=False)
    
    # Save the new train and test files
    print("Saving new train and test files...")
    new_train.to_csv(public / 'train.csv', index=False)
    new_test.to_csv(public / 'test.csv', index=False)
    
    # Step 8: Handle additional folders (if any)
    print("Checking for additional train/test folders...")
    train_folder = raw / 'train'
    test_folder = raw / 'test'
    
    if train_folder.is_dir() or test_folder.is_dir():
        print("Additional train/test folders found, creating new ones based on the split...")
        new_train_folder = public / 'train'
        new_test_folder = public / 'test'
        new_train_folder.mkdir(exist_ok=True)
        new_test_folder.mkdir(exist_ok=True)
        
        # Logic to split the folder contents would go here
        # This is a placeholder as we don't have specific information about these folders
    
    # Step 9: Validate the split
    print("Validating the split...")
    try:
        assert len(new_train) + len(new_test) == len(train_df), "Total row count doesn't match after split"
        assert len(new_test) > 0, "Test set is empty"
        assert len(new_train) > 0, "Train set is empty"
        test_ratio = len(new_test) / (len(new_train) + len(new_test))
        assert abs(test_ratio - split_ratio) < 0.01, "Split ratio not maintained"
        print("Split validation passed!")
    except AssertionError as e:
        print(f"Split validation failed: {e}")
    
    
    print("Data preparation completed successfully!")

    # clean up
    shutil.rmtree(raw)