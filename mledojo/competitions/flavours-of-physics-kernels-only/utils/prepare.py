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
    os.system(f"unzip {raw / 'training.csv.zip'} -d {raw}")
    os.system(f"unzip {raw / 'test.csv.zip'} -d {raw}")
    os.system(f"unzip {raw / 'sample_submission.csv.zip'} -d {raw}")
    os.system(f"unzip {raw / 'check_agreement.csv.zip'} -d {raw}")
    os.system(f"unzip {raw / 'check_correlation.csv.zip'} -d {raw}")
    
    # Load the original data
    print("Loading original data...")
    try:
        training_df = pd.read_csv(raw / "training.csv")
        test_df = pd.read_csv(raw / "test.csv")
        sample_submission = pd.read_csv(raw / "sample_submission.csv")
        
        # Check if additional files exist
        check_agreement_path = raw / "check_agreement.csv"
        check_correlation_path = raw / "check_correlation.csv"
        
        if os.path.exists(check_agreement_path):
            print("Found check_agreement.csv, will copy to public folder")
            shutil.copy(check_agreement_path, public / "check_agreement.csv")
            
        if os.path.exists(check_correlation_path):
            print("Found check_correlation.csv, will copy to public folder")
            shutil.copy(check_correlation_path, public / "check_correlation.csv")
        
        print(f"Loaded training.csv with {len(training_df)} rows and {len(training_df.columns)} columns")
        print(f"Loaded test.csv with {len(test_df)} rows and {len(test_df.columns)} columns")
        print(f"Loaded sample_submission.csv with {len(sample_submission)} rows and {len(sample_submission.columns)} columns")
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Calculate split ratio based on original test/train sizes
    try:
        original_train_size = len(training_df)
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
        # For this competition, we need to ensure we have both signal and background in our train and test sets
        train_signal = training_df[training_df['signal'] == 1]
        train_background = training_df[training_df['signal'] == 0]
        
        # Split signal and background separately to maintain their proportions
        train_signal_new, test_signal_new = train_test_split(
            train_signal, test_size=split_ratio, random_state=42
        )
        
        train_background_new, test_background_new = train_test_split(
            train_background, test_size=split_ratio, random_state=42
        )
        
        # Combine the split datasets
        new_train_df = pd.concat([train_signal_new, train_background_new])
        new_test_df = pd.concat([test_signal_new, test_background_new])
        
        print(f"New train size: {len(new_train_df)}, New test size: {len(new_test_df)}")
        print(f"Train signal: {len(train_signal_new)}, Train background: {len(train_background_new)}")
        print(f"Test signal: {len(test_signal_new)}, Test background: {len(test_background_new)}")
        
        # Validate the split
        assert len(new_train_df) + len(new_test_df) == len(training_df), "Split sizes don't match original size"
        assert len(new_test_df) > 0, "Test set is empty"
        assert len(new_train_df) > 0, "Train set is empty"
        
        # Create test_answer.csv (private) first
        print("Creating test_answer.csv...")
        test_answer = pd.DataFrame()
        test_answer['id'] = new_test_df['id']
        test_answer['prediction'] = new_test_df['signal'].astype(float)  # Ground truth as prediction
        test_answer['min_ANNmuon'] = new_test_df['min_ANNmuon'].astype(float)
        
        # Then create new test data (public) without signal column
        test_columns = test_df.columns.tolist()  # Get columns from original test
        new_test_public = new_test_df[test_columns].copy()
        
        # Create sample_submission.csv (public)
        print("Creating sample_submission.csv...")
        sample_submission_new = pd.DataFrame()
        sample_submission_new['id'] = new_test_df['id']
        sample_submission_new['prediction'] = 0.5  # Default value, typically 0.5 for binary classification
        
        # Validate that test_answer and sample_submission have the same columns
        # assert set(test_answer.columns) == set(sample_submission_new.columns), "Columns in test_answer and sample_submission don't match"
        
        # Save the files
        print("Saving new files...")
        new_train_df.to_csv(public / "training.csv", index=False)
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
