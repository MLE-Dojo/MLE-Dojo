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
    public.mkdir(exist_ok=True, parents=True)
    private.mkdir(exist_ok=True, parents=True)
    
    # unzip the zip files
    os.system(f"unzip {raw / 'train.csv.zip'} -d {raw}")
    os.system(f"unzip {raw / 'test.csv.zip'} -d {raw}")
    os.system(f"unzip {raw / 'bids.csv.zip'} -d {raw}")
    
    # Load the original data
    print("Loading original data...")
    try:
        train_df = pd.read_csv(raw / "train.csv")
        test_df = pd.read_csv(raw / "test.csv")
        
        # Check if bids.csv exists and copy it if it does
        if (raw / "bids.csv").exists():
            print("Copying bids.csv to public folder...")
            shutil.copy(raw / "bids.csv", public / "bids.csv")
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
        # Split the data
        new_train_df, new_test_df = train_test_split(
            train_df, test_size=split_ratio, random_state=42
        )
        
        print(f"New train size: {len(new_train_df)}, New test size: {len(new_test_df)}")
        
        # Validate the split
        assert len(new_train_df) + len(new_test_df) == len(train_df), "Split sizes don't match original size"
        assert len(new_test_df) > 0, "Test set is empty"
        assert len(new_train_df) > 0, "Train set is empty"
        
        # Create a test_answer.csv (private)
        print("Creating test_answer.csv...")
        test_answer = new_test_df[["bidder_id", "outcome"]].rename(columns={"outcome": "prediction"})
        
        # Create a sample_submission.csv (public)
        print("Creating sample_submission.csv...")
        sample_submission_new = pd.DataFrame({"bidder_id": new_test_df["bidder_id"]})
        sample_submission_new["prediction"] = 0.0  # Default prediction
        
        # Validate that test_answer and sample_submission have the same columns
        assert set(test_answer.columns) == set(sample_submission_new.columns), "Columns in test_answer and sample_submission don't match"
        
        # Remove the outcome column from the new test set
        new_test_df = new_test_df.drop(columns=["outcome"])
        
        # Save the files
        print("Saving new files...")
        new_train_df.to_csv(public / "train.csv", index=False)
        new_test_df.to_csv(public / "test.csv", index=False)
        sample_submission_new.to_csv(public / "sample_submission.csv", index=False)
        test_answer.to_csv(private / "test_answer.csv", index=False)
        
        print("Data preparation completed successfully!")
    except Exception as e:
        print(f"Error during data preparation: {e}")

    # clean up
    shutil.rmtree(raw)
