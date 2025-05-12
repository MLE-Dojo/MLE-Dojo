#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np
from pathlib import Path
import shutil
from tqdm import tqdm

def prepare(raw: Path, public: Path, private: Path):
    """
    Prepare data for the pubg-finish-placement-prediction competition.
    
    Args:
        raw: Path to raw data directory
        public: Path to public data directory
        private: Path to private data directory
    """
    # Set fixed seed for reproducibility
    SEED = 42
    np.random.seed(SEED)
    
    # Create output directories
    public.mkdir(exist_ok=True, parents=True)
    private.mkdir(exist_ok=True, parents=True)
    
    # Extract zip files if needed
    os.system(f"unzip -o {raw / 'train_V2.csv.zip'} -d {raw}")
    os.system(f"unzip -o {raw / 'test_V2.csv.zip'} -d {raw}")
    
    # Read the original CSV files
    print("Reading original CSV files...")
    train_df = pd.read_csv(raw / "train_V2.csv")
    test_original_df = pd.read_csv(raw / "test_V2.csv")

    # Determine the sizes for splitting
    N_total = len(train_df)
    N_test_original = len(test_original_df)
    print(f"Total rows in train_V2.csv: {N_total}")
    print(f"Number of rows in original test_V2.csv: {N_test_original}")

    # Split train_df into new test set and new train set
    print("Splitting the training data...")
    new_test_set = train_df.sample(n=N_test_original, random_state=SEED)
    new_train_set = train_df.drop(new_test_set.index)

    # Verify split integrity
    assert len(new_test_set) == N_test_original, "New test set row count does not match original test count."
    assert len(new_train_set) == (N_total - N_test_original), "New train set row count not matching expected."
    assert (len(new_train_set) + len(new_test_set)) == N_total, "Total row count mismatch after splitting."

    # Save the new CSV files
    print("Saving new CSV files...")
    new_train_set.to_csv(public / "train.csv", index=False)
    
    # Create test.csv (without winPlacePerc)
    test_df = new_test_set.copy()
    if 'winPlacePerc' in test_df.columns:
        test_df = test_df.drop(columns=['winPlacePerc'])
    test_df.to_csv(public / "test.csv", index=False)

    # Create test_answer.csv with Id and winPlacePerc
    if 'Id' in new_test_set.columns and 'winPlacePerc' in new_test_set.columns:
        test_answer_df = new_test_set[['Id', 'winPlacePerc']]
    else:
        raise KeyError("Required columns ('Id' and 'winPlacePerc') not found in train_V2.csv.")
    
    test_answer_df.to_csv(private / "test_answer.csv", index=False)

    # Create sample_submission.csv with dummy answers
    sample_submission_df = test_answer_df.copy()
    sample_submission_df["winPlacePerc"] = 0.0
    sample_submission_df.to_csv(public / "sample_submission.csv", index=False)

    print("Data reorganization completed successfully.")
    
    # Clean up
    shutil.rmtree(raw)
