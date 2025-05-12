#!/usr/bin/env python3
import os
import shutil
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

def prepare(raw: Path, public: Path, private: Path):
    """
    Prepare data for the predicting-red-hat-business-value competition.
    
    Args:
        raw: Path to raw data directory
        public: Path to public data directory
        private: Path to private data directory
    """
    # Set random seed for reproducibility
    np.random.seed(42)
    os.system("unzip -o " + str(raw / "act_train.csv.zip") + " -d " + str(raw))
    os.system("unzip -o " + str(raw / "act_test.csv.zip") + " -d " + str(raw))
    os.system("unzip -o " + str(raw / "people.csv.zip") + " -d " + str(raw))
    os.system("unzip -o " + str(raw / "sample_submission.csv.zip") + " -d " + str(raw))
    
    # Create output directories
    public.mkdir(exist_ok=True, parents=True)
    private.mkdir(exist_ok=True, parents=True)
    
    # Define paths for input files
    act_train_path = raw / "act_train.csv"
    act_test_path = raw / "act_test.csv"
    people_path = raw / "people.csv"
    
    print("Reading activity data...")
    act_train = pd.read_csv(act_train_path)
    act_test = pd.read_csv(act_test_path)
    N_train = act_train.shape[0]
    N_test_orig = act_test.shape[0]
    
    # Verify required columns exist
    assert "activity_id" in act_train.columns, "'activity_id' column missing in act_train.csv"
    assert "outcome" in act_train.columns, "'outcome' column missing in act_train.csv"
    assert N_train > N_test_orig, "act_train.csv does not have more rows than act_test.csv for splitting"
    
    print("Splitting training data into new train and test sets...")
    # Perform random permutation split
    shuffled_indices = np.random.permutation(N_train)
    test_indices = shuffled_indices[:N_test_orig]
    train_indices = shuffled_indices[N_test_orig:]
    
    new_train = act_train.iloc[train_indices].copy()
    new_test = act_train.iloc[test_indices].copy()
    
    # Verify split integrity
    assert new_train.shape[0] + new_test.shape[0] == N_train, "Split size mismatch"
    assert new_test.shape[0] == N_test_orig, "New test set size mismatch"
    
    print("Saving new train and test files...")
    # Save new training set
    new_train.to_csv(public / "act_train.csv", index=False)
    
    # Save new test set (without outcome)
    new_test.drop(columns=['outcome']).to_csv(public / "act_test.csv", index=False)
    
    print("Copying people.csv...")
    # Copy people.csv to public directory
    shutil.copy2(people_path, public / "people.csv")
    
    print("Creating test_answer.csv...")
    # Create test_answer.csv with activity_id and outcome
    test_answer = new_test[["activity_id", "outcome"]].copy()
    test_answer.to_csv(private / "test_answer.csv", index=False)
    
    print("Creating sample_submission.csv...")
    # Create sample_submission.csv with same activity_ids and dummy outcomes
    dummy_submission = test_answer[["activity_id"]].copy()
    dummy_submission["outcome"] = 0.0
    dummy_submission.to_csv(public / "sample_submission.csv", index=False)
    
    print("Data preparation complete!")

    # Clean up
    shutil.rmtree(raw)