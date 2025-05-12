#!/usr/bin/env python3
import os
import shutil
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split


def prepare(raw: Path, public: Path, private: Path):
    """
    Prepares data for the quora-question-pairs competition by
    splitting raw data into public and private datasets.
    
    Args:
        raw: Path to the raw data directory
        public: Path to the public output directory
        private: Path to the private output directory
    """
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Create output directories
    public.mkdir(parents=True, exist_ok=True)
    private.mkdir(parents=True, exist_ok=True)
    
    os.system(f"rm quora-question-pairs.zip")
    os.system("unzip train.csv.zip")
    # os.system("unzip test.csv.zip")
    os.system("unzip sample_submission.csv.zip")
    os.system("rm train.csv.zip test.csv.zip sample_submission.csv.zip")
    # Load the original train.csv and test.csv
    train_df = pd.read_csv(raw / "train.csv")
    competition_test_df = pd.read_csv(raw / "test.csv")
    
    # Determine split ratio based on original competition ratio
    m = len(train_df)
    n = len(competition_test_df)
    test_ratio = n / (m + n)
    
    # Shuffle and split the training data
    shuffled_train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
    split_idx = int(m * (1 - test_ratio))
    local_train = shuffled_train_df.iloc[:split_idx].copy()
    local_test = shuffled_train_df.iloc[split_idx:].copy()
    
    # Save local training set to public/train.csv
    local_train.to_csv(public / "train.csv", index=False)
    
    # Save local test set to public/test.csv
    local_test.to_csv(public / "test.csv", index=False)
    
    # Create test_answer.csv in private directory
    test_answer = local_test[['id', 'is_duplicate']].copy()
    test_answer.rename(columns={'id': 'test_id'}, inplace=True)
    test_answer.to_csv(private / "test_answer.csv", index=False)
    
    # Create sample_submission.csv in public directory
    sample_submission = pd.DataFrame({
        'test_id': local_test['id'],
        'is_duplicate': 0.5  # dummy predictions
    })
    sample_submission.to_csv(public / "sample_submission.csv", index=False)
    
    # Verify data integrity
    _verify_data_integrity(train_df, local_train, local_test, public, private, test_ratio)
    
    print("Data preparation completed successfully.")
    
    # Clean up
    shutil.rmtree(raw)


def _verify_data_integrity(train_df, local_train, local_test, public, private, expected_ratio):
    """Helper function to verify data integrity with assertions"""
    
    # Total Count Check
    total_split_rows = len(local_train) + len(local_test)
    assert total_split_rows == len(train_df), "Total row count after split does not match original train.csv."
    
    # Ratio Check
    actual_ratio = len(local_test) / len(train_df)
    assert abs(actual_ratio - expected_ratio) < 0.05, "Local test split ratio does not approximate the expected ratio."
    
    # Column Check
    expected_columns = ['id', 'qid1', 'qid2', 'question1', 'question2', 'is_duplicate']
    assert list(local_train.columns) == expected_columns, "Train columns do not match expected"
    assert list(local_test.columns) == expected_columns, "Test columns do not match expected"
    
    # ID Consistency Check
    test_answer = pd.read_csv(private / "test_answer.csv")
    sample_submission = pd.read_csv(public / "sample_submission.csv")
    assert set(sample_submission['test_id']) == set(test_answer['test_id']), "ID mismatch between sample_submission.csv and test_answer.csv"
    
    # Value Check for is_duplicate
    assert test_answer['is_duplicate'].isin([0, 1]).all(), "is_duplicate column values not in expected set {0,1}"
    
    # Sample Submission Format Check
    assert list(sample_submission.columns) == ['test_id', 'is_duplicate'], "sample_submission.csv headers mismatch"
    assert sample_submission['is_duplicate'].dtype == float, "is_duplicate column is not of float type"