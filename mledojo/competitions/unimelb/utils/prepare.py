#!/usr/bin/env python3
import os
import shutil
from math import isclose
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def prepare(raw: Path, public: Path, private: Path):
    """
    Prepare data for unimelb competition.
    
    Args:
        raw: Path to raw data directory
        public: Path to public data directory
        private: Path to private data directory
    """
    # Create destination directories
    for d in [private, public]:
        d.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Process training data splitting
    training_file = raw / "unimelb_training.csv"
    df = pd.read_csv(training_file)
    
    # Define test split size based on original competition ratio
    # Original: 10,883 = 8,707 (training) + 2,176 (test)
    local_test_fraction = 2176 / 10883
    N = len(df)
    expected_test_rows = round(N * local_test_fraction)
    
    # Use stratified splitting by "Grant Status"
    train_df, test_df = train_test_split(
        df,
        test_size=expected_test_rows,
        stratify=df["Grant.Status"],
        random_state=42,
        shuffle=True,
    )
    
    # Step 2: Write the newly split CSV files
    train_output_path = public / "train.csv"
    test_output_path = public / "test.csv"
    
    # Save training and test splits
    train_df.to_csv(train_output_path, index=False)
    
    # Create test file without the target column
    test_df_copy = test_df.copy()
    test_df_copy["Grant.Status"] = ""
    test_df_copy.to_csv(test_output_path, index=False)
    
    # Extract ground truth answers
    test_answer_columns = [test_df.columns[0], "Grant.Status"]
    test_answer_df = test_df[test_answer_columns].copy()
    test_answer_output_path = private / "test_answer.csv"
    test_answer_df.to_csv(test_answer_output_path, index=False)
    
    # Step 3: Create sample_submission.csv
    example_submission_path = raw / "unimelb_example.csv"
    example_df = pd.read_csv(example_submission_path)
    submission_columns = example_df.columns.tolist()
    
    sample_submission_df = pd.DataFrame({
        submission_columns[0]: test_answer_df[test_answer_columns[0]],
        submission_columns[1]: 0.0
    })
    
    sample_submission_df[submission_columns[1]] = sample_submission_df[submission_columns[1]].astype(float)
    sample_submission_output_path = public / "sample_submission.csv"
    sample_submission_df.to_csv(sample_submission_output_path, index=False)
    
    
    # Assertions for data integrity
    assert len(train_df) + len(test_df) == N, "Total rows in splits do not sum up to original count."
    assert abs(len(test_df) - expected_test_rows) <= 1, "Test split row count is not as expected."
    
    test_df_read = pd.read_csv(test_output_path)
    test_answer_read = pd.read_csv(test_answer_output_path)
    assert len(test_df_read) == len(test_answer_read), "Row counts between test split and test_answer do not match."
    
    id_col = test_df_read.columns[0]
    assert (test_df_read[id_col].values == test_answer_read[id_col].values).all(), "Mismatch in IDs between test split and test_answer."
    assert len(sample_submission_df) == len(test_answer_df), "sample_submission row count does not match test_answer row count."

    # Clean up
    shutil.rmtree(raw)