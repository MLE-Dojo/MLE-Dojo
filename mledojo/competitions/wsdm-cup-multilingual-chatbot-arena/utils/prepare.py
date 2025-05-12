#!/usr/bin/env python3
import os
import shutil
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

def prepare(raw: Path, public: Path, private: Path):
    """
    Prepare data for wsdm-cup-multilingual-chatbot-arena competition.
    
    Args:
        raw: Path to raw data directory
        public: Path to public data directory
        private: Path to private data directory
    """

    # Step 1: Load and split the original train.parquet
    print("Loading the original train.parquet...")
    df_train_original = pd.read_parquet(raw / "train.parquet")
    N_train = df_train_original.shape[0]
    
    print("Loading the original test.parquet to get row count...")
    df_test_original = pd.read_parquet(raw / "test.parquet")
    N_test = df_test_original.shape[0]
    
    # Compute the test fraction and new test rows count
    test_fraction = max(N_test / (N_train + N_test), 0.2)
    new_test_rows = int(round(N_train * test_fraction))
    new_train_rows = N_train - new_test_rows
    
    print(f"Original train rows: {N_train}")
    print(f"Original test rows: {N_test}")
    print(f"Computed new test rows: {new_test_rows}")
    print(f"Computed new train rows: {new_train_rows}")
    
    # Use a random shuffle with fixed seed for reproducibility
    print("Shuffling and splitting the original train data...")
    df_train_shuffled = df_train_original.sample(frac=1, random_state=42).reset_index(drop=True)
    df_new_test = df_train_shuffled.iloc[:new_test_rows].copy()
    df_new_train = df_train_shuffled.iloc[new_test_rows:].copy()
    
    # Final assertions to check row counts
    assert df_new_train.shape[0] + df_new_test.shape[0] == N_train, "Sum of new train and test rows does not equal original train rows."
    
    # Check that both splits have exactly the expected columns.
    expected_columns = ["id", "prompt", "response_a", "response_b", "winner", "model_a", "model_b", "language"]
    assert list(df_new_train.columns) == expected_columns, "Columns in new training set do not match expected columns."
    assert list(df_new_test.columns) == expected_columns, "Columns in new test set do not match expected columns."
    
    # Step 2: Save new training and test sets to public
    print("Saving new training set to public/train/train.parquet...")
    train_out_path = public / "train.parquet"
    df_new_train.to_parquet(train_out_path, index=False)
    
    print("Saving new test set to public/test/test.parquet...")
    test_out_path = public / "test.parquet"
    df_new_test.drop(columns=["winner"]).to_parquet(test_out_path, index=False)
    
    # Step 3: Create new sample_submission.csv for the test set in public
    print("Creating new sample_submission.csv for the test set...")
    submission_df = pd.DataFrame()
    submission_df["id"] = df_new_test["id"]
    submission_df["winner"] = "model_a"  # Fill dummy valid answers
    
    # Assertion: Check header and row count
    assert list(submission_df.columns) == ["id", "winner"], "sample_submission header does not match required format."
    assert submission_df.shape[0] == new_test_rows, "sample_submission row count does not equal new test rows."
    
    submission_out_path = public / "sample_submission.csv"
    submission_df.to_csv(submission_out_path, index=False)
    
    # Step 4: Create private/test_answer.csv from new test set's ground-truth answers
    print("Creating private/test_answer.csv with ground-truth answers...")
    test_answer_df = df_new_test[["id", "winner"]].copy()
    
    # Assertions: Check header and row count
    assert list(test_answer_df.columns) == ["id", "winner"], "test_answer.csv header does not match required format."
    assert test_answer_df.shape[0] == new_test_rows, "test_answer.csv row count does not equal new test rows."
    
    test_answer_out_path = private / "test_answer.csv"
    test_answer_df.to_csv(test_answer_out_path, index=False)
    
    print("Data reorganization is complete.")

    # Clean up
    shutil.rmtree(raw)