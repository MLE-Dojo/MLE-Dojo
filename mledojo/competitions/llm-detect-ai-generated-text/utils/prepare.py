#!/usr/bin/env python3
import os
import shutil
import pandas as pd
import numpy as np
from pathlib import Path


def prepare(raw: Path, public: Path, private: Path):
    """
    Prepares data for the llm-detect-ai-generated-text competition by
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
    
    # Read the CSV files from the raw data folder
    train_essays_df = pd.read_csv(raw / "train_essays.csv")
    train_prompts_df = pd.read_csv(raw / "train_prompts.csv")
    
    # Set fixed 8:2 train:test ratio
    test_ratio = 0.2
    
    # Shuffle train_essays_df and split according to test_ratio
    shuffled_df = train_essays_df.sample(frac=1, random_state=42).reset_index(drop=True)
    n_test = int(round(len(shuffled_df) * test_ratio))
    test_partition_df = shuffled_df.iloc[:n_test].copy()
    train_partition_df = shuffled_df.iloc[n_test:].copy()
    
    # Export the organized training set with all columns (including "generated")
    train_partition_df.to_csv(public / "train.csv", index=False)
    
    # Create test.csv (features only) and test_answer.csv (answers)
    public_test_df = test_partition_df[["id", "prompt_id", "text"]].copy()
    private_test_answer_df = test_partition_df[["id", "generated"]].copy()
    
    public_test_df.to_csv(public / "test.csv", index=False)
    private_test_answer_df.to_csv(private / "test_answer.csv", index=False)
    
    # Copy train_prompts.csv into public directory unchanged
    shutil.copyfile(raw / "train_prompts.csv", public / "train_prompts.csv")
    
    # Build sample_submission.csv based on test_answer.csv
    sample_submission_df = private_test_answer_df.copy()
    sample_submission_df["generated"] = 0.0  # assign dummy value for each row
    sample_submission_df.to_csv(public / "sample_submission.csv", index=False)
    
    # Run assertions for verification
    _verify_data_integrity(
        train_essays_df, 
        train_partition_df, 
        public_test_df, 
        private_test_answer_df, 
        sample_submission_df, 
        train_prompts_df,
        public,
        test_ratio
    )
    
    print("Data reorganization completed successfully.")

    # Clean up
    shutil.rmtree(raw)


def _verify_data_integrity(
    train_essays_df, 
    train_partition_df, 
    public_test_df, 
    private_test_answer_df, 
    sample_submission_df, 
    train_prompts_df,
    public,
    test_ratio
):
    """Helper function to verify data integrity with assertions"""
    
    # 1. Verify total rows of original train_essays equals public/train + public/test rows
    assert len(train_essays_df) == (len(train_partition_df) + len(public_test_df)), \
        "Row count mismatch: original train_essays.csv rows not equal to sum of train.csv and test.csv rows."
    
    # 2. Verify that public/test.csv and private/test_answer.csv have the same number of rows
    assert len(public_test_df) == len(private_test_answer_df), \
        "Row count mismatch: test.csv and test_answer.csv row counts differ."
    
    # 3. Verify that public/test.csv and sample_submission.csv have the same number of rows
    assert len(public_test_df) == len(sample_submission_df), \
        "Row count mismatch: test.csv and sample_submission.csv row counts differ."
    
    # 4. Verify test_ratio is preserved (allowing a small rounding difference)
    computed_ratio = len(public_test_df) / len(train_essays_df)
    assert abs(computed_ratio - test_ratio) < 0.05, \
        f"Test ratio mismatch: expected {test_ratio}, but got {computed_ratio}"
    
    # 5. Verify headers for public/test.csv (should be: id, prompt_id, text) and no "generated"
    expected_public_test_columns = ["id", "prompt_id", "text"]
    assert list(public_test_df.columns) == expected_public_test_columns, \
        f"public/test.csv header mismatch: expected {expected_public_test_columns}, got {list(public_test_df.columns)}"
    
    # 6. Verify headers for private/test_answer.csv (should be exactly: id,generated)
    expected_private_columns = ["id", "generated"]
    assert list(private_test_answer_df.columns) == expected_private_columns, \
        f"private/test_answer.csv header mismatch: expected {expected_private_columns}, got {list(private_test_answer_df.columns)}"
    
    # 7. Verify that IDs between public/test.csv, private/test_answer.csv, and sample_submission.csv are identical
    ids_public_test = public_test_df["id"].tolist()
    ids_private = private_test_answer_df["id"].tolist()
    ids_sample = sample_submission_df["id"].tolist()
    assert ids_public_test == ids_private == ids_sample, \
        "ID mismatch between test.csv, test_answer.csv and sample_submission.csv."
    
    # 8. Check that the "generated" column in private/test_answer.csv can be converted to float
    try:
        private_test_answer_df["generated"].astype(float)
    except Exception as e:
        raise ValueError("Conversion of 'generated' column to float failed.") from e
    
    # 9. Verify sample_submission.csv header and type of dummy predictions
    with open(public / "sample_submission.csv", "r") as f:
        header_line = f.readline().strip()
    assert header_line == "id,generated", "sample_submission.csv header is not 'id,generated'."
    
    dummy_values = sample_submission_df["generated"].tolist()
    for val in dummy_values:
        try:
            float(val)
        except ValueError:
            raise ValueError("Not all dummy 'generated' values in sample_submission.csv are valid floats.")
    
    # 10. Verify that train_prompts.csv copied without data loss
    copied_train_prompts_df = pd.read_csv(public / "train_prompts.csv")
    assert copied_train_prompts_df.shape == train_prompts_df.shape, \
        "train_prompts.csv shape mismatch after copying."
    pd.testing.assert_frame_equal(copied_train_prompts_df, train_prompts_df, check_dtype=False)