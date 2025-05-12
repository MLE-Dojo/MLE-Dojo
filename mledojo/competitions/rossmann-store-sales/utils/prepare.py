#!/usr/bin/env python3
import os
import shutil
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

def prepare(raw: Path, public: Path, private: Path):
    """
    Prepare data for the rossmann-store-sales competition.
    
    Args:
        raw: Path to raw data directory
        public: Path to public data directory
        private: Path to private data directory
    """
    # Set fixed random seed for reproducibility
    RANDOM_SEED = 42
    np.random.seed(RANDOM_SEED)
    
    # Create destination directories if they don't exist
    public.mkdir(parents=True, exist_ok=True)
    private.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Read the necessary CSV files
    print("Reading original train.csv and test.csv ...")
    train_df = pd.read_csv(raw / "train.csv")
    test_df = pd.read_csv(raw / "test.csv")
    
    # Get counts and compute split ratio
    T = len(train_df)
    P = len(test_df)
    total = T + P
    test_ratio = P / total
    new_test_size = int(round(T * test_ratio))
    new_train_size = T - new_test_size
    
    # Assert that the split sizes add up to T
    assert new_train_size + new_test_size == T, "New train and test sizes do not sum to T."
    
    print(f"Original train.csv rows: {T}")
    print(f"Original test.csv rows: {P}")
    print(f"Splitting train.csv into new training set with {new_train_size} rows and new test set with {new_test_size} rows.")
    
    # Step 2: Shuffle train.csv with a fixed random seed and split it
    train_shuffled = train_df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    new_train_df = train_shuffled.iloc[:new_train_size, :].reset_index(drop=True)
    new_test_df = train_shuffled.iloc[new_train_size:, :].reset_index(drop=True)
    assert len(new_train_df) + len(new_test_df) == T, "Combined split does not equal original train size."
    
    # Step 3: Save the new training set (features and answers) into public/train.csv
    print("Saving new training set to", public / "train.csv")
    new_train_df.to_csv(public / "train.csv", index=False)
    
    # Step 4: Process the new test set
    # Create new test feature file (public/test.csv) by dropping the 'Sales' column
    if "Sales" not in new_test_df.columns:
        raise ValueError("Column 'Sales' not found in train.csv")
    new_test_features_df = new_test_df.drop(columns=["Sales"])
    
    print("Saving new test features file to", public / "test.csv")
    new_test_features_df.to_csv(public / "test.csv", index=False)
    
    # Create the ground truth file for new test set (private/test_answer.csv)
    # Generate an 'Id' column: sequential integers starting at 1
    new_test_answer_df = new_test_df[["Sales"]].copy()
    new_test_answer_df.insert(0, "Id", range(1, new_test_size + 1))
    
    print("Saving new test ground truth file to", private / "test_answer.csv")
    new_test_answer_df.to_csv(private / "test_answer.csv", index=False)
    
    # Assertions for new test set split:
    assert len(new_test_features_df) == len(new_test_answer_df), "Features and ground truth row counts mismatch."
    assert "Id" in new_test_answer_df.columns, "Id column missing in test_answer.csv."
    assert list(new_test_answer_df.columns) == ["Id", "Sales"], "test_answer.csv must contain only 'Id' and 'Sales' columns."
    assert "Sales" not in new_test_features_df.columns, "Sales column should not be present in test features file."
    
    # Step 5: Create new sample_submission.csv in public using the same Id values and dummy Sales answers (0)
    print("Creating new sample_submission.csv ...")
    sample_submission_df = pd.DataFrame({
        "Id": new_test_answer_df["Id"],
        "Sales": np.zeros(new_test_size, dtype=new_test_answer_df["Sales"].dtype)
    })
    sample_submission_df.to_csv(public / "sample_submission.csv", index=False)
    
    # Assertions for sample_submission.csv:
    assert list(sample_submission_df.columns) == ["Id", "Sales"], "sample_submission.csv should have columns 'Id' and 'Sales'."
    assert len(sample_submission_df) == len(new_test_answer_df), "sample_submission.csv row count does not match test_answer.csv."
    assert (sample_submission_df["Id"] == new_test_answer_df["Id"]).all(), "Id values in sample_submission.csv do not match test_answer.csv."
    
    # Step 6: Copy store.csv to public without modifications
    print("Copying store.csv to", public / "store.csv")
    original_store_df = pd.read_csv(raw / "store.csv")
    with tqdm(total=len(original_store_df), desc="Copying store.csv rows") as pbar:
        original_store_df.to_csv(public / "store.csv", index=False)
        pbar.update(len(original_store_df))
    
    # Verification of store.csv copying using pandas
    copied_store_df = pd.read_csv(public / "store.csv")
    assert original_store_df.shape == copied_store_df.shape, "store.csv row/column count mismatch after copying."
    pd.testing.assert_frame_equal(original_store_df, copied_store_df)
    
    # Final Verification: Check that splitting is proper
    combined_rows = len(new_train_df) + len(new_test_df)
    assert combined_rows == T, "Combined rows of public/train.csv and new test set do not equal original train.csv rows."
    print("\nData reorganization completed successfully.")

    # Clean up
    shutil.rmtree(raw)