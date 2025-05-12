#!/usr/bin/env python3
import os
import shutil
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split


def prepare(raw: Path, public: Path, private: Path):
    """
    Prepares data for the StumbleUpon competition by splitting raw data into public and private datasets.
    
    Args:
        raw: Path to the raw data directory
        public: Path to the public output directory
        private: Path to the private output directory
    """
    # Create output directories
    public.mkdir(parents=True, exist_ok=True)
    private.mkdir(parents=True, exist_ok=True)
    
    os.system("rm stumbleupon.zip")
    os.system("unzip raw_content.zip")
    os.system("rm raw_content.zip")
    # Create raw content directories
    train_raw_content_dir = public / "train_raw_content"
    test_raw_content_dir = public / "test_raw_content"
    train_raw_content_dir.mkdir(exist_ok=True)
    test_raw_content_dir.mkdir(exist_ok=True)
    
    # Load train.tsv
    train_df = pd.read_csv(raw / "train.tsv", delimiter="\t")
    raw_content_dir = raw / "raw_content"
    
    # Perform stratified split (70% train, 30% test)
    test_split_ratio = 0.3
    train_split, test_split = train_test_split(
        train_df,
        test_size=test_split_ratio,
        random_state=42,
        stratify=train_df["label"]
    )
    
    # Reset indices
    train_split = train_split.reset_index(drop=True)
    test_split = test_split.reset_index(drop=True)
    
    # Save train and test feature files
    train_split.to_csv(public / "train_features.tsv", sep="\t", index=False)
    test_split.to_csv(public / "test_features.tsv", sep="\t", index=False)
    
    # Copy raw content files
    copied_train = 0
    for urlid in train_split["urlid"]:
        src_file = raw_content_dir / str(urlid)
        dest_file = train_raw_content_dir / str(urlid)
        if src_file.exists():
            shutil.copy2(src_file, dest_file)
            copied_train += 1
    
    copied_test = 0
    for urlid in test_split["urlid"]:
        src_file = raw_content_dir / str(urlid)
        dest_file = test_raw_content_dir / str(urlid)
        if src_file.exists():
            shutil.copy2(src_file, dest_file)
            copied_test += 1
    
    print(f"Copied {copied_train} raw content files to train_raw_content directory.")
    print(f"Copied {copied_test} raw content files to test_raw_content directory.")
    
    # Create sample_submission.csv
    submission_df = pd.DataFrame({
        "urlid": test_split["urlid"],
        "label": 0.0,  # dummy prediction
    })
    submission_df.to_csv(public / "sample_submission.csv", index=False)
    
    # Create test_answer.csv
    answer_df = test_split[["urlid", "label"]].copy()
    answer_df.to_csv(private / "test_answer.csv", index=False)
    
    # Verify data integrity
    _verify_data_integrity(train_df, train_split, test_split, public, private, test_split_ratio, 
                          copied_train, copied_test)
    
    print("Data preparation completed successfully.")
    # Clean up
    shutil.rmtree(raw)


def _verify_data_integrity(original_df, train_df, test_df, public, private, expected_ratio, 
                          copied_train, copied_test):
    """Helper function to verify data integrity with assertions"""
    
    # Total Count Check
    total_split_rows = len(train_df) + len(test_df)
    assert total_split_rows == len(original_df), "Total row count after split does not match original train.tsv."
    
    # Ratio Check
    actual_ratio = len(test_df) / len(original_df)
    assert abs(actual_ratio - expected_ratio) < 0.05, "Test split ratio does not approximate the expected ratio."
    
    # Stratification Check
    orig_pct = original_df["label"].mean()
    train_pct = train_df["label"].mean()
    test_pct = test_df["label"].mean()
    assert abs(orig_pct - train_pct) < 0.05, "Train split label distribution deviates more than 5%."
    assert abs(orig_pct - test_pct) < 0.05, "Test split label distribution deviates more than 5%."
    
    # File Existence Check
    assert (public / "train_features.tsv").exists(), "train_features.tsv not found in public directory."
    assert (public / "test_features.tsv").exists(), "test_features.tsv not found in public directory."
    assert (public / "sample_submission.csv").exists(), "sample_submission.csv not found in public directory."
    assert (private / "test_answer.csv").exists(), "test_answer.csv not found in private directory."
    
    # Raw Content Files Check
    assert copied_train > 0, "No raw content files were copied for training set."
    assert copied_test > 0, "No raw content files were copied for test set."
    
    # Sample Submission Format Check
    sample_sub_df = pd.read_csv(public / "sample_submission.csv")
    assert list(sample_sub_df.columns) == ["urlid", "label"], "sample_submission.csv headers mismatch."
    assert len(sample_sub_df) == len(test_df), "Number of rows in sample_submission.csv does not match test set."
    
    # Test Answer Format Check
    test_answer_df = pd.read_csv(private / "test_answer.csv")
    assert list(test_answer_df.columns) == ["urlid", "label"], "test_answer.csv headers mismatch."
    assert len(test_answer_df) == len(test_df), "Number of rows in test_answer.csv does not match test set."
    assert test_answer_df["label"].isin([0, 1]).all(), "Not all label values in test_answer.csv are 0 or 1."